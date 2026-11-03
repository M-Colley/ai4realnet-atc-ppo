"""
Train one shared policy for both competition tracks with PPO on BlueSky-Batch.

    python -m atc_agent.train --run main --total-steps 300e6
    python -m atc_agent.train --run ft --resume runs/main/best.zip --ma-slots 8 --lr 1e-4

Each worker process runs one BatchedCompetitionSim with ``--sa-slots``
single-agent scenarios and ``--ma-slots`` 10-agent scenarios, so the vectorised
env has workers * (sa_slots + 10 * ma_slots) seats. All seats share one policy
(parameter sharing = self-play in the multi-agent scenarios).

Safety is handled as a constrained problem: the per-category penalty
multipliers (intrusion / restricted area / sector exit) are adapted by dual
ascent in log space towards target violation rates (events per episode).
By default the constraint counts only *avoidable* violations: events already
present at spawn (about 0.1 intrusions per SA episode, from intruders spawned
within 5 NM of the start) are excluded, otherwise the targets are unreachable
and the multipliers sit at --lam-max forever (``--avoidable-constraint 0``
restores the original behaviour, used for runs/main).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from atc_agent.policy import AtcPolicy
from atc_agent.vec_env import BlueSkyBatchVecEnv


def parse_choices(txt):
    out = []
    for part in txt.split(","):
        n, p = part.split(":")
        out.append((int(n), float(p)))
    return tuple(out)


class CompetitionCallback(BaseCallback):
    """Logs competition metrics, adapts safety multipliers, anneals entropy,
    runs the fast held-out evaluation and keeps checkpoints."""

    LAMS = {"lam_intr": "intrusion_events", "lam_ra": "restricted_area_events",
            "lam_sec": "sector_exit_events"}

    def __init__(self, args, out_dir):
        super().__init__(verbose=0)
        self.a = args
        self.out = out_dir
        self.buf = []
        self.lam = {k: 1.0 for k in self.LAMS}
        self.best = -math.inf
        self.next_eval = args.eval_every
        self.next_save = args.save_every
        self.t0 = time.time()
        self.evaluator = None
        state = os.path.join(out_dir, "train_state.json")
        if args.resume and os.path.exists(os.path.join(os.path.dirname(args.resume), "train_state.json")):
            state = os.path.join(os.path.dirname(args.resume), "train_state.json")
        if os.path.exists(state) and not args.reset_lambda:
            with open(state) as f:
                st = json.load(f)
            self.lam.update(st.get("lam", {}))

    def _on_training_start(self):
        self.t_start = self.num_timesteps
        # count from where this run starts (a resumed model is already past 1e9 steps)
        self.next_eval = self.t_start + self.a.eval_every
        self.next_save = self.t_start + self.a.save_every
        self.training_env.set_reward_params(self.lam)
        self.jsonl = open(os.path.join(self.out, "progress.jsonl"), "a")
        if self.a.eval_every > 0:
            from atc_agent.fast_eval import FastEvaluator
            self.evaluator = FastEvaluator.heldout(n_sa=self.a.eval_sa, n_ma=self.a.eval_ma)

    def _on_rollout_start(self):
        frac = min(1.0, (self.num_timesteps - self.t_start) / max(1.0, self.a.total_steps))
        self.model.ent_coef = self.a.ent_start + (self.a.ent_end - self.a.ent_start) * frac

    def _on_step(self):
        for info in self.locals["infos"]:
            if "metrics" in info:
                m = dict(info["metrics"])
                m["mode"], m["orig"], m["nint"] = info["mode"], info["orig"], info["n_intruders"]
                m["timeout"] = float(info.get("timeout", False))
                spawn = info.get("spawn", {})
                for k in self.LAMS.values():
                    m[k + "_avoidable"] = m[k] - spawn.get(k, 0.0)
                self.buf.append(m)
        return True

    def _group_stats(self):
        groups = {
            "sa10": [m for m in self.buf if m["mode"] == "sa" and m["nint"] == 10],
            "sa_all": [m for m in self.buf if m["mode"] == "sa"],
            "ma_orig": [m for m in self.buf if m["mode"] == "ma" and m["orig"]],
            "ma_all": [m for m in self.buf if m["mode"] == "ma"],
        }
        keys = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
                "restricted_area_events", "time_in_restricted_area", "sector_exit_events",
                "time_outside_sector", "total_reward", "intrusion_events_avoidable", "timeout"]
        stats = {}
        for g, recs in groups.items():
            if recs:
                stats[g] = {k: float(np.mean([r[k] for r in recs])) for k in keys}
                stats[g]["n"] = len(recs)
        return stats, groups

    def _on_rollout_end(self):
        stats, groups = self._group_stats()
        for g, st in stats.items():
            for k, v in st.items():
                self.logger.record(f"{g}/{k}", v)

        # dual ascent on the safety multipliers (log space, bounded)
        recs = groups["sa_all"] + groups["ma_all"]
        if self.a.adapt and recs and self.num_timesteps - self.t_start >= self.a.adapt_warmup:
            for lam, key in self.LAMS.items():
                if self.a.avoidable_constraint:
                    key += "_avoidable"
                rate = float(np.mean([r[key] for r in recs]))
                target = getattr(self.a, f"target_{lam[4:]}")
                logl = math.log(self.lam[lam]) + self.a.adapt_lr * (rate - target)
                self.lam[lam] = float(np.clip(math.exp(logl), self.a.lam_min, self.a.lam_max))
            self.training_env.set_reward_params(self.lam)
        for k, v in self.lam.items():
            self.logger.record(f"safety/{k}", v)
        self.logger.record("train/ent_coef_now", self.model.ent_coef)
        fps = self.num_timesteps / max(1e-6, time.time() - self.t0)
        self.logger.record("time/steps_per_s_total", fps)

        row = {"t": self.num_timesteps, "wall": time.time() - self.t0, "lam": dict(self.lam), **stats}

        if self.evaluator is not None and self.num_timesteps >= self.next_eval:
            self.next_eval += self.a.eval_every
            from atc_agent.fast_eval import score

            def predict(o):
                return self.model.predict(o, deterministic=True)[0]
            res = self.evaluator.evaluate(predict)
            sc = np.mean([score(res[k]) for k in ("sa", "ma") if res.get(k)])
            for mode, st in res.items():
                for k, v in st.items():
                    self.logger.record(f"eval_{mode}/{k}", v)
            self.logger.record("eval/score", sc)
            row["eval"] = res
            row["eval_score"] = float(sc)
            if sc > self.best:
                self.best = sc
                self.model.save(os.path.join(self.out, "best.zip"))
                self._save_state("best_state.json", sc)
            print(f"[eval @ {self.num_timesteps/1e6:.1f}M] score {sc:.2f} (best {self.best:.2f}) "
                  f"SA {res.get('sa')}  MA {res.get('ma')}", flush=True)

        if self.num_timesteps >= self.next_save:
            self.next_save += self.a.save_every
            self.model.save(os.path.join(self.out, "checkpoints", f"step_{self.num_timesteps // 1_000_000:05d}M.zip"))
        self.model.save(os.path.join(self.out, "latest.zip"))
        self._save_state("train_state.json")
        self.jsonl.write(json.dumps(row) + "\n")
        self.jsonl.flush()
        self.buf = []

    def _save_state(self, name, sc=None):
        with open(os.path.join(self.out, name), "w") as f:
            json.dump({"lam": self.lam, "timesteps": self.num_timesteps, "score": sc, "best": self.best}, f)

    def _on_training_end(self):
        self.jsonl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="main", help="run name (output in runs/<run>)")
    p.add_argument("--total-steps", type=float, default=300e6)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--sa-slots", type=int, default=48, help="single-agent scenarios per worker")
    p.add_argument("--ma-slots", type=int, default=4, help="10-agent scenarios per worker")
    p.add_argument("--sa-intruders", default="10:0.6,12:0.2,14:0.2",
                   help="n_intruders:prob list for SA training scenarios")
    p.add_argument("--action-frequency", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=3e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-start", type=float, default=0.01)
    p.add_argument("--ent-end", type=float, default=0.002)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--share-trunk", action="store_true",
                   help="one transformer for actor and critic (~2x faster updates)")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--adapt", type=int, default=1, help="adaptive safety multipliers on/off")
    p.add_argument("--adapt-warmup", type=float, default=5e6)
    p.add_argument("--adapt-lr", type=float, default=0.5)
    p.add_argument("--target-intr", type=float, default=0.02, help="target intrusion events / episode")
    p.add_argument("--target-ra", type=float, default=0.01)
    p.add_argument("--target-sec", type=float, default=0.005)
    p.add_argument("--lam-min", type=float, default=1.0)
    p.add_argument("--lam-max", type=float, default=8.0)
    p.add_argument("--reset-lambda", action="store_true")
    p.add_argument("--avoidable-constraint", type=int, default=1,
                   help="safety constraint on avoidable violations only (excludes those present at spawn)")
    p.add_argument("--timeout-penalty", type=float, default=20.0,
                   help="penalty when the 3000 s budget runs out (terminal, not bootstrapped)")
    p.add_argument("--geodesic-progress", type=int, default=1,
                   help="progress potential on the restricted-area-free route length (0: straight line)")
    p.add_argument("--eval-every", type=float, default=10e6)
    p.add_argument("--eval-sa", type=int, default=200)
    p.add_argument("--eval-ma", type=int, default=20)
    p.add_argument("--save-every", type=float, default=25e6)
    p.add_argument("--resume", default=None, help="model .zip to continue from")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    args.total_steps = int(args.total_steps)
    args.eval_every = int(args.eval_every)
    args.save_every = int(args.save_every)

    out = os.path.join("runs", args.run)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    # argument validation in torch.distributions forces GPU syncs in every minibatch (~15-25% of the update)
    th.distributions.Distribution.set_default_validate_args(False)
    sa_choices = parse_choices(args.sa_intruders)
    cfgs = [dict(sa_slots=args.sa_slots, ma_slots=args.ma_slots, sa_intruders=sa_choices,
                 action_frequency=args.action_frequency, seed=args.seed * 100_003 + 7919 * i,
                 reward=dict(timeout=args.timeout_penalty, geodesic=float(args.geodesic_progress)))
            for i in range(args.workers)]
    print(f"starting {args.workers} BlueSky workers ...", flush=True)
    venv = BlueSkyBatchVecEnv(cfgs)
    print(f"{venv.num_envs} seats ({int(venv.seat_is_ma.sum())} multi-agent), "
          f"rollout = {venv.num_envs * args.n_steps} samples", flush=True)

    def lr_schedule(progress_remaining):
        return args.lr_final + (args.lr - args.lr_final) * progress_remaining

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device,
                         custom_objects={"learning_rate": lr_schedule, "lr_schedule": lr_schedule,
                                         "n_steps": args.n_steps, "batch_size": args.batch_size,
                                         "n_epochs": args.epochs, "clip_range": lambda _: args.clip,
                                         "target_kl": args.target_kl, "gamma": args.gamma,
                                         "gae_lambda": args.gae_lambda})
    else:
        model = PPO(
            AtcPolicy, venv,
            n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.epochs,
            learning_rate=lr_schedule, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_range=args.clip, ent_coef=args.ent_start, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
            policy_kwargs=dict(share_features_extractor=args.share_trunk,
                               features_extractor_kwargs=dict(
                                   d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                                   amp=not args.no_amp)),
            device=args.device, seed=args.seed, verbose=0)
    fmt = ["stdout", "csv"]
    try:
        import tensorboard  # noqa: F401
        fmt.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(out, fmt))
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"policy parameters: {n_params/1e6:.2f}M on {model.device}", flush=True)

    cb = CompetitionCallback(args, out)
    try:
        model.learn(total_timesteps=args.total_steps, callback=cb,
                    reset_num_timesteps=not bool(args.resume), progress_bar=False)
    finally:
        model.save(os.path.join(out, "final.zip"))
        venv.close()


if __name__ == "__main__":
    main()

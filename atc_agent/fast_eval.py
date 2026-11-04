"""
Fast, exact policy evaluation in BlueSky-Batch.

* ``FastEvaluator.heldout()``  -- fixed held-out scenarios (seed 2024, NOT the
  reporting seed) used for checkpoint selection during training.
* ``FastEvaluator.official(n)`` -- the exact scenario stream of the official
  protocol: gymnasium ``seeding.np_random(42)`` on the first reset, then the RNG
  continues (identical to ``scripts/evaluate_competition.py``). Because
  BlueSky-Batch is step-for-step equivalent to the official envs
  (tests/test_equivalence.py), this reproduces the official numbers, but
  hundreds of scenarios are flown simultaneously.

    python -m atc_agent.fast_eval --model runs/main/best.zip --env both
"""

import argparse
import contextlib
import csv
import io
import os

import numpy as np

from atc_agent.batched_env import METRIC_KEYS

SCORED = [k for k in METRIC_KEYS if k != "total_reward"]


def summarize(records):
    if not records:
        return {}
    out = {k: float(np.mean([r[k] for r in records])) for k in SCORED}
    out["n"] = len(records)
    return out


def score(summary):
    """Scalar used only for checkpoint selection: completion first, then safety
    (one violation event ~ 1000 s of flight time), then efficiency."""
    if not summary:
        return -np.inf
    return (100.0 * summary["waypoint_reached"]
            - 10.0 * (summary["intrusion_events"] + summary["restricted_area_events"]
                      + summary["sector_exit_events"])
            - 0.01 * summary["flight_time"])


def _generators(n_intruders=10, n_agents=10, n_obstacles=5):
    from core.scenario import ScenarioGenerator
    return (ScenarioGenerator(n_obstacles=n_obstacles, n_intruders=n_intruders, agent_speed=150.0),
            ScenarioGenerator(n_obstacles=n_obstacles, n_intruders=0, n_agents=n_agents, agent_speed=150.0))


class FastEvaluator:
    def __init__(self, sa_scenarios, ma_scenarios, n_agents=10):
        self.sa = list(sa_scenarios)
        self.ma = list(ma_scenarios)
        self.n_agents = n_agents

    @classmethod
    def heldout(cls, n_sa=200, n_ma=20, seed=2024):
        g_sa, g_ma = _generators()
        rng = np.random.default_rng(seed)
        return cls([g_sa.generate(rng) for _ in range(n_sa)], [g_ma.generate(rng) for _ in range(n_ma)])

    @classmethod
    def official(cls, n_sa=1000, n_ma=1000, seed=42):
        from gymnasium.utils import seeding
        g_sa, g_ma = _generators()
        rng, _ = seeding.np_random(seed)
        sa = [g_sa.generate(rng) for _ in range(n_sa)]
        rng, _ = seeding.np_random(seed)
        ma = [g_ma.generate(rng) for _ in range(n_ma)]
        return cls(sa, ma)

    def run(self, predict, chunk=250):
        """predict(obs (B, D)) -> actions (B, 2). Returns per-episode records
        {'sa': [...], 'ma': [...]} in scenario order (MA: agent order)."""
        recs = {"sa": [], "ma": []}
        for kind, scens in (("sa", self.sa), ("ma", self.ma)):
            step = chunk if kind == "sa" else max(1, chunk // self.n_agents)
            for i in range(0, len(scens), step):
                recs[kind] += self._run_chunk(kind, scens[i:i + step], predict)
        return recs

    def _run_chunk(self, kind, scens, predict):
        with contextlib.redirect_stdout(io.StringIO()):
            from atc_agent.batched_env import BatchedCompetitionSim
            sim = BatchedCompetitionSim(sa_slots=len(scens) if kind == "sa" else 0,
                                        ma_slots=len(scens) if kind == "ma" else 0,
                                        ma_agents=self.n_agents, respawn=False, auto_reset=False)
        sim.forced = {k: [s] for k, s in enumerate(scens)}
        obs = sim.reset()
        per_seat = getattr(predict, "per_seat", False)   # stateful decoders see seat ids
        if per_seat:
            predict.reset(sim.n_seats)
        out = {}
        live = np.ones(sim.n_seats, dtype=bool)
        while live.any():
            act = np.zeros((sim.n_seats, 2), dtype=np.int64)
            idx = np.flatnonzero(sim.active)
            if len(idx):
                act[idx] = predict(obs[idx], idx) if per_seat else predict(obs[idx])
            obs, _, _, _, infos = sim.step(act)
            for seat, info in infos.items():
                if live[seat]:
                    out[seat] = info["metrics"]
                    live[seat] = False
        return [out[s] for s in range(sim.n_seats)]

    def evaluate(self, predict, chunk=250):
        recs = self.run(predict, chunk)
        return {k: summarize(v) for k, v in recs.items()}


class StickyDecoder:
    """Greedy decoding with hysteresis, a diagnostic for indecision.

    Adds ``side`` to the log-probability of every heading bin that does not
    reverse the previous decision's side of the goal bearing (centre counts as
    either side) and ``speed`` to every speed bin within one bin of the previous
    speed, then takes the argmax. Both 0 is exactly ``model.predict(...,
    deterministic=True)``; larger values only change decisions whose top-two
    margin is smaller than the bonus."""

    per_seat = True

    def __init__(self, model, side=0.0, speed=0.0):
        import torch as th
        from atc_agent.actions import HEADING_OFFSETS, SPEED_TARGETS
        self.th, self.model = th, model
        sgn = np.sign(HEADING_OFFSETS)
        self.keep_side = th.as_tensor(sgn[:, None] * sgn[None, :] >= 0)               # [prev, new]
        sp = np.arange(len(SPEED_TARGETS))
        self.keep_speed = th.as_tensor(np.abs(sp[:, None] - sp[None, :]) <= 1)
        self.side, self.speed = float(side), float(speed)

    def reset(self, n_seats):
        self.prev = np.full((n_seats, 2), -1, dtype=np.int64)

    def __call__(self, obs, seats):
        th = self.th
        policy = self.model.policy
        with th.no_grad():
            dists = policy.get_distribution(policy.obs_to_tensor(np.asarray(obs, dtype=np.float32))[0]).distribution
            prev = th.as_tensor(self.prev[seats])
            has = (prev[:, 0] >= 0).unsqueeze(1)
            out = []
            for d, keep, bonus, p in ((dists[0], self.keep_side, self.side, prev[:, 0]),
                                      (dists[1], self.keep_speed, self.speed, prev[:, 1])):
                w = d.probs.cpu()
                if bonus:
                    w = w * th.exp(bonus * (keep[p.clamp(min=0)] & has).to(w.dtype))
                out.append(th.argmax(w, dim=1))     # same op as SB3's mode()
        a = th.stack(out, dim=1).numpy()
        self.prev[seats] = a
        return a


ROWS = [
    ("Goal completion rate [%]", "waypoint_reached", 100.0),
    ("Flight time [s]", "flight_time", 1.0),
    ("Intrusion events", "intrusion_events", 1.0),
    ("Intrusion time [s]", "intrusion_time", 1.0),
    ("Restricted-area events", "restricted_area_events", 1.0),
    ("Restricted-area time [s]", "time_in_restricted_area", 1.0),
    ("Sector-exit events", "sector_exit_events", 1.0),
    ("Time outside sector [s]", "time_outside_sector", 1.0),
]


def print_table(records, label, seed):
    print(f"\n### {label} - mean over {len(records)} {'episodes' if label.startswith('Single') else 'agent-episodes'} (seed {seed})\n")
    print("| Metric | Mean | Std |")
    print("| --- | ---: | ---: |")
    for name, key, f in ROWS:
        v = np.array([r[key] for r in records], dtype=float) * f
        print(f"| {name} | {v.mean():.3f} | {v.std():.3f} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--env", choices=["sa", "ma", "both"], default="both")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--out", default=None, help="CSV prefix")
    p.add_argument("--sticky-side", type=float, default=0.0, help="StickyDecoder heading-side bonus (log-prob)")
    p.add_argument("--sticky-speed", type=float, default=0.0, help="StickyDecoder speed bonus (log-prob)")
    args = p.parse_args()

    from atc_agent.inference import load_model
    model = load_model(args.model, device=args.device)

    def predict(o):
        return model.predict(o, deterministic=not args.stochastic)[0]

    if args.sticky_side or args.sticky_speed:
        assert not args.stochastic
        predict = StickyDecoder(model, args.sticky_side, args.sticky_speed)

    n_sa = args.episodes if args.env in ("sa", "both") else 0
    n_ma = args.episodes if args.env in ("ma", "both") else 0
    ev = FastEvaluator.official(n_sa, n_ma, seed=args.seed)
    recs = ev.run(predict, chunk=args.chunk)
    for kind, label in (("sa", "Single-agent"), ("ma", "Multi-agent")):
        if recs[kind]:
            print_table(recs[kind], label, args.seed)
            if args.out:
                path = f"{args.out}_{kind}.csv"
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["episode_index"] + SCORED)
                    w.writeheader()
                    for i, r in enumerate(recs[kind]):
                        w.writerow({"episode_index": i, **{k: r[k] for k in SCORED}})
                print(f"wrote {path}")


if __name__ == "__main__":
    main()

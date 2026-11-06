"""
One command per training-seed repeat: train, then evaluate on the official
seed-42 stream and compare (paired) with the submitted policy.

Recipes
-------
finetune (default, ~10 h of training on a 28-core + GPU PC, ~8.5k samples/s)
    The stage that produced the submitted policy: geodesic-progress fine-tune
    of the time-aware 1.07B-step checkpoint (submission/geo_init, including its
    safety multipliers), 300M steps, identical hyperparameters, only the seed
    differs. The submitted policy is this stage after 115.1M steps (an
    interruption split it into three resumes with seeds 1, 2, 3); the repeat
    runs uninterrupted and is evaluated at the same step and at the end.

scratch (~45 h)
    The method from scratch with the final code: stage 1 = 1.0B steps with the
    main-run hyperparameters (time-aware observation, timeout penalty,
    geodesic progress and avoidable-violation constraint on from step 0), then
    stage 2 = the fine-tune above from stage 1's final model. NOT the exact
    lineage of the submitted policy (which added these features in stages);
    it tests the method, not the checkpoint.

    python -m scripts.seed_repeat --seed 11
    python -m scripts.seed_repeat --seed 12 --recipe scratch
    python -m scripts.seed_repeat --seed 11 --eval-only      # (re-)evaluate a finished run

Outputs in runs/rep_<recipe>_s<seed>/ (stage 2): eval/fast_{matched,final}_{sa,ma}.csv
and repeat_summary.json. Needs ~40 GiB RAM with the default 26 workers; the
rollout size (26 workers x 88 seats x 128 steps) is kept identical to the
original run, so keep --workers 26 even on fewer cores (it is just slower).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_INIT = "submission/geo_init/init.zip"
GEO_INIT_STEPS = 1_070_125_056      # num_timesteps stored in GEO_INIT
MATCHED_STEPS = 1_185_220_608       # num_timesteps of submission/policy.zip
REFERENCE = "submission/results/fast"

COMMON = ["--sa-slots", "48", "--ma-slots", "4", "--sa-intruders", "10:0.6,12:0.2,14:0.2",
          "--action-frequency", "10", "--n-steps", "128", "--batch-size", "4096", "--epochs", "4",
          "--gamma", "0.99", "--gae-lambda", "0.95", "--clip", "0.2", "--vf-coef", "0.5",
          "--max-grad-norm", "0.5", "--target-kl", "0.05", "--d-model", "128", "--layers", "2", "--heads", "4",
          "--adapt", "1", "--adapt-lr", "0.5", "--target-intr", "0.02", "--target-ra", "0.01",
          "--target-sec", "0.005", "--lam-min", "1.0", "--lam-max", "8.0", "--avoidable-constraint", "1",
          "--timeout-penalty", "20.0", "--geodesic-progress", "1", "--eval-sa", "200", "--eval-ma", "20"]
# runs/main (stage 1): defaults of atc_agent.train at the time
STAGE1 = ["--total-steps", "1e9", "--lr", "3e-4", "--lr-final", "3e-5", "--ent-start", "0.01",
          "--ent-end", "0.002", "--adapt-warmup", "5e6", "--eval-every", "1e7", "--save-every", "2.5e7"]
# runs/ft_geo (the stage the submitted policy comes from); checkpoints every 5M for the matched step
FINETUNE = ["--total-steps", "3e8", "--lr", "1e-4", "--lr-final", "1e-5", "--ent-start", "0.003",
            "--ent-end", "0.001", "--adapt-warmup", "0", "--eval-every", "1e7", "--save-every", "5e6"]
SMOKE = ["--total-steps", "5e4", "--sa-slots", "4", "--ma-slots", "1", "--eval-every", "1e12", "--save-every", "2e4"]


def run(cmd, log=None):
    print("+", " ".join(cmd), flush=True)
    if log:
        with open(log, "a") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, cwd=ROOT, check=True)


def arg(args, name):
    return args[len(args) - 1 - args[::-1].index(name) + 1]       # last occurrence wins, as in argparse


def kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def train(name, args, extra, a, resume=None, stop_at=None):
    """Train stage ``name``. If the run dies (e.g. host out of memory), it is resumed
    from its latest checkpoint with the remaining steps, the LR / entropy schedules
    continued and a new seed (up to --tries attempts). With ``stop_at`` the run is
    stopped once a checkpoint within 3M steps of that step exists (schedules unchanged)."""
    out = os.path.join(ROOT, "runs", name)
    os.makedirs(out, exist_ok=True)
    args = args + extra
    total = int(float(arg(args, "--total-steps")))
    ent0, ent1 = float(arg(args, "--ent-start")), float(arg(args, "--ent-end"))
    spec_path = os.path.join(out, "stage.json")
    if not os.path.exists(spec_path):
        t0 = 0
        if resume:
            import zipfile
            with zipfile.ZipFile(os.path.join(ROOT, resume)) as z:
                t0 = int(json.loads(z.read("data"))["num_timesteps"])
        with open(spec_path, "w") as f:
            json.dump(dict(t0=t0, t_end=t0 + total, resume=resume), f)
    with open(spec_path) as f:
        spec = json.load(f)
    for attempt in range(a.tries):
        state = os.path.join(out, "train_state.json")
        t = spec["t0"]
        if os.path.exists(os.path.join(out, "latest.zip")) and os.path.exists(state):
            with open(state) as f:
                t = int(json.load(f)["timesteps"])
        if (stop_at and matched_checkpoint(out, stop_at)) or spec["t_end"] - t < 1_000_000:
            return
        start = "runs/" + name + "/latest.zip" if t > spec["t0"] else spec["resume"]
        frac = (t - spec["t0"]) / (spec["t_end"] - spec["t0"])
        sched = ["--total-steps", str(spec["t_end"] - t), "--ent-start", f"{ent0 + (ent1 - ent0) * frac:.6g}"]
        cmd = [sys.executable, "-u", "-m", "atc_agent.train", "--run", name, "--seed", str(a.seed + 1000 * attempt),
               "--workers", str(a.workers), "--device", a.device] + COMMON + args + sched
        if start:
            cmd += ["--resume", start]
        print("+", " ".join(cmd), flush=True)
        with open(os.path.join(out, "train.log"), "a") as log:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=os.name != "nt")
            while p.poll() is None:
                if stop_at and matched_checkpoint(out, stop_at):
                    time.sleep(30)                      # let the checkpoint finish writing
                    kill_tree(p.pid)
                    p.wait()
                    print(f"stopped at the matched checkpoint {matched_checkpoint(out, stop_at)}", flush=True)
                    return
                time.sleep(60)
        if p.returncode == 0:
            return
        print(f"training exited with {p.returncode}; resuming from the latest checkpoint", flush=True)
        time.sleep(5 if a.smoke else 300)
    raise SystemExit(f"{name}: training failed {a.tries} times")


def matched_checkpoint(run_dir, target):
    ck = glob.glob(os.path.join(run_dir, "checkpoints", "step_*M.zip"))
    if not ck:
        return None
    step = lambda p: int(re.search(r"step_(\d+)M", p).group(1)) * 1_000_000    # noqa: E731
    best = min(ck, key=lambda p: abs(step(p) - target))
    return best if abs(step(best) - target) <= 3_000_000 else None


def evaluate(run_dir, a, target=MATCHED_STEPS):
    out = {}
    models = {"matched": matched_checkpoint(run_dir, target)}
    if not a.until_matched:
        models["final"] = os.path.join(run_dir, "final.zip")
    ev = os.path.join(run_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    for tag, model in models.items():
        if not model or not os.path.exists(model):
            print(f"[{tag}] no checkpoint, skipped", flush=True)
            continue
        prefix = os.path.join(ev, f"fast_{tag}")
        run([sys.executable, "-u", "-m", "atc_agent.fast_eval", "--model", model, "--env", "both",
             "--episodes", str(a.episodes), "--out", prefix], prefix + ".log")
        from atc_agent.paired import compare, summary
        res = dict(model=os.path.relpath(model, ROOT), **summary(prefix))
        if a.episodes == 1000:
            res["vs_submitted"] = compare(os.path.join(ROOT, REFERENCE), prefix)
        out[tag] = res
        print(f"[{tag}] {json.dumps(res)}", flush=True)
    with open(os.path.join(run_dir, "repeat_summary.json"), "w") as f:
        json.dump(dict(seed=a.seed, recipe=a.recipe, reference=REFERENCE, results=out), f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--recipe", choices=["finetune", "scratch"], default="finetune")
    p.add_argument("--workers", type=int, default=26, help="keep 26 for the original rollout size")
    p.add_argument("--device", default="auto")
    p.add_argument("--episodes", type=int, default=1000, help="official-stream scenarios per track for the evaluation")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny pipeline check (minutes), not a repeat")
    p.add_argument("--until-matched", action="store_true",
                   help="stop the fine-tune at the submitted policy's step (115M of 300M; same schedules) "
                        "and evaluate only that checkpoint")
    p.add_argument("--tries", type=int, default=3, help="training attempts per stage (resumed after a crash)")
    a = p.parse_args()
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    tag = f"rep_{a.recipe}_s{a.seed}" + ("_smoke" if a.smoke else "")
    extra = SMOKE if a.smoke else []
    run_dir = os.path.join(ROOT, "runs", tag)
    target = GEO_INIT_STEPS if a.smoke else MATCHED_STEPS     # smoke: stop at the first checkpoint
    if not a.eval_only:
        resume = GEO_INIT
        if a.recipe == "scratch":
            train(tag + "_stage1", STAGE1, extra, a)
            resume = f"runs/{tag}_stage1/final.zip"
        train(tag, FINETUNE, extra, a, resume=resume, stop_at=target if a.until_matched else None)
    evaluate(run_dir, a, target)


if __name__ == "__main__":
    main()

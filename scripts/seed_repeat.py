"""
One command per training-seed repeat: train, then evaluate on the official
seed-42 stream and compare (paired) with the submitted policy.

The submitted policy is the end of two fine-tuning stages:
  ft_geo  geodesic-progress fine-tune of the time-aware 1.07B-step checkpoint
          (submission/geo_init, incl. its safety multipliers); a 300M-step schedule,
          of which the first 115.1M steps were run (-> the parent, 1185.2M steps);
  ft_rev  the rest of that schedule (to 1370.1M) from the parent, adding small
          costs for reversing the turn direction and for speed jumps.

Recipes (times for a 28-core + GPU PC at ~8.5k samples/s; ~2.5x longer at 3.5k)
-------
finetune (default, ~10 h)
    ft_geo from geo_init with a new seed, stopped at the parent's step, then ft_rev
    from that checkpoint. Evaluates the parent-step checkpoint (vs the submitted
    parent's results) and the end (vs the submitted policy). Identical
    hyperparameters; the originals were interrupted and resumed with new seeds
    (ft_geo seeds 1-3, ft_rev seeds 4-5), the repeat runs straight through.
rev (~6 h)
    ft_rev only, from the submitted parent (submission/rev_init) with a new seed:
    the variance of the last stage alone.
scratch (~45 h)
    The method from scratch with the final code: stage 1 = 1.0B steps with the
    main-run hyperparameters (time-aware observation, timeout penalty, geodesic
    progress and avoidable-violation constraint on from step 0), then ft_geo and
    ft_rev as in `finetune`. NOT the exact lineage of the submitted policy (which
    added these features in stages); it tests the method, not the checkpoint.

    python -m scripts.seed_repeat --seed 11
    python -m scripts.seed_repeat --seed 12 --recipe rev
    python -m scripts.seed_repeat --seed 11 --eval-only      # (re-)evaluate a finished run
    python -m scripts.seed_repeat --seed 1 --smoke           # pipeline check, minutes

Outputs in runs/rep_<recipe>_s<seed>/ (ft_geo) and runs/rep_<recipe>_s<seed>_rev/
(ft_rev; for `rev` only the latter, without the suffix): eval/fast_{matched,final}_*.csv
and repeat_summary.json in the first. Needs ~40 GiB RAM with the default 26
workers; the rollout size (26 workers x 88 seats x 128 steps) is kept identical
to the original runs, so keep --workers 26 even on fewer cores (it is just slower).
A crashed stage is resumed from its latest checkpoint (schedules continued).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_INIT = "submission/geo_init/init.zip"
GEO_INIT_STEPS = 1_070_125_056      # num_timesteps stored in GEO_INIT
PARENT_STEPS = 1_185_220_608        # num_timesteps of the parent (start of ft_rev)
PARENT_OFFSET = PARENT_STEPS - GEO_INIT_STEPS
REV_INIT = "submission/rev_init/init.zip"   # the submitted parent, with its safety multipliers
REFERENCE = "submission/results/fast"                  # the submitted policy
PARENT_REFERENCE = "submission/results/parent/fast"    # the submitted parent

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
# ft_geo; checkpoints every 5M so that one lands at the parent's step
FINETUNE = ["--total-steps", "3e8", "--lr", "1e-4", "--lr-final", "1e-5", "--ent-start", "0.003",
            "--ent-end", "0.001", "--adapt-warmup", "0", "--eval-every", "1e7", "--save-every", "5e6"]
# ft_rev = the rest of FINETUNE's schedule plus the indecision costs. The LR follows
# from (step, schedule end) alone (SB3 progress = 1 - t / t_end); the entropy start
# is FINETUNE's value at the start step (set in rev_stage).
REV = ["--lr", "1e-4", "--lr-final", "1e-5", "--ent-end", "0.001", "--adapt-warmup", "0",
       "--reversal-penalty", "0.02", "--speed-jump-penalty", "0.01", "--eval-every", "1e7", "--save-every", "2.5e7"]
SMOKE_SHAPE = ["--sa-slots", "4", "--ma-slots", "1", "--eval-every", "1e12", "--save-every", "2e4"]
SMOKE = ["--total-steps", "5e4"] + SMOKE_SHAPE


def run(cmd, log=None):
    print("+", " ".join(cmd), flush=True)
    if log:
        with open(log, "a") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, cwd=ROOT, check=True)


def arg(args, name):
    return args[len(args) - 1 - args[::-1].index(name) + 1]       # last occurrence wins, as in argparse


def steps_of(zip_path):
    with zipfile.ZipFile(os.path.join(ROOT, zip_path)) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])


def kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def stage_spec(name, total=None, resume=None):
    """{t0, t_end, resume} of a stage, fixed when the stage is first started."""
    out = os.path.join(ROOT, "runs", name)
    path = os.path.join(out, "stage.json")
    if not os.path.exists(path):
        os.makedirs(out, exist_ok=True)
        t0 = steps_of(resume) if resume else 0
        with open(path, "w") as f:
            json.dump(dict(t0=t0, t_end=t0 + total, resume=resume), f)
    with open(path) as f:
        return json.load(f)


def train(name, args, extra, a, resume=None, stop_at=None):
    """Train stage ``name``. If the run dies (e.g. host out of memory), it is resumed
    from its latest checkpoint with the remaining steps, the LR / entropy schedules
    continued and a new seed (up to --tries attempts). With ``stop_at`` the run is
    stopped once a checkpoint within 3M steps of that step exists (schedules unchanged)."""
    out = os.path.join(ROOT, "runs", name)
    args = args + extra
    ent0, ent1 = float(arg(args, "--ent-start")), float(arg(args, "--ent-end"))
    spec = stage_spec(name, int(float(arg(args, "--total-steps"))), resume)
    for attempt in range(a.tries):
        state = os.path.join(out, "train_state.json")
        t = spec["t0"]
        if os.path.exists(os.path.join(out, "latest.zip")) and os.path.exists(state):
            with open(state) as f:
                t = int(json.load(f)["timesteps"])
        # done = within 1M steps of the end (a few rollouts); tiny smoke stages use half their length
        if (stop_at and matched_checkpoint(out, stop_at)) or \
                spec["t_end"] - t < min(1_000_000, (spec["t_end"] - spec["t0"]) // 2):
            return spec
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
                    return spec
                time.sleep(60)
        if p.returncode == 0:
            return spec
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


def rev_init(name, ckpt):
    """Copy the parent-step checkpoint of stage ``name`` to runs/<name>/rev_init/ with the
    safety multipliers it was trained with (progress.jsonl row of its step), so that
    atc_agent.train picks them up on resume, as the original ft_rev did."""
    rel = os.path.relpath(ckpt, ROOT).replace(os.sep, "/")
    t = steps_of(rel)
    lam = None
    with open(os.path.join(ROOT, "runs", name, "progress.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if row["t"] == t:
                lam = row["lam"]
    if lam is None:
        raise SystemExit(f"no progress.jsonl row at step {t} for {rel}")
    d = os.path.join(ROOT, "runs", name, "rev_init")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(ROOT, rel), os.path.join(d, "init.zip"))
    with open(os.path.join(d, "train_state.json"), "w") as f:
        json.dump({"lam": lam, "timesteps": t, "score": None, "best": None, "source": rel}, f)
    return f"runs/{name}/rev_init/init.zip"


def rev_stage(name, init, geo_spec, a):
    """ft_rev from ``init`` to the end of the ft_geo schedule ``geo_spec`` (smoke: 30k steps)."""
    t = steps_of(init)
    frac = min(1.0, (t - geo_spec["t0"]) / (geo_spec["t_end"] - geo_spec["t0"]))
    e0, e1 = float(arg(FINETUNE, "--ent-start")), float(arg(FINETUNE, "--ent-end"))
    total = 30_000 if a.smoke else geo_spec["t_end"] - t
    args = REV + ["--total-steps", str(total), "--ent-start", f"{e0 + (e1 - e0) * frac:.6g}"]
    return train(name, args, SMOKE_SHAPE if a.smoke else [], a, resume=init)


def complete(prefix, episodes):
    """fast_eval output for ``episodes`` scenarios per track already present?"""
    try:
        with open(prefix + "_sa.csv") as f:
            n_sa = sum(1 for _ in f) - 1
        with open(prefix + "_ma.csv") as f:
            n_ma = sum(1 for _ in f) - 1
    except OSError:
        return False
    return n_sa == episodes and n_ma == 10 * episodes


def evaluate(geo_dir, rev_dir, a, target):
    out = {}
    models = {}
    if geo_dir:
        models["matched"] = (matched_checkpoint(geo_dir, target), PARENT_REFERENCE)
    if rev_dir and not a.until_matched:
        models["final"] = (os.path.join(rev_dir, "final.zip"), REFERENCE)
    summary_dir = geo_dir or rev_dir
    ev = os.path.join(summary_dir, "eval")
    os.makedirs(ev, exist_ok=True)
    for tag, (model, ref) in models.items():
        if not model or not os.path.exists(model):
            print(f"[{tag}] no checkpoint, skipped", flush=True)
            continue
        prefix = os.path.join(ev, f"fast_{tag}")
        if not complete(prefix, a.episodes):
            run([sys.executable, "-u", "-m", "atc_agent.fast_eval", "--model", model, "--env", "both",
                 "--episodes", str(a.episodes), "--out", prefix], prefix + ".log")
        from atc_agent.paired import compare, summary
        res = dict(model=os.path.relpath(model, ROOT), steps=steps_of(os.path.relpath(model, ROOT)),
                   **summary(prefix))
        if a.episodes == 1000:
            res["reference"] = ref
            res["vs_reference"] = compare(os.path.join(ROOT, ref), prefix)
        out[tag] = res
        print(f"[{tag}] {json.dumps(res)}", flush=True)
    with open(os.path.join(summary_dir, "repeat_summary.json"), "w") as f:
        json.dump(dict(seed=a.seed, recipe=a.recipe, results=out), f, indent=2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--recipe", choices=["finetune", "rev", "scratch"], default="finetune")
    p.add_argument("--workers", type=int, default=26, help="keep 26 for the original rollout size")
    p.add_argument("--device", default="auto")
    p.add_argument("--episodes", type=int, default=1000, help="official-stream scenarios per track for the evaluation")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny pipeline check (minutes), not a repeat")
    p.add_argument("--until-matched", "--parent-only", dest="until_matched", action="store_true",
                   help="stop after ft_geo at the parent's step (skip ft_rev) and evaluate only that checkpoint")
    p.add_argument("--tries", type=int, default=3, help="training attempts per stage (resumed after a crash)")
    a = p.parse_args()
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)

    tag = f"rep_{a.recipe}_s{a.seed}" + ("_smoke" if a.smoke else "")
    extra = SMOKE if a.smoke else []
    offset = 0 if a.smoke else PARENT_OFFSET          # smoke: stop ft_geo at its first checkpoint
    geo_name = None if a.recipe == "rev" else tag
    rev_name = tag if a.recipe == "rev" else tag + "_rev"
    target = None
    if geo_name:
        resume = GEO_INIT
        if a.recipe == "scratch":
            if not a.eval_only:
                train(tag + "_stage1", STAGE1, extra, a)
            resume = f"runs/{tag}_stage1/final.zip"
        if a.eval_only:
            geo_spec = stage_spec(geo_name)
        else:
            geo_spec = train(geo_name, FINETUNE, extra, a, resume=resume, stop_at=steps_of(resume) + offset)
        target = geo_spec["t0"] + offset
    if not a.eval_only and not a.until_matched:
        if geo_name:
            ck = matched_checkpoint(os.path.join(ROOT, "runs", geo_name), target)
            if not ck:
                raise SystemExit(f"{geo_name}: no checkpoint at the parent's step {target}")
            init = rev_init(geo_name, ck)
        else:
            init = REV_INIT
            geo_spec = dict(t0=GEO_INIT_STEPS, t_end=GEO_INIT_STEPS + int(float(arg(FINETUNE, "--total-steps"))))
        rev_stage(rev_name, init, geo_spec, a)
    evaluate(os.path.join(ROOT, "runs", geo_name) if geo_name else None,
             os.path.join(ROOT, "runs", rev_name), a, target)


if __name__ == "__main__":
    main()

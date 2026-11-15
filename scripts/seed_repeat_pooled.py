"""Seed repeats pooled over evaluation streams, and the last stage averaged over training runs.

``scripts.seed_repeat`` evaluates each repeat on the official seed-42 stream. This script adds
further streams: it evaluates the submitted policy, its parent and each repeat's two
checkpoints with the deterministic BlueSky-Batch evaluator (skipped when the CSVs exist) and
reports paired differences pooled over the streams (one unit = one SA episode or one MA
scenario; 1000 per track and stream):

* per repeat: its end against the submitted policy, its parent-step checkpoint against the
  submitted parent, and its own last stage (end against its parent-step checkpoint);
* the last stage (ft_rev, with the indecision costs) averaged over all runs, i.e. the
  submitted lineage and the repeats, and the repeats' ends averaged against the submitted
  policy. Per-unit differences are averaged across runs, so the BCa bootstrap resamples
  evaluation units with all runs of a unit kept together; the interval is conditional on
  these runs. A one-sample t-test over the run means treats the training run as the unit.

    python -m scripts.seed_repeat_pooled --seeds 11 12 13               # streams 42 and 777
    python -m scripts.seed_repeat_pooled --seeds 11 12 13 --streams 42   # no new evaluations

Missing evaluations take ~20 min each (set OMP_NUM_THREADS to share the machine).
Writes runs/seed_repeats_pooled.json.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atc_agent.paired import boot_ci, load, per_unit_score  # noqa: E402
from scripts.seed_repeat import PARENT_REFERENCE, REFERENCE, REV_INIT, complete, run  # noqa: E402

POLICY = "submission/policy.zip"
VIOLATIONS = ["intrusion_events", "restricted_area_events", "sector_exit_events"]
METRICS = ("score", "MA failures /1000 scenarios", "MA flight time [s]", "SA flight time [s]",
           "MA violations /aircraft", "SA violations /episode")


def suffix(stream):
    return "" if stream == 42 else f"_s{stream}"


def ensure(model, prefix, stream, episodes):
    """CSV prefix of ``model`` on ``stream``, evaluated first if missing."""
    path = os.path.join(ROOT, prefix)
    if not complete(path, episodes):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        run([sys.executable, "-u", "-m", "atc_agent.fast_eval", "--model", model, "--env", "both",
             "--episodes", str(episodes), "--seed", str(stream), "--out", prefix], path + ".log")
    return path


def unit_diffs(ref, new):
    """Paired per-unit differences new - ref (SA episode i, MA scenario i)."""
    rsa, rma = load(ref)
    sa, ma = load(new)
    assert (sa.episode_index.to_numpy() == rsa.episode_index.to_numpy()).all()
    assert (ma.episode_index.to_numpy() == rma.episode_index.to_numpy()).all()
    assert len(sa) == ma.scenario.nunique(), "the score pairs SA episode i with MA scenario i"

    def scen(df, values):
        return values.groupby(df.scenario).mean().to_numpy()

    return {
        "score": ((per_unit_score(sa) - per_unit_score(rsa)).to_numpy()
                  + scen(ma, per_unit_score(ma)) - scen(rma, per_unit_score(rma))) / 2,
        "MA failures /1000 scenarios": 1000 * ((ma.waypoint_reached < 1).groupby(ma.scenario).sum().to_numpy()
                                               - (rma.waypoint_reached < 1).groupby(rma.scenario).sum().to_numpy()),
        "MA flight time [s]": scen(ma, ma.flight_time) - scen(rma, rma.flight_time),
        "SA flight time [s]": (sa.flight_time - rsa.flight_time).to_numpy(),
        "MA violations /aircraft": scen(ma, ma[VIOLATIONS].sum(axis=1)) - scen(rma, rma[VIOLATIONS].sum(axis=1)),
        "SA violations /episode": (sa[VIOLATIONS].sum(axis=1) - rsa[VIOLATIONS].sum(axis=1)).to_numpy(),
    }


def pooled(pairs):
    """Per-unit differences of (ref, new) prefix pairs, one pair per stream, concatenated."""
    per = [unit_diffs(ref, new) for ref, new in pairs]
    return {k: np.concatenate([p[k] for p in per]) for k in METRICS}


def describe(d):
    lo, hi = boot_ci(d)
    return dict(mean=float(d.mean()), ci=[lo, hi])


def across_runs(runs):
    """Average per-unit differences over runs; bootstrap over units, t-test over runs."""
    out = {}
    for k in METRICS:
        stack = np.stack([r[k] for r in runs])
        means = stack.mean(axis=1)
        res = describe(stack.mean(axis=0))
        t = stats.ttest_1samp(means, 0.0)
        ci = t.confidence_interval()
        res.update(runs=[float(m) for m in means], sd=float(means.std(ddof=1)),
                   t_test=dict(p=float(t.pvalue), ci=[float(ci.low), float(ci.high)]))
        out[k] = res
    return out


def show(title, table):
    print(title)
    for k, r in table.items():
        line = f"  {k:28s} {r['mean']:+8.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
        if "runs" in r:
            line += f"   runs {' '.join(f'{m:+.3f}' for m in r['runs'])}   t-test p={r['t_test']['p']:.3f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="training seeds of finished repeats")
    p.add_argument("--recipe", choices=["finetune", "rev"], default="finetune")
    p.add_argument("--streams", type=int, nargs="+", default=[42, 777], help="evaluation seeds")
    p.add_argument("--episodes", type=int, default=1000)
    a = p.parse_args()
    os.chdir(ROOT)

    ref = {s: ensure(POLICY, REFERENCE + suffix(s), s, a.episodes) for s in a.streams}
    parent = {s: ensure(REV_INIT, PARENT_REFERENCE + suffix(s), s, a.episodes) for s in a.streams}
    report = dict(streams=a.streams, recipe=a.recipe, repeats={})
    stage_runs = [pooled([(parent[s], ref[s]) for s in a.streams])]     # the submitted lineage
    end_runs = []
    for seed in a.seeds:
        tag = f"rep_{a.recipe}_s{seed}"
        with open(os.path.join(ROOT, "runs", tag, "repeat_summary.json")) as f:
            models = {k: v["model"].replace("\\", "/") for k, v in json.load(f)["results"].items()}
        ev = f"runs/{tag}/eval/"
        final = {s: ensure(models["final"], ev + "fast_final" + suffix(s), s, a.episodes) for s in a.streams}
        if "matched" in models:
            matched = {s: ensure(models["matched"], ev + "fast_matched" + suffix(s), s, a.episodes) for s in a.streams}
        else:                                                           # recipe rev starts from the parent
            matched = parent
        cases = {"end vs submitted policy": pooled([(ref[s], final[s]) for s in a.streams]),
                 "own last stage (end vs parent-step)": pooled([(matched[s], final[s]) for s in a.streams])}
        if "matched" in models:
            cases["parent-step vs submitted parent"] = pooled([(parent[s], matched[s]) for s in a.streams])
        end_runs.append(cases["end vs submitted policy"])
        stage_runs.append(cases["own last stage (end vs parent-step)"])
        report["repeats"][seed] = {}
        for name, d in cases.items():
            table = {k: describe(v) for k, v in d.items()}
            report["repeats"][seed][name] = table
            show(f"seed {seed}: {name} (streams {a.streams})", table)
    report["last stage, all runs"] = across_runs(stage_runs)
    show(f"last stage averaged over {len(stage_runs)} runs (submitted + seeds {a.seeds})", report["last stage, all runs"])
    if len(end_runs) > 1:
        report["repeat ends vs submitted"] = across_runs(end_runs)
        show(f"repeat ends vs submitted policy, averaged over {len(end_runs)} runs", report["repeat ends vs submitted"])
    with open(os.path.join(ROOT, "runs", "seed_repeats_pooled.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

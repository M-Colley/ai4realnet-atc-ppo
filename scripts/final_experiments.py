"""Pre-registered final experiments (docs/solution/EXPERIMENTS_2026-11.md): decision rule and
exploratory analyses, from the shipped per-episode CSVs.

Candidates (``submission/results/final_experiments/<name>/fast[_s777|_s9001]_{sa,ma}.csv``) are
paired against the submitted policy (``submission/results/fast[_s777|_s9001]``) per evaluation
unit (SA episode i, MA scenario i), pooled over the streams 42 and 777, with the
scenario-cluster BCa bootstrap of ``scripts.seed_repeat_pooled``:

1. gain: the MA-failure difference has its whole 95% CI below 0, or the score difference has
   its whole CI above 0 with an MA-failure point estimate <= 0;
2. safety: MA and SA violation differences have point estimates <= 0 or CIs reaching 0, and SA
   failures do not rise significantly (exact McNemar over the pooled discordant pairs, p < 0.05);
3. confirmation on stream 9001: the qualifying effect has the same sign, and the score CI does
   not lie entirely below 0;
4. among qualifying candidates, the larger pooled score difference wins.

Exploratory (not part of the decision): ft_time and ft_hard against the equal-compute control
ft_cont, and MA failures split by whether the aircraft's goal lies inside a restricted area
(scenarios regenerated with the official generator; ``--no-split`` skips this, ~1 min).

    python -m scripts.final_experiments
    python -m scripts.final_experiments --build-pool runs/pool/pool_hard_v2.pkl   # test 4's pool

Writes runs/final_experiments.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atc_agent.paired import load  # noqa: E402
from scripts.seed_repeat_pooled import METRICS, describe, pooled  # noqa: E402

RESULTS = os.path.join(ROOT, "submission", "results")
CANDIDATES = ("soup4", "ens4", "ft_hard", "ft_time", "ft_cont")
CONTROL = "ft_cont"
POOLED, CONFIRM = (42, 777), 9001
MA_FAIL, SCORE = "MA failures /1000 scenarios", "score"
VIOLATIONS = ("MA violations /aircraft", "SA violations /episode")


def suffix(stream):
    return "" if stream == 42 else f"_s{stream}"


def prefix(run, stream):
    base = os.path.join(RESULTS, "fast") if run == "submitted" else \
        os.path.join(RESULTS, "final_experiments", run, "fast")
    return base + suffix(stream)


def effects(ref, new, streams):
    """{metric: dict(mean, ci)} of ``new - ref`` pooled over ``streams``."""
    d = pooled([(prefix(ref, s), prefix(new, s)) for s in streams])
    return {k: describe(d[k]) for k in METRICS}


def sa_mcnemar(ref, new, streams):
    """Exact McNemar test on SA failures over the pooled discordant pairs."""
    new_fail = fixed = 0
    for s in streams:
        a = (load(prefix(ref, s))[0].waypoint_reached < 1).to_numpy()
        b = (load(prefix(new, s))[0].waypoint_reached < 1).to_numpy()
        new_fail, fixed = new_fail + int((~a & b).sum()), fixed + int((a & ~b).sum())
    p = float(stats.binomtest(new_fail, new_fail + fixed).pvalue) if new_fail + fixed else 1.0
    return dict(new=new_fail, fixed=fixed, p=p)


def decide(e, mc, confirm=None):
    """Rules 1-3 for one candidate; ``e`` / ``confirm``: effects on the pooled / confirmation streams."""
    via_ma = e[MA_FAIL]["ci"][1] < 0
    via_score = e[SCORE]["ci"][0] > 0 and e[MA_FAIL]["mean"] <= 0
    r1 = via_ma or via_score
    r2 = (all(e[k]["mean"] <= 0 or e[k]["ci"][0] <= 0 for k in VIOLATIONS)
          and not (mc["p"] < 0.05 and mc["new"] > mc["fixed"]))
    r3 = None
    if r1 and r2 and confirm is not None:
        key = MA_FAIL if via_ma else SCORE
        same_sign = confirm[key]["mean"] < 0 if via_ma else confirm[key]["mean"] > 0
        r3 = bool(same_sign and confirm[SCORE]["ci"][1] >= 0)
    return dict(rule1=bool(r1), rule2=bool(r2), rule3=r3, qualifies=bool(r1 and r2 and r3))


def goal_in_area(streams):
    """Per MA aircraft of the official streams: is its goal inside a restricted area?"""
    from atc_agent.fast_eval import FastEvaluator
    from atc_agent.mine_hard import goal_depths
    return np.concatenate([[d > 0 for sc in FastEvaluator.official(0, 1000, seed=s).ma for d in goal_depths(sc)]
                           for s in streams])


def failure_split(runs, streams):
    """MA failures by goal class, and paired differences vs the submitted policy per class."""
    inside = goal_in_area(streams)
    fails, scen = {}, None
    for run in runs:
        mas = [load(prefix(run, s))[1] for s in streams]
        fail = np.concatenate([(m.waypoint_reached < 1).to_numpy() for m in mas])
        assert len(fail) == len(inside), (run, len(fail), len(inside))
        fails[run] = fail
        scen = np.concatenate([m.scenario.to_numpy() + i * 10**6 for i, m in enumerate(mas)])
    out = dict(aircraft=int(len(inside)), goal_inside=int(inside.sum()), runs={})
    for run, fail in fails.items():
        row = dict(failures=int(fail.sum()), goal_inside=int((fail & inside).sum()),
                   other=int((fail & ~inside).sum()))
        if run != "submitted":
            for name, mask in (("d_goal_inside", inside), ("d_other", ~inside)):
                d = pd.Series((fail & mask).astype(int) - (fails["submitted"] & mask).astype(int))
                per_scenario = 1000.0 * d.groupby(scen).sum().to_numpy()
                # BCa is undefined for constant data (no discordant aircraft at all)
                row[name] = describe(per_scenario) if np.ptp(per_scenario) else \
                    dict(mean=float(per_scenario[0]), ci=[float(per_scenario[0])] * 2)
        out["runs"][run] = row
    return out


def build_hard_pool(path):
    """Test 4's replay pool from the shipped mining rows (seeds 3001-3020, mined with the
    submitted policy; SA rows and MA timeout rows), no goal inside a restricted area."""
    import pickle
    from atc_agent.mine_hard import build_pool
    mined = pd.read_csv(os.path.join(RESULTS, "final_experiments", "ft_hard", "mined_seeds_3001-3020.csv"))
    assert not mined.seed.isin([42, 777, 9001]).any()
    pool = build_pool(mined, [], max_depth=1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pool, f)
    print(path, json.dumps(pool["meta"]))


def fmt(x):
    return f"{x['mean']:+8.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-split", action="store_true", help="skip the goal-inside-area failure split")
    p.add_argument("--build-pool", metavar="PATH", help="only rebuild test 4's replay pool to PATH")
    a = p.parse_args()
    if a.build_pool:
        build_hard_pool(a.build_pool)
        return
    report = dict(candidates={}, vs_control={})
    for run in CANDIDATES:
        e = effects("submitted", run, POOLED)
        mc = sa_mcnemar("submitted", run, POOLED)
        c = effects("submitted", run, (CONFIRM,))
        report["candidates"][run] = dict(effects=e, sa_mcnemar=mc, confirm_9001=c, **decide(e, mc, c))
        print(f"{run} vs submitted (streams 42 + 777 pooled)")
        for k in METRICS:
            print(f"  {k:28s} {fmt(e[k])}")
        r = report["candidates"][run]
        print(f"  SA failures new {mc['new']}, fixed {mc['fixed']}, exact McNemar p = {mc['p']:.3f}")
        print(f"  rule 1 {r['rule1']}, rule 2 {r['rule2']}, rule 3 {r['rule3']} -> qualifies {r['qualifies']}\n")
    ok = [r for r in CANDIDATES if report["candidates"][r]["qualifies"]]
    best = max(ok, key=lambda r: report["candidates"][r]["effects"][SCORE]["mean"]) if ok else None
    report["decision"] = best or "submitted policy stays"
    print("decision:", report["decision"], "\n")

    print("exploratory: against the equal-compute control", CONTROL)
    for run in ("ft_time", "ft_hard"):
        e = effects(CONTROL, run, POOLED)
        report["vs_control"][run] = e
        print(f"{run} vs {CONTROL}")
        for k in METRICS:
            print(f"  {k:28s} {fmt(e[k])}")
    if not a.no_split:
        s = failure_split(("submitted",) + CANDIDATES, POOLED)
        report["failure_split"] = s
        print(f"\nexploratory: MA failures by goal class ({s['aircraft']} aircraft, "
              f"{s['goal_inside']} goals inside a restricted area)")
        for run, row in s["runs"].items():
            extra = "" if run == "submitted" else \
                f"   vs submitted per 1000 scenarios: inside {fmt(row['d_goal_inside'])}, other {fmt(row['d_other'])}"
            print(f"  {run:9s} {row['failures']:4d} = {row['goal_inside']:3d} inside + {row['other']:3d} other{extra}")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "final_experiments.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

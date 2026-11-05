"""
Paired comparison of evaluation runs on the same official scenario stream.

Every model flies the same scenarios, so differences are paired per scenario:
exact McNemar (binomial) test on single-agent failures, and scenario-cluster BCa
bootstrap (scipy) for multi-agent failures and for the selection score.

Inputs are fast_eval / evaluate_competition CSV prefixes (``<prefix>_sa.csv``,
``<prefix>_ma.csv``; 1000 SA episodes, 1000 x 10 MA agent-episodes).

    python -m atc_agent.paired --ref submission/results/fast runs/rep_s11/eval/fast_matched
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from atc_agent.fast_eval import score

KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "restricted_area_events", "sector_exit_events"]
AGENTS = 10


def load(prefix):
    sa = pd.read_csv(prefix + "_sa.csv").sort_values("episode_index").reset_index(drop=True)
    ma = pd.read_csv(prefix + "_ma.csv").sort_values("episode_index").reset_index(drop=True)
    ma["scenario"] = ma["episode_index"] // AGENTS
    return sa, ma


def per_unit_score(df):
    """score() is linear in the means, so the mean of per-unit scores equals score(means)."""
    return (100.0 * df.waypoint_reached
            - 10.0 * (df.intrusion_events + df.restricted_area_events + df.sector_exit_events)
            - 0.01 * df.flight_time)


def boot_ci(d, seed=0):
    r = stats.bootstrap((np.asarray(d, float),), np.mean, n_resamples=10000,
                        confidence_level=0.95, method="BCa", random_state=seed)
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def summary(prefix):
    sa, ma = load(prefix)
    return dict(SA_fail=int((sa.waypoint_reached < 1).sum()), MA_fail=int((ma.waypoint_reached < 1).sum()),
                SA_flight_time=float(sa.flight_time.mean()), MA_flight_time=float(ma.flight_time.mean()),
                SA_violations=float(sa[KEYS[2:]].sum(axis=1).mean()),
                MA_violations=float(ma[KEYS[2:]].sum(axis=1).mean()),
                score=(score({k: sa[k].mean() for k in KEYS}) + score({k: ma[k].mean() for k in KEYS})) / 2)


def compare(ref_prefix, prefix):
    """Paired differences ``prefix - ref_prefix`` (positive MA failures = worse)."""
    ref_sa, ref_ma = load(ref_prefix)
    sa, ma = load(prefix)
    assert (sa.episode_index.to_numpy() == ref_sa.episode_index.to_numpy()).all()
    assert (ma.episode_index.to_numpy() == ref_ma.episode_index.to_numpy()).all()
    a, b = (ref_sa.waypoint_reached < 1).to_numpy(), (sa.waypoint_reached < 1).to_numpy()
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())          # new-only, ref-only failures
    d_ma = ((ma.waypoint_reached < 1).groupby(ma.scenario).sum()
            - (ref_ma.waypoint_reached < 1).groupby(ref_ma.scenario).sum())
    lo, hi = boot_ci(d_ma)
    d = ((per_unit_score(sa) - per_unit_score(ref_sa)).to_numpy()
         + (per_unit_score(ma).groupby(ma.scenario).mean()
            - per_unit_score(ref_ma).groupby(ref_ma.scenario).mean()).to_numpy()) / 2
    slo, shi = boot_ci(d)
    return dict(SA_new_fail=n01, SA_fixed=n10,
                SA_mcnemar_p=float(stats.binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0,
                dMA_fail_per_1000=1000 * float(d_ma.mean()), dMA_fail_ci=(1000 * lo, 1000 * hi),
                dscore=float(d.mean()), dscore_ci=(slo, shi))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="reference CSV prefix")
    p.add_argument("prefixes", nargs="+", help="CSV prefixes to compare against --ref")
    a = p.parse_args()
    rows = [dict(run=a.ref, **summary(a.ref))]
    for pre in a.prefixes:
        c = compare(a.ref, pre)
        rows.append(dict(run=pre, **summary(pre),
                         SA_disc=f"+{c['SA_new_fail']}/-{c['SA_fixed']} (p={c['SA_mcnemar_p']:.2f})",
                         dMA_fail_per_1000=f"{c['dMA_fail_per_1000']:+.0f} [{c['dMA_fail_ci'][0]:+.0f}, "
                                           f"{c['dMA_fail_ci'][1]:+.0f}]",
                         dscore=f"{c['dscore']:+.3f} [{c['dscore_ci'][0]:+.3f}, {c['dscore_ci'][1]:+.3f}]"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

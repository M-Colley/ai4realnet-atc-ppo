"""Paired per-unit differences and the across-run average of scripts.seed_repeat_pooled."""

import numpy as np
import pandas as pd

from atc_agent.paired import AGENTS
from scripts.seed_repeat_pooled import METRICS, across_runs, pooled, unit_diffs


def _write(prefix, sa_reached, sa_time, ma_reached, ma_time, sa_intr=0):
    n_sa, n_ma = len(sa_reached), len(ma_reached)
    zeros = lambda n: np.zeros(n, int)  # noqa: E731
    pd.DataFrame(dict(episode_index=np.arange(n_sa), waypoint_reached=sa_reached, flight_time=sa_time,
                      intrusion_events=np.full(n_sa, sa_intr), restricted_area_events=zeros(n_sa),
                      sector_exit_events=zeros(n_sa))).to_csv(prefix + "_sa.csv", index=False)
    # agents in reverse order: load() sorts by episode_index
    pd.DataFrame(dict(episode_index=np.arange(n_ma)[::-1], waypoint_reached=ma_reached[::-1],
                      flight_time=ma_time[::-1], intrusion_events=zeros(n_ma),
                      restricted_area_events=zeros(n_ma), sector_exit_events=zeros(n_ma))
                 ).to_csv(prefix + "_ma.csv", index=False)


def _pair(tmp, name, ma_fail_new, time_shift):
    """Reference and new run: 2 SA episodes, 2 MA scenarios of AGENTS aircraft."""
    n_ma = 2 * AGENTS
    ref, new = str(tmp / f"{name}_ref"), str(tmp / f"{name}_new")
    _write(ref, np.ones(2), np.full(2, 1000.0), np.ones(n_ma), np.full(n_ma, 900.0))
    ma_new = np.ones(n_ma)
    ma_new[:ma_fail_new] = 0                                  # failures in scenario 0
    _write(new, np.ones(2), np.full(2, 1000.0 + time_shift), ma_new, np.full(n_ma, 900.0 + time_shift), sa_intr=1)
    return ref, new


def test_unit_diffs(tmp_path):
    ref, new = _pair(tmp_path, "a", ma_fail_new=2, time_shift=-5.0)
    d = unit_diffs(ref, new)
    assert set(d) == set(METRICS)
    np.testing.assert_allclose(d["MA failures /1000 scenarios"], [2000.0, 0.0])
    np.testing.assert_allclose(d["MA flight time [s]"], [-5.0, -5.0])
    np.testing.assert_allclose(d["SA flight time [s]"], np.full(2, -5.0))
    np.testing.assert_allclose(d["SA violations /episode"], np.ones(2))
    # score: SA -10 (intrusion) + 0.05 (time); MA scenario 0 loses 20% of its goals
    sa = -10.0 + 0.05
    np.testing.assert_allclose(d["score"], [(sa + (-100.0 * 2 / AGENTS + 0.05)) / 2, (sa + 0.05) / 2])


def test_pooled_and_across_runs(tmp_path):
    runs = []
    for i, (fails, shift) in enumerate([(1, -4.0), (2, -6.0), (3, -8.0)]):
        runs.append(pooled([_pair(tmp_path, f"r{i}s{s}", fails, shift) for s in (42, 777)]))
    assert all(len(r["MA flight time [s]"]) == 4 for r in runs)              # 2 scenarios x 2 streams
    res = across_runs(runs)["MA flight time [s]"]
    np.testing.assert_allclose(res["runs"], [-4.0, -6.0, -8.0])
    np.testing.assert_allclose(res["mean"], -6.0)
    np.testing.assert_allclose(res["sd"], 2.0)
    assert res["t_test"]["p"] < 0.05
    np.testing.assert_allclose(across_runs(runs)["MA failures /1000 scenarios"]["runs"], [500.0, 1000.0, 1500.0])


if __name__ == "__main__":
    import pathlib
    import tempfile

    for f in (test_unit_diffs, test_pooled_and_across_runs):
        with tempfile.TemporaryDirectory() as d:
            f(pathlib.Path(d))
        print(f.__name__, "ok")

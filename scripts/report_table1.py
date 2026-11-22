"""Table 1 of the report: official-harness metrics per flight, seed 42, with 95% BCa CIs for the
policy (SA: over episodes; MA: over scenarios, the 10 aircraft of a scenario are one cluster), and
the policy on the confirmation stream (seed 9001, BlueSky-Batch), which no decision used.

    python -m scripts.report_table1
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atc_agent.paired import boot_ci, load  # noqa: E402

RES = os.path.join(ROOT, "submission", "results")
RUNS = {"direct": os.path.join(RES, "reference", "direct"), "MVP": os.path.join(RES, "reference", "mvp_route"),
        "ours": os.path.join(RES, "official"), "ours s9001": os.path.join(RES, "fast_s9001")}
# (label, column, scale, decimals)
ROWS = [("Goal completion [%]", "waypoint_reached", 100.0, 2), ("Flight time [s]", "flight_time", 1.0, 1),
        ("Intrusion events", "intrusion_events", 1.0, 3), ("Intrusion time [s]", "intrusion_time", 1.0, 2),
        ("RA events", "restricted_area_events", 1.0, 3), ("RA time [s]", "time_in_restricted_area", 1.0, 2),
        ("Sector-exit events", "sector_exit_events", 1.0, 3), ("Time outside sector [s]", "time_outside_sector", 1.0, 2)]
EVENTS = ["intrusion_events", "restricted_area_events", "sector_exit_events"]


def units(df, track, col):
    """Per-unit values: SA episodes, MA scenario means (equal cluster sizes, so the mean is unchanged)."""
    s = df[col].astype(float) if isinstance(col, str) else col.astype(float)
    return s.to_numpy() if track == "sa" else s.groupby(df["scenario"]).mean().to_numpy()


def ci(x):
    return boot_ci(x) if np.ptp(x) else (float(x[0]), float(x[0]))


def main():
    data = {k: dict(zip(("sa", "ma"), load(p))) for k, p in RUNS.items()}
    for track in ("sa", "ma"):
        print(f"\n{track.upper()}: " + " | ".join(RUNS))
        for label, col, scale, dp in ROWS:
            cells = []
            for k in RUNS:
                df = data[k][track]
                x = units(df, track, col) * scale
                cell = f"{x.mean():.{dp}f}"
                if k.startswith("ours"):
                    lo, hi = ci(x)
                    cell += f" [{lo:.{dp}f}, {hi:.{dp}f}]"
                cells.append(cell)
            print(f"  {label:24s} " + " | ".join(cells))
        for k in RUNS:
            df = data[k][track]
            x = units(df, track, df[EVENTS].sum(axis=1))
            lo, hi = ci(x)
            print(f"  safety events per flight, {k:10s} {x.mean():.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()

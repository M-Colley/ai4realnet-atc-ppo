"""Report Fig. 2: the submitted policy on seed-42 MA scenarios 261 and 788 (BlueSky-Batch replay).

    python -m scripts.report_fig_trajectories
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atc_agent.fast_eval import FastEvaluator  # noqa: E402
from atc_agent.inference import load_model  # noqa: E402
from atc_agent.plot_flights import plot, replay  # noqa: E402

IDS = (261, 788)
TITLES = ("(a) #261: 10/10 reached, no violations", "(b) #788: goal 0.13 NM inside an RA")


def main():
    model = load_model(os.path.join(ROOT, "submission", "policy.zip"), device="cpu")
    ev = FastEvaluator.official(0, max(IDS) + 1, seed=42)
    scens = [ev.ma[i] for i in IDS]
    res = replay(lambda o: model.predict(o, deterministic=True)[0], scens, "ma")
    for i, r in zip(IDS, res):
        ms = [r["metrics"][j] for j in sorted(r["metrics"])]
        print(i, "reached", sum(m["waypoint_reached"] >= 1 for m in ms), "violations",
              sum(m["intrusion_events"] + m["restricted_area_events"] + m["sector_exit_events"] for m in ms),
              "mean flight", np.mean([m["flight_time"] for m in ms]))
    out = os.path.join(ROOT, "docs", "solution", "report", "figures", "trajectories")
    for ext in ("pdf", "png"):
        plot(scens, res, TITLES, f"{out}.{ext}", ncols=2, size=2.3, legend=False)
    print("wrote", out + ".pdf")


if __name__ == "__main__":
    main()

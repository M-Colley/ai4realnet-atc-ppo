"""Print a compact learning-curve table from runs/<run>/progress.jsonl.

    python -m atc_agent.progress runs/main            # every rollout
    python -m atc_agent.progress runs/main --evals    # held-out evaluations only
"""

import argparse
import json
import os


def fmt(st):
    if not st:
        return "-"
    return (f"n={st.get('n', 0):4d} goal {st.get('waypoint_reached', 0):.3f} ft {st.get('flight_time', 0):5.0f} "
            f"intr {st.get('intrusion_events', 0):.3f} ra {st.get('restricted_area_events', 0):.3f} "
            f"sec {st.get('sector_exit_events', 0):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--evals", action="store_true")
    p.add_argument("--every", type=int, default=1)
    a = p.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.run, "progress.jsonl"))]
    for i, r in enumerate(rows):
        if a.evals:
            if "eval" not in r:
                continue
            print(f"{r['t']/1e6:7.2f}M {r['wall']/3600:5.2f}h score {r['eval_score']:7.2f} | "
                  f"SA {fmt(r['eval'].get('sa'))} | MA {fmt(r['eval'].get('ma'))}")
        elif i % a.every == 0:
            lam = " ".join(f"{v:.2f}" for v in r["lam"].values())
            print(f"{r['t']/1e6:7.2f}M {r['wall']/3600:5.2f}h lam[{lam}] | SA10 {fmt(r.get('sa10'))} | "
                  f"MA {fmt(r.get('ma_orig'))}")


if __name__ == "__main__":
    main()

"""Report Sec. 7: MA-failure and score change of the geodesic stage against its start (1070M,
``submission/geo_init``), pooled over seeds 42 and 777, for the submitted lineage (the parent, 1185M)
and the seed repeats' checkpoints at the same step (``runs/rep_finetune_s<seed>/eval/fast_matched*``,
written by ``scripts.seed_repeat``).

    python -m scripts.geodesic_replication --seeds 11 12 13
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.seed_repeat_pooled import describe, pooled  # noqa: E402

RES = os.path.join(ROOT, "submission", "results")
START = os.path.join(RES, "geo_init", "fast")
STREAMS = ("", "_s777")          # seeds 42 and 777
KEYS = ("MA failures /1000 scenarios", "score")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="*", default=[11, 12, 13])
    a = p.parse_args()
    runs = {"submitted lineage (1185M)": os.path.join(RES, "parent", "fast")}
    runs.update({f"repeat seed {s}": os.path.join(ROOT, "runs", f"rep_finetune_s{s}", "eval", "fast_matched")
                 for s in a.seeds})
    for name, prefix in runs.items():
        d = pooled([(START + sfx, prefix + sfx) for sfx in STREAMS])
        print(name + ": " + ", ".join(
            f"{k} {e['mean']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]" for k, e in
            ((k, describe(d[k])) for k in KEYS)))


if __name__ == "__main__":
    main()

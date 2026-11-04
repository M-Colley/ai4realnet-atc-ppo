"""
Uniform weight averaging ("model soup") of checkpoints from one training run.

Consecutive checkpoints of one fine-tune share a loss basin, so their mean is
usually a valid policy that is less sensitive to the last updates' noise.
Floating-point policy tensors are averaged; everything else (integer buffers,
optimizer state, hyper-parameters) is taken from the last checkpoint.

    python -m atc_agent.average_checkpoints runs/ft_geo_r/best.zip runs/ft_geo_r2/best.zip \
        runs/ft_geo_r2/final.zip --out runs/soup_geo/soup.zip
"""

import argparse
import os

import torch as th
from stable_baselines3 import PPO


def average(paths, out):
    models = [PPO.load(p, device="cpu") for p in paths]
    states = [m.policy.state_dict() for m in models]
    avg = {}
    for k, v in states[-1].items():
        avg[k] = th.stack([s[k] for s in states]).mean(0) if v.is_floating_point() else v
    models[-1].policy.load_state_dict(avg)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    models[-1].save(out)
    return models[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    average(args.models, args.out)
    print(f"wrote {args.out} (mean of {len(args.models)} checkpoints)")


if __name__ == "__main__":
    main()

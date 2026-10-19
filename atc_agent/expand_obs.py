"""
Carry a trained policy over to a wider ego observation (new entries appended at
the end of the ego block, e.g. the elapsed-time feature). The new input columns
of the ego embedding get zero weights, so the converted policy computes exactly
the same actions and values as the original until training changes them.

    python -m atc_agent.expand_obs runs/main/final.zip runs/ta_init/init.zip
"""

import argparse
import os
import shutil

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO

from atc_agent.features import OBS_DIM, EGO_DIM


def expand(src, dst):
    from stable_baselines3.common import base_class
    orig_loader = base_class.load_from_zip_file
    info = {}

    def pad(t, old_ego, extra):
        return th.cat([t, th.zeros(t.shape[0], extra, dtype=t.dtype)], dim=1)

    def loader(*args, **kwargs):
        data, params, pt_vars = orig_loader(*args, **kwargs)
        old_dim = data["observation_space"].shape[0]
        extra = OBS_DIM - old_dim
        if extra <= 0:
            raise SystemExit(f"nothing to do: model obs dim {old_dim}, current {OBS_DIM}")
        old_ego = EGO_DIM - extra
        shapes = set()
        for k, v in params["policy"].items():
            if k.endswith("ego.0.weight"):
                assert v.shape[1] == old_ego, (k, v.shape)
                params["policy"][k] = pad(v, old_ego, extra)
                shapes.add(tuple(v.shape))
        for st in params["policy.optimizer"]["state"].values():     # Adam moments of those weights
            for m in ("exp_avg", "exp_avg_sq"):
                if m in st and tuple(st[m].shape) in shapes:
                    st[m] = pad(st[m], old_ego, extra)
        data["observation_space"] = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        info.update(old=old_dim, extra=extra)
        return data, params, pt_vars

    base_class.load_from_zip_file = loader
    try:
        model = PPO.load(src, device="cpu")
    finally:
        base_class.load_from_zip_file = orig_loader
    model.save(dst)
    st = os.path.join(os.path.dirname(src), "train_state.json")
    if os.path.exists(st):
        shutil.copy(st, os.path.join(os.path.dirname(dst), "train_state.json"))
    print(f"{src} ({info['old']}) -> {dst} ({OBS_DIM}), {info['extra']} zero-weight ego input(s)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    expand(a.src, a.dst)


if __name__ == "__main__":
    main()

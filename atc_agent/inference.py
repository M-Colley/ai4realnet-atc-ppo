"""Load a trained policy as an ``act(obs) -> action`` callable."""

import numpy as np


def load_model(model_path, device="cpu"):
    from stable_baselines3 import PPO
    return PPO.load(model_path, device=device)


def load_act(model_path, device="cpu", deterministic=True):
    model = load_model(model_path, device=device)

    def act(obs):
        a, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
        return a

    return act

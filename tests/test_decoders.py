"""Decoders that are documented to reduce to plain greedy decoding do so."""

import numpy as np

from atc_agent.fast_eval import EnsembleDecoder, StickyDecoder
from atc_agent.inference import load_model

MODEL = "submission/policy.zip"


def _obs(model, n=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n,) + model.observation_space.shape).astype(np.float32)


def test_one_member_ensemble_is_greedy():
    m = load_model(MODEL)
    obs = _obs(m)
    np.testing.assert_array_equal(EnsembleDecoder([m])(obs), m.predict(obs, deterministic=True)[0])


def test_ensemble_of_copies_is_greedy():
    m = load_model(MODEL)
    obs = _obs(m)
    np.testing.assert_array_equal(EnsembleDecoder([m, m])(obs), m.predict(obs, deterministic=True)[0])


def test_zero_bonus_sticky_is_greedy():
    m = load_model(MODEL)
    obs = _obs(m)
    dec = StickyDecoder(m)
    dec.reset(len(obs))
    for _ in range(2):          # second call has a previous action
        np.testing.assert_array_equal(dec(obs, np.arange(len(obs))), m.predict(obs, deterministic=True)[0])

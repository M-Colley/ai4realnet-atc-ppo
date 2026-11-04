"""The indecision costs (reversal / speed_jump) are exactly the documented
amounts, are reset with each episode, and leave everything else unchanged."""

import contextlib
import io

import numpy as np

from atc_agent.actions import HEADING_OFFSETS, DIRECT_BIN

R = dict(reversal=0.05, speed_jump=0.01)


def _sim(reward):
    with contextlib.redirect_stdout(io.StringIO()):
        from atc_agent.batched_env import BatchedCompetitionSim
        return BatchedCompetitionSim(sa_slots=2, ma_slots=0, seed=3, reward=reward)


def _bin(offset):
    return int(np.flatnonzero(HEADING_OFFSETS == offset)[0])


def _rollout(reward, seq):
    sim = _sim(reward)       # BlueSky is a singleton: one sim at a time
    sim.reset()
    out = []
    for h, s in seq:
        _, r, d, _, _ = sim.step(np.tile([h, s], (sim.n_seats, 1)))
        assert not d.any()
        out.append(r)
    return np.array(out)


def test_costs_match_definition():
    seq = [(DIRECT_BIN, 5), (_bin(120), 5), (_bin(-120), 5), (_bin(-5), 0), (_bin(12), 1), (DIRECT_BIN, 1)]
    expect = [0.0,                       # first decision: nothing to compare with
              0.0,                       # 0 -> +120: not a flip
              0.05 * 120 / 90,           # +120 -> -120
              0.01 * 4,                  # -120 -> -5: same side; speed bin 5 -> 0
              0.05 * 5 / 90,             # -5 -> +12; speed 0 -> 1 is within one bin
              0.0]
    diff = _rollout(None, seq) - _rollout(R, seq)
    np.testing.assert_allclose(diff, np.tile(np.array(expect)[:, None], (1, diff.shape[1])), atol=1e-6)


def test_reset_clears_previous_action():
    sim = _sim(R)
    sim.reset()
    sim.step(np.tile([_bin(120), 5], (sim.n_seats, 1)))
    sim._init_seats(np.array([0]))
    assert sim.prev_act[0, 0] == -1 and sim.prev_act[1, 0] == _bin(120)

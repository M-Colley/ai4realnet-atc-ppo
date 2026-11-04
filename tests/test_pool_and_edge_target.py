"""Replay pool restarts slots from pool scenarios (and changes nothing when
unused); the edge target only moves goals that lie shallowly inside a
restricted area, to a capturable point outside it."""

import contextlib
import io

import numpy as np

from atc_agent import routing
from atc_agent.fast_eval import FastEvaluator


def _sim(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        from atc_agent.batched_env import BatchedCompetitionSim
        return BatchedCompetitionSim(sa_slots=3, ma_slots=0, seed=5, **kw)


def test_pool_draws_and_is_inert_when_unused():
    plain = _sim().reset()
    unused = _sim(pool={"sa": FastEvaluator.official(2, 0, seed=9).sa}, pool_prob={"sa": 0.0}).reset()
    np.testing.assert_array_equal(plain, unused)
    target = FastEvaluator.official(1, 0, seed=9).sa[0]
    sim = _sim(pool={"sa": [target]}, pool_prob={"sa": 1.0})
    sim.reset()
    for k in range(sim.K):
        assert sim.scen[k] is not target                     # deep copy, the pool stays untouched
        assert sim.scen[k].sector == target.sector and sim.scen[k].agents[0].goal == target.agents[0].goal


def _geo():
    sq = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])     # 10 NM restricted square
    sector = np.array([[-50.0, -50.0], [-50.0, 50.0], [50.0, 50.0], [50.0, -50.0]])
    return {"obst": sq[None], "obst_valid": np.array([True]), "sector": sector}


def test_edge_target():
    geo, old = _geo(), routing.EDGE_TARGET
    try:
        routing.EDGE_TARGET = False
        assert routing.route_target((5.0, 1.0), geo) == (5.0, 1.0)
        routing.EDGE_TARGET = True
        t = routing.route_target((5.0, 1.0), geo)                 # 1 NM inside the south edge
        assert abs(t[0] - 5.0) < 1e-9 and abs(t[1] + routing.EDGE_OFFSET_NM) < 1e-9
        assert np.hypot(t[0] - 5.0, t[1] - 1.0) < routing.CAPTURE_NM
        assert routing.route_target((5.0, 5.0), geo) == (5.0, 5.0)   # 5 NM deep: unchanged
        assert routing.route_target((20.0, 20.0), geo) == (20.0, 20.0)   # outside: unchanged
    finally:
        routing.EDGE_TARGET = old

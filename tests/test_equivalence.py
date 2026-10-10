"""
Equivalence of the batched training simulator with the OFFICIAL environments.

For identical scenarios and identical action sequences, BatchedCompetitionSim
(with other scenarios sharing the simulator) must reproduce the official
CompetitionEnv / CompetitionZooEnv (with our hooks) step by step:
observations, aircraft states, and the nine scored metrics.

Run:  python -m pytest tests/test_equivalence.py -q     (or python tests/test_equivalence.py)
"""

import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with contextlib.redirect_stdout(io.StringIO()):
    import bluesky as bs  # noqa: F401
    from core.scenario import ScenarioGenerator
    from atc_agent.envs import AgentCompetitionEnv, AgentCompetitionZooEnv
    from atc_agent.batched_env import BatchedCompetitionSim, METRIC_KEYS

SCORED = [k for k in METRIC_KEYS if k != "total_reward"]


def _actions(rng, n, shape=()):
    """Random but smooth-ish action sequence (mostly direct, some manoeuvres)."""
    h = rng.choice(17, size=(n,) + shape, p=None)
    h = np.where(rng.random((n,) + shape) < 0.5, 8, h)
    s = rng.integers(0, 6, size=(n,) + shape)
    return np.stack([h, s], axis=-1)


def _official_sa(scen, acts):
    with contextlib.redirect_stdout(io.StringIO()):
        env = AgentCompetitionEnv(scenario=scen)
    obs, info = env.reset(seed=0)
    out = [obs]
    for a in acts:
        obs, _, term, trunc, info = env.step(a)
        out.append(obs)
        if term or trunc:
            break
    return np.array(out), info


def _batched_sa(scen, acts, extra_slots=3):
    sim = BatchedCompetitionSim(sa_slots=1 + extra_slots, ma_slots=0, seed=123)
    sim.forced = {0: [scen]}
    obs = sim.reset()
    out = [obs[0]]
    info = None
    for a in acts:
        A = np.tile(np.array([8, 5]), (sim.n_seats, 1))
        A[0] = a
        obs, _, done, _, infos = sim.step(A)
        if done[0]:
            info = infos[0]["metrics"]
            break
        out.append(obs[0])
    return np.array(out), info


def test_single_agent_equivalence(n_scen=3):
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=10, agent_speed=150.0)
    rng = np.random.default_rng(7)
    for i in range(n_scen):
        scen = gen.generate(rng)
        acts = _actions(rng, 300)
        o_obs, o_info = _official_sa(scen, acts)
        b_obs, b_info = _batched_sa(scen, acts)
        n = len(b_obs)
        assert np.allclose(o_obs[:n], b_obs, atol=1e-5), \
            f"SA scenario {i}: obs mismatch, max err {np.abs(o_obs[:n] - b_obs).max()}"
        for k in SCORED:
            assert abs(o_info[k] - b_info[k]) < 1e-9, f"SA scenario {i}: {k} {o_info[k]} vs {b_info[k]}"
        print(f"SA scenario {i}: {n} steps identical; metrics", {k: o_info[k] for k in SCORED})


def test_multi_agent_equivalence(n_scen=2, n_agents=10):
    gen = ScenarioGenerator(n_obstacles=5, n_intruders=0, n_agents=n_agents, agent_speed=150.0)
    rng = np.random.default_rng(11)
    for i in range(n_scen):
        scen = gen.generate(rng)
        acts = _actions(rng, 300, (n_agents,))
        # official
        with contextlib.redirect_stdout(io.StringIO()):
            env = AgentCompetitionZooEnv(scenario=scen)
        obs, _ = env.reset(seed=0)
        o_final, o_obs = {}, [np.array([obs[a] for a in env.possible_agents])]
        for t, a in enumerate(acts):
            names = list(env.agents)
            obs, _, term, trunc, infos = env.step({ag: a[env.possible_agents.index(ag)] for ag in names})
            for ag in infos:
                if term[ag] or trunc[ag]:
                    o_final[ag] = infos[ag]
            o_obs.append({ag: obs[ag] for ag in env.agents})
            if not env.agents:
                break
        # batched (respawn off so the scenario stays identical)
        sim = BatchedCompetitionSim(sa_slots=2, ma_slots=1, seed=5, respawn=False)
        sim.forced = {2: [scen]}
        bobs = sim.reset()
        seats = sim.slot_seats[2]
        assert np.allclose(bobs[seats], o_obs[0], atol=1e-5), "MA reset obs mismatch"
        b_final = {}
        for t, a in enumerate(acts):
            A = np.tile(np.array([8, 5]), (sim.n_seats, 1))
            A[seats] = a
            alive_before = sim.active[seats].copy()
            bobs, _, done, _, infos = sim.step(A)
            for n_, s in enumerate(seats):
                if s in infos and env.possible_agents[n_] not in b_final:
                    b_final[env.possible_agents[n_]] = infos[s]["metrics"]
            if t + 1 < len(o_obs):
                for n_, s in enumerate(seats):
                    ag = env.possible_agents[n_]
                    if ag in o_obs[t + 1] and sim.active[s] and alive_before[n_]:
                        err = np.abs(o_obs[t + 1][ag] - bobs[s]).max()
                        assert err < 1e-5, f"MA scenario {i} step {t} {ag}: obs err {err}"
            if len(b_final) == n_agents:
                break
        for ag in env.possible_agents:
            for k in SCORED:
                assert abs(o_final[ag][k] - b_final[ag][k]) < 1e-9, \
                    f"MA scenario {i} {ag}: {k} {o_final[ag][k]} vs {b_final[ag][k]}"
        tot = {k: sum(o_final[a][k] for a in env.possible_agents) for k in SCORED}
        print(f"MA scenario {i}: all {n_agents} agents identical; summed metrics", tot)


if __name__ == "__main__":
    test_single_agent_equivalence()
    test_multi_agent_equivalence()
    print("ALL EQUIVALENCE TESTS PASSED")

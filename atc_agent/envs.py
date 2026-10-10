"""
Our MDP plugged into the OFFICIAL competition environments, via the three
sanctioned hooks only (``_get_obs``, ``_get_action``, ``_get_reward``). The
scoring layer (``_update_metrics`` / ``_get_info``), the scenario distribution
and all fixed parameters are untouched. These classes are what the official
evaluation / GIF scripts instantiate.
"""

import numpy as np
import bluesky as bs
from gymnasium import spaces

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition.competition import CompetitionZooEnv
from atc_agent.actions import action_space, action_targets, apply_targets
from atc_agent.features import build_observations, scenario_geometry, OBS_DIM
from atc_agent.batched_env import DEFAULT_REWARD, GOAL_MARGIN_KM
from atc_agent.geometry import kwikqdrdist


def traffic_obs_inputs(own, others, others_is_agent, others_goal, goal, geo, mode_ma):
    """Observation inputs for one ownship read straight from BlueSky state.
    Mirrors ``BatchedCompetitionSim._obs_inputs`` field by field."""
    own = int(own)
    ot = np.asarray(others, dtype=int).reshape(1, -1)
    is_ag = np.asarray(others_is_agent, dtype=bool).reshape(1, -1)
    if ot.shape[1] == 0:            # keep one masked column so shapes stay valid
        ot = np.array([[own]])
        is_ag = np.zeros((1, 1), dtype=bool)
        valid = np.zeros((1, 1), dtype=bool)
        og = np.zeros((1, 1, 2))
    else:
        valid = np.ones_like(ot, dtype=bool)
        og = np.asarray(others_goal, dtype=float).reshape(1, -1, 2)
    swl = bs.traf.swlnav[ot]
    return {
        "own_lat": bs.traf.lat[[own]], "own_lon": bs.traf.lon[[own]],
        "own_hdg": bs.traf.hdg[[own]], "own_tas": bs.traf.tas[[own]],
        "own_cas": bs.traf.cas[[own]], "own_seltrk": bs.traf.ap.trk[[own]],
        "own_selspd": bs.traf.selspd[[own]],
        "goal_lat": np.array([goal[0]], dtype=float), "goal_lon": np.array([goal[1]], dtype=float),
        "mode_ma": np.array([float(mode_ma)]),
        "center": geo["center"][None], "sector": geo["sector"][None],
        "obst": geo["obst"][None], "obst_valid": geo["obst_valid"][None],
        "obst_centroid": geo["obst_centroid"][None], "obst_radius": geo["obst_radius"][None],
        "oth_lat": bs.traf.lat[ot], "oth_lon": bs.traf.lon[ot],
        "oth_hdg": bs.traf.hdg[ot], "oth_tas": bs.traf.tas[ot],
        "oth_seltrk": bs.traf.ap.trk[ot],
        "oth_valid": valid, "oth_is_agent": is_ag,
        "oth_has_intent": np.where(is_ag, True, swl),
        "oth_intent_lat": np.where(is_ag, og[..., 0], bs.traf.actwp.lat[ot]),
        "oth_intent_lon": np.where(is_ag, og[..., 1], bs.traf.actwp.lon[ot]),
    }


class _HooksMixin:
    """Shared hook implementations (all keyed by aircraft id)."""

    def _setup_spaces(self):
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._act_space = action_space()
        self._geo_key = None
        self._phi = {}

    def _geometry(self):
        if self._geo_key is not self.scenario:
            self._geo = scenario_geometry(self.scenario)
            self._geo_key = self.scenario
        return self._geo

    def _apply(self, ac_id, goal, action):
        i = bs.traf.id2idx(ac_id)
        hdg, cas = action_targets(np.asarray(action).reshape(1, 2),
                                  bs.traf.lat[[i]], bs.traf.lon[[i]],
                                  np.array([goal[0]]), np.array([goal[1]]))
        apply_targets([i], hdg, cas)

    def _reward_from(self, ac_id, goal, base):
        """Our training reward minus the per-second soft-buffer terms (those need
        1 s sampling, which the hook interface does not expose). Reported as
        ``total_reward`` only; it does not influence the scored metrics."""
        r = DEFAULT_REWARD
        m = self.metrics[ac_id]
        i = bs.traf.id2idx(ac_id)
        _, dg = kwikqdrdist(bs.traf.lat[i], bs.traf.lon[i], goal[0], goal[1])
        phi = -max(dg * 1852.0 - GOAL_MARGIN_KM * 1000.0, 0.0) / r["v_ref"]
        phi0 = self._phi.get(ac_id, phi)
        self._phi[ac_id] = phi
        d = {k: m[k] - base[k] for k in m}
        return (-r["time"] * d["flight_time"] + r["progress"] * (phi - phi0) + r["goal"] * d["waypoint_reached"]
                - r["intr_event"] * d["intrusion_events"] - r["intr_time"] * d["intrusion_time"]
                - r["ra_event"] * d["restricted_area_events"] - r["ra_time"] * d["time_in_restricted_area"]
                - r["sec_event"] * d["sector_exit_events"] - r["sec_time"] * d["time_outside_sector"])


class AgentCompetitionEnv(_HooksMixin, CompetitionEnv):
    """Single-agent official env with our observation / action / reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_spaces()
        self.observation_space = self._obs_space
        self.action_space = self._act_space

    def _get_obs(self, ac_id):
        others = [bs.traf.id2idx(x) for x in self.intruder_ids]
        others = [o for o in others if o >= 0]
        s = traffic_obs_inputs(bs.traf.id2idx(ac_id), others, [False] * len(others),
                               np.zeros((len(others), 2)), (self.goal_lat, self.goal_lon),
                               self._geometry(), mode_ma=0)
        return build_observations(s)[0][0]

    def _get_action(self, ac_id, action):
        self._apply(ac_id, (self.goal_lat, self.goal_lon), action)

    def _get_reward(self, ac_id):
        return self._reward_from(ac_id, (self.goal_lat, self.goal_lon), self._reward_baseline)

    def reset(self, seed=None, options=None):
        self._phi = {}
        return super().reset(seed=seed, options=options)


class AgentCompetitionZooEnv(_HooksMixin, CompetitionZooEnv):
    """Multi-agent official env with our observation / action / reward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_spaces()
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

    def _get_obs(self, ac_id):
        others = [a for a in self.agents if a != ac_id]
        idx = [bs.traf.id2idx(a) for a in others]
        keep = [n for n, i in enumerate(idx) if i >= 0]
        idx = [idx[n] for n in keep]
        goals = np.array([self._goal[others[n]] for n in keep], dtype=float).reshape(-1, 2)
        s = traffic_obs_inputs(bs.traf.id2idx(ac_id), idx, [True] * len(idx), goals,
                               self._goal[ac_id], self._geometry(), mode_ma=1)
        return build_observations(s)[0][0]

    def _get_action(self, ac_id, action):
        self._apply(ac_id, self._goal[ac_id], action)

    def _get_reward(self, ac_id):
        return self._reward_from(ac_id, self._goal[ac_id], self._reward_baseline[ac_id])

    def reset(self, seed=None, options=None):
        self._phi = {}
        return super().reset(seed=seed, options=options)

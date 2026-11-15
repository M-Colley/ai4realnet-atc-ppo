"""
Classical reference: fly direct to the goal with BlueSky's Modified Voltage Potential (MVP)
conflict resolution, in the official environments and harness loop (seed 42).

Each aircraft is commanded, every decision step at its initial speed, the heading to its goal
(``--nav direct``) or to a point 5 NM along the shortest route around the restricted areas
(``--nav route``: the grid route of atc_agent.routing, also the policy's route-hint input);
BlueSky's state-based conflict detection (5 NM, 300 s look-ahead) and MVP resolution
(heading and speed, no altitude changes) take over while a conflict is active and hand back
after the closest point of approach. Single-agent: only the agent resolves; the intruders fly
their routes unchanged (non-cooperative, as in the competition). Multi-agent: every agent
resolves (cooperative). MVP itself knows nothing about restricted areas or the sector.

    python -m atc_agent.mvp_baseline --env sa --nav route --out runs/mvp/route_sa.csv
    python -m atc_agent.mvp_baseline --env ma --nav direct --out runs/mvp/direct_ma.csv
"""

import argparse

import bluesky as bs
import numpy as np
from bluesky.tools.geo import kwikqdrdist

from atc_agent.features import scenario_geometry
from atc_agent.geometry import latlon_to_nm
from atc_agent.routing import ScenarioRoutes, route_query
from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition_v0 import CompetitionZooEnv
from scripts import evaluate_competition as ec

NAV = "direct"
SETTINGS = ["CDMETHOD STATEBASED", "ZONER 5", "DTLOOK 300", "RESO MVP", "RMETHH BOTH", "RMETHV OFF"]


def enable_mvp(non_cooperative=()):
    # queued after the env's own "RESO OFF", so processed after it on the next sim step
    for cmd in SETTINGS:
        bs.stack.stack(cmd)
    for ac in non_cooperative:          # RESOOFF toggles; aircraft are recreated every reset
        bs.stack.stack(f"RESOOFF {ac}")


class Navigator:
    """Heading command towards the goal, directly or along the area-free grid route."""

    def __init__(self, scenario):
        self.center = scenario.center
        self.routes = ScenarioRoutes(scenario_geometry(scenario)) if NAV == "route" else None

    def command(self, ac_id, goal):
        i = bs.traf.id2idx(ac_id)
        lat, lon = float(bs.traf.lat[i]), float(bs.traf.lon[i])
        aim = goal
        if self.routes is not None:
            p = np.array(latlon_to_nm(lat, lon, *self.center))
            g = np.array(latlon_to_nm(goal[0], goal[1], *self.center))
            _, point, ok = route_query(p[:1], p[1:], g[:1], g[1:], self.routes.grid_params()[None],
                                       self.routes.node_dist(g)[None])
            if ok[0]:
                n, e = point[0]
                aim = (self.center[0] + n / 60.0, self.center[1] + e / (60.0 * np.cos(np.radians(self.center[0]))))
        qdr, _ = kwikqdrdist(lat, lon, aim[0], aim[1])
        bs.stack.stack(f"HDG {ac_id} {float(qdr) % 360.0:.2f}")


class MvpSingleAgent(CompetitionEnv):
    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        enable_mvp(self.intruder_ids)
        self.nav = Navigator(self.scenario)
        return out

    def _get_action(self, ac_id, action):
        self.nav.command(ac_id, (self.goal_lat, self.goal_lon))


class MvpMultiAgent(CompetitionZooEnv):
    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        enable_mvp()
        self.nav = Navigator(self.scenario)
        return out

    def _get_action(self, ac_id, action):
        self.nav.command(ac_id, self._goal[ac_id])


def make_env(kind, n_agents=ec.N_AGENTS_MA):
    return MvpSingleAgent() if kind == "sa" else MvpMultiAgent(n_agents=n_agents)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["sa", "ma"], required=True)
    p.add_argument("--nav", choices=["direct", "route"], default="direct")
    p.add_argument("--episodes", type=int, default=ec.N_EPISODES)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    global NAV
    NAV = a.nav
    ec.make_env = make_env                     # the official loop, with the MVP environments

    def act(_obs):
        return np.zeros(2, dtype=np.float32)   # ignored by _get_action above
    if a.env == "sa":
        recs, label = ec.run_single_agent(a.episodes, act), f"Single-agent ({a.nav} + MVP)"
    else:
        recs, label = ec.run_multi_agent(a.episodes, act, ec.N_AGENTS_MA), f"Multi-agent ({a.nav} + MVP)"
    ec.summarize(recs, label)
    ec.write_csv(recs, a.out)


if __name__ == "__main__":
    main()

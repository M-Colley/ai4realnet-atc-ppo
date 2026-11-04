"""
BlueSky-Batch: many independent competition scenarios inside ONE BlueSky
simulation, for high-throughput training with *exact* BlueSky dynamics.

Why this is exact
-----------------
BlueSky is vectorised over aircraft, and with conflict resolution OFF
(``RESO OFF``, a fixed competition rule) aircraft never interact. Scenarios that
share the simulator therefore evolve exactly as they would alone; only the
per-call Python overhead of ``bs.sim.step()`` is shared. Measured: one step with
11 aircraft costs ~0.8 ms, with 1100 aircraft ~1.9 ms, i.e. ~40x more samples
per CPU second than one scenario per process.

Everything that defines an episode is reproduced from the official envs:
scenarios come from the official ``core.scenario.ScenarioGenerator``, aircraft
are created with the same ``bs.traf.cre`` calls, intruder routes with the same
``ADDWPT`` stack commands, and the nine metrics are recomputed per 1 s sim step
with the official deterministic ``kwikqdrdist`` and an affine-equivalent
point-in-polygon test. ``tests/test_equivalence.py`` checks trajectories and
metrics against the official ``CompetitionEnv`` / ``CompetitionZooEnv``.

Slots and seats
---------------
A *slot* is one scenario. A *seat* is one controlled aircraft = one entry of
the vectorised env. Single-agent ("sa") slots have one seat; multi-agent ("ma")
slots have ``n_agents`` seats.

Multi-agent slots use *continuous traffic* so every seat always holds a live
agent (no zero-padded "dead" samples): when an agent reaches its goal, a fresh
agent is spawned with the official placement routine (``_generate_agent``,
start >= 10 NM from all live aircraft). The scenario itself is regenerated once
its original cohort has finished (or the 3000 s clock runs out), so the dense
simultaneous-start phase of the evaluation distribution is trained on as well.
"""

import copy

import numpy as np
import bluesky as bs
from bluesky.stack.stackbase import Stack
from shapely.geometry import Polygon

import bluesky_gym.envs.common.functions as fn
from core.scenario import ScenarioGenerator, _bearing_nm, _nm_to_latlon
from atc_agent.actions import HEADING_OFFSETS, action_targets, apply_targets
from atc_agent.features import (
    build_observations, scenario_geometry, N_OBS, OBS_DIM, SEP_NM,
)
from atc_agent.geometry import (
    NM2KM, kwikqdrdist, latlon_to_nm, points_in_polygons, nearest_on_polygons,
)
from atc_agent.routing import ScenarioRoutes, route_query, route_target, GRID

METRIC_KEYS = [
    "total_reward", "flight_time", "waypoint_reached",
    "intrusion_events", "intrusion_time",
    "sector_exit_events", "time_outside_sector",
    "restricted_area_events", "time_in_restricted_area",
]

# Reward weights. Unit: 0.01 = one second of flight time.
DEFAULT_REWARD = dict(
    time=0.01,            # per second flown
    progress=0.01,        # per second of nominal flight time saved (potential on distance-to-goal)
    v_ref=152.4,          # m/s TAS at 150 m/s CAS, converts distance to nominal time
    goal=1.0,             # waypoint captured
    intr_event=2.0, intr_time=0.02,     # per loss-of-separation event / per intruder-second
    ra_event=2.0, ra_time=0.02,         # restricted area
    sec_event=2.0, sec_time=0.02,       # sector exit
    prox_ac=0.005, prox_ac_nm=8.0,      # soft buffer around the 5 NM disc (per second, quadratic)
    prox_obs=0.005, prox_obs_nm=3.0,    # soft buffer outside obstacles
    prox_sec=0.005, prox_sec_nm=3.0,    # soft buffer inside the sector edge
    pred_conflict=0.002,                # per second per predicted conflict within 120 s
    timeout=20.0,                       # goal not reached within the 3000 s budget (terminal, no bootstrap)
    geodesic=1.0,                       # progress potential on the restricted-area-free route length (0: straight line)
    lam_intr=1.0, lam_ra=1.0, lam_sec=1.0,   # adaptive (Lagrangian) safety multipliers
    reversal=0.0,       # per 90 deg of a direct left<->right flip of the heading offset (min of the two)
    speed_jump=0.0,     # per speed bin changed beyond one bin between consecutive decisions
)

# violations already present when an aircraft spawns (the generator does not keep
# scripted intruders 5 NM away from the SA start): counted by the official
# metrics, but no policy can avoid them
SPAWN_KEYS = ["intrusion_events", "restricted_area_events", "sector_exit_events"]

EPISODE_TIME_LIMIT = 3000.0
GOAL_MARGIN_KM = 5.0
AC_SPD = 150.0
ALTITUDE = 350
CENTER = (52.0, 4.0)


def _poly_nm(verts_ll, center):
    return [latlon_to_nm(la, lo, center[0], center[1]) for la, lo in verts_ll]


class BatchedCompetitionSim:
    """K competition scenarios in one BlueSky instance (see module docstring).

    Parameters
    ----------
    sa_slots, ma_slots : number of single-/multi-agent scenarios.
    sa_intruders : list of (n_intruders, prob) for SA scenario randomisation.
    ma_agents : agents per MA scenario (competition: 10).
    ma_intruders : extra scripted intruders in MA scenarios (0 = competition).
    n_obstacles : obstacles per scenario (competition: 5).
    action_frequency : sim seconds per decision.
    reward : overrides for DEFAULT_REWARD.
    pool, pool_prob : prioritized replay. ``pool`` maps "sa"/"ma" to lists of
        Scenario; a slot of that kind restarts from a uniformly drawn pool
        scenario with probability ``pool_prob[kind]``, else from a fresh one.
    """

    def __init__(self, sa_slots=32, ma_slots=0, sa_intruders=((10, 1.0),),
                 ma_agents=10, ma_intruders=0, n_obstacles=5,
                 action_frequency=10, reward=None, seed=0, respawn=True, auto_reset=True,
                 pool=None, pool_prob=None):
        if bs.sim is None:
            bs.init(mode="sim", detached=True)
        # BlueSky is a process singleton: drop commands a previous instance queued
        # but never stepped (e.g. ADDWPT for intruders whose ids this instance reuses)
        Stack.clear()
        bs.stack.stack("DT 1;FF")
        bs.stack.stack("RESO OFF")
        bs.traf.reset()

        self.rng = np.random.default_rng(seed)
        self.respawn = respawn          # False only for equivalence tests (MA seats then idle)
        self.forced = {}                # slot -> list of Scenario to use next (tests / eval)
        self.auto_reset = auto_reset    # False: finished scenarios are cleared, not regenerated (eval)
        self.action_frequency = int(action_frequency)
        self.reward = dict(DEFAULT_REWARD)
        if reward:
            self.reward.update(reward)
        self.pool = {k: v for k, v in (pool or {}).items() if len(v)}
        self.pool_prob = dict(pool_prob or {})

        self.sa_choices = np.array([n for n, _ in sa_intruders], dtype=int)
        p = np.array([q for _, q in sa_intruders], dtype=float)
        self.sa_probs = p / p.sum()
        self.ma_agents = ma_agents
        self.ma_intruders = ma_intruders
        self.n_obstacles = n_obstacles
        self._gens = {}

        self.kinds = ["sa"] * sa_slots + ["ma"] * ma_slots
        self.K = len(self.kinds)
        a_sa = 1 + int(self.sa_choices.max()) if sa_slots else 0
        a_ma = ma_agents + ma_intruders if ma_slots else 0
        self.A = max(a_sa, a_ma)

        # seats
        seat_slot, seat_pos = [], []
        self.slot_seats = []
        for k, kind in enumerate(self.kinds):
            n = 1 if kind == "sa" else ma_agents
            self.slot_seats.append(np.arange(len(seat_slot), len(seat_slot) + n))
            seat_slot += [k] * n
            seat_pos += list(range(n))
        self.seat_slot = np.array(seat_slot, dtype=int)
        self.seat_pos = np.array(seat_pos, dtype=int)
        self.n_seats = len(seat_slot)
        self.seat_is_ma = np.array([self.kinds[k] == "ma" for k in seat_slot])

        B, K, A = self.n_seats, self.K, self.A
        self.metrics = {key: np.zeros(B) for key in METRIC_KEYS}
        self.in_intr = np.zeros((B, A), dtype=bool)
        self.obs_occ = np.zeros((B, N_OBS), dtype=bool)
        self.outside = np.zeros(B, dtype=bool)
        self.reached = np.zeros(B, dtype=bool)
        self.active = np.zeros(B, dtype=bool)
        self.orig = np.zeros(B, dtype=bool)
        self.goal = np.zeros((B, 2))
        self.ep_len = np.zeros(B, dtype=int)
        self.phi = np.zeros(B)
        self.prev_act = np.full((B, 2), -1, dtype=np.int64)    # previous decision (-1: none yet)
        self.spawn_viol = np.zeros((B, len(SPAWN_KEYS)))
        self.routes = [None] * len(self.kinds)
        self.route_grid = np.ones((len(self.kinds), 3))
        self.route_D = np.zeros((B, GRID * GRID), dtype=np.float32)
        self.route_tgt = np.zeros((B, 2))       # point the route field aims at (= goal unless EDGE_TARGET)
        self.acc = {k: np.zeros(B) for k in ("dt", "prox_ac", "prox_obs", "prox_sec")}

        # slot tables
        self.ids = [[None] * A for _ in range(K)]
        self.tidx = -np.ones((K, A), dtype=int)
        self.is_agent = np.zeros((K, A), dtype=bool)
        self.ac_goal = np.zeros((K, A, 2))
        self.slot_t = np.zeros(K)
        self.slot_nint = np.zeros(K, dtype=int)
        self.slot_gen = np.zeros(K, dtype=int)
        self.scen = [None] * K
        self.shapely = [None] * K
        self.g_center = np.tile(np.array(CENTER, dtype=float), (K, 1))
        self.g = None
        from atc_agent.features import S_MAX
        self.sec_n = np.zeros((K, S_MAX, 2))
        self.sec_off = np.zeros((K, S_MAX))

    # ------------------------------------------------------------ helpers
    def _gen(self, n_intruders, n_agents):
        key = (n_intruders, n_agents)
        if key not in self._gens:
            self._gens[key] = ScenarioGenerator(
                n_obstacles=self.n_obstacles, n_intruders=n_intruders,
                n_agents=n_agents, agent_speed=AC_SPD, center=CENTER)
        return self._gens[key]

    def _alloc_geometry(self, geo):
        K = self.K
        self.g = {k: np.zeros((K,) + np.shape(v), dtype=np.asarray(v).dtype)
                  for k, v in geo.items()}

    def _sector_halfplanes(self, k):
        """Outward unit normals n and offsets c of the sector edges: n.p <= c inside."""
        v = self.g["sector"][k]
        a, b = v, np.roll(v, -1, axis=0)
        dvec = b - a
        L = np.hypot(dvec[:, 0], dvec[:, 1])
        n = np.stack([dvec[:, 1], -dvec[:, 0]], axis=1) / np.maximum(L, 1e-12)[:, None]
        cen = v.mean(axis=0)
        flip = np.sum(n * (cen - a), axis=1) > 0
        n[flip] *= -1.0
        c = np.sum(n * a, axis=1)
        c[L < 1e-9] = np.inf          # padded zero-length edges never bind
        self.sec_n[k] = n
        self.sec_off[k] = c

    def _refresh_index(self):
        idmap = {acid: i for i, acid in enumerate(bs.traf.id)}
        for k in range(self.K):
            row = self.ids[k]
            for j in range(self.A):
                acid = row[j]
                self.tidx[k, j] = idmap.get(acid, -1) if acid is not None else -1

    def set_reward_params(self, params):
        self.reward.update(params)

    # --------------------------------------------------------- slot reset
    def _reset_slots(self, slots):
        """Regenerate the scenarios of ``slots`` (delete + create aircraft)."""
        if len(slots) == 0:
            return
        dele = [self.tidx[k, j] for k in slots for j in range(self.A) if self.tidx[k, j] >= 0]
        if dele:
            bs.traf.delete(np.array(dele, dtype=int))
        cre_id, cre_lat, cre_lon, cre_hdg, cre_spd, routes = [], [], [], [], [], []
        new_seats = []
        for k in slots:
            kind = self.kinds[k]
            if self.forced.get(k):
                scen = self.forced[k].pop(0)
                nint = len(scen.intruder_routes)
            elif self.pool_prob.get(kind, 0.0) > 0 and kind in self.pool and self.rng.random() < self.pool_prob[kind]:
                scen = copy.deepcopy(self.pool[kind][self.rng.integers(len(self.pool[kind]))])
                nint = len(scen.intruder_routes)
            elif kind == "sa":
                nint = int(self.rng.choice(self.sa_choices, p=self.sa_probs))
                scen = self._gen(nint, 1).generate(self.rng)
            else:
                nint = self.ma_intruders
                scen = self._gen(nint, self.ma_agents).generate(self.rng)
            self.scen[k] = scen
            self.slot_nint[k] = nint
            self.slot_t[k] = 0.0
            self.slot_gen[k] += 1
            geo = scenario_geometry(scen)
            if self.g is None:
                self._alloc_geometry(geo)
            for key, v in geo.items():
                self.g[key][k] = v
            self._sector_halfplanes(k)
            self.routes[k] = ScenarioRoutes({key: self.g[key][k] for key in geo})
            self.route_grid[k] = self.routes[k].grid_params()
            self.shapely[k] = None
            self.ids[k] = [None] * self.A
            self.is_agent[k] = False
            for j, spec in enumerate(scen.agents):
                acid = f"K{k}A{j}G{self.slot_gen[k] % 100}"
                self.ids[k][j] = acid
                self.is_agent[k, j] = True
                self.ac_goal[k, j] = spec.goal
                cre_id.append(acid)
                cre_lat.append(spec.start[0]); cre_lon.append(spec.start[1])
                cre_hdg.append(spec.heading); cre_spd.append(spec.speed)
            n_ag = len(scen.agents)
            for i, route in enumerate(scen.intruder_routes):
                acid = f"K{k}I{i}G{self.slot_gen[k] % 100}"
                j = n_ag + i
                self.ids[k][j] = acid
                spawn, nxt = route.waypoints[0], route.waypoints[1]
                hdg = fn.get_hdg(np.array(spawn), np.array(nxt))
                cre_id.append(acid)
                cre_lat.append(spawn[0]); cre_lon.append(spawn[1])
                cre_hdg.append(hdg); cre_spd.append(route.speed)
                routes.append((acid, route.waypoints[1:]))
            for seat in self.slot_seats[k]:
                j = self.seat_pos[seat]
                self.goal[seat] = scen.agents[j].goal
                self.orig[seat] = True
                new_seats.append(seat)
        bs.traf.cre(cre_id, actype="A320", aclat=np.array(cre_lat), aclon=np.array(cre_lon),
                    achdg=np.array(cre_hdg), acspd=np.array(cre_spd), acalt=ALTITUDE)
        for acid, wpts in routes:
            for lat, lon in wpts:
                bs.stack.stack(f"{acid} ADDWPT {lat} {lon}")
        self._refresh_index()
        self._init_seats(np.array(new_seats, dtype=int))

    def _clear_slots(self, slots):
        dele = [self.tidx[k, j] for k in slots for j in range(self.A) if self.tidx[k, j] >= 0]
        if dele:
            bs.traf.delete(np.array(dele, dtype=int))
        for k in slots:
            self.ids[k] = [None] * self.A
            self.active[self.slot_seats[k]] = False
        self._refresh_index()

    def _respawn(self, seats):
        """Continuous-traffic MA: put a fresh agent into each finished seat.
        Returns the set of slots where placement failed (they get regenerated)."""
        failed, pending, plans = set(), {}, []
        for seat in seats:
            k = int(self.seat_slot[seat])
            if k in failed:
                continue
            c = self.scen[k].center
            if self.shapely[k] is None:
                self.shapely[k] = (
                    Polygon(_poly_nm(self.scen[k].sector, c)),
                    [Polygon(_poly_nm(o.vertices, c)) for o in self.scen[k].obstacles])
            sector_poly, obstacle_polys = self.shapely[k]
            live = self.tidx[k][self.tidx[k] >= 0]
            ex, ey = latlon_to_nm(bs.traf.lat[live], bs.traf.lon[live], c[0], c[1])
            existing = list(zip(ex.tolist(), ey.tolist())) + pending.get(k, [])
            try:
                start, goal = self._gen(0, 1)._generate_agent(
                    self.rng, sector_poly, obstacle_polys, existing_starts=existing)
            except ValueError:
                failed.add(k)
                continue
            pending.setdefault(k, []).append(start)
            plans.append((seat, k, start, goal))
        plans = [p for p in plans if p[1] not in failed]
        if plans:
            ids, lats, lons, hdgs, new = [], [], [], [], []
            for seat, k, start, goal in plans:
                j = self.seat_pos[seat]
                c = self.scen[k].center
                s_ll = _nm_to_latlon(c, start)
                g_ll = _nm_to_latlon(c, goal)
                self.slot_gen[k] += 1
                acid = f"K{k}A{j}G{self.slot_gen[k] % 100}"
                self.ids[k][j] = acid
                self.ac_goal[k, j] = g_ll
                self.goal[seat] = g_ll
                self.orig[seat] = False
                ids.append(acid)
                lats.append(s_ll[0])
                lons.append(s_ll[1])
                hdgs.append(_bearing_nm(start, goal))
                new.append(seat)
            bs.traf.cre(ids, actype="A320", aclat=np.array(lats), aclon=np.array(lons),
                        achdg=np.array(hdgs), acspd=AC_SPD, acalt=ALTITUDE)
            self._refresh_index()
            self._init_seats(np.array(new, dtype=int))
        return failed

    def _init_seats(self, seats):
        if len(seats) == 0:
            return
        for key in METRIC_KEYS:
            self.metrics[key][seats] = 0.0
        self.in_intr[seats] = False
        self.obs_occ[seats] = False
        self.outside[seats] = False
        self.reached[seats] = False
        self.active[seats] = True
        self.ep_len[seats] = 0
        self.prev_act[seats] = -1
        for v in self.acc.values():
            v[seats] = 0.0
        # the official envs call _update_metrics(dt=0) at reset
        self._update_metrics(seats, 0.0)
        self.spawn_viol[seats] = np.stack([self.metrics[k][seats] for k in SPAWN_KEYS], axis=1)
        for s in seats.tolist():
            k = self.seat_slot[s]
            geo = {key: self.g[key][k] for key in ("obst", "obst_valid", "sector")}
            self.route_tgt[s] = route_target(self._goal_nm(np.array([s]))[0], geo)
            self.route_D[s] = self.routes[k].node_dist(self.route_tgt[s])
        if self.reward["geodesic"]:
            self._geodesic_phi(seats)

    # ------------------------------------------------------------ metrics
    def _update_metrics(self, seats, dt):
        """Vectorised copy of the official ``_update_metrics`` for ``seats``.
        Returns the seats that captured their waypoint during this update."""
        if len(seats) == 0:
            return seats
        n = len(seats)
        k = self.seat_slot[seats]
        j = self.seat_pos[seats]
        own = self.tidx[k, j]
        lat, lon = bs.traf.lat[own], bs.traf.lon[own]
        m = self.metrics
        m["flight_time"][seats] += dt

        # losses of separation with all other live aircraft of the scenario
        oth = self.tidx[k]
        ok = oth >= 0
        ok[np.arange(n), j] = False
        ot = np.where(ok, oth, own[:, None])
        _, d = kwikqdrdist(lat[:, None], lon[:, None], bs.traf.lat[ot], bs.traf.lon[ot])
        now = (d < SEP_NM) & ok
        m["intrusion_events"][seats] += np.sum(now & ~self.in_intr[seats], axis=1)
        m["intrusion_time"][seats] += np.sum(now, axis=1) * dt
        self.in_intr[seats] = now

        # restricted areas + sector, in the (affine-equivalent) local NM frame
        c = self.g["center"][k]
        x, y = latlon_to_nm(lat, lon, c[:, 0], c[:, 1])
        verts = self.g["obst"][k]
        valid = self.g["obst_valid"][k]
        inside = points_in_polygons(x, y, verts) & valid
        m["restricted_area_events"][seats] += np.sum(inside & ~self.obs_occ[seats], axis=1)
        m["time_in_restricted_area"][seats] += inside.any(axis=1) * dt
        self.obs_occ[seats] = inside
        sector = self.g["sector"][k][:, None]
        ins = points_in_polygons(x, y, sector)[:, 0]
        out = ~ins
        m["sector_exit_events"][seats] += out & ~self.outside[seats]
        m["time_outside_sector"][seats] += out * dt
        self.outside[seats] = out

        # waypoint capture
        _, dg = kwikqdrdist(lat, lon, self.goal[seats, 0], self.goal[seats, 1])
        newly = (dg * NM2KM < GOAL_MARGIN_KM) & ~self.reached[seats]
        self.reached[seats] |= newly
        m["waypoint_reached"][seats] = np.maximum(m["waypoint_reached"][seats], newly)

        # reward shaping accumulators (not part of the official metrics)
        if dt > 0:
            r = self.reward
            prox = np.clip((r["prox_ac_nm"] - d) / (r["prox_ac_nm"] - SEP_NM), 0.0, 1.0) ** 2
            self.acc["prox_ac"][seats] += np.sum(prox * ok, axis=1) * dt
            # obstacles: exact boundary distance only where the enclosing circle is close
            cen = self.g["obst_centroid"][k]
            dc = np.hypot(cen[..., 0] - x[:, None], cen[..., 1] - y[:, None]) - self.g["obst_radius"][k]
            near = np.flatnonzero(np.any(valid & (dc < r["prox_obs_nm"]), axis=1))
            if len(near):
                d_ob, _ = nearest_on_polygons(x[near], y[near], verts[near])
                sd = np.where(inside[near], -d_ob, d_ob)
                sd = np.min(np.where(valid[near], sd, np.inf), axis=1)
                self.acc["prox_obs"][seats[near]] += np.clip((r["prox_obs_nm"] - sd) / r["prox_obs_nm"], 0, 1) ** 2 * dt
            # sector is convex (cyclic polygon): inside distance = min half-plane distance
            sd_s = np.min(self.sec_off[k] - (self.sec_n[k][..., 0] * x[:, None] + self.sec_n[k][..., 1] * y[:, None]), axis=1)
            sd_s = np.where(ins, sd_s, -1.0)
            self.acc["prox_sec"][seats] += np.clip((r["prox_sec_nm"] - sd_s) / r["prox_sec_nm"], 0, 1) ** 2 * dt
            self.acc["dt"][seats] += dt
        if not self.reward["geodesic"]:
            self.phi[seats] = -np.maximum(dg * 1852.0 - GOAL_MARGIN_KM * 1000.0, 0.0) / self.reward["v_ref"]
        return seats[newly]

    # ------------------------------------------------------------ routing
    def _goal_nm(self, seats):
        c = self.g["center"][self.seat_slot[seats]]
        return np.stack(latlon_to_nm(self.goal[seats, 0], self.goal[seats, 1], c[:, 0], c[:, 1]), axis=1)

    def _route(self, seats):
        k = self.seat_slot[seats]
        own = self.tidx[k, self.seat_pos[seats]]
        c = self.g["center"][k]
        px, py = latlon_to_nm(bs.traf.lat[own], bs.traf.lon[own], c[:, 0], c[:, 1])
        gl = self.route_tgt[seats]
        return route_query(px, py, gl[:, 0], gl[:, 1], self.route_grid[k], self.route_D[seats])

    def _geodesic_phi(self, seats):
        """Progress potential on the route length (replaces the straight-line one)."""
        if len(seats) == 0:
            return
        geo, _, _ = self._route(seats)
        phi = -np.maximum(geo * 1852.0 - GOAL_MARGIN_KM * 1000.0, 0.0) / self.reward["v_ref"]
        self.phi[seats] = np.where(self.reached[seats], 0.0, phi)

    # -------------------------------------------------------- observation
    def _obs_inputs(self, seats):
        n = len(seats)
        k = self.seat_slot[seats]
        j = self.seat_pos[seats]
        own = self.tidx[k, j]
        oth = self.tidx[k]
        ok = oth >= 0
        ok[np.arange(n), j] = False
        ot = np.where(ok, oth, own[:, None])
        is_ag = self.is_agent[k]
        swl = bs.traf.swlnav[ot]
        s = {
            "own_lat": bs.traf.lat[own], "own_lon": bs.traf.lon[own],
            "own_hdg": bs.traf.hdg[own], "own_tas": bs.traf.tas[own],
            "own_cas": bs.traf.cas[own], "own_seltrk": bs.traf.ap.trk[own],
            "own_selspd": bs.traf.selspd[own],
            "goal_lat": self.goal[seats, 0], "goal_lon": self.goal[seats, 1],
            "mode_ma": self.seat_is_ma[seats].astype(float),
            # episode clock as the official envs define it: SA flight time,
            # MA shared scenario clock (the time limit applies to it)
            "elapsed_s": np.where(self.seat_is_ma[seats], self.slot_t[k], self.metrics["flight_time"][seats]),
            "center": self.g["center"][k], "sector": self.g["sector"][k],
            "obst": self.g["obst"][k], "obst_valid": self.g["obst_valid"][k],
            "obst_centroid": self.g["obst_centroid"][k], "obst_radius": self.g["obst_radius"][k],
            "oth_lat": bs.traf.lat[ot], "oth_lon": bs.traf.lon[ot],
            "oth_hdg": bs.traf.hdg[ot], "oth_tas": bs.traf.tas[ot],
            "oth_seltrk": bs.traf.ap.trk[ot],
            "oth_valid": ok, "oth_is_agent": is_ag,
            "oth_has_intent": np.where(is_ag, True, swl),
            "oth_intent_lat": np.where(is_ag, self.ac_goal[k, :, 0], bs.traf.actwp.lat[ot]),
            "oth_intent_lon": np.where(is_ag, self.ac_goal[k, :, 1], bs.traf.actwp.lon[ot]),
        }
        s["route_dist"], s["route_next"], s["route_has"] = self._route(seats)
        return s

    def _observe(self, seats):
        if len(seats) == 0:
            return np.zeros((0, OBS_DIM), np.float32), {"n_pred_short": np.zeros(0)}
        return build_observations(self._obs_inputs(seats))

    # ----------------------------------------------------------- gym API
    def reset(self):
        self._reset_slots(list(range(self.K)))
        obs, _ = self._observe(np.arange(self.n_seats))
        return obs

    def step(self, actions):
        """actions (n_seats, 2) -> obs, rewards, dones, truncated, infos(dict seat->info)."""
        r = self.reward
        B = self.n_seats
        act = np.flatnonzero(self.active)
        k, j = self.seat_slot[act], self.seat_pos[act]
        own = self.tidx[k, j]
        hdg, cas = action_targets(np.asarray(actions)[act], bs.traf.lat[own], bs.traf.lon[own],
                                  self.goal[act, 0], self.goal[act, 1])
        apply_targets(own, hdg, cas)

        base = {key: v.copy() for key, v in self.metrics.items()}
        phi0 = self.phi.copy()
        for v in self.acc.values():
            v[:] = 0.0
        stepped = self.active.copy()
        terminated = np.zeros(B, dtype=bool)

        for _ in range(self.action_frequency):
            bs.sim.step()
            self.slot_t += 1.0
            act = np.flatnonzero(self.active)
            newly = self._update_metrics(act, 1.0)
            if len(newly):
                self.active[newly] = False
                terminated[newly] = True
                ma_new = newly[self.seat_is_ma[newly]]
                if len(ma_new):   # MA: the reached aircraft leaves the simulation
                    kk, jj = self.seat_slot[ma_new], self.seat_pos[ma_new]
                    bs.traf.delete(np.array(self.tidx[kk, jj], dtype=int))
                    for a, b in zip(kk, jj):
                        self.ids[a][b] = None
                    self._refresh_index()

        if r["geodesic"]:     # reached MA aircraft have left the simulation: potential 0
            self._geodesic_phi(np.flatnonzero(stepped & ~self.reached & (self.tidx[self.seat_slot, self.seat_pos] >= 0)))
            self.phi[stepped & self.reached] = 0.0

        # truncation
        truncated = np.zeros(B, dtype=bool)
        sa_seat = ~self.seat_is_ma
        # running out of the time budget is a task failure: the elapsed time is
        # observed, so it is treated as terminal (penalised, not bootstrapped)
        timeout = sa_seat & self.active & (self.metrics["flight_time"] >= EPISODE_TIME_LIMIT)
        truncated |= timeout
        ma_reset_slots = []
        for kk, kind in enumerate(self.kinds):
            if kind != "ma":
                continue
            seats = self.slot_seats[kk]
            orig_left = np.any(self.active[seats] & self.orig[seats])
            if (self.slot_t[kk] >= EPISODE_TIME_LIMIT or not orig_left
                    or (not self.respawn and not np.any(self.active[seats]))):
                ma_reset_slots.append(kk)
                truncated[seats] |= self.active[seats]
                if self.slot_t[kk] >= EPISODE_TIME_LIMIT:
                    timeout[seats] |= self.active[seats]

        # reward over this decision interval
        m = self.metrics
        d = {key: m[key] - base[key] for key in METRIC_KEYS}
        rew = (-r["time"] * self.acc["dt"]
               + r["progress"] * (self.phi - phi0)
               + r["goal"] * d["waypoint_reached"]
               - r["lam_intr"] * (r["intr_event"] * d["intrusion_events"] + r["intr_time"] * d["intrusion_time"])
               - r["lam_ra"] * (r["ra_event"] * d["restricted_area_events"] + r["ra_time"] * d["time_in_restricted_area"])
               - r["lam_sec"] * (r["sec_event"] * d["sector_exit_events"] + r["sec_time"] * d["time_outside_sector"])
               - r["prox_ac"] * self.acc["prox_ac"]
               - r["prox_obs"] * self.acc["prox_obs"]
               - r["prox_sec"] * self.acc["prox_sec"])
        rew -= r["timeout"] * timeout
        # indecision costs: flipping the heading offset straight across the goal
        # bearing, and jumping across speed bins (not potential-based: they break
        # near-ties between equally good options, so keep them small)
        a = np.asarray(actions)
        had = stepped & (self.prev_act[:, 0] >= 0)
        if r["reversal"] or r["speed_jump"]:
            o_now = HEADING_OFFSETS[a[:, 0]]
            o_prev = HEADING_OFFSETS[np.maximum(self.prev_act[:, 0], 0)]
            rev = np.where(o_now * o_prev < 0, np.minimum(np.abs(o_now), np.abs(o_prev)) / 90.0, 0.0)
            jump = np.maximum(np.abs(a[:, 1] - self.prev_act[:, 1]) - 1, 0)
            rew -= np.where(had, r["reversal"] * rev + r["speed_jump"] * jump, 0.0)
        self.prev_act[stepped] = a[stepped]
        rew = np.where(stepped, rew, 0.0)

        done = terminated | truncated
        cont = np.flatnonzero(stepped & ~done)
        obs_cont, aux = self._observe(cont)
        rew[cont] -= r["pred_conflict"] * aux["n_pred_short"] * self.acc["dt"][cont]
        m["total_reward"] += rew
        self.ep_len += stepped

        infos = {}
        done_idx = np.flatnonzero(done)
        if len(done_idx):
            trunc_idx = np.flatnonzero(truncated)
            if len(trunc_idx):
                term_obs, _ = self._observe(trunc_idx)
                term_map = dict(zip(trunc_idx.tolist(), term_obs))
            else:
                term_map = {}
            for seat in done_idx.tolist():
                info = {
                    "episode": {"r": float(m["total_reward"][seat]), "l": int(self.ep_len[seat]), "t": 0.0},
                    "metrics": {key: float(m[key][seat]) for key in METRIC_KEYS},
                    "mode": "ma" if self.seat_is_ma[seat] else "sa",
                    "orig": bool(self.orig[seat]),
                    "spawn": dict(zip(SPAWN_KEYS, self.spawn_viol[seat].tolist())),
                    "n_intruders": int(self.slot_nint[self.seat_slot[seat]]),
                    "TimeLimit.truncated": bool(truncated[seat] and not timeout[seat]),
                    "timeout": bool(timeout[seat]),
                }
                if seat in term_map:
                    info["terminal_observation"] = term_map[seat]
                infos[seat] = info
            self.active[done_idx] = False

        # start new episodes
        sa_reset = sorted({int(self.seat_slot[s]) for s in done_idx if not self.seat_is_ma[s]})
        resets = sa_reset + ma_reset_slots
        if not self.auto_reset:
            self._clear_slots(resets)
            obs = np.zeros((B, OBS_DIM), np.float32)
            obs[cont] = obs_cont
            return obs, rew.astype(np.float32), done, truncated, infos
        self._reset_slots(resets)
        respawn = [s for s in done_idx.tolist()
                   if self.seat_is_ma[s] and self.seat_slot[s] not in ma_reset_slots]
        if respawn and self.respawn:
            failed = self._respawn(respawn)
            if failed:
                extra = sorted(failed)
                for kk in extra:
                    seats = self.slot_seats[kk]
                    live = seats[self.active[seats]]
                    if len(live):   # truncate live agents of the failed slot (rare)
                        tob, _ = self._observe(live)
                        for s, o in zip(live.tolist(), tob):
                            infos[s] = {
                                "episode": {"r": float(m["total_reward"][s]), "l": int(self.ep_len[s]), "t": 0.0},
                                "metrics": {key: float(m[key][s]) for key in METRIC_KEYS},
                                "mode": "ma", "orig": bool(self.orig[s]),
                                "spawn": dict(zip(SPAWN_KEYS, self.spawn_viol[s].tolist())),
                                "n_intruders": int(self.slot_nint[kk]),
                                "TimeLimit.truncated": True, "terminal_observation": o,
                            }
                        done[live] = True
                        truncated[live] = True
                        self.active[live] = False
                self._reset_slots(extra)

        obs = np.zeros((B, OBS_DIM), np.float32)
        obs[cont] = obs_cont
        new_ep = np.flatnonzero(done & self.active)
        if len(new_ep):
            o, _ = self._observe(new_ep)
            obs[new_ep] = o
        return obs, rew.astype(np.float32), done, truncated, infos

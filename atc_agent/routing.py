"""
Geodesic (obstacle-aware) distance to the goal, for reward shaping and as a
route hint in the observation. Not a controller: the policy still chooses every
heading / speed command; this only replaces the straight-line distance in the
progress potential (which pulls aircraft *into* concave obstacle pockets, where
the 1B-step policy times out) and tells the policy which way the shortest
restricted-area-avoiding route goes.

Per scenario and goal, a cost-to-go field is computed on a GRID x GRID grid
covering the sector (16-connected Dijkstra, scipy). Restricted areas and the
outside of the sector are not forbidden but cost ``COST_INSIDE`` times more per
NM, so the field is finite everywhere (enclosed goals, aircraft inside an
area) and a crossing is only "chosen" when going around is much longer.
Queries interpolate the field bilinearly, so the potential is continuous in
the aircraft position; the route hint is the steepest-descent direction.
All distances in NM, in the scenario's local (north, east) frame.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from atc_agent.geometry import points_in_polygons

GRID = 96
COST_INSIDE = 5.0
MARGIN_NM = 10.0
_OFFS = [(di, dj) for di in range(-2, 3) for dj in range(-2, 3)
         if (di, dj) != (0, 0) and np.gcd(abs(di), abs(dj)) == 1]       # 16-connectivity


class ScenarioRoutes:
    """Grid graph of one scenario (``geo`` = features.scenario_geometry output)."""

    def __init__(self, geo):
        sec = geo["sector"]
        lo = sec.min(axis=0) - MARGIN_NM
        hi = sec.max(axis=0) + MARGIN_NM
        self.res = float(np.max(hi - lo) / (GRID - 1))
        self.origin = lo
        ii, jj = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
        x = self.origin[0] + ii.ravel() * self.res
        y = self.origin[1] + jj.ravel() * self.res
        n = GRID * GRID
        inside_ob = (points_in_polygons(x, y, np.broadcast_to(geo["obst"], (n,) + geo["obst"].shape))
                     & geo["obst_valid"][None]).any(axis=1)
        in_sec = points_in_polygons(x, y, np.broadcast_to(geo["sector"][None], (n, 1) + geo["sector"].shape))[:, 0]
        cost = np.where(inside_ob | ~in_sec, COST_INSIDE, 1.0)
        ii, jj = ii.ravel(), jj.ravel()
        rows, cols, w = [], [], []
        for di, dj in _OFFS:
            ok = (ii + di >= 0) & (ii + di < GRID) & (jj + dj >= 0) & (jj + dj < GRID)
            a = (ii * GRID + jj)[ok]
            b = ((ii + di) * GRID + (jj + dj))[ok]
            rows.append(a)
            cols.append(b)
            w.append(np.hypot(di, dj) * self.res * 0.5 * (cost[a] + cost[b]))
        self.graph = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self._cache = {}

    def node_dist(self, goal):
        """Cost-to-go field (GRID*GRID,) to ``goal`` (NM frame)."""
        key = (round(float(goal[0]), 6), round(float(goal[1]), 6))
        if key in self._cache:
            return self._cache[key]
        fi = np.clip((goal[0] - self.origin[0]) / self.res, 0, GRID - 1.001)
        fj = np.clip((goal[1] - self.origin[1]) / self.res, 0, GRID - 1.001)
        i0, j0 = int(fi), int(fj)
        src = [(i0 + a) * GRID + (j0 + b) for a in (0, 1) for b in (0, 1)]
        field = dijkstra(self.graph, directed=True, indices=src, min_only=True).astype(np.float32)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = field
        return field

    def grid_params(self):
        return np.array([self.origin[0], self.origin[1], self.res])


def route_query(px, py, gx, gy, grid, field):
    """Batched query.

    px, py, gx, gy : (B,) own position / goal (NM frame)
    grid  : (B, 3) [origin_n, origin_e, res] of each aircraft's scenario grid
    field : (B, GRID*GRID) cost-to-go fields
    Returns geodesic distance (B,), a point 5 NM along the steepest descent (B, 2),
    and has_route (B,) (False only outside the grid, then straight line).
    """
    B = len(px)
    fi = (px - grid[:, 0]) / grid[:, 2]
    fj = (py - grid[:, 1]) / grid[:, 2]
    has = (fi >= 0) & (fi <= GRID - 1) & (fj >= 0) & (fj <= GRID - 1)
    fi = np.clip(fi, 0, GRID - 1.001)
    fj = np.clip(fj, 0, GRID - 1.001)
    i0, j0 = fi.astype(int), fj.astype(int)
    u, v = fi - i0, fj - j0
    r = np.arange(B)
    f00 = field[r, i0 * GRID + j0]
    f10 = field[r, (i0 + 1) * GRID + j0]
    f01 = field[r, i0 * GRID + j0 + 1]
    f11 = field[r, (i0 + 1) * GRID + j0 + 1]
    geo = (1 - u) * (1 - v) * f00 + u * (1 - v) * f10 + (1 - u) * v * f01 + u * v * f11
    dn = ((1 - v) * (f10 - f00) + v * (f11 - f01)) / grid[:, 2]
    de = ((1 - u) * (f01 - f00) + u * (f11 - f10)) / grid[:, 2]
    norm = np.hypot(dn, de)
    euclid = np.hypot(gx - px, gy - py)
    near = euclid < 2 * grid[:, 2]                      # inside the source cells: aim at the goal
    dirn = np.where(near | (norm < 1e-9), (gx - px) / np.maximum(euclid, 1e-9), -dn / np.maximum(norm, 1e-9))
    dire = np.where(near | (norm < 1e-9), (gy - py) / np.maximum(euclid, 1e-9), -de / np.maximum(norm, 1e-9))
    geo = np.where(has, np.maximum(geo, euclid), euclid)
    nxt = np.stack([px + 5.0 * dirn, py + 5.0 * dire], axis=1)
    return geo, nxt, has

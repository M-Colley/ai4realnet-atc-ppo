"""
Where does multi-agent flight time above the obstacle-aware route go?

Replays the official seed-42 MA scenarios with the deterministic policy, logging
every decision's position and action. For each agent that reaches its goal, the
time above the route bound (route length at the policy's own top ground speed
``v_top``) is split exactly into

    path  = (flown length - route length) / v_top
    speed = flight time - flown length / v_top     (slow flying, turns, acceleration)

where the route is the shortest path that never enters a restricted area: the
least-cost path on the route grid (atc_agent.routing) restricted to area-free
edges, string-pulled, ending at the 5 km capture radius. Each flight is also
tagged with

* other_side: it passes some restricted area on the other side than the route
  (non-zero winding number of flown path + reversed route around the area),
* loop: the ground track crosses itself,
* side_flips: changes of the heading offset's sign (zero offsets skipped; also
  counted for offsets of at least 22 deg only),
* turn_deg / net_turn_deg: total absolute / net turning of the ground track,
* hint_depth_nm: how deep the route-hint field's own least-cost path enters an area,
* traffic: decisions with another aircraft within 10 NM,
* top_speed: share of decisions at the fastest speed bin.

Flights whose straight line crosses a restricted area are split into the
slowest 10% (by time above the bound) and the rest.

    python -m atc_agent.detour_diag --model runs/ft_geo_r2/final.zip --out runs/diag/detour
"""

import argparse
import os

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from atc_agent.actions import HEADING_OFFSETS, SPEED_TARGETS
from atc_agent.fast_eval import FastEvaluator
from atc_agent.features import scenario_geometry, latlon_to_nm
from atc_agent.plot_flights import replay, plot
from atc_agent.routing import ScenarioRoutes, CAPTURE_NM, GRID

DT = 10.0            # sim seconds per decision
TRAFFIC_NM = 10.0


def node_xy(R):
    n = np.arange(GRID * GRID)
    return R.origin + np.stack([n // GRID, n % GRID], axis=1) * R.res


def free_graph(R, areas):
    """The route grid with only the edges that stay clear of the restricted areas
    (Euclidean weights): routes on it never enter an area."""
    G = R.graph.tocoo()
    xy = node_xy(R)
    keep = ~shapely.intersects(shapely.linestrings(np.stack([xy[G.row], xy[G.col]], axis=1)), areas)
    w = np.hypot(*(xy[G.row] - xy[G.col]).T)
    return csr_matrix((w[keep], (G.row[keep], G.col[keep])), shape=G.shape)


def grid_path(graph, xy, start, goal):
    """Least-cost node path on ``graph`` from the node nearest ``start`` to the
    cheapest node within the capture radius of ``goal`` (None if unreachable)."""
    s = int(np.argmin(np.hypot(*(xy - start).T)))
    dist, pred = dijkstra(graph, directed=True, indices=s, return_predecessors=True)
    cand = np.flatnonzero(np.hypot(*(xy - goal).T) <= CAPTURE_NM)
    cand = cand[np.isfinite(dist[cand])]
    if not len(cand):
        return None
    seq = [int(cand[np.argmin(dist[cand])])]
    while seq[-1] != s:
        seq.append(int(pred[seq[-1]]))
    return xy[seq[::-1]]


def pull(pts, areas):
    """String pulling: replace path pieces by straight segments that miss ``areas``."""
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        segs = shapely.linestrings(np.stack([np.repeat(pts[i:i + 1], len(pts) - i - 1, axis=0), pts[i + 1:]], axis=1))
        clear = np.flatnonzero(~shapely.intersects(segs, areas))
        i = i + 1 + (clear.max() if len(clear) else 0)
        out.append(pts[i])
    return np.array(out)


def length_to_capture(path, goal, step=0.05):
    """Length along ``path`` until it first comes within the capture radius of ``goal``."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.arange(0.0, seg.sum(), step)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    pts = np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], axis=1)
    inside = np.flatnonzero(np.hypot(*(pts - goal).T) <= CAPTURE_NM)
    return float(s[inside[0]]) if len(inside) else float(seg.sum())


def winding(loop, q):
    """Winding number of the closed polyline ``loop`` (N, 2) around point ``q``."""
    a = np.arctan2(loop[:, 1] - q[1], loop[:, 0] - q[0])
    d = np.diff(np.append(a, a[:1]))
    return float(np.sum((d + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi))


def analyse(scens, results):
    rows = []
    for k, (sc, res) in enumerate(zip(scens, results)):
        geo = scenario_geometry(sc)
        areas = unary_union([Polygon(ob) for ob, ok in zip(geo["obst"], geo["obst_valid"]) if ok])
        parts = list(getattr(areas, "geoms", [areas]))
        shapely.prepare(areas)
        R = ScenarioRoutes(geo)
        xy = node_xy(R)
        G = free_graph(R, areas)
        tr = res["tracks"]
        ids = sorted(tr)
        P = np.full((len(ids), max(len(t) for t in tr.values()), 2), np.nan)
        for n, j in enumerate(ids):
            P[n, :len(tr[j])] = tr[j]
        D = np.hypot(*(P[:, None] - P[None]).transpose(3, 0, 1, 2))      # (A, A, T)
        D[np.arange(len(ids)), np.arange(len(ids))] = np.nan
        with np.errstate(invalid="ignore"):
            near_t = np.nan_to_num(D, nan=np.inf).min(axis=1) < TRAFFIC_NM  # (A, T)
        for n, j in enumerate(ids):
            track = tr[j]
            if not res["agent"][j]:
                continue
            m = res["metrics"][j]
            start = track[0]
            goal = np.array(latlon_to_nm(*sc.agents[j].goal, *sc.center))
            crosses = any(LineString([start, goal]).intersects(p) for p in parts)
            seg = np.hypot(*np.diff(track, axis=0).T)
            nodes = grid_path(G, xy, start, goal)
            if nodes is None:           # no area-free route (goal deep inside an area)
                route_nm, other = np.nan, np.nan
            else:
                route = pull(np.vstack([start, nodes]), areas)
                route_nm = length_to_capture(route, goal)
                loop = np.vstack([track, goal[None], route[::-1]])
                other = any(abs(winding(loop, np.array(p.representative_point().coords[0]))) > 0.5 for p in parts)
            hint = grid_path(R.graph, xy, start, goal)      # the policy's route-hint field
            inside = shapely.contains_xy(areas, hint[:, 0], hint[:, 1])
            hint_depth = float(shapely.distance(areas.boundary, shapely.points(hint[inside])).max()) if inside.any() else 0.0
            off = HEADING_OFFSETS[res["actions"][j][:, 0]]
            s = np.sign(off[off != 0])
            s22 = np.sign(off[np.abs(off) >= 22])
            hdg = np.unwrap(np.arctan2(*np.diff(track, axis=0)[:, ::-1].T)) if len(track) > 2 else np.zeros(1)
            near = int(near_t[n, :len(track)].sum())
            rows.append(dict(
                scenario=k, agent=j, reached=int(m["waypoint_reached"] == 1), flight_time=m["flight_time"],
                violations=m["intrusion_events"] + m["restricted_area_events"] + m["sector_exit_events"],
                crosses_ra=crosses, d_str=float(np.hypot(*(goal - start))), route_nm=route_nm, hint_depth_nm=hint_depth,
                flown_nm=float(seg.sum()), gs_kt_p99=float(np.quantile(seg / DT * 3600, 0.99)) if len(seg) else np.nan,
                other_side=other, side_flips=int(np.sum(s[1:] != s[:-1])), side_flips_22=int(np.sum(s22[1:] != s22[:-1])),
                turn_deg=float(np.degrees(np.abs(np.diff(hdg)).sum())),
                net_turn_deg=float(np.degrees(abs(hdg[-1] - hdg[0]))),
                loop=len(track) > 3 and not LineString(track).is_simple,
                traffic=int(near), top_speed=float(np.mean(res["actions"][j][:, 1] == len(SPEED_TARGETS) - 1)),
                n_decisions=len(off)))
    return pd.DataFrame(rows)


def summarise(df, v_top_kt):
    v = v_top_kt / 3600.0
    ok = df[df.reached == 1].copy()
    ok["bound_s"] = ok.route_nm / v
    ok["excess_s"] = ok.flight_time - ok.bound_s
    ok["path_s"] = (ok.flown_nm - ok.route_nm) / v
    ok["speed_s"] = ok.flight_time - ok.flown_nm / v
    groups = {}
    cr = ok[ok.crosses_ra]
    cut = cr.excess_s.quantile(0.9)
    groups["crosses RA, slowest 10%"] = cr[cr.excess_s >= cut]
    groups["crosses RA, other 90%"] = cr[cr.excess_s < cut]
    groups["clear of RA"] = ok[~ok.crosses_ra]
    lines = [f"v_top = {v_top_kt:.1f} kt (median over agents of each one's 99th-percentile 10 s ground speed); "
             f"{len(ok)} of {len(df)} agents reached the goal; {len(cr)} of them have a straight line crossing an RA.", "",
             "| Group | n | excess [s] mean | path [s] | speed [s] | share of all excess | other side [%] | loop [%] | "
             "side flips (all / >=22 deg) | turn total / net [deg] | traffic decisions | top-speed share |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    tot = ok.excess_s.sum()
    for name, g in groups.items():
        lines.append(f"| {name} | {len(g)} | {g.excess_s.mean():.0f} | {g.path_s.mean():.0f} | {g.speed_s.mean():.0f} | "
                     f"{g.excess_s.sum() / tot:.2f} | {100 * g.other_side.mean():.0f} | {100 * g.loop.mean():.0f} | "
                     f"{g.side_flips.mean():.1f} / {g.side_flips_22.mean():.1f} | {g.turn_deg.mean():.0f} / "
                     f"{g.net_turn_deg.mean():.0f} | {g.traffic.mean():.1f} | {g.top_speed.mean():.2f} |")
    tail = groups["crosses RA, slowest 10%"]
    other, loop = tail.other_side.astype(bool), tail.loop.astype(bool)
    lines += ["", "Slowest 10% of RA-crossing flights: flights (share of the tail's excess)", "",
              "| | loop | no loop |", "| --- | ---: | ---: |"]
    for name, o in (("other side than the shortest route", other), ("same side", ~other)):
        cells = [f"{int((o & l).sum())} ({tail.excess_s[o & l].sum() / tail.excess_s.sum():.2f})" for l in (loop, ~loop)]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines.append(f"\nSpeed part larger than path part: {int((tail.speed_s > tail.path_s).sum())} tail flights.")
    return ok, "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=25, help="MA scenarios flown together")
    p.add_argument("--check", default=None, help="fast_eval MA CSV that the replay must reproduce")
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    from atc_agent.inference import load_model
    model = load_model(a.model, device="cpu")
    ev = FastEvaluator.official(0, a.episodes, seed=a.seed)
    dfs, kept = [], {}
    for i in range(0, len(ev.ma), a.chunk):
        scens = ev.ma[i:i + a.chunk]
        res = replay(lambda o: model.predict(o, deterministic=True)[0], scens, "ma")
        d = analyse(scens, res)
        d["scenario"] += i
        dfs.append(d)
        kept.update({i + k: (sc, r) for k, (sc, r) in enumerate(zip(scens, res))})
        print(f"{i + len(scens)}/{len(ev.ma)} scenarios", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    if a.check:
        ref = pd.read_csv(a.check).sort_values("episode_index")
        idx = df.scenario * 10 + df.agent
        r = ref.set_index("episode_index").loc[idx]
        bad = int((np.abs(r.flight_time.to_numpy() - df.flight_time.to_numpy()) > 1e-6).sum()
                  + (r.waypoint_reached.to_numpy() != df.reached.to_numpy()).sum())
        print(f"replay vs {a.check}: {bad} mismatching agent-episodes of {len(df)}")
    v_top = float(np.nanmedian(df.gs_kt_p99))
    ok, text = summarise(df, v_top)
    df.to_csv(a.out + "_agents.csv", index=False)
    with open(a.out + ".md", "w") as f:
        f.write(f"# Detour diagnosis: {a.model}, seed {a.seed}, {a.episodes} MA scenarios\n\n{text}\n")
    print(text)

    # eight slowest RA-crossing flights, one per scenario
    worst = ok[ok.crosses_ra].sort_values("excess_s", ascending=False).drop_duplicates("scenario").head(8)
    scens = [kept[s][0] for s in worst.scenario]
    results = [kept[s][1] for s in worst.scenario]
    titles = [f"ma #{s} agent {j}: +{e:.0f} s, path {pth:.0f} s, speed {spd:.0f} s, flips {f}"
              for s, j, e, pth, spd, f in zip(worst.scenario, worst.agent, worst.excess_s, worst.path_s,
                                               worst.speed_s, worst.side_flips)]
    plot(scens, results, titles, a.out + "_worst.png")


if __name__ == "__main__":
    main()

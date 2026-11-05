"""
Why do multi-agent flights fail? A causal taxonomy of the official-stream MA
failures (agents that time out at 3000 s).

Each failed agent is re-flown in three counterfactual settings:

* full / greedy   -- the original scenario, deterministic policy (must reproduce the failure),
* solo / greedy   -- the same agent alone (the other nine aircraft removed),
* solo / sampled  -- alone, actions sampled from the policy's own distribution (K seeds),
* full / sampled  -- the original scenario, every agent sampling (K seeds),

and tagged with the geometry of its goal and of where it gets stuck. Classes,
assigned in this order:

1. goal deep in RA    -- the goal lies inside a restricted area and the 5 km capture
                         circle lies entirely inside it: reaching it needs an RA entry.
2. goal shallow in RA -- the goal lies inside a restricted area, but part of the
                         capture circle is outside it.
3. no free route      -- the goal is outside every area, yet no route on the
                         area-free grid reaches the capture circle.
4. traffic            -- the agent reaches its goal when flown alone: the other
                         aircraft cause the failure.
5. greedy trap        -- fails alone with greedy decoding, but at least half of the
                         sampled solo flights reach the goal: the policy's own
                         distribution knows a way out, the argmax cycles.
6. navigation         -- fails alone under greedy and (mostly) sampled decoding.

    python -m atc_agent.failure_diag --model runs/ft_geo_r2/final.zip \
        --csv runs/ft_geo_r2/eval/fast_final_ma.csv --agents-csv runs/diag/detour_1185_v2_agents.csv \
        --out runs/diag/fail_1185
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import shapely
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from atc_agent.detour_diag import free_graph, grid_path, length_to_capture, node_xy, pull
from atc_agent.fast_eval import FastEvaluator
from atc_agent.features import latlon_to_nm, scenario_geometry
from atc_agent.plot_flights import replay
from atc_agent.routing import CAPTURE_NM, ScenarioRoutes

DT = 10.0              # sim seconds per decision
LATE = 180             # decisions in the last 30 min of a 3000 s flight
TRAFFIC_NM = 10.0
CLASSES = ["goal deep in RA", "goal shallow in RA", "no free route", "traffic", "greedy trap", "navigation"]


def fly(predict, scens, chunk, n_agents=10):
    out = []
    for i in range(0, len(scens), chunk):
        out += replay(predict, scens[i:i + chunk], "ma", n_agents=n_agents)
    return out


def reached(res, j):
    m = res["metrics"].get(j)
    return bool(m is not None and m["waypoint_reached"] == 1)


def free_route(sc, start, goal):
    """Shortest route from ``start`` to the capture circle of ``goal`` that never enters a
    restricted area (string-pulled area-free grid path; None if there is none), plus the
    pieces needed for distance-to-go queries."""
    geo = scenario_geometry(sc)
    areas = unary_union([Polygon(ob) for ob, ok in zip(geo["obst"], geo["obst_valid"]) if ok])
    shapely.prepare(areas)
    R = ScenarioRoutes(geo)
    xy = node_xy(R)
    G = free_graph(R, areas)
    nodes = grid_path(G, xy, start, goal)
    route = pull(np.vstack([start, nodes]), areas) if nodes is not None else None
    return route, areas, Polygon(geo["sector"]), xy, G


def geometry(sc, res, j):
    """Goal and track geometry of agent ``j`` in its (greedy, full) replay ``res``."""
    track = res["tracks"][j]
    start = track[0]
    goal = np.array(latlon_to_nm(*sc.agents[j].goal, *sc.center))
    route, areas, sector, xy, G = free_route(sc, start, goal)
    gp = Point(goal)
    in_ra = bool(areas.contains(gp))
    depth = float(areas.boundary.distance(gp)) if in_ra else 0.0
    route_nm = length_to_capture(route, goal) if route is not None else np.nan
    # area-free distance-to-go field (reversed graph, multi-source from the capture circle)
    cand = np.flatnonzero(np.hypot(*(xy - goal).T) <= CAPTURE_NM)
    d2g_nodes = dijkstra(G.T.tocsr(), directed=True, indices=cand, min_only=True) if len(cand) else np.full(len(xy), np.inf)
    fin = np.isfinite(d2g_nodes)

    def d2go(p):
        if not fin.any():
            return np.nan
        d = np.hypot(*(xy[fin] - p).T)
        return float((d2g_nodes[fin] + d).min())

    d_goal = np.hypot(*(track - goal).T)
    late = track[-LATE:]
    hdg = np.unwrap(np.arctan2(*np.diff(late, axis=0)[:, ::-1].T))
    cen = late.mean(axis=0)
    others = [k for k in res["tracks"] if k != j and res["agent"].get(k)]
    T = len(track)
    near = np.zeros(T, dtype=bool)
    nearest = np.full(T, np.inf)
    for k in others:
        tk = res["tracks"][k]
        n = min(T, len(tk))
        d = np.hypot(*(tk[:n] - track[:n]).T)
        nearest[:n] = np.minimum(nearest[:n], d)
        near[:n] |= d < TRAFFIC_NM
    alive_late = sum(len(res["tracks"][k]) > T - LATE for k in others)
    d0, dmin = d2go(start), min(d2go(p) for p in track[::3])
    if not np.isfinite(route_nm):           # no area-free route: distance-to-go undefined
        d0 = dmin = np.nan
    return dict(
        goal_in_ra=in_ra, goal_depth_nm=depth, goal_to_sector_edge_nm=float(sector.exterior.distance(gp)),
        goal_in_sector=bool(sector.contains(gp)), d_straight_nm=float(np.hypot(*(goal - start))),
        crosses_ra=bool(LineString([start, goal]).intersects(areas)), route_nm=route_nm,
        detour_ratio=route_nm / float(np.hypot(*(goal - start))),
        closest_nm=float(d_goal.min()), t_closest_s=float(np.argmin(d_goal) * DT), final_nm=float(d_goal[-1]),
        d2go_start_nm=d0, d2go_best_nm=dmin, progress=(1.0 - dmin / d0) if d0 and np.isfinite(d0) else np.nan,
        late_radius_nm=float(np.sqrt(((late - cen) ** 2).sum(axis=1).mean())),
        late_orbits=float(abs(hdg[-1] - hdg[0]) / (2 * np.pi)) if len(hdg) else 0.0,
        late_centre_to_goal_nm=float(np.hypot(*(cen - goal))),
        late_centre_to_ra_nm=float(areas.boundary.distance(Point(cen))),
        late_centre_to_sector_nm=float(sector.exterior.distance(Point(cen))),
        late_traffic_share=float(near[-LATE:].mean()), late_nearest_nm=float(np.median(nearest[-LATE:])),
        others_alive_late=int(alive_late), loop=len(track) > 3 and not LineString(track).is_simple)


def classify(r):
    if r.goal_in_ra:
        return CLASSES[0] if r.goal_depth_nm >= CAPTURE_NM else CLASSES[1]
    if not np.isfinite(r.route_nm):
        return CLASSES[2]
    if r.solo_greedy:
        return CLASSES[3]
    return CLASSES[4] if r.solo_sampled >= 0.5 else CLASSES[5]


def markdown(tab):
    """DataFrame -> markdown table (index as first column, floats to 2 decimals)."""
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)    # noqa: E731
    head = ["class"] + list(tab.columns)
    rows = [[str(i)] + [fmt(v) for v in r] for i, r in zip(tab.index, tab.itertuples(index=False))]
    return "\n".join(["| " + " | ".join(head) + " |", "| --- |" + " ---: |" * len(tab.columns)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def hazard(agents_csv, df, bins=(0.0, 1.05, 1.2, 1.5, 2.0, np.inf)):
    """Failure rate of agents whose goal is outside every area, by detour ratio
    (area-free route / straight line), over all agents of the stream (detour_diag output)."""
    d = pd.read_csv(agents_csv)
    key = d.scenario * 10 + d.agent
    ra_goal = set((df.scenario * 10 + df.agent)[df.cls.isin(CLASSES[:2])])
    d = d[~key.isin(ra_goal)].copy()
    d["fail"] = d.reached == 0
    d["ratio"] = d.route_nm / d.d_str
    d["bin"] = pd.cut(d.ratio, list(bins), right=False)
    g = d.groupby("bin", observed=True)
    tab = pd.DataFrame({"agents": g.size(), "share of agents": g.size() / len(d), "failures": g.fail.sum(),
                        "failure rate [%]": 100 * g.fail.mean()})
    tab.index = [f"{iv.left:g} - {iv.right:g}" for iv in tab.index]
    hi, lo = d.ratio >= 1.5, d.ratio < 1.5
    return "\n".join([f"Failure rate by detour ratio (area-free route / straight line), {len(d)} agents whose goal "
                      f"is outside every area ({int(d.ratio.isna().sum())} without an area-free route on the grid "
                      f"left out, {int(d.fail[d.ratio.isna()].sum())} of them failed):", "",
                      markdown(tab).replace("| class |", "| detour ratio |"), "",
                      f"Detour ratio >= 1.5: {100 * hi.mean():.1f}% of these agents, {int(d.fail[hi].sum())} of "
                      f"{int(d.fail.sum())} failures; failure rate {100 * d.fail[hi].mean():.2f}% vs "
                      f"{100 * d.fail[lo].mean():.3f}% below."])


def plot_cases(df, scens, full, solo, path, ncols=6, size=3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(df)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, df.iterrows()):
        sc, res = scens[r.scenario], full[r.scenario]
        geo = scenario_geometry(sc)
        sec = np.vstack([geo["sector"], geo["sector"][:1]])
        ax.plot(sec[:, 1], sec[:, 0], color="0.3", lw=0.8)
        for ob, ok in zip(geo["obst"], geo["obst_valid"]):
            if ok:
                ax.fill(ob[:, 1], ob[:, 0], color="#d6604d", alpha=0.35, lw=0)
        for k, tk in res["tracks"].items():
            if k != r.agent:
                ax.plot(tk[:, 1], tk[:, 0], color="0.7", lw=0.5)
        st = solo[(r.scenario, r.agent)]["tracks"][0]
        ax.plot(st[:, 1], st[:, 0], color="#2a78d6", lw=0.7, ls="--")
        tr = res["tracks"][r.agent]
        goal = np.array(latlon_to_nm(*sc.agents[r.agent].goal, *sc.center))
        route = free_route(sc, tr[0], goal)[0]
        if route is not None:
            ax.plot(route[:, 1], route[:, 0], color="k", lw=0.7, ls=":")
        ax.scatter(tr[:, 1], tr[:, 0], c=np.arange(len(tr)), s=1.5, cmap="viridis", lw=0, zorder=3)
        ax.add_patch(plt.Circle((goal[1], goal[0]), CAPTURE_NM, fill=False, color="k", lw=0.8))
        ax.plot(goal[1], goal[0], "k*", ms=5)
        ax.set_title(f"#{r.scenario}/{r.agent}: {r.cls}\nsolo {'ok' if r.solo_greedy else 'fail'}, "
                     f"sampled {r.solo_sampled:.0%} / {r.full_sampled:.0%}", fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True, help="fast_eval / official MA CSV (episode_index = 10*scenario + agent)")
    p.add_argument("--seed", type=int, default=42, help="official scenario stream")
    p.add_argument("--samples", type=int, default=16, help="sampled flights per failure (solo and full)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=25, help="MA scenarios flown together")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--limit", type=int, default=None, help="first N failures only (quick check)")
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads (leave cores to other jobs)")
    p.add_argument("--agents-csv", default=None,
                   help="detour_diag *_agents.csv of the same model and stream: adds failure rates by detour ratio")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    import torch as th
    th.set_num_threads(a.threads)
    from atc_agent.inference import load_model
    model = load_model(a.model, device="cpu")
    greedy = lambda o: model.predict(o, deterministic=True)[0]        # noqa: E731
    sampled = lambda o: model.predict(o, deterministic=False)[0]      # noqa: E731

    ref = pd.read_csv(a.csv)
    fails = ref[ref.waypoint_reached == 0].episode_index.to_numpy()
    cases = [(int(e) // 10, int(e) % 10) for e in fails][:a.limit]
    sids = sorted({s for s, _ in cases})
    ev = FastEvaluator.official(0, max(sids) + 1, seed=a.seed)
    scens = {s: ev.ma[s] for s in sids}
    print(f"{len(cases)} failed agents in {len(sids)} scenarios", flush=True)

    full = dict(zip(sids, fly(greedy, [scens[s] for s in sids], a.chunk)))
    solo_sc = []
    for s, j in cases:
        sc = copy.deepcopy(scens[s])
        sc.agents = [sc.agents[j]]
        solo_sc.append(sc)
    solo = dict(zip(cases, fly(greedy, solo_sc, a.chunk * 10, n_agents=1)))
    print("greedy replays done", flush=True)

    th.manual_seed(a.sample_seed)
    solo_s = fly(sampled, [sc for sc in solo_sc for _ in range(a.samples)], a.chunk * 10, n_agents=1)
    solo_rate = {c: np.mean([reached(solo_s[i * a.samples + k], 0) for k in range(a.samples)])
                 for i, c in enumerate(cases)}
    print("solo sampled done", flush=True)
    th.manual_seed(a.sample_seed + 1)
    full_s = fly(sampled, [scens[s] for s in sids for _ in range(a.samples)], a.chunk)
    full_rate, full_all = {}, {}
    for i, s in enumerate(sids):
        runs = full_s[i * a.samples:(i + 1) * a.samples]
        for c in cases:
            if c[0] == s:
                full_rate[c] = np.mean([reached(r, c[1]) for r in runs])
        full_all[s] = np.mean([[reached(r, j) for j in range(len(scens[s].agents))] for r in runs])
    print("full sampled done", flush=True)

    rows = []
    for s, j in cases:
        rows.append(dict(scenario=s, agent=j, reproduced=not reached(full[s], j),
                         solo_greedy=reached(solo[(s, j)], 0), solo_sampled=solo_rate[(s, j)],
                         full_sampled=full_rate[(s, j)], scenario_sampled_completion=full_all[s],
                         **geometry(scens[s], full[s], j)))
    df = pd.DataFrame(rows)
    df["cls"] = [classify(r) for r in df.itertuples()]
    df.to_csv(a.out + "_failures.csv", index=False)

    g = df.groupby("cls")
    tab = pd.DataFrame({
        "n": g.size(),
        "solo greedy reach": g.solo_greedy.mean(), "solo sampled reach": g.solo_sampled.mean(),
        "full sampled reach": g.full_sampled.mean(), "progress": g.progress.mean(),
        "closest [NM]": g.closest_nm.median(), "late orbits": g.late_orbits.median(),
        "late radius [NM]": g.late_radius_nm.median(), "late centre to RA [NM]": g.late_centre_to_ra_nm.median(),
        "late traffic share": g.late_traffic_share.mean(), "detour ratio": g.detour_ratio.median(),
    }).reindex([c for c in CLASSES if c in g.groups])
    lines = [f"# MA failure taxonomy: {a.model}, seed {a.seed}", "",
             f"{len(df)} failed agents in {len(sids)} scenarios ({a.csv}); greedy replay reproduces "
             f"{int(df.reproduced.sum())} of them. Sampled rates over {a.samples} seeds.", "",
             markdown(tab), "",
             "progress = 1 - (best area-free distance-to-go reached) / (at start); 'late' = last 30 min "
             "of the 3000 s budget; traffic share = decisions with another aircraft within 10 NM.", "",
             f"{os.path.basename(a.out)}_cases.png (classes 3-6): greedy track coloured by time (dark = early), "
             "solo greedy track blue dashed, shortest area-free route black dotted, other aircraft grey, "
             "goal capture circle black.", "",
             "Scenario-level completion of all agents under sampling, failing scenarios only: "
             f"{np.mean(list(full_all.values())):.3f} (greedy: "
             f"{1 - len(df) / sum(len(scens[s].agents) for s in sids):.3f})."]
    if a.agents_csv:
        lines += ["", hazard(a.agents_csv, df)]
    text = "\n".join(lines)
    with open(a.out + ".md", "w") as f:
        f.write(text + "\n")
    print(text)
    show = df[~df.cls.isin(CLASSES[:2])].sort_values(["cls", "scenario", "agent"])
    if len(show):
        plot_cases(show, scens, full, solo, a.out + "_cases.png")


if __name__ == "__main__":
    main()

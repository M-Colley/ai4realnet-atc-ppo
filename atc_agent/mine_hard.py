"""
Mine hard scenarios for prioritized replay (Jiang et al., 2021, "Prioritized
Level Replay"): fly a policy over fresh official-protocol scenario streams and
record, per agent-episode, the official metrics and the obstacle-aware route
length, so "hard" (timeout, large excess over the route at top speed) can be
decided afterwards without re-flying.

Scenarios are not stored: ``load_pool`` regenerates them exactly from
(seed, kind, scenario index) with the official generator.

    python -m atc_agent.mine_hard --model runs/ft_geo_r2/final.zip --seed 1001 --out runs/mined
"""

import argparse
import os

import numpy as np
import pandas as pd

from atc_agent.fast_eval import FastEvaluator, SCORED
from atc_agent.features import scenario_geometry, latlon_to_nm
from atc_agent.routing import ScenarioRoutes, route_query

# seeds never to mine from: reporting (42), decoding tuning (777)
RESERVED_SEEDS = {42, 777}


def route_lengths(scen):
    """[(straight NM, geodesic NM)] per agent of one scenario."""
    R = ScenarioRoutes(scenario_geometry(scen))
    c = scen.center
    out = []
    for ag in scen.agents:
        s = np.array(latlon_to_nm(*ag.start, *c))
        g = np.array(latlon_to_nm(*ag.goal, *c))
        d_geo = route_query(s[:1], s[1:], g[:1], g[1:], R.grid_params()[None], R.node_dist(g)[None])[0][0]
        out.append((float(np.hypot(*(g - s))), float(d_geo)))
    return out


def mine(predict, seed, n_sa, n_ma):
    ev = FastEvaluator.official(n_sa, n_ma, seed=seed)
    recs = ev.run(predict)
    rows = []
    for kind, scens in (("sa", ev.sa), ("ma", ev.ma)):
        r = iter(recs[kind])
        for i, sc in enumerate(scens):
            for a, (d_str, d_geo) in enumerate(route_lengths(sc)):
                rows.append(dict(seed=seed, kind=kind, scenario=i, agent=a, d_str=d_str, d_geo=d_geo, **next(r)))
    df = pd.DataFrame(rows)
    return df[["seed", "kind", "scenario", "agent", "d_str", "d_geo"] + SCORED]


def goal_depths(scen):
    """Per agent: how deep [NM] its goal lies inside a restricted area (0 = outside)."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    areas = unary_union([Polygon([latlon_to_nm(la, lo, *scen.center) for la, lo in ob.vertices])
                         for ob in scen.obstacles])
    out = []
    for ag in scen.agents:
        p = Point(latlon_to_nm(*ag.goal, *scen.center))
        out.append(areas.boundary.distance(p) if areas.contains(p) else 0.0)
    return out


def shallow_goal_scenarios(seed, n_ma, max_depth):
    """Geometry-only scan (no flying) of an official-protocol MA stream: scenarios
    with a goal inside a restricted area but less than ``max_depth`` NM deep,
    i.e. capturable from outside, and no deeper (entry-only) goal."""
    ev = FastEvaluator.official(0, n_ma, seed=seed)
    keep = []
    for i, sc in enumerate(ev.ma):
        d = np.array(goal_depths(sc))
        if ((d > 0) & (d < max_depth)).any() and not (d >= max_depth).any():
            keep.append((i, sc))
    return keep


def load_pool(df):
    """Regenerate the scenarios listed in ``df`` (columns seed, kind, scenario).
    Returns {"sa": [Scenario], "ma": [Scenario]}, one entry per distinct scenario."""
    pool = {"sa": [], "ma": []}
    for (seed, kind), g in df.groupby(["seed", "kind"], sort=False):
        n = int(g.scenario.max()) + 1
        ev = FastEvaluator.official(n if kind == "sa" else 0, n if kind == "ma" else 0, seed=int(seed))
        scens = ev.sa if kind == "sa" else ev.ma
        pool[kind] += [scens[i] for i in g.scenario.drop_duplicates()]
    return pool


def build_pool(mined, shallow, max_depth, sa_top_frac=0.05, v_kt=332.0, sa_timeout_weight=1, ma_timeout_weight=1):
    """Replay pool from mine_hard outputs.

    mined : DataFrame of mine_hard CSVs. SA: timeouts plus the ``sa_top_frac``
        largest excesses over the route at ``v_kt``; MA: scenarios with a timeout.
    shallow : list of MA Scenario from ``--shallow-only`` scans.
    Timeout scenarios are repeated ``*_timeout_weight`` times (uniform sampling
    from the pool then weights them accordingly). Scenarios with any goal >=
    ``max_depth`` inside a restricted area (reachable only by entering it) are
    left out, so the pool never rewards an entry."""
    m = mined.copy()
    m["excess"] = m.flight_time - m.d_geo / v_kt * 3600.0
    sa = m[m.kind == "sa"]
    ok = sa[sa.waypoint_reached >= 1]
    parts = {"sa_timeout": (sa[sa.waypoint_reached < 1], sa_timeout_weight),
             "sa_slow": (ok[ok.excess >= ok.excess.quantile(1 - sa_top_frac)], 1),
             "ma_timeout": (m[(m.kind == "ma") & (m.waypoint_reached < 1)], ma_timeout_weight)}
    pool, meta = {"sa": [], "ma": []}, {}
    for name, (part, weight) in parts.items():
        kind = name[:2]
        scens = load_pool(part[["seed", "kind", "scenario"]].drop_duplicates())[kind]
        keep = [s for s in scens if max(goal_depths(s)) < max_depth]
        meta[name] = len(keep)
        meta[name + "_dropped_deep_goal"] = len(scens) - len(keep)
        meta[name + "_weight"] = weight
        pool[kind] += keep * weight
    pool["ma"] += list(shallow)
    meta["ma_shallow"] = len(shallow)
    pool["meta"] = meta
    return pool


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-sa", type=int, default=1000)
    p.add_argument("--n-ma", type=int, default=250)
    p.add_argument("--out", default="runs/mined")
    p.add_argument("--shallow-only", action="store_true",
                   help="no flying: pickle MA scenarios whose goal lies shallowly inside a restricted area")
    args = p.parse_args()
    assert args.seed not in RESERVED_SEEDS, "reserved evaluation seed"

    if args.shallow_only:
        import pickle
        from atc_agent.routing import CAPTURE_NM, EDGE_OFFSET_NM
        keep = shallow_goal_scenarios(args.seed, args.n_ma, CAPTURE_NM - EDGE_OFFSET_NM)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"shallow_seed{args.seed}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"seed": args.seed, "index": [i for i, _ in keep], "ma": [s for _, s in keep]}, f)
        print(f"wrote {path}: {len(keep)} of {args.n_ma} MA scenarios")
        return

    from atc_agent.inference import load_model
    model = load_model(args.model, device="cpu")
    df = mine(lambda o: model.predict(o, deterministic=True)[0], args.seed, args.n_sa, args.n_ma)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"seed{args.seed}.csv")
    df.to_csv(path, index=False)
    print(f"wrote {path}: {len(df)} agent-episodes, timeouts SA {int(((df.kind == 'sa') & (df.waypoint_reached < 1)).sum())}"
          f" MA {int(((df.kind == 'ma') & (df.waypoint_reached < 1)).sum())}")


if __name__ == "__main__":
    main()

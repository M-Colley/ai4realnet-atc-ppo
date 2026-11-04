"""
Replay chosen official-protocol scenarios and plot what the policy does:
sector, restricted areas, own path(s) coloured by time, other traffic in grey.

    python -m atc_agent.plot_flights --model runs/ft_geo_r2/final.zip --seed 42 \
        --kind sa --scenarios 216,235,548 --out runs/diag/slow_sa.png
"""

import argparse
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import bluesky as bs  # noqa: E402

from atc_agent.batched_env import GOAL_MARGIN_KM  # noqa: E402
from atc_agent.fast_eval import FastEvaluator  # noqa: E402
from atc_agent.features import scenario_geometry, latlon_to_nm  # noqa: E402


def replay(predict, scens, kind, n_agents=10):
    """Fly ``scens`` together; return per scenario {'tracks': {j: (T, 2) NM}, 'agent': {j: bool},
    'metrics': {j: dict}, 'actions': {j: (n_decisions, 2)}} with positions every decision."""
    with contextlib.redirect_stdout(io.StringIO()):
        from atc_agent.batched_env import BatchedCompetitionSim
        sim = BatchedCompetitionSim(sa_slots=len(scens) if kind == "sa" else 0,
                                    ma_slots=len(scens) if kind == "ma" else 0,
                                    ma_agents=n_agents, respawn=False, auto_reset=False)
    sim.forced = {k: [s] for k, s in enumerate(scens)}
    obs = sim.reset()
    out = [dict(tracks={}, agent={j: bool(sim.is_agent[k, j]) for j in range(sim.A)}, metrics={}, actions={})
           for k in range(len(scens))]
    seat_of = {(int(sim.seat_slot[s]), int(sim.seat_pos[s])): s for s in range(sim.n_seats)}

    def record():
        for k, sc in enumerate(scens):
            c = sc.center
            for j in range(sim.A):
                i = sim.tidx[k, j]
                if i >= 0:
                    p = latlon_to_nm(bs.traf.lat[i], bs.traf.lon[i], c[0], c[1])
                    out[k]["tracks"].setdefault(j, []).append(p)

    record()
    live = np.ones(sim.n_seats, dtype=bool)
    while live.any():
        act = np.zeros((sim.n_seats, 2), dtype=np.int64)
        idx = np.flatnonzero(sim.active)
        if len(idx):
            act[idx] = predict(obs[idx])
        for s in idx:
            out[sim.seat_slot[s]]["actions"].setdefault(int(sim.seat_pos[s]), []).append(act[s].copy())
        obs, _, _, _, infos = sim.step(act)
        for seat, info in infos.items():
            if live[seat]:
                k, j = int(sim.seat_slot[seat]), int(sim.seat_pos[seat])
                out[k]["metrics"][j] = info["metrics"]
                live[seat] = False
        record()
    for o in out:
        o["tracks"] = {j: np.array(t) for j, t in o["tracks"].items()}
        o["actions"] = {j: np.array(a) for j, a in o["actions"].items()}
    assert len(seat_of) == sim.n_seats
    return out


def plot(scens, results, titles, path, ncols=4, size=4.2, legend=True):
    n = len(scens)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, sc, res, title in zip(axes.flat, scens, results, titles):
        geo = scenario_geometry(sc)
        sec = np.vstack([geo["sector"], geo["sector"][:1]])
        ax.plot(sec[:, 1], sec[:, 0], color="0.3", lw=1)
        for ob, ok in zip(geo["obst"], geo["obst_valid"]):
            if ok:
                ax.fill(ob[:, 1], ob[:, 0], color="tab:red", alpha=0.25, lw=0)
        for j, tr in res["tracks"].items():
            if not res["agent"][j]:
                ax.plot(tr[:, 1], tr[:, 0], color="0.6", lw=0.6)
        for j, tr in res["tracks"].items():
            if not res["agent"][j]:
                continue
            m = res["metrics"].get(j, {})
            failed = m.get("waypoint_reached", 1) < 1
            ax.scatter(tr[:, 1], tr[:, 0], c=np.arange(len(tr)), cmap="viridis", s=3, zorder=3)
            ax.plot(tr[:, 1], tr[:, 0], color="tab:red" if failed else "tab:blue", lw=1.2 if failed else 0.5,
                    zorder=5 if failed else 2)
            ax.plot(tr[0, 1], tr[0, 0], "k^", ms=5, zorder=4)
            g = latlon_to_nm(*sc.agents[j].goal, *sc.center)
            ax.add_patch(plt.Circle((g[1], g[0]), GOAL_MARGIN_KM / 1.852, fill=False,
                                    color="black" if failed else "tab:green", lw=1.2, zorder=4))
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
    if legend:
        fig.suptitle("east [NM] vs north [NM]; own path dots coloured by time (dark = early); "
                     "grey = other traffic; shaded = restricted areas; circle = goal; red path / black circle = timeout",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kind", choices=["sa", "ma"], default="sa")
    p.add_argument("--scenarios", required=True, help="comma-separated scenario indices")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from atc_agent.inference import load_model
    model = load_model(args.model, device="cpu")
    ids = [int(x) for x in args.scenarios.split(",")]
    n = max(ids) + 1
    ev = FastEvaluator.official(n if args.kind == "sa" else 0, n if args.kind == "ma" else 0, seed=args.seed)
    scens = [(ev.sa if args.kind == "sa" else ev.ma)[i] for i in ids]
    res = replay(lambda o: model.predict(o, deterministic=True)[0], scens, args.kind)
    titles = []
    for i, r in zip(ids, res):
        ms = [r["metrics"][j] for j in sorted(r["metrics"])]
        fails = sum(m["waypoint_reached"] < 1 for m in ms)
        viol = sum(m["intrusion_events"] + m["restricted_area_events"] + m["sector_exit_events"] for m in ms)
        ft = np.mean([m["flight_time"] for m in ms])
        titles.append(f"{args.kind} #{i}: mean flight {ft:.0f} s, timeouts {fails}, violations {viol:.0f}")
    plot(scens, res, titles, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

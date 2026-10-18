"""
Where do the remaining single-agent violations come from, and could they have
been avoided?

Phase 1 flies fresh SA scenarios (default seed 777: neither the reporting seed
42 nor the held-out selection seed 2024) with the deterministic policy and logs
every violation onset (time, decision step, other aircraft, geometry, distance
of that aircraft at t=0).

Phase 2 is an *escape search*, used only as a diagnostic (the controller stays
the pure RL policy): every violating episode is replayed exactly in
BlueSky-Batch with the recorded actions, and branched L decision steps before
each onset. A branch holds one of the 102 discrete actions until shortly after
the original onset and then hands control back to the policy. A violation is
counted as avoidable at lag L if some branch reaches the goal with strictly
fewer violations and no category worse. "Not found" is a lower bound on
difficulty, not a proof of infeasibility. One control replay per episode (no
branch) must reproduce the original metrics exactly.

    python -m atc_agent.diagnose_sa --model runs/main/best.zip --episodes 1000 --out runs/diag
"""

import argparse
import contextlib
import csv
import io
import json
import os
import time

import numpy as np

from atc_agent.actions import HEADING_OFFSETS, SPEED_TARGETS
from atc_agent.geometry import kwikqdrdist, wrap180

KINDS = ("intrusion", "restricted_area", "sector_exit")
METRIC_OF = {"intrusion": "intrusion_events", "restricted_area": "restricted_area_events",
             "sector_exit": "sector_exit_events"}


def _make_sim(scens):
    with contextlib.redirect_stdout(io.StringIO()):
        from atc_agent.batched_env import BatchedCompetitionSim
        sim = BatchedCompetitionSim(sa_slots=len(scens), ma_slots=0, respawn=False, auto_reset=False)
    sim.forced = {k: [s] for k, s in enumerate(scens)}
    return sim


class OnsetRecorder:
    """Wraps sim._update_metrics to log violation onsets per seat."""

    def __init__(self, sim):
        import bluesky as bs
        self.bs, self.sim = bs, sim
        self.dec = 0
        self.events = {s: [] for s in range(sim.n_seats)}
        self.init_dist = {}
        self._orig = sim._update_metrics
        sim._update_metrics = self._hook

    def _hook(self, seats, dt):
        sim, bs = self.sim, self.bs
        b_int, b_obs, b_out = sim.in_intr[seats].copy(), sim.obs_occ[seats].copy(), sim.outside[seats].copy()
        res = self._orig(seats, dt)
        k = sim.seat_slot[seats]
        own = sim.tidx[k, sim.seat_pos[seats]]
        if dt == 0:      # episode start: distance to every other aircraft
            oth = sim.tidx[k]
            ok = oth >= 0
            ot = np.where(ok, oth, own[:, None])
            _, d = kwikqdrdist(bs.traf.lat[own][:, None], bs.traf.lon[own][:, None],
                               bs.traf.lat[ot], bs.traf.lon[ot])
            for i, s in enumerate(seats.tolist()):
                self.init_dist[s] = np.where(ok[i], d[i], np.inf)
            # no return: the official reset-time update already counts violations present at t=0
        new_int = sim.in_intr[seats] & ~b_int
        new_obs = sim.obs_occ[seats] & ~b_obs
        new_out = sim.outside[seats] & ~b_out
        for i in np.flatnonzero(new_int.any(1) | new_obs.any(1) | new_out):
            s = int(seats[i])
            base = dict(t=float(sim.slot_t[k[i]]), dec=self.dec, own_hdg=float(bs.traf.hdg[own[i]]),
                        own_cas=float(bs.traf.cas[own[i]]))
            for j in np.flatnonzero(new_int[i]):
                o = sim.tidx[k[i], j]
                qdr, d = kwikqdrdist(bs.traf.lat[own[i]], bs.traf.lon[own[i]], bs.traf.lat[o], bs.traf.lon[o])
                self.events[s].append(dict(base, kind="intrusion", other=int(j),
                                           cross_deg=float(wrap180(bs.traf.hdg[o] - bs.traf.hdg[own[i]])),
                                           rel_brg_deg=float(wrap180(qdr - bs.traf.hdg[own[i]])),
                                           other_tas=float(bs.traf.tas[o]),
                                           other_init_nm=float(self.init_dist[s][j])))
            for j in np.flatnonzero(new_obs[i]):
                self.events[s].append(dict(base, kind="restricted_area", other=int(j)))
            if new_out[i]:
                self.events[s].append(dict(base, kind="sector_exit", other=-1))
        return res


def phase1(predict, scens, chunk):
    eps = []
    for i0 in range(0, len(scens), chunk):
        sc = scens[i0:i0 + chunk]
        sim = _make_sim(sc)
        rec = OnsetRecorder(sim)
        obs = sim.reset()
        acts, out = [], {}
        live = np.ones(sim.n_seats, dtype=bool)
        while live.any():
            act = np.zeros((sim.n_seats, 2), dtype=np.int64)
            idx = np.flatnonzero(sim.active)
            if len(idx):
                act[idx] = predict(obs[idx])
            acts.append(act)
            obs, _, _, _, infos = sim.step(act)
            rec.dec += 1
            for s, info in infos.items():
                if live[s]:
                    out[s] = (info["metrics"], rec.dec)
                    live[s] = False
        acts = np.stack(acts, 1)                 # (seats, n_dec, 2)
        for s in range(sim.n_seats):
            m, n_dec = out[s]
            eps.append(dict(idx=i0 + s, metrics=m, actions=acts[s, :n_dec], events=rec.events[s]))
        print(f"  phase 1: {i0 + len(sc)}/{len(scens)} episodes", flush=True)
    return eps


def phase2(predict, scens, eps, lags, chunk):
    """Escape search. Returns {(ep, k0, hold): best result} and control checks."""
    jobs = []
    plans = {}
    for e in eps:
        m = e["metrics"]
        if not any(m[METRIC_OF[kd]] > 0 for kd in KINDS):
            continue
        jobs.append((e["idx"], None, None, 0))            # control replay
        for ev in e["events"]:
            if ev["t"] == 0:          # present at spawn: no policy can avoid it
                continue
            for L in lags:
                k0 = max(0, ev["dec"] - L + 1)
                hold = ev["dec"] - k0 + 3
                if (e["idx"], k0, hold) in plans:
                    continue
                plans[(e["idx"], k0, hold)] = None
                for h in range(len(HEADING_OFFSETS)):
                    for v in range(len(SPEED_TARGETS)):
                        jobs.append((e["idx"], k0, (h, v), hold))
    by_idx = {e["idx"]: e for e in eps}
    print(f"  phase 2: {len(jobs)} branches over {len(plans)} branch points", flush=True)
    results = {}
    control = {}
    t0 = time.time()
    for j0 in range(0, len(jobs), chunk):
        jb = jobs[j0:j0 + chunk]
        sim = _make_sim([scens[j[0]] for j in jb])
        obs = sim.reset()
        n = sim.n_seats
        rec_acts = [by_idx[j[0]]["actions"] for j in jb]
        k0 = np.array([j[1] if j[1] is not None else 10**9 for j in jb])
        hold = np.array([j[3] for j in jb])
        const = np.array([j[2] if j[2] is not None else (0, 0) for j in jb])
        live = np.ones(n, dtype=bool)
        dec = 0
        out = {}
        while live.any():
            act = np.zeros((n, 2), dtype=np.int64)
            idx = np.flatnonzero(sim.active)
            pol = []
            for s in idx.tolist():
                if dec < k0[s] and dec < len(rec_acts[s]):
                    act[s] = rec_acts[s][dec]
                elif dec < k0[s] + hold[s]:
                    act[s] = const[s]
                else:
                    pol.append(s)
            if pol:
                act[pol] = predict(obs[pol])
            obs, _, _, _, infos = sim.step(act)
            dec += 1
            for s, info in infos.items():
                if live[s]:
                    out[s] = info["metrics"]
                    live[s] = False
        for s, j in enumerate(jb):
            m = out[s]
            if j[1] is None:
                control[j[0]] = m
                continue
            key = (j[0], j[1], j[3])
            o = by_idx[j[0]]["metrics"]
            ok = (m["waypoint_reached"] >= 1
                  and all(m[METRIC_OF[kd]] <= o[METRIC_OF[kd]] for kd in KINDS)
                  and sum(m[METRIC_OF[kd]] for kd in KINDS) < sum(o[METRIC_OF[kd]] for kd in KINDS))
            cand = dict(ok=bool(ok), action=j[2], flight_time=m["flight_time"],
                        dviol=sum(m[METRIC_OF[kd]] - o[METRIC_OF[kd]] for kd in KINDS))
            cur = results.get(key)
            if cur is None or (cand["ok"] and (not cur["ok"] or cand["flight_time"] < cur["flight_time"])):
                results[key] = cand
        el = time.time() - t0
        done = min(j0 + chunk, len(jobs))
        print(f"  phase 2: {done}/{len(jobs)} branches, {el/60:.1f} min, eta {el/done*(len(jobs)-done)/60:.1f} min",
              flush=True)
    return results, control


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--lags", default="1,2,4,8,16", help="decision steps before onset to branch at")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--branch-chunk", type=int, default=512)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="runs/diag")
    a = p.parse_args()
    lags = [int(x) for x in a.lags.split(",")]
    os.makedirs(a.out, exist_ok=True)

    import torch as th
    th.set_num_threads(a.threads)
    from atc_agent.inference import load_model
    from atc_agent.fast_eval import _generators
    model = load_model(a.model, device="cpu")

    def predict(o):
        return model.predict(np.asarray(o, dtype=np.float32), deterministic=True)[0]

    g_sa, _ = _generators()
    rng = np.random.default_rng(a.seed)
    scens = [g_sa.generate(rng) for _ in range(a.episodes)]

    t0 = time.time()
    eps = phase1(predict, scens, a.chunk)
    print(f"phase 1 done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open(os.path.join(a.out, "events.csv"), "w", newline="") as f:
        keys = ["episode", "kind", "t", "dec", "other", "own_hdg", "own_cas", "cross_deg", "rel_brg_deg",
                "other_tas", "other_init_nm"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for e in eps:
            for ev in e["events"]:
                w.writerow({"episode": e["idx"], **{k: ev.get(k, "") for k in keys if k != "episode"}})

    t1 = time.time()
    res, control = phase2(predict, scens, eps, lags, a.branch_chunk)
    print(f"phase 2 done in {(time.time()-t1)/60:.1f} min", flush=True)

    # ------------------------------------------------------------- summary
    by_idx = {e["idx"]: e for e in eps}
    n_ctrl_bad = sum(any(abs(control[i][k] - by_idx[i]["metrics"][k]) > 1e-9
                         for k in by_idx[i]["metrics"] if k != "total_reward") for i in control)
    rows = []
    for e in eps:
        for ev in e["events"]:
            r = dict(episode=e["idx"], kind=ev["kind"], t=ev["t"], dec=ev["dec"],
                     other_init_nm=ev.get("other_init_nm"), cross_deg=ev.get("cross_deg"))
            r["min_lag"] = None
            for L in lags:
                k0 = max(0, ev["dec"] - L + 1)
                hit = res.get((e["idx"], k0, ev["dec"] - k0 + 3))
                if hit and hit["ok"]:
                    r["min_lag"] = L
                    r["escape_action"] = [float(HEADING_OFFSETS[hit["action"][0]]), float(SPEED_TARGETS[hit["action"][1]])]
                    break
            rows.append(r)
    with open(os.path.join(a.out, "escape.json"), "w") as f:
        json.dump(rows, f, indent=1)

    n = len(eps)
    lines = [f"# SA violation diagnosis: {a.model}, {n} episodes, seed {a.seed}", "",
             f"Control replays reproducing the original metrics: {len(control) - n_ctrl_bad}/{len(control)}", ""]
    mean = {k: np.mean([e["metrics"][METRIC_OF[k]] for e in eps]) for k in KINDS}
    lines.append("| Kind | Events/episode | Events | At t=0 (spawn) | Onset in first 60 s | Avoidable (cumulative, by branch lag) | Not found (t>0) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for kd in KINDS:
        rr = [r for r in rows if r["kind"] == kd]
        if not rr:
            lines.append(f"| {kd} | 0 | 0 | - | - | - | - |")
            continue
        early = sum(r["t"] <= 60 for r in rr)
        at0 = sum(r["t"] == 0 for r in rr)
        cum = []
        for L in lags:
            c = sum(r["min_lag"] is not None and r["min_lag"] <= L for r in rr)
            cum.append(f"{10*L}s: {c}")
        nf = sum(r["min_lag"] is None and r["t"] > 0 for r in rr)
        lines.append(f"| {kd} | {mean[kd]:.3f} | {len(rr)} | {at0} | {early} | {', '.join(cum)} | {nf} |")
    ir = [r for r in rows if r["kind"] == "intrusion"]
    if ir:
        d0 = np.array([r["other_init_nm"] for r in ir], dtype=float)
        tt = np.array([r["t"] for r in ir], dtype=float)
        lines += ["", "Intrusions: other aircraft distance at t=0 [NM] (quartiles): "
                  + ", ".join(f"{q:.1f}" for q in np.percentile(d0, [0, 25, 50, 75, 100])),
                  "Intrusions: onset time [s] (quartiles): "
                  + ", ".join(f"{q:.0f}" for q in np.percentile(tt, [0, 25, 50, 75, 100])),
                  f"Intrusions with the other aircraft < 10 NM at t=0: {int((d0 < 10).sum())}",
                  f"Episodes with any violation: {sum(1 for e in eps if e['events'])}/{n}"]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()

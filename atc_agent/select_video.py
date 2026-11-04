"""
Pick the multi-agent video scenarios by a fixed rule, then record them.

Candidate k is environment seed k (k = 1..N): ``CompetitionZooEnv.reset(seed=k)``
flies the first scenario of the official generator seeded with k, which is
``FastEvaluator.official(0, 1, seed=k).ma[0]``. Every candidate is flown in
BlueSky-Batch by the policy and by an approximate fly-direct (direct heading
bin at 150 m/s, close to the stock zero action). Eligible: the policy brings all
10 aircraft to their goals without a violation. Five seeds are then picked, in
this order, each the eligible seed with the highest value not already picked:

1. conflicts resolved: fly-direct intrusion events,
2. restricted areas avoided: fly-direct restricted-area events,
3. busiest: fly-direct events of both kinds,
4. converging goals: goal pairs closer than 10 NM,
5. typical: fly-direct events closest to the eligible median.

The picks are flown again in the official environment with our MDP wrapper (as
scripts/evaluate_competition.py and scripts/record_competition_gifs.py do),
rendered, captioned with the outcome of that official episode and with the
stock environment's own fly-direct (zero action) violation counts, and written
as MP4 (ffmpeg) and GIF.

    python -m atc_agent.select_video --model runs/ft_geo_r2/final.zip --out runs/video/ma
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from atc_agent.actions import DIRECT_BIN, SPEED_TARGETS
from atc_agent.fast_eval import FastEvaluator, SCORED
from atc_agent.features import latlon_to_nm

AGENTS = 10
VIOL = ["intrusion_events", "restricted_area_events", "sector_exit_events"]
CRITERIA = [
    ("conflicts resolved", lambda d: d.fd_intrusions),
    ("restricted areas avoided", lambda d: d.fd_ra),
    ("busiest", lambda d: d.fd_intrusions + d.fd_ra),
    ("converging goals", lambda d: d.close_goal_pairs),
    ("typical", lambda d: -(d.fd_intrusions + d.fd_ra - (d.fd_intrusions + d.fd_ra).median()).abs()),
]


def candidates(predict, n):
    scens = [FastEvaluator.official(0, 1, seed=k).ma[0] for k in range(1, n + 1)]
    fd = lambda o: np.tile([DIRECT_BIN, int(np.flatnonzero(SPEED_TARGETS == 150.0)[0])], (len(o), 1))  # noqa: E731
    pol = pd.DataFrame(FastEvaluator([], scens).run(predict)["ma"])[SCORED]
    dir_ = pd.DataFrame(FastEvaluator([], scens).run(fd)["ma"])[SCORED]
    rows = []
    for i, sc in enumerate(scens):
        p, f = pol.iloc[i * AGENTS:(i + 1) * AGENTS], dir_.iloc[i * AGENTS:(i + 1) * AGENTS]
        g = np.array([latlon_to_nm(*a.goal, *sc.center) for a in sc.agents])
        dg = np.hypot(*(g[:, None] - g[None]).transpose(2, 0, 1))
        rows.append(dict(seed=i + 1, reached=int(p.waypoint_reached.sum()), violations=float(p[VIOL].sum().sum()),
                         flight_time=float(p.flight_time.mean()), fd_intrusions=float(f.intrusion_events.sum()),
                         fd_ra=float(f.restricted_area_events.sum()), fd_flight_time=float(f.flight_time.mean()),
                         close_goal_pairs=int((dg[np.triu_indices(AGENTS, 1)] < 10.0).sum())))
    return pd.DataFrame(rows)


def pick(df):
    ok = df[(df.reached == AGENTS) & (df.violations == 0)].copy()
    chosen = []
    for name, f in CRITERIA:
        v = f(ok).drop(index=[c[1] for c in chosen])
        chosen.append((name, int(v.idxmax())))
    return [(name, int(ok.loc[i, "seed"])) for name, i in chosen]


def rollout(seed, act, max_steps=400):
    """Official MA env with our MDP wrapper: rgb frames and each agent's final info."""
    from atc_agent.envs import AgentCompetitionZooEnv
    env = AgentCompetitionZooEnv(render_mode="rgb_array", n_agents=AGENTS)
    obs, _ = env.reset(seed=seed)
    frames, records, steps = [env.render()], [], 0
    while env.agents and steps < max_steps:
        obs, _, term, trunc, infos = env.step({a: act(obs[a]) for a in env.agents})
        records += [dict(infos[a]) for a in infos if term[a] or trunc[a]]
        frames.append(env.render())
        steps += 1
    env.close()
    return frames, records


def stock_fly_direct(seed):
    """Violation counts of the stock environment's zero action (fly direct) on ``seed``."""
    from bluesky_zoo.competition_v0 import CompetitionZooEnv
    env = CompetitionZooEnv(n_agents=AGENTS)
    env.reset(seed=seed)
    recs = []
    while env.agents:
        _, _, term, trunc, infos = env.step({a: np.zeros(2, dtype=np.float32) for a in env.agents})
        recs += [dict(infos[a]) for a in infos if term[a] or trunc[a]]
    env.close()
    r = pd.DataFrame(recs)
    return int(r.intrusion_events.sum()), int(r.restricted_area_events.sum())


def caption(frames, lines, bar=56):
    out = []
    font = ImageFont.load_default(size=18)
    for f in frames:
        im = Image.fromarray(f)
        canvas = Image.new("RGB", (im.width, im.height + bar), "white")
        canvas.paste(im, (0, bar))
        d = ImageDraw.Draw(canvas)
        for k, t in enumerate(lines):
            d.text((10, 6 + 24 * k), t, fill="black", font=font)
        out.append(np.asarray(canvas))
    return out


def write_mp4(frames, path, fps=12):
    tmp = tempfile.mkdtemp()
    try:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", os.path.join(tmp, "%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_gif(frames, path, scale=0.5, duration=80):
    ims = [Image.fromarray(f) for f in frames]
    ims = [im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR) for im in ims]
    ims[0].save(path, save_all=True, append_images=ims[1:], loop=0, duration=duration, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--candidates", type=int, default=300)
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--record", action="store_true", help="only record the picks of an earlier selection run")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if not a.record:
        from atc_agent.inference import load_model
        model = load_model(a.model, device="cpu")
        df = candidates(lambda o: model.predict(o, deterministic=True)[0], a.candidates)
        df.to_csv(a.out + "_candidates.csv", index=False)
        print(f"{int(((df.reached == AGENTS) & (df.violations == 0)).sum())} of {len(df)} candidates eligible", flush=True)
        # the official env gets a fresh BlueSky process
        subprocess.run([sys.executable, "-m", "atc_agent.select_video", "--model", a.model, "--out", a.out, "--record"],
                       check=True)
        return

    from atc_agent.inference import load_act
    df = pd.read_csv(a.out + "_candidates.csv")
    picks = pick(df)
    act = load_act(a.model, deterministic=True)
    reel, meta = [], []
    for n, (name, seed) in enumerate(picks, 1):
        frames, recs = rollout(seed, act)
        r = pd.DataFrame(recs)
        fd_intr, fd_ra = stock_fly_direct(seed)
        lines = [f"Scenario {n}/5 (seed {seed}): {name}",
                 f"{int(r.waypoint_reached.sum())}/{len(r)} goals, {int(r[VIOL].sum().sum())} violations "
                 f"(fly-direct: {fd_intr} intrusions, {fd_ra} RA)"]
        reel += caption(frames, lines)
        meta.append(dict(scenario=n, seed=seed, criterion=name, goals=int(r.waypoint_reached.sum()), agents=len(r),
                         violations=int(r[VIOL].sum().sum()), mean_flight_time=float(r.flight_time.mean()),
                         fd_intrusions=fd_intr, fd_ra=fd_ra, frames=len(frames)))
        print(meta[-1], flush=True)
    with open(a.out + "_picks.json", "w") as f:
        json.dump(meta, f, indent=1)
    write_mp4(reel, a.out + ".mp4")
    write_gif(reel, a.out + ".gif")
    print(f"wrote {a.out}.mp4 and {a.out}.gif ({len(reel)} frames)")


if __name__ == "__main__":
    main()

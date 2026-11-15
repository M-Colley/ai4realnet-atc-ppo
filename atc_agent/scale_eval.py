"""
Denser traffic than in training (not scored): the policy flies scenarios with more aircraft
than the competition uses, next to a fly-direct reference (straight at the goal, initial speed)
on the same scenarios.

Training saw 10 aircraft per multi-agent scenario and 10-14 intruders in single-agent ones.
The observation keeps the 10 nearest aircraft as tokens, so denser traffic is representable;
restricted areas stay at 5 because the observation has exactly 5 obstacle tokens.

    python -m atc_agent.scale_eval --model runs/ft_rev/final.zip --agents 10 15 20 30 \
        --intruders 10 20 30 --n 300 --out runs/scale/ft_rev

Writes <out>_{ma,sa}<n>_{policy,direct}.csv, <out>_summary.csv and <out>.txt. Scenario streams are seeded per
size (seed + size), independent of the official seed-42 stream.
"""

import argparse
import functools
import os

import numpy as np
import pandas as pd

from atc_agent.actions import DIRECT_BIN, SPEED_TARGETS
from atc_agent.fast_eval import FastEvaluator

DIRECT_SPEED_BIN = int(np.argmin(np.abs(SPEED_TARGETS - 150.0)))    # the generator's agent speed
COLS = ["waypoint_reached", "flight_time", "intrusion_events", "restricted_area_events", "sector_exit_events"]


def fly_direct(obs):
    a = np.empty((len(obs), 2), dtype=np.int64)
    a[:, 0], a[:, 1] = DIRECT_BIN, DIRECT_SPEED_BIN
    return a


def scenarios(kind, size, n, seed):
    from core.scenario import ScenarioGenerator
    g = (ScenarioGenerator(n_obstacles=5, n_intruders=0, n_agents=size, agent_speed=150.0) if kind == "ma"
         else ScenarioGenerator(n_obstacles=5, n_intruders=size, agent_speed=150.0))
    rng = np.random.default_rng(seed + size)
    return [g.generate(rng) for _ in range(n)]


def run(kind, size, scens, predict, path):
    if os.path.exists(path):
        return pd.read_csv(path)
    import atc_agent.batched_env as be
    sim_cls = be.BatchedCompetitionSim
    if kind == "sa":            # seats per SA slot follow the intruder count
        be.BatchedCompetitionSim = functools.partial(sim_cls, sa_intruders=((size, 1.0),))
    try:
        ev = FastEvaluator(scens if kind == "sa" else [], scens if kind == "ma" else [],
                           n_agents=size if kind == "ma" else 10)
        df = pd.DataFrame(ev.run(predict)[kind])
    finally:
        be.BatchedCompetitionSim = sim_cls
    df.to_csv(path, index=False)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--agents", type=int, nargs="*", default=[10, 15, 20, 30])
    p.add_argument("--intruders", type=int, nargs="*", default=[10, 20, 30])
    p.add_argument("--n", type=int, default=300, help="scenarios per size")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--out", required=True, help="output prefix")
    args = p.parse_args()

    from atc_agent.inference import load_model
    model = load_model(args.model, device="cpu")

    def policy(obs):
        return model.predict(obs, deterministic=True)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for kind, sizes in (("ma", args.agents), ("sa", args.intruders)):
        for size in sizes:
            scens = scenarios(kind, size, args.n, args.seed)
            for name, pred in (("policy", policy), ("direct", fly_direct)):
                df = run(kind, size, scens, pred, f"{args.out}_{kind}{size}_{name}.csv")
                m = df[COLS].mean()
                rows.append(dict(track=kind.upper(), aircraft=size, controller=name,
                                 goal_pct=100 * m.waypoint_reached, flight_time=m.flight_time,
                                 intrusions=m.intrusion_events, ra=m.restricted_area_events,
                                 sector=m.sector_exit_events, n=len(df)))
                print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out + "_summary.csv", index=False)
    with open(args.out + ".txt", "w") as f:
        f.write(f"model {args.model}; {args.n} scenarios per size; seed {args.seed} + size\n\n")
        f.write(table.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

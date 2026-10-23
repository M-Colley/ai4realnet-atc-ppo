# atc_agent — our competition entry

One shared, intent-aware entity-transformer policy for **both** tracks of the
BlueSky-Gym ATC competition, trained with PPO on **BlueSky-Batch**: an exact,
batched re-hosting of the official environments that runs hundreds of
scenarios inside one BlueSky process.

## What is in here

| File | Purpose |
| --- | --- |
| `batched_env.py` | **BlueSky-Batch**: K scenarios in one BlueSky instance (exact physics, ~40x more samples per CPU-second), official scenario generator, vectorised copy of the official metrics, reward, continuous-traffic multi-agent training. |
| `vec_env.py` | SB3 `VecEnv` over N worker processes (one BlueSky each). |
| `features.py` | Observation: ego + 32-ray lidar, ranked aircraft tokens with CPA + **intent**, obstacle tokens. Shared verbatim by training and evaluation. |
| `actions.py` | Goal-relative categorical action space `MultiDiscrete([17, 6])` (heading offset from the goal bearing x CAS target). |
| `policy.py` | Entity-transformer actor-critic (SB3 `ActorCriticPolicy`). |
| `envs.py` | Our hooks (`_get_obs`, `_get_action`, `_get_reward`) on the **official** `CompetitionEnv` / `CompetitionZooEnv`; used by the official scoring scripts. |
| `train.py` | PPO training with adaptive safety multipliers, held-out evaluation, checkpoint selection. |
| `fast_eval.py` | Exact fast evaluation (held-out set, or the official seed-42 stream). |
| `routing.py` | Geodesic cost-to-go field (grid Dijkstra) for progress shaping and the route hint. |
| `expand_obs.py` | Converts a trained model to a wider ego observation (zero-weight new inputs, exact function preservation). |
| `diagnose_sa.py` | Violation diagnosis: onset time / geometry of every SA violation, plus an escape search (exact replay + branching) that tells which violations any manoeuvre could have avoided. |
| `../tests/test_equivalence.py` | Proves BlueSky-Batch == official envs (obs, states, all metrics) step by step. |

## Setup on the training PC

The code lives in the private repo `M-Colley/ai4realnet-atc-competition`
(branch `main`, built on top of the official `AI4REAL-NET-Competition`
branch). Linux and Windows both work (workers use `forkserver` / `spawn`).

```bash
gh repo clone M-Colley/ai4realnet-atc-competition
cd ai4realnet-atc-competition
git remote add upstream https://github.com/TUDelft-CNS-ATM/bluesky-gym.git   # to pull organiser updates
uv sync
# GPU build of torch (the lockfile pins the CPU wheel). cu130 needs an NVIDIA driver >= 580;
# use .../whl/cu126 for older drivers (not RTX 50xx).
uv pip install --reinstall "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
```

Always run with `uv run --no-sync ...` afterwards (plain `uv run` would re-sync
torch back to the CPU wheel), or call `.venv/bin/python` (Linux) /
`.venv\Scripts\python` (Windows) directly.

Sanity checks (2-5 minutes):

```bash
uv run --no-sync python tests/test_equivalence.py
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"
```

## Training

Workers default to `cpu_count - 2`. Each worker holds `--sa-slots` single-agent
and `--ma-slots` 10-agent scenarios (default 48 + 4 = 88 seats per worker).
Simulation is CPU-bound; the PPO update runs on the GPU.

```bash
# main run (stage 1): both tracks jointly
uv run --no-sync python -m atc_agent.train --run main --total-steps 1e9

# optional stage 2: multi-agent emphasis, lower LR, continue from the best checkpoint
uv run --no-sync python -m atc_agent.train --run ma_ft --resume runs/main/best.zip \
    --sa-slots 24 --ma-slots 8 --lr 1e-4 --lr-final 1e-5 --ent-start 0.003 --ent-end 0.001 \
    --adapt-warmup 0 --total-steps 300e6
```

Monitor with `python -m atc_agent.progress runs/main --evals` (held-out
deterministic evaluations) or `python -m atc_agent.progress runs/main --every 20`.

### Sizing for big hardware

The rollout is CPU-bound: about 1.5k samples/s per worker on a modern core.
The update is GPU-bound: a laptop RTX 4050 does ~28k samples/s per epoch (bf16,
separate actor/critic); a desktop RTX 4090/5090 or A100/H100 should do 5–10x
that. Check `time/fps` in the log, and if the update dominates:

- add `--share-trunk` (one transformer for actor and critic, ~2x faster), or
- use `--batch-size 8192 --epochs 3`.

| Machine | Suggested flags |
| --- | --- |
| 32 cores + 1 big GPU | `--workers 30` (2640 seats; ~170k samples per rollout) |
| 64+ cores + 1 big GPU | `--workers 60 --n-steps 64` (~340k samples per rollout) |
| Many cores, modest GPU | add `--share-trunk --epochs 3` |

More parallel data helps PPO keep improving (Beukman et al., RLC 2026), so
prefer more workers over longer rollouts. Budget: the local smoke run already
roughly halved violations within 3M steps. Plan for 0.5–1B steps for the final
policy (~3–6 h at ~50k samples/s).

### What we verified locally (RTX 4050 laptop, 16 workers, 3M steps, ~40 min)

- The batched sim equals the official envs step by step
  (`tests/test_equivalence.py`).
- `fast_eval` reproduces the official harness per episode (SA, 12 episodes)
  and in aggregate (MA, 3 episodes × 10 agents).
- Learning progress over the first 3M steps, deterministic held-out policy:

  | | SA | MA |
  | --- | ---: | ---: |
  | Goal completion | 100% | 100% |
  | Intrusion events per (agent-)episode | 0.76 | 0.28 |
  | Restricted-area events per (agent-)episode | 0.54 | 0.24 |
  | Sector exits | 0 | 0 |

  For comparison, the official SA neutral baseline (1000 seed-42 episodes,
  `results/baseline_neutral_sa.csv`) gets 1.127 intrusions and 0.813
  restricted-area entries per episode. The run was stopped at 3M steps on purpose; it is
  a pipeline check, not the final model.

Outputs in `runs/<run>/`: `best.zip` (best held-out score), `latest.zip`,
`checkpoints/`, `progress.csv` (all logged scalars), `progress.jsonl`
(per-rollout metric groups + held-out evals), `train_state.json` (safety
multipliers; reused on `--resume`).

Logged groups: `sa10/*` (single-agent episodes in the exact competition config),
`sa_all/*` (incl. denser traffic), `ma_orig/*` (original 10-agent cohorts),
`ma_all/*` (incl. respawned agents, which are truncated at scenario resets),
`eval_sa/*`, `eval_ma/*` (deterministic policy on 200 + 20 held-out scenarios,
seed 2024), `safety/lam_*` (adaptive multipliers).

### Unavoidable violations and the safety constraint

The official generator does not keep scripted intruders 5 NM away from the SA
start, so some SA episodes begin in loss of separation. These count as
intrusion events but no policy can avoid them:

| Scenario stream (1000 SA episodes) | Intrusions present at t=0 per episode |
| --- | ---: |
| official seed 42 | 0.100 |
| held-out seed 2024 | 0.103 |
| diagnosis seed 777 | 0.116 |

`python -m atc_agent.diagnose_sa --model runs/main/best.zip --out runs/diag` on
the 490M-step checkpoint (seed 777): 0.151 intrusions/episode, of which 116/151
are present at spawn, 14 more start within 33 s with the intruder spawned
5-8 NM away, and only ~18 could be removed by some manoeuvre 10-80 s earlier.
Restricted-area (0.015) and sector-exit (0.008) events are mostly avoidable.

Because the original constraint targets (`--target-intr 0.02`) sit below this
floor, the multipliers stayed at `--lam-max` during `runs/main`. The training
now constrains *avoidable* violations only (`--avoidable-constraint 1`,
default); `--avoidable-constraint 0` reproduces `runs/main`. Logged groups
include `*/intrusion_events_avoidable`.

### Time limit as part of the task (branch `time-aware`)

On the official seed-42 stream the 1B-step policy reaches the goal in 99.2% (SA)
and 99.17% (MA agent-episodes) of cases. Every failure is a 3000 s timeout: the
aircraft circles or follows obstacle edges inside clusters of restricted areas
(`runs/diag/sa_failures_seed42.png`). Stochastic actions barely help (99.5%).
The time limit was invisible to the policy (truncation was bootstrapped), so
nothing pushed it out of these traps. Following Pardo et al. (ICML 2018, "Time
Limits in Reinforcement Learning"):

- the ego observation ends with the elapsed fraction of the 3000 s budget
  (SA flight time / MA scenario clock, as the official envs define it);
- running out of time is terminal with `--timeout-penalty` (default 20), not
  bootstrapped.

`python -m atc_agent.expand_obs runs/main/final.zip runs/ta_init/init.zip`
carries a trained policy over to the wider observation with zero weights on the
new input (verified: identical actions and values), so the time-aware policy is
fine-tuned from the 1B-step model instead of trained from scratch.

### Geodesic progress shaping and route hint (branch `geodesic`)

The time-aware fine-tune did not reduce timeouts (seed 42: 9/1000 SA). The
trapped episodes are the ones whose shortest restricted-area-free route is a
large detour: for 4 of the 6 routable SA failures it is 2.1-2.8x the straight
line (seed-42 median 1.008, 99th percentile 2.16); 2 failures have no free route
at all. The straight-line progress potential rewards flying *into* those
pockets. `routing.py` computes a cost-to-go field per scenario and goal (96x96
grid, 16-connected Dijkstra; restricted areas and the outside of the sector
cost 5x per NM instead of being forbidden), interpolated bilinearly so the
potential is continuous everywhere, including inside areas:

- the progress potential uses this geodesic distance (`--geodesic-progress 1`);
- the ego observation gets 4 route entries: geodesic distance, steepest-descent
  direction (cos/sin, body frame), detour ratio.

The policy still issues every command; the field is reward / observation design.

## Evaluation

```bash
# fast, exact: the official seed-42 stream flown in BlueSky-Batch (minutes)
uv run --no-sync python -m atc_agent.fast_eval --model runs/main/best.zip --env both --out results/fast

# official reporting protocol (slow: single-scenario BlueSky, ~0.5 h SA, a few h MA)
uv run --no-sync python -m scripts.evaluate_competition --env sa --model runs/main/best.zip
uv run --no-sync python -m scripts.evaluate_competition --env ma --model runs/main/best.zip

# video reel for the submission
uv run --no-sync python -m scripts.record_competition_gifs --env sa --seeds 1 2 3 4 5 --model runs/main/best.zip --out reel_sa.gif
uv run --no-sync python -m scripts.record_competition_gifs --env ma --seeds 1 2 3 --model runs/main/best.zip --out reel_ma.gif
```

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

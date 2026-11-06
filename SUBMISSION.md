# Team MATCH — AI4REALNET BlueSky-Gym ATC competition entry

Mark Colley, University College London. Branch `submission` of the private repository
`M-Colley/ai4realnet-atc-competition` (built on the official `AI4REAL-NET-Competition`
branch of BlueSky-Gym).

One reinforcement-learning policy flies both tracks: a PPO-trained entity-transformer
actor-critic that, every 10 s, picks a heading offset from the goal bearing and a speed
target for each aircraft (`MultiDiscrete([17, 6])`). Training ran in BlueSky-Batch, an
exact batched re-hosting of the official environments (`atc_agent/batched_env.py`,
verified step by step against them in `tests/test_equivalence.py`). Method and results
are described in the technical report (LaTeX source in `docs/solution/report/`, build command
below; the PDF goes with the submission email); the agent code
is documented in [`atc_agent/README.md`](atc_agent/README.md).

## What is in `submission/`

| Path | Content |
| --- | --- |
| `policy.zip` | The submitted policy (SB3 PPO zip, 1,185,220,608 training steps; sha256 `dd22d299…250fa4`). |
| `results/official_{sa,ma}.csv` | Its per-episode metrics from the official harness, seed 42 (1000 SA episodes, 1000 × 10 MA agent-episodes). |
| `results/fast_{sa,ma}.csv` | The same stream flown in BlueSky-Batch (`atc_agent.fast_eval`): SA identical episode by episode, MA identical in 998 of 1000 scenarios (agents are listed in a different order). |
| `geo_init/` | Starting point of the stage that produced the policy (time-aware 1.07B-step model with the route-hint inputs added, plus its safety multipliers); used by the seed repeats. |

Official harness, seed 42 (`scripts/evaluate_competition.py`):

| Metric (mean per flight) | SA policy | SA fly-direct | MA policy | MA fly-direct |
| --- | ---: | ---: | ---: | ---: |
| Goal completion [%] | 99.60 | 100.0 | 99.37 | 100.0 |
| Flight time [s] | 987.6 | 852.0 | 921.4 | 847.7 |
| Intrusion events | 0.127 | 1.127 | 0.010 | 0.877 |
| Restricted-area events | 0.013 | 0.813 | 0.008 | 0.799 |
| Sector-exit events | 0.008 | 0.000 | 0.005 | 0.000 |

Most SA intrusions cannot be avoided: on a separate 1000-episode stream (seed 777), 116 of the
policy's 141 intrusion events are already present at spawn, because the scenario generator can
place intruders within 5 NM of the start (`atc_agent.diagnose_sa`).

## Install (fresh clone)

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. Linux and Windows both work.

```bash
git clone -b submission https://github.com/M-Colley/ai4realnet-atc-competition.git
cd ai4realnet-atc-competition
uv sync --frozen
```

The lockfile pins the CPU build of torch, which is all evaluation needs. For training on an
NVIDIA GPU, add the CUDA build (driver >= 580; use `.../whl/cu126` for older drivers):

```bash
uv pip install --reinstall "torch==2.12.0" --index-url https://download.pytorch.org/whl/cu130
```

Afterwards always use `uv run --no-sync ...` (plain `uv run` re-syncs torch to the CPU wheel).

## Evaluate

```bash
# official protocol, single scenario at a time (slow: ~0.5 h SA, a few hours MA)
uv run --no-sync python -m scripts.evaluate_competition --env sa --model submission/policy.zip --out results/official_sa.csv
uv run --no-sync python -m scripts.evaluate_competition --env ma --model submission/policy.zip --out results/official_ma.csv

# the same seed-42 stream in BlueSky-Batch (~20 min for both tracks)
uv run --no-sync python -m atc_agent.fast_eval --model submission/policy.zip --env both --out results/fast

# paired comparison with the shipped results (zero differences expected)
uv run --no-sync python -m atc_agent.paired --ref submission/results/fast results/fast
```

`atc_agent.paired` compares any evaluation runs on the same stream: exact McNemar test on SA
failures, scenario-cluster BCa bootstrap on MA failures and on a selection score
(100 x goal - 10 x violations - 0.01 x flight time; our own checkpoint-selection proxy, not
the competition's scoring).

## Seed repeats

One command trains a repeat with a new seed, evaluates it on the official seed-42 stream at
the submitted policy's step and at the end of the stage, and writes a paired comparison with
the submitted policy to `runs/rep_<recipe>_s<seed>/repeat_summary.json`:

```bash
uv run --no-sync python -m scripts.seed_repeat --seed 11                    # ~10 h training + 40 min eval
uv run --no-sync python -m scripts.seed_repeat --seed 12 --recipe scratch   # ~45 h
```

`--until-matched` stops the fine-tune at the submitted policy's step (115M of its 300M steps, same
schedules; ~40% of the time). A run that dies (e.g. out of memory) is resumed from its latest
checkpoint with the schedules continued (`--tries`, default 3).

- `finetune` (default) repeats the stage that produced the policy, from the same starting
  checkpoint (`submission/geo_init`) with identical hyperparameters: it measures the
  run-to-run spread of the final stage.
- `scratch` trains the method from scratch with the final code (all features on from step
  0), then the same fine-tune. It is not the exact staged lineage below, so it tests the
  method rather than this checkpoint.

Training holds ~40 GiB of RAM with the default 26 simulation workers. Keep `--workers 26`
on smaller machines (it is only slower), so each PPO rollout has the original size
(26 x 88 seats x 128 steps). Check the pipeline first with `--smoke` (a few minutes).

## How the submitted policy was trained

| Stage | Steps (cumulative) | Code | What changed |
| --- | --- | --- | --- |
| main | 0 – 1.0B | `dfb3d76` | PPO from scratch, both tracks, adaptive safety multipliers (seed 1). |
| ft_ta | 1.0B – 1.07B | `5108938` | `expand_obs` adds the elapsed-time input; timeouts terminal with a penalty; only avoidable violations constrained. Best held-out checkpoint kept. |
| ft_geo | 1.07B – 1.185B | `05406ca`, `f1a9bc7` | `expand_obs` adds the route hint; geodesic progress shaping. Interrupted twice and resumed (seeds 1, 2, 3). |

Every stage's arguments are recorded in `scripts/seed_repeat.py` (stage 1 and fine-tune) and
in the report. The training logs themselves (`runs/`) are not in the repository and are
available on request.

## Video and figures

```bash
# picks 5 of 300 seeds by a fixed rule, records them in the official MA env (MP4 needs ffmpeg)
uv run --no-sync python -m atc_agent.select_video --model submission/policy.zip --out runs/video/ma
# trajectories of chosen seed-42 scenarios (report Fig. 2)
uv run --no-sync python -m atc_agent.plot_flights --model submission/policy.zip --seed 42 --kind ma --scenarios 261,788 --out runs/diag/flights.png
# diagnostics used in the report
uv run --no-sync python -m atc_agent.detour_diag --model submission/policy.zip --check submission/results/fast_ma.csv --out runs/diag/detour
uv run --no-sync python -m atc_agent.failure_diag --model submission/policy.zip --csv submission/results/fast_ma.csv \n    --agents-csv runs/diag/detour_agents.csv --out runs/diag/fail
uv run --no-sync python -m atc_agent.diagnose_sa --model submission/policy.zip --out runs/diag/sa
# the report (TeX Live or TinyTeX)
cd docs/solution/report && latexmk -pdf report.tex
```

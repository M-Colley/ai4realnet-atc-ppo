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
| `policy.zip` | The submitted policy (SB3 PPO zip, 1,370,310,656 training steps; sha256 `d5532eab…d08bb`). |
| `results/official_{sa,ma}.csv` | Its per-episode metrics from the official harness, seed 42 (1000 SA episodes, 1000 × 10 MA agent-episodes). |
| `results/fast_{sa,ma}.csv` | The same stream flown in BlueSky-Batch (`atc_agent.fast_eval`). For the parent below, SA was identical episode by episode and MA identical in 998 of 1000 scenarios (agents are listed in a different order). |
| `rev_init/` | The parent: the policy before the last stage (1,185,220,608 steps; sha256 `dd22d299…250fa4`; submitted until 2026-11-07), with its safety multipliers. Start of the last stage; used by the seed repeats. |
| `results/parent/` | The parent's official and fast results on the same stream. |
| `geo_init/` | Start of the ft_geo stage (time-aware 1.07B-step model with the route-hint inputs added, plus its safety multipliers); used by the seed repeats. |

Official harness, seed 42 (`scripts/evaluate_competition.py`):

| Metric (mean per flight) | SA policy | SA fly-direct | MA policy | MA fly-direct |
| --- | ---: | ---: | ---: | ---: |
| Goal completion [%] | TODO | 100.0 | TODO | 100.0 |
| Flight time [s] | TODO | 852.0 | TODO | 847.7 |
| Intrusion events | TODO | 1.127 | TODO | 0.877 |
| Restricted-area events | TODO | 0.813 | TODO | 0.799 |
| Sector-exit events | TODO | 0.000 | TODO | 0.000 |

TODO (official run in progress): BlueSky-Batch on the same stream gives SA 99.70 % / 978.7 s /
0.126 / 0.012 / 0.005 and MA 99.39 % / 916.7 s / 0.008 / 0.009 / 0.007. Against the parent,
paired over seeds 42 and 777 (2000 scenarios per track): flight time -10.5 s SA [-16.9, -4.1]
and -4.2 s MA [-6.0, -2.4], violations and MA failures unchanged.

Most SA intrusions cannot be avoided: on a separate 1000-episode stream (seed 777), 116 of the
parent's 141 intrusion events (TODO: submitted policy) are already present at spawn, because the scenario generator can
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

The policy is the end of two fine-tuning stages (table below): ft_geo, stopped after 115.1M of
its 300M-step schedule (the parent), and ft_rev, the rest of that schedule with the
indecision costs. One command repeats them with a new seed, evaluates the result on the
official seed-42 stream and writes paired comparisons to
`runs/rep_<recipe>_s<seed>/repeat_summary.json` (the parent-step checkpoint against
`results/parent`, the end against `results/`):

```bash
uv run --no-sync python -m scripts.seed_repeat --seed 11                    # ~10 h at 8.5k samples/s
uv run --no-sync python -m scripts.seed_repeat --seed 12 --recipe rev       # ~6 h
uv run --no-sync python -m scripts.seed_repeat --seed 13 --recipe scratch   # ~50 h
```

- `finetune` (default): ft_geo from `submission/geo_init` to the parent's step, then ft_rev from
  that checkpoint (with the safety multipliers it was trained with), identical
  hyperparameters. The originals were interrupted and resumed with new seeds; a repeat runs
  straight through.
- `rev`: ft_rev only, from the submitted parent (`submission/rev_init`): the spread of the
  last stage alone.
- `scratch`: the method from scratch with the final code (all features on from step 0), then
  both fine-tuning stages. It is not the exact staged lineage below, so it tests the method
  rather than this checkpoint.

`--parent-only` stops after ft_geo. A run that dies (e.g. out of memory) is resumed from its
latest checkpoint with the schedules continued (`--tries`, default 3).

Training holds ~40 GiB of RAM with the default 26 simulation workers. Keep `--workers 26`
on smaller machines (it is only slower), so each PPO rollout has the original size
(26 x 88 seats x 128 steps). Check the pipeline first with `--smoke` (a few minutes).

## How the submitted policy was trained

| Stage | Steps (cumulative) | Code | What changed |
| --- | --- | --- | --- |
| main | 0 – 1.0B | `dfb3d76` | PPO from scratch, both tracks, adaptive safety multipliers (seed 1). |
| ft_ta | 1.0B – 1.07B | `5108938` | `expand_obs` adds the elapsed-time input; timeouts terminal with a penalty; only avoidable violations constrained. Best held-out checkpoint kept. |
| ft_geo | 1.07B – 1.185B | `05406ca`, `f1a9bc7` | `expand_obs` adds the route hint; geodesic progress shaping. Interrupted twice and resumed (seeds 1, 2, 3). Ends in the parent (`rev_init/`). |
| ft_rev | 1.185B – 1.370B | `d96fd81` | The rest of ft_geo's schedule plus costs for flipping the heading offset directly between left and right (0.02 = 2 s per 90°) and for speed changes beyond one bin (0.01 = 1 s per extra bin). Interrupted once (host out of memory at 1.197B) and resumed (seeds 4, 5). |

Every stage's arguments are recorded in `scripts/seed_repeat.py` (stage 1, ft_geo, ft_rev) and
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
uv run --no-sync python -m atc_agent.failure_diag --model submission/policy.zip --csv submission/results/fast_ma.csv \
    --agents-csv runs/diag/detour_agents.csv --out runs/diag/fail
uv run --no-sync python -m atc_agent.diagnose_sa --model submission/policy.zip --out runs/diag/sa
# the report (TeX Live or TinyTeX)
cd docs/solution/report && latexmk -pdf report.tex
```

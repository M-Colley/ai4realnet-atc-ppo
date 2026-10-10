# Exact Batched Simulation and Intent-Aware Entity Transformers for Safe Single- and Multi-Agent Air Traffic Control

*Draft technical report (3–4 pages target). Results tables are filled in from
`python -m scripts.evaluate_competition` once training is complete.*

## Abstract

We present a reinforcement-learning controller for the BlueSky-Gym ATC
competition that solves the single-agent task (one aircraft among ten scripted
intruders) and the multi-agent task (ten cooperating learners) with **one
shared policy**. Four elements make this work. (i) **BlueSky-Batch**: we
re-host the official environments so that hundreds of independent scenarios run
inside a single BlueSky process. With conflict resolution disabled, as the rules
require, aircraft never interact, so the dynamics are *exactly* those of the
official environment. A step-by-step equivalence test confirms identical
observations, states and all nine metrics, at ~40x the sample throughput per
CPU core. (ii) An **intent-aware entity observation** with a transformer
encoder. Other aircraft are ranked by predicted miss distance and carry CPA
features and their intent (the active route waypoint of scripted intruders,
the goal of other agents). Obstacles are represented by tokens and a lidar
over their true polygons. (iii) A **goal-relative categorical action space**,
which makes "fly direct" the default and represents "pass left *or* right" as
two modes. (iv) **Constrained PPO** with dual-ascent safety multipliers, and
**continuous-traffic self-play** for the multi-agent task.

## 1. Problem

Per the competition rules: 2-D A320 dynamics in BlueSky, a 5 NM separation
minimum, 5 km goal capture, a 3000 s budget and 1 s metric sampling, with
`RESO OFF`. The design surface is the observation, reward and action mapping,
plus the RL algorithm. Scoring uses the first 1000 seed-42 episodes and is
reported per (agent-)episode as: goal completion, flight time, intrusion
events/time, restricted-area events/time, and sector-exit events/time.

## 2. BlueSky-Batch: exact high-throughput simulation

*Observation.* One `bs.sim.step()` of the official env costs ~1.8 ms for 11
aircraft, but almost all of that is fixed per-call overhead: 1100 aircraft cost
1.9 ms (measured). The official env therefore runs at ~48 decision steps/s per
core.

*Method.* A slot holds one scenario, drawn from the official
`ScenarioGenerator`, and many slots share one simulator. Aircraft are created
with the same `traf.cre` calls and intruder routes with the same `ADDWPT` stack
commands. Commands go through BlueSky's own HDG/SPD autopilot implementations,
vectorised. The nine metrics are recomputed every simulated second with the
official deterministic `kwikqdrdist`, plus a point-in-polygon test in the local
NM frame, which is affine-equivalent to BlueSky's lat/lon test. Single-agent
slots reset on goal capture or at the time limit. Multi-agent slots remove an
aircraft at goal capture, exactly as the official env does.

*Exactness.* `tests/test_equivalence.py` flies identical scenarios and action
sequences in the official environments (with our hooks) and in BlueSky-Batch,
with other scenarios sharing the simulator. Observations agree at every step to
< 1e-5, and all scored metrics are identical, for both tracks.

*Throughput.* One process with 48 single-agent and 4 ten-agent scenarios runs
~1.5k decision steps/s, against ~48/s for the official env (~30x). We
parallelise over processes with a custom SB3 `VecEnv`. TODO: report the
training-PC numbers.

## 3. MDP design

**Observation (404-D, body frame).**
- *Ego:* goal distance and bearing; own TAS/CAS; the autopilot's selected
  heading and speed (turn/acceleration state); sector and obstacle status;
  predicted-conflict counts; and a 32-ray lidar with two channels (obstacle
  edges, sector boundary), computed on the true polygons.
- *Aircraft tokens (up to 10):* ranked by the miss distance predicted over a
  300 s horizon. Each token holds relative position and velocity, heading,
  tcpa/dcpa, predicted conflict at 120 s and 300 s, the intrusion flag, a
  learner-vs-scripted flag, and **intent**: the active waypoint for scripted
  intruders (route legs bend mid-route) and the goal for other agents.
- *Obstacle tokens (5):* nearest boundary point, signed distance, centroid,
  enclosing radius, left/right tangent bearings, and whether the obstacle
  blocks the direct line to the goal (with the entry distance).

**Action.** `MultiDiscrete([17, 6])`.
- *Heading:* target heading = bearing-to-goal + {0, ±5, ±12, ±22, ±35, ±50,
  ±70, ±90, ±120}°.
- *Speed:* CAS target in {110, 125, 140, 150, 157, 163} m/s. 163 m/s is the
  OpenAP A320 VMO at the competition altitude, so it is time-optimal.

The default action is "direct to goal". A categorical head can represent
bimodal avoidance (pass left or pass right), which a Gaussian would average.

**Reward** (unit: 0.01 = one second of flight time).
- Time: −0.01 per second.
- Progress: a potential on distance-to-goal, converted to nominal flight time.
  Flying direct at the reference speed nets ≈ 0, and faster nets a gain.
- Goal capture: +1.
- Safety costs: per event (2) and per second (0.02), for intrusions,
  restricted areas and sector exits, each scaled by an adaptive multiplier λ.
- Soft buffers: small quadratic penalties inside 8 NM of other aircraft and
  within 3 NM of obstacles or the sector edge.
- Predicted conflicts: a penalty per predicted conflict within 120 s.

None of these terms affect the scored metrics.

## 4. Learning

**Policy.** An entity transformer: type-specific token MLPs plus type
embeddings feed a 2-layer pre-LN transformer (d = 128, 4 heads, key-padding
masks). The features are the ego-token output concatenated with a masked mean
pool. Actor and critic are separate; the encoders run in bf16.

**Algorithm.** PPO (Stable-Baselines3) with a large rollout (~90k–180k
samples), 4 epochs, 4096-sample minibatches, γ = 0.99 per 10 s decision,
GAE λ = 0.95, linear LR decay and annealed entropy.

**Constrained RL.** The safety multipliers λ_intr, λ_ra and λ_sec follow dual
ascent in log space towards target event rates per episode (0.02 / 0.01 /
0.005), bounded to [1, 8].

**Single policy, both tracks.** Seats from single-agent and multi-agent
scenarios are mixed in every batch, distinguished by a mode flag and per-token
learner flags. In the multi-agent scenarios, parameter sharing is self-play.

**Continuous traffic.** When an agent reaches its goal, a new agent is spawned
using the official placement routine (≥ 10 NM from all live aircraft), so no
training samples are wasted on "dead" agents. A scenario is regenerated once
its original cohort has finished, so the dense simultaneous start is trained
on too.

**Domain randomisation.** Single-agent scenarios use {10, 12, 14} intruders
(p = 0.6 / 0.2 / 0.2).

**Model selection.** A deterministic policy is scored on 200 single-agent and
20 multi-agent held-out scenarios (seed 2024, not the reporting seed).

## 5. Results (seed 42, first 1000 episodes; official harness)

| Metric | SA neutral | SA ours | MA neutral | MA ours |
| --- | ---: | ---: | ---: | ---: |
| Goal completion rate [%] | 100.0 | | | |
| Flight time [s] | 852.0 | | | |
| Intrusion events | 1.127 | | | |
| Intrusion time [s] | 96.3 | | | |
| Restricted-area events | 0.813 | | | |
| Restricted-area time [s] | 129.9 | | | |
| Sector-exit events | 0.000 | | | |
| Time outside sector [s] | 0.0 | | | |

SA neutral = stock env and zero action, from the unmodified official script
(`results/baseline_neutral_sa.csv`). MA neutral: TODO, run
`python -m scripts.evaluate_competition --env ma` (no `--model`) on the
training machine.

Planned ablations, if compute allows: no intent features; Gaussian vs
categorical actions; fixed vs adaptive λ; single- vs multi-agent-only training.

## 6. Discussion

TODO: emergent behaviours (speed-based sequencing at shared goals, consistent
passing side, early small deviations), residual failure modes (e.g.
unavoidable conflicts at spawn, where an intruder starts within 5 NM), and
limitations.

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.
- de Witt et al., *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*, 2020.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Groot, Ellerbroek, Hoekstra, *Relative-state transformers for MARL in ATC*, SESAR Innovation Days 2023.
- Aziz & Wei, *Transformer-based MARL for Separation Assurance in Structured and Unstructured Airspaces*, arXiv:2601.04401, 2026.
- Stooke, Achiam, Abbeel, *Responsive Safety in RL by PID Lagrangian Methods*, ICML 2020.
- Beukman et al., *Scaling PPO to a million parallel environments*, RLC 2026 (arXiv:2603.06009).
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
- Hoekstra & Ellerbroek, *BlueSky ATC Simulator Project*, ICRAT 2016.
- Groot et al., *BlueSky-Gym*, SESAR Innovation Days 2024.
- Sun et al., *OpenAP: An open-source aircraft performance model*, 2020.

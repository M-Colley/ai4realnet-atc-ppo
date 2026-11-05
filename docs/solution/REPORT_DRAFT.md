# Exact Batched Simulation, Intent-Aware Entity Transformers and Failure-Driven Fine-Tuning for Safe Multi-Agent Air Traffic Control

*Draft technical report (3–4 pages target). TODO(authors): names, team,
affiliation. Numbers marked TODO are pending runs; all others are measured. The
"ours" results are for the current candidate (1185M steps); TODO(final):
replace them with the final checkpoint.*

## Abstract

We present a reinforcement-learning controller for the BlueSky-Gym ATC
competition. It solves the single-agent task (one aircraft among ten scripted
intruders) and the multi-agent task (ten learning aircraft) with **one shared
policy**. (i) **BlueSky-Batch** runs hundreds of independent scenarios inside
one BlueSky process. With conflict resolution off, as the rules require, this
is *exactly* the official environment (equivalence-tested), at ~30x the
throughput. One desktop PC trains 1B steps in 33 h. (ii) An **intent-aware
entity transformer** observes ranked traffic tokens with closest-approach
features and intent, obstacle tokens, a lidar and an obstacle-aware route hint.
(iii) A **goal-relative categorical action space** makes "fly direct" the
default and represents "pass left *or* right" as two modes. (iv) **Constrained
PPO** with adaptive safety multipliers that count only *avoidable* violations,
a time-aware observation with timeouts treated as failures, and a
**geodesic progress potential** around restricted areas.

On the official seed-42 protocol the policy completes 99.37% of multi-agent
flights with 0.023 safety events per aircraft, against 1.68 for a fly-direct
baseline. It completes 99.6% of single-agent flights with 0.148 events per
episode (1.94 for fly-direct). A failure analysis shows that most remaining
single-agent intrusions exist at spawn, and that about half of the multi-agent
failures have a goal *inside* a restricted area.

## 1. Problem

The rules fix 2-D A320 dynamics in BlueSky, 5 NM separation, 5 km goal capture,
a 3000 s budget and 1 s metric sampling, with `RESO OFF`. The design surface is
the observation, reward and action mapping, plus the RL algorithm. Results are
the first 1000 episodes of seed 42 from the official harness, per
(agent-)episode: goal completion, flight time, and intrusion, restricted-area
and sector-exit events and durations. No scalar score is defined, so we report
all metrics, and for checkpoint selection only we use
100·goal − 10·events − 0.01·flight time (one event weighs as much as 1000 s).

## 2. BlueSky-Batch: exact high-throughput simulation

*Observation.* One `bs.sim.step()` of the official env costs ~1.8 ms for 11
aircraft, and almost all of that is fixed per-call overhead: 1100 aircraft
cost 1.9 ms (measured). The official env therefore runs at ~48 decision
steps/s per core.

*Method.* A slot holds one scenario from the official `ScenarioGenerator`, and
many slots share one simulator. Aircraft are created with the same `traf.cre`
calls and intruder routes with the same `ADDWPT` stack commands. Commands go
through BlueSky's own HDG/SPD autopilot functions, vectorised. The nine metrics
are recomputed every simulated second with the official `kwikqdrdist`, plus a
point-in-polygon test in the local NM frame, which is affine-equivalent to
BlueSky's. Multi-agent slots remove an aircraft at goal capture, as the
official env does.

*Exactness.* `tests/test_equivalence.py` flies identical scenarios and actions
in the official environments (with our hooks) and in BlueSky-Batch, with other
scenarios sharing the simulator. Observations agree to < 1e-5 at every step and
all scored metrics are identical, on both tracks. Our batched evaluator
reproduces the official harness on 1000/1000 single-agent episodes and 998/1000
multi-agent scenarios. In the other two, trajectories diverge slightly: one
aircraft times out in one run but not the other, and mean metrics change in
the fourth decimal.

*Throughput.* One process with 48 single-agent and 4 ten-agent scenarios runs
~1.5k decision steps/s, against ~48/s for the official env. With 26 such
processes (custom SB3 `VecEnv`) on an i7-14700K with an RTX 5080, PPO trains at
8.5k samples/s: 1B steps in 32.6 h.

## 3. MDP design

**Observation (409-D, body frame).**
- *Ego:*
  - goal distance and bearing; own TAS/CAS;
  - the autopilot's selected heading and speed (turn/acceleration state);
  - sector and obstacle status; predicted-conflict counts;
  - a 32-ray lidar with two channels (obstacle edges, sector boundary), on the true polygons;
  - **elapsed time**, so the remaining budget is observable;
  - an **obstacle-aware route hint**: geodesic distance to the goal and the direction of steepest descent of a cost-to-go field (16-connected Dijkstra on a 96×96 grid per scenario, `scipy`).
- *Aircraft tokens (up to 10):* ranked by the miss distance predicted over 300 s.
  - relative position and velocity, heading, tcpa/dcpa, predicted conflict at 120 s and 300 s, the intrusion flag, a learner-vs-scripted flag;
  - **intent:** the active waypoint for scripted intruders (route legs bend mid-route) and the goal for other agents.
- *Obstacle tokens (5):* nearest boundary point, signed distance, centroid, enclosing radius, left/right tangent bearings, and whether the obstacle blocks the direct line to the goal.

**Action.** `MultiDiscrete([17, 6])`.
- *Heading:* bearing-to-goal + {0, ±5, ±12, ±22, ±35, ±50, ±70, ±90, ±120}°.
- *Speed:* CAS in {110, 125, 140, 150, 157, 163} m/s. 163 m/s is the OpenAP A320 VMO at the competition altitude.

Decisions are made every 10 s. A categorical head represents bimodal avoidance,
which a Gaussian would average into a path straight through the conflict.

**Reward** (unit: 0.01 = one second of flight time).
- Time: −0.01 per second.
- Goal capture: +1.
- **Timeout:** −20, terminal and not bootstrapped. Because the elapsed time is observed, running out of budget is a failure of the task rather than an artificial truncation (Pardo et al., 2018).
- **Progress:** a potential-based term (Ng et al., 1999) on the *geodesic* route length around restricted areas. The straight-line potential pulled aircraft into concave obstacle pockets, where they timed out.
- Safety: per event (2) and per second (0.02), for intrusions, restricted areas and sector exits, each scaled by an adaptive multiplier λ (§4).
- Small shaping terms: quadratic soft buffers within 8 NM of aircraft and 3 NM of obstacles or the sector edge, and a cost per conflict predicted within 120 s.

None of these terms affect the scored metrics.

## 4. Learning

**Policy.**
- Type-specific token MLPs and type embeddings feed a 2-layer pre-LN transformer (d = 128, 4 heads, padding masks).
- The features are the ego token concatenated with a masked mean pool.
- Actor and critic are separate, and the encoders run in bf16.

**Algorithm.** PPO (Stable-Baselines3):
- rollouts of 292,864 samples;
- 4 epochs of 4096-sample minibatches;
- γ = 0.99 per 10 s decision, GAE λ = 0.95;
- linear LR decay and annealed entropy.

**Constrained RL on avoidable violations.**
- The multipliers λ_intr, λ_ra and λ_sec follow dual ascent in log space towards target rates of 0.02 / 0.01 / 0.005 events per episode, bounded to [1, 8].
- The generator does not keep scripted intruders 5 NM away from the single-agent start. Some intrusions therefore exist at t = 0, and no policy can avoid them (§6).
- Counting them in the constraint drives λ to its cap regardless of behaviour, so the constraint statistic **excludes violations present at spawn**. They still appear in the reported metrics.

**One policy for both tracks.** Single- and multi-agent seats are mixed in every batch, distinguished by a mode flag and per-token learner flags. Parameter sharing in the multi-agent scenarios is self-play.

**Continuous traffic.** A finished multi-agent seat respawns using the official placement routine (≥ 10 NM from live aircraft), so no samples are wasted on dead agents. Scenarios are regenerated once the original cohort is done, so the dense simultaneous start is trained on too.

**Domain randomisation.** Single-agent scenarios use {10, 12, 14} intruders (p = 0.6 / 0.2 / 0.2).

**Training schedule.** A 1B-step run from scratch, then fine-tuning stages, each resumed from the previous one:

| Stage | Steps | Change | Obs |
| --- | ---: | --- | ---: |
| main | 0 → 1000M | — | 404 |
| time-aware | 1000M → 1070M (best of 300M) | elapsed time, timeout = failure, avoidable-only constraint | 405 |
| geodesic | 1070M → 1185M | geodesic potential, route hint | 409 |

When the observation grows, the new input columns get zero weights, so the
converted policy acts identically until training changes it.

**Model selection.** A deterministic policy is scored every 10M steps on 200
single-agent and 20 multi-agent held-out scenarios (seed 2024, not the
reporting seed).

## 5. Results (seed 42, first 1000 episodes, official harness)

| Metric | SA fly-direct | SA ours | MA fly-direct | MA ours |
| --- | ---: | ---: | ---: | ---: |
| Goal completion rate [%] | 100.0 | 99.6 | 100.0 | 99.37 |
| Flight time [s] | 852.0 | 987.6 | 847.7 | 921.4 |
| Intrusion events | 1.127 | 0.127 | 0.877 | 0.010 |
| Intrusion time [s] | 96.3 | 5.14 | 64.8 | 0.36 |
| Restricted-area events | 0.813 | 0.013 | 0.799 | 0.008 |
| Restricted-area time [s] | 129.9 | 0.74 | 121.7 | 0.51 |
| Sector-exit events | 0.000 | 0.008 | 0.000 | 0.005 |
| Time outside sector [s] | 0.0 | 0.64 | 0.0 | 0.21 |

Fly-direct is the stock env with the zero action (the harness default). The
multi-agent figures are per agent-episode, over 10,000 agent-episodes.

**What each stage contributed.**
- Every checkpoint is flown on the same 1000 + 1000 seed-42 scenarios.
- Changes in multi-agent failures are paired per scenario, with a scenario-cluster BCa bootstrap (`scipy`).
- Single-agent failure changes use an exact McNemar test.
- `ft_ctrl` continues the main run for 300M steps with no change: a control for extra training.

| Checkpoint | SA failures /1000 | MA failures /10,000 | Δ MA failures per 1000 scenarios vs main [95% CI] |
| --- | ---: | ---: | --- |
| main (950M, best) | 8 | 83 | — |
| + 300M steps, no change (control) | 9 | 88 | +5 [−8, +19] |
| + avoidable-only constraint | 5 | 75 | −8 [−21, +4] |
| time-aware (1070M) | 9 | 78 | −5 [−22, +11] |
| time-aware, +230M steps (control) | 3 | 79 | −4 [−21, +12] |
| **geodesic (1185M)** | **4** | **62** | **−21 [−41, −5]** |

Only the geodesic stage reduces multi-agent failures significantly. Its effect
grew monotonically over the fine-tune: 82 → 72 → 63 → 62 failures at 1087M,
1165M, 1177M and 1185M. Safety events were unchanged throughout (0.022–0.025
per agent). Single-agent changes are not significant: 8 → 4, McNemar p = 0.29.
Against the time-aware control, which trained twice as long (230M vs 115M
steps), the geodesic checkpoint has 17 fewer MA failures per 1000 scenarios
[5, 31]. The control reaches 3 SA failures, so on an equal SA/MA weighting of
100·goal − 10·violations − 0.01·flight time the two tie: +0.02 [−0.22, +0.21].

## 6. Failure analysis

**Single-agent violations are mostly unavoidable.**
- We replayed every violating episode exactly and branched each of the 102 actions 10–160 s before the violation's onset (a diagnostic only; the controller stays the RL policy).
- Of 141 intrusions on 1000 fresh episodes (1185M policy, seed 777), **116 exist at t = 0**: the intruder spawns within 5 NM.
- Of the 25 later ones, an escape was found for 9. The same search finds escapes for 14/18 restricted-area and 4/6 sector-exit events.
- All 147 control replays reproduce the original metrics exactly.

**Every failure is a timeout; half of the multi-agent ones are infeasible goals.**
- All 4 single-agent and 62 multi-agent failures run out of the 3000 s budget.
- A small fraction of generated goals lie *inside* a restricted area: 29 of 10,000 multi-agent goals on seed 42, and none of them was reached.
  - **23** lie deeper than the 5 km capture radius, so they can be reached only by entering the area.
  - **6** are shallower and could be captured from outside.

| Failure class (all timeouts) | MA, seed 42 | MA, 50k mined | SA, seed 42 | SA, 20k mined |
| --- | ---: | ---: | ---: | ---: |
| goal inside an area, deeper than capture radius | 23 | 94 | 1 | 16 |
| goal inside an area, capturable from outside | 6 | 34 | 0 | 6 |
| goal within 10 NM of an area | 1 | 9 | 1 | 0 |
| goal more than 10 NM from any area | 32 | 196 | 2 | 64 |

**The other multi-agent failures are long detours, not congestion** (`atc_agent.failure_diag`, `runs/diag/fail_1185.md`).
- 33 failures have their goal outside every area. Each was re-flown alone (other aircraft removed), and alone and in its scenario with actions sampled from the policy (16 seeds each).
- All 33 have a restricted-area cluster across the straight line. Their area-free route is a median 1.76 times the straight line.
- Flights with a detour ratio of at least 1.5 are 3.8% of multi-agent flights but hold 27 of the 33 failures: a failure rate of 7.2%, against 0.06% below that ratio.
- 20 of the 33 reach the goal when flown alone. Yet in their last 30 min another aircraft is within 10 NM only 18% of the time; one pair is a lasting standoff. An encounter pushes them off their route, and they then shuttle back and forth along an area edge or circle in a concave pocket between areas, a median 5 NM from an area.
- The other 13 also fail alone. Sampling rescues 2 of them in 62% of seeds and the remaining 11 in 11%.

**We chose not to enter restricted areas.** Completion would rise by about
0.3 points, but at the cost of safety events. For the shallow goals the
original policy does poorly. On 249 held-out scenarios with such goals, it
reaches 14% and enters the area in 37% of cases, ~50x its usual rate. The
reason is that its route hint aims at the goal point inside the area, while the
safety cost keeps it out.

**Fix (TODO results):**
- The route field now targets the nearest point just outside the area whenever the goal is less than 5 km inside it. Deep goals keep their target.
- We fine-tune with **prioritized replay** (Jiang et al., 2021) of mined hard scenarios: shallow-goal multi-agent scenarios, multi-agent timeouts, and single-agent timeouts and the slowest 5%. Scenarios with deep goals are excluded, so replay never rewards an entry.

**Remaining inefficiency.**
- We bound each flight by its geodesic route length flown at the policy's own 90th-percentile clear-path speed.
- Against that bound, the excess is 139 s per single-agent and 81 s per multi-agent flight. The slowest 10% of flights carry about half of it (loops near the start, waiting for traffic, long detours).
- These flights reverse turn direction twice as often. But decoding with hysteresis does not help: only a weak bonus is neutral-to-positive (+0.13 selection score on seed 42 [+0.00, +0.38]), and strong commitment lengthens flights. The reversals are mostly purposeful steering.
- Biasing decoding towards top speed and the direct heading raises multi-agent violations by 45% without fewer failures. So the learned safety/efficiency trade-off is not over-cautious.
- Uniform weight averaging of the last three checkpoints ("model soup") changes nothing.

## 7. Limitations

- One training seed per configuration. Paired tests control for scenario noise, not for seed variance. TODO: repeat the geodesic stage with more seeds.
- The scenario generator occasionally produces goals inside restricted areas. We report them separately rather than trade safety for completion.
- The selection score is our own; the official judging weighs the metrics jointly.

## References

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.
- de Witt et al., *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*, 2020.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Ng, Harada, Russell, *Policy Invariance under Reward Transformations*, ICML 1999.
- Pardo et al., *Time Limits in Reinforcement Learning*, ICML 2018.
- Altman, *Constrained Markov Decision Processes*, 1999.
- Stooke, Achiam, Abbeel, *Responsive Safety in RL by PID Lagrangian Methods*, ICML 2020.
- Jiang, Grefenstette, Rocktäschel, *Prioritized Level Replay*, ICML 2021.
- Wortsman et al., *Model Soups*, ICML 2022.
- Groot, Ellerbroek, Hoekstra, *Relative-state transformers for MARL in ATC*, SESAR Innovation Days 2023.
- Hoekstra & Ellerbroek, *BlueSky ATC Simulator Project*, ICRAT 2016.
- Groot et al., *BlueSky-Gym*, SESAR Innovation Days 2024.
- Sun et al., *OpenAP: An open-source aircraft performance model*, 2020.

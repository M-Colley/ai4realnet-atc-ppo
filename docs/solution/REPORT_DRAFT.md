# Batched Simulation, Intent-Aware Entity Transformers and Failure-Driven Fine-Tuning for Safe Multi-Agent Air Traffic Control

*Markdown draft of the technical report. The submitted version is the LaTeX
source in `report/report.tex`; this draft follows it. Numbers marked TODO are
pending runs; all others are measured. "Ours" is the submitted policy (1370M
steps).*

## Abstract

We present a reinforcement-learning controller for the BlueSky-Gym ATC
competition. It solves the single-agent task (one aircraft among ten scripted
intruders) and the multi-agent task (ten learning aircraft) with **one shared
policy**. (i) **BlueSky-Batch** runs hundreds of independent scenarios inside
one BlueSky process. With conflict resolution off, as the rules require, it
reproduces the official environment (observations within 1e-5, identical scored
metrics for the submitted policy) at 31x the throughput. One desktop PC trains 1B steps in 33 h. (ii) An **intent-aware
entity transformer** observes ranked traffic tokens with closest-approach
features and intent, obstacle tokens, a lidar and an obstacle-aware route hint.
(iii) A **goal-relative categorical action space** makes "fly direct" the
default and represents "pass left *or* right" as two modes. (iv) **Constrained
PPO** with adaptive safety multipliers that count only *avoidable* violations,
a time-aware observation with timeouts treated as failures, a
**geodesic progress potential** around restricted areas, and small **costs on
indecision**.

On the official seed-42 protocol the policy completes 99.39% of multi-agent
flights with 0.024 safety events per aircraft, against 1.68 for a fly-direct
baseline. It completes 99.7% of single-agent flights with 0.143 events per
episode (1.94 for fly-direct). BlueSky's Modified Voltage Potential (MVP)
conflict resolution along an area-free route, a classical reference, has 0.72
and 1.33. A failure analysis shows that most remaining single-agent intrusions
exist at spawn, and that over a third of the multi-agent failures have a goal
*inside* a restricted area.

## 1. Problem

The rules fix 2-D A320 dynamics in BlueSky, 5 NM separation, 5 km goal capture,
a 3000 s budget and 1 s metric sampling, with `RESO OFF`. The design surface is
the observation, reward and action mapping, plus the RL algorithm. Results are
the first 1000 episodes of seed 42 from the official harness, per
(agent-)episode: goal completion, flight time, and intrusion, restricted-area
and sector-exit events and durations. No scalar score is defined, so we report
all metrics, and for checkpoint selection only we use
100·goal − 10·events − 0.01·flight time (one event weighs as much as 1000 s).

## 2. BlueSky-Batch: high-throughput simulation

*Observation.* One `bs.sim.step()` of the official env costs ~1.8 ms for 11
aircraft, and almost all of that is fixed per-call overhead: 1100 aircraft
cost 1.9 ms (measured). The official env therefore runs at ~48 decision
steps/s per core.

*Method.* A slot holds one scenario from the official `ScenarioGenerator`, and
many slots share one simulator. Aircraft are created with the same `traf.cre`
calls and intruder routes with the same `ADDWPT` stack commands. Commands go
through BlueSky's own HDG/SPD autopilot functions, vectorised. The eight metrics
are recomputed every simulated second with the official `kwikqdrdist`, plus a
point-in-polygon test in the local NM frame, which is affine-equivalent to
BlueSky's. Multi-agent slots remove an aircraft at goal capture, as the
official env does.

*Equivalence.* `tests/test_equivalence.py` flies identical scenarios and actions
in the official environments (with our hooks) and in BlueSky-Batch, with other
scenarios sharing the simulator. Observations agree to < 1e-5 at every step and
all scored metrics are identical, on both tracks. Our batched evaluator
reproduces the official harness on 1000/1000 single-agent episodes and on
998–1000 of the 1000 multi-agent scenarios, depending on the checkpoint (all
1000 for the submitted policy). Where it differs, one aircraft's trajectory
diverges slightly, and mean metrics change in the fourth decimal.

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
- *Speed:* CAS in {110, 125, 140, 150, 157, 163} m/s. 163 m/s (317 kt) is the A320's en-route CAS limit (`vmaxer`) in BlueSky's OpenAP performance model; at the competition's 350 m altitude it is 322 kt true airspeed.

Decisions are made every 10 s. A categorical head represents bimodal avoidance,
which a Gaussian would average into a path straight through the conflict.

**Reward** (unit: 0.01 = one second of flight time).
- Time: −0.01 per second.
- Goal capture: +1.
- **Timeout:** −20, terminal and not bootstrapped. Because the elapsed time is observed, running out of budget is a failure of the task rather than an artificial truncation (Pardo et al., 2018).
- **Progress:** a potential-based term (Ng et al., 1999) on the *geodesic* route length around restricted areas. The straight-line potential pulled aircraft into concave obstacle pockets, where they timed out.
- Safety: per event (2) and per second (0.02), for intrusions, restricted areas and sector exits, each scaled by an adaptive multiplier λ (§4).
- Small shaping terms: quadratic soft buffers within 8 NM of aircraft and 3 NM of obstacles or the sector edge, and a cost per conflict predicted within 120 s.
- **Indecision costs** (last stage only): 0.02 per 90° when the heading offset flips directly from one side of the goal bearing to the other (of the smaller of the two offsets), and 0.01 per speed bin changed beyond one between consecutive decisions.

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
| indecision costs | 1185M → 1370M (submitted) | rest of the geodesic stage's 300M-step schedule, plus the indecision costs | 409 |

When the observation grows, the new input columns get zero weights, so the
converted policy acts identically until training changes it.

**Model selection.** A deterministic policy is scored every 10M steps on 200
single-agent and 20 multi-agent held-out scenarios (seed 2024, not the
reporting seed). We submit the last stage's final checkpoint; its best
held-out checkpoint does not differ significantly on seeds 42 and 777 (score
+0.00 [−0.11, +0.11]).

## 5. Results (seed 42, first 1000 episodes, official harness)

| Metric | SA fly-direct | SA direct + MVP | SA route + MVP | SA ours | MA fly-direct | MA direct + MVP | MA route + MVP | MA ours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Goal completion rate [%] | 100.0 | 97.2 | 95.6 | 99.7 | 100.0 | 99.96 | 99.44 | 99.39 |
| Flight time [s] | 852.0 | 1019.6 | 1183.9 | 978.7 | 847.7 | 858.6 | 986.3 | 916.7 |
| Intrusion events | 1.127 | 0.519 | 0.587 | 0.126 | 0.877 | 0.004 | 0.098 | 0.008 |
| Intrusion time [s] | 96.3 | 54.6 | 55.7 | 5.09 | 64.8 | 0.01 | 0.12 | 0.35 |
| Restricted-area events | 0.813 | 0.836 | 0.605 | 0.012 | 0.799 | 0.804 | 0.594 | 0.009 |
| Restricted-area time [s] | 129.9 | 134.0 | 31.5 | 0.50 | 121.7 | 122.6 | 39.0 | 0.60 |
| Sector-exit events | 0.000 | 0.076 | 0.135 | 0.005 | 0.000 | 0.001 | 0.030 | 0.007 |
| Time outside sector [s] | 0.0 | 71.7 | 104.4 | 0.21 | 0.0 | 0.9 | 8.6 | 0.30 |

Fly-direct is the stock env with the zero action (the harness default). The
multi-agent figures are per agent-episode, over 10,000 agent-episodes.

The MVP columns are a classical reference (`atc_agent.mvp_baseline`), not a
competition entry: the rules require `RESO OFF` for the agent.
- Each aircraft flies at its initial speed towards its goal (*direct*) or towards a point 5 NM along the area-free grid route that is also the policy's route hint (*route*).
- BlueSky's state-based conflict detection (5 NM, 300 s look-ahead) and MVP resolution (heading and speed) take over during a conflict.
- In multi-agent scenarios every aircraft resolves, so resolution is cooperative. In single-agent scenarios only the agent resolves and the intruders fly their routes.
- MVP knows nothing about restricted areas or the sector.

**What each stage contributed.**
- Every checkpoint is flown on the same 1000 + 1000 seed-42 scenarios.
- Changes in multi-agent failures are paired per scenario, with a scenario-cluster BCa bootstrap (`scipy`).
- Single-agent failure changes use an exact McNemar test.
- Controls continue a stage for as many steps with no change.

| Checkpoint | SA failures /1000 | MA failures /10,000 | Flight time SA / MA [s] | Δ MA failures per 1000 scenarios vs main [95% CI] |
| --- | ---: | ---: | ---: | --- |
| main (950M, best) | 8 | 83 | 1002 / 932 | — |
| + 300M steps, no change (control) | 9 | 88 | 1003 / 930 | +5 [−8, +19] |
| + avoidable-only constraint | 5 | 75 | 998 / 930 | −8 [−21, +4] |
| time-aware (1070M) | 9 | 78 | 996 / 932 | −5 [−22, +11] |
| time-aware, +230M steps (control) | 3 | 79 | 990 / 927 | −4 [−21, +12] |
| geodesic (1185M) | 4 | 62 | 988 / 921 | −21 [−41, −5] |
| geodesic, +185M steps (control) | 3 | 65 | 987 / 919 | −18 [−36, −4] |
| **+ indecision costs (1370M, submitted)** | **3** | **61** | **979 / 917** | **−22 [−40, −8]** |

Only the geodesic stage reduces multi-agent failures significantly. Its effect
grew monotonically over the fine-tune: 82 → 72 → 63 → 62 failures at 1087M,
1165M, 1177M and 1185M. Safety events were unchanged throughout (0.022–0.025
per agent). Single-agent changes are not significant: 8 → 4, McNemar p = 0.29.
Against the time-aware control, which trained twice as long (230M vs 115M
steps), the geodesic checkpoint has 17 fewer MA failures per 1000 scenarios
[5, 31]. The control reaches 3 SA failures, so on an equal SA/MA weighting of
100·goal − 10·violations − 0.01·flight time the two tie: +0.02 [−0.22, +0.21].
On seed 777 the control again has more MA failures, +26 [11, 43] per 1000
scenarios, and a lower score, −0.32 [−0.64, −0.03].

The indecision costs shorten flights instead. Pooled over seeds 42 and 777
(2000 scenarios per track), the submitted policy flies 4.2 s [2.4, 6.0] less
per MA flight and 10.5 s [4.1, 16.9] less per SA flight than the 1185M
checkpoint. Violations and MA failures are unchanged (−2.5 [−12, +6.5] per 1000
scenarios); the selection score rises by +0.17 [+0.01, +0.35]. The control
that continues the geodesic stage for the same 185M steps without the costs
flies 2.6 s [0.8, 4.6] (MA) and 7.5 s [1.9, 13.4] (SA) longer than the
submitted policy.

**Classical reference.** MVP has perfect state information, and in multi-agent
scenarios every aircraft resolves cooperatively.
- Along the area-free route it completes as many multi-agent flights as the policy (99.44% vs 99.39%). It has 0.72 safety events per aircraft against the policy's 0.024, mostly restricted-area entries, because its manoeuvres ignore the areas.
- Flown direct, MVP has fewer intrusions than the policy (0.004 vs 0.008) and 99.96% completion, at 0.80 restricted-area events per aircraft.
- Against the non-cooperative single-agent intruders it leaves 4–5 times the policy's intrusions and fails 2.8–4.4% of flights.

**Denser traffic** (`atc_agent.scale_eval`, BlueSky-Batch, 300 scenarios per size, 5 restricted areas, submitted policy).
- The policy observes at most ten aircraft tokens (the nearest by predicted miss distance), so with more traffic it never sees all of it.

| Track | Traffic | Goal [%] | Intrusions (fly-direct) | RA events | Flight time [s] |
| --- | --- | ---: | ---: | ---: | ---: |
| MA | 10 aircraft | 99.3 | 0.005 (0.84) | 0.013 | 930 |
| MA | 15 aircraft | 98.8 | 0.016 (1.32) | 0.025 | 957 |
| MA | 20 aircraft | 98.8 | 0.019 (1.75) | 0.023 | 970 |
| MA | 30 aircraft | 97.6 | 0.036 (2.67) | 0.062 | 1017 |
| SA | 10 intruders | 100.0 | 0.09 (1.11) | 0.013 | 991 |
| SA | 20 intruders | 99.3 | 0.27 (2.25) | 0.043 | 1131 |
| SA | 30 intruders | 99.3 | 0.54 (3.49) | 0.057 | 1265 |

Training used 10 aircraft (MA) and 10–14 intruders (SA). With three times the
density the policy degrades gradually: it still resolves 98.7% (MA) and 85%
(SA) of the fly-direct intrusions.

**Pre-registered final experiments** (`docs/solution/EXPERIMENTS_2026-11.md`, fixed 2026-11-15 before any result).
- Five candidates were tested against the submitted policy:
  - a uniform weight average of the four last-stage runs (ours and seed repeats 11–13);
  - their decision-time ensemble, which averages action probabilities;
  - 150M further steps with prioritized replay of 576 mined hard scenarios (315 SA, 261 MA; none with a goal inside a restricted area);
  - 150M further steps with the time cost doubled (0.02 per second);
  - an unchanged 150M-step continuation as the control.
- A candidate would replace the submission only on a pooled seed 42 + 777 gain in MA failures (or score), with no safety regression and confirmation on seed 9001.
- None qualified, so the submission is unchanged.

| Candidate | Δ MA failures /1000 | Δ score | Δ MA time [s] | Δ SA time [s] |
| --- | --- | --- | --- | --- |
| weight average | −1.0 [−11.0, +8.5] | +0.00 [−0.14, +0.15] | −1.6 [−3.5, +0.2] | +2.6 [−2.6, +8.2] |
| ensemble | +0.5 [−9.0, +10.0] | +0.01 [−0.15, +0.17] | −1.3 [−3.1, +0.5] | +2.9 [−2.4, +9.1] |
| hard replay | −3.0 [−13.0, +6.0] | −0.07 [−0.25, +0.10] | +1.5 [−0.3, +3.4] | +10.6 [+4.3, +17.7] |
| doubled time cost | +3.5 [−6.5, +14.5] | −0.02 [−0.18, +0.14] | −3.2 [−5.2, −1.2] | −3.0 [−8.7, +3.0] |
| unchanged control | +3.0 [−7.0, +14.0] | −0.14 [−0.33, +0.04] | −0.6 [−2.5, +1.2] | +7.1 [+1.8, +13.0] |

Exploratory comparisons, not corrected for multiplicity:
- Against the equal-compute control, the doubled time cost shortens MA flights by 2.6 s [0.7, 4.5] and SA flights by 10.1 s [3.8, 16.4]. It raises the score by 0.12 [0.01, 0.25] and adds 0.004 [0.000, 0.008] MA violations per aircraft.
- Hard replay against the control: −6 [−17, +4] MA failures per 1000 scenarios.
- More training alone does not improve the submitted policy, and SA flights get longer.
- Earlier, averaging two checkpoints had added MA failures. Averaging all four last-stage runs does not degrade the policy.

## 6. Failure analysis

**Single-agent violations are mostly unavoidable** (`atc_agent.diagnose_sa`, submitted policy, 1000 episodes of seed 777).
- We replayed every violating episode exactly and branched each of the 102 actions 10–160 s before the violation's onset (a diagnostic only; the controller stays the RL policy).
- Of 136 intrusions, **116 exist at t = 0**: the intruder spawns within 5 NM.
- Of the 20 later ones, an escape was found for 5. The same search finds escapes for 9/15 restricted-area and 4/6 sector-exit events.
- All 138 control replays reproduce the original metrics exactly.

**Every failure is a timeout; over a third of the multi-agent ones are infeasible goals.**
- All 3 single-agent and 61 multi-agent failures run out of the 3000 s budget.
- 29 of 10,000 multi-agent goals on seed 42 lie *inside* a restricted area. The policy reaches one of them, after entering the area.
  - **23** lie deeper than the 5 km capture radius, so they can be reached only by entering the area.
  - **6** are shallower and could be captured from outside.
- This is a property of the generator: on 50k fresh multi-agent agent-episodes, 128 of the 1185M checkpoint's 333 failures have their goal inside an area (single-agent: 22 of 86 on 20k episodes; seed 42, submitted policy: 1 of 3).
- For the submitted policy on seeds 42 and 777 (20,000 MA aircraft), 50 of its 136 MA failures have a goal inside an area; 52 aircraft have such a goal. None of the pre-registered candidates reduces this class.

| MA failure class, seed 42 (all timeouts) | 1185M | submitted |
| --- | ---: | ---: |
| goal inside an area, deeper than capture radius | 23 | 23 |
| goal inside an area, capturable from outside | 6 | 5 |
| reaches the goal when flown alone (traffic) | 20 | 21 |
| fails alone greedily, ≥ 50% of sampled solo flights reach it (greedy trap) | 2 | 4 |
| fails alone (navigation) | 11 | 8 |
| total | 62 | 61 |

**We chose not to enter restricted areas.** Completion would rise by about
0.3 points, but at the cost of safety events. For the shallow goals the policy
does poorly. On 249 held-out scenarios (265 shallow goals), the 1185M
checkpoint reaches 14% and enters the area in 37% of cases. The reason is that
its route hint aims at the goal point inside the area, while the safety cost
keeps it out.

**Shallow-goal fix (ablation, not submitted).**
- The route field targets the nearest point just outside the area whenever the goal is less than 5 km inside it. Deep goals keep their target.
- Fine-tune 1185M → 1370M with **prioritized replay** (Jiang et al., 2021) of mined hard scenarios: shallow-goal multi-agent scenarios, multi-agent timeouts, and single-agent timeouts and the slowest 5%. Scenarios with deep goals are excluded, so replay never rewards an entry.
- Shallow-goal capture nearly doubles, to 26.8% [21.6, 32.6] (Clopper–Pearson). Entries stay at 36%.
- Pooled over seeds 42 and 777, multi-agent flights become 2.6 s [0.5, 4.7] longer, multi-agent violations fall slightly (−0.004 per aircraft [−0.007, −0.001]), and failures do not change (+4.5 [−7.5, +16.5] per 1000 scenarios). We did not submit it.

**The other multi-agent failures are long detours, not congestion** (`atc_agent.failure_diag`, `runs/diag/fail_rev.md`).
- 33 failures have their goal outside every area. Each was re-flown alone (other aircraft removed), and alone and in its scenario with actions sampled from the policy (16 seeds each).
- All 33 have a restricted-area cluster across the straight line. Their area-free route is a median 1.64 times the straight line.
- Flights with a detour ratio of at least 1.5 are 3.8% of multi-agent flights but hold 26 of the 33 failures: a failure rate of 6.9%, against 0.07% below that ratio.
- 21 of the 33 reach the goal when flown alone. Yet in their last 30 min another aircraft is within 10 NM only 15% of the time. An encounter pushes them off their route, and they then shuttle back and forth along an area edge or circle in a concave pocket between areas, a median 5 NM from an area.
- Of the other 12, sampling rescues 4 in 66% of seeds and the remaining 8 in 11%.

**The indecision costs remove chatter, not wrong-side detours** (`atc_agent.detour_diag`, `runs/diag/detour_1185_v2.md`, `runs/diag/detour_rev.md`).
- Each successful seed-42 multi-agent flight is compared with the shortest route that never enters an area (area-free grid path, string-pulled), flown at the policy's own top ground speed (322 kt).
- For the 1185M checkpoint the excess averages 54 s per flight. Half of it comes from the slowest 10% of the flights whose straight line crosses an area (5% of all flights, +518 s each, 86% of it extra path).
- Two thirds of these pass an area on the other side than the shortest route, a quarter loop, and they flip the heading offset across the goal bearing (≥ 22°) 40 times per flight, against 18 for the other crossing flights.
- Decoding with hysteresis does not fix this: only a weak bonus is neutral-to-positive (+0.13 selection score on seed 42 [+0.00, +0.38]), and strong commitment lengthens flights. Biasing decoding towards top speed and the direct heading raises multi-agent violations by 45% without fewer failures.
- Trained in as costs, commitment works: the flips fall to 16 (slow tail) and 3 (other crossing flights) per flight, the tail flies at top speed in 77% instead of 67% of its decisions, and the mean excess drops to 50 s.
- The tail's extra path (449 vs 446 s) and its wrong-side share (65% vs 66%) do not change: the costs remove hesitation, not the choice of the wrong side.
- Uniform weight averaging of the last three checkpoints ("model soup") changes nothing. Combining the 1185M checkpoint with the 1300M control, by averaging weights or action probabilities, adds 14 [1, 29] and 19 [5, 36] multi-agent failures per 1000 scenarios on seed 42.

## 7. Limitations

- Paired tests control for scenario noise, not for seed variance. Seed repeats re-train both fine-tuning stages from the 1070M checkpoint with new seeds (`scripts/seed_repeat.py`); the earlier stages were trained once.
  - Pooled over seeds 42 and 777 (2000 scenarios per track), the three repeats end with more multi-agent failures than ours: 5.5 [−4, +16] per 1000 scenarios for seed 11, 7 [−3.5, +18.5] for seed 12 and 1.5 [−8.5, +11] for seed 13. Their scores are −0.14 [−0.31, +0.01], −0.13 [−0.31, +0.04] and −0.07 [−0.24, +0.09]. Averaged over the three repeats: +4.7 [−4, +13.5] failures (0.05 points of multi-agent completion), score −0.11 [−0.23, +0.03]. Every scenario-bootstrap interval includes 0, but with the run as the unit the repeats score lower than ours (t-test p = 0.03): our run is a favourable draw.
  - The indecision-cost stage replicates. Each run against its own 1185M checkpoint shortens multi-agent flights: 4.2 s (ours), 7.2 s [5.3, 9.0] (seed 11), 6.3 s [4.4, 8.3] (seed 12) and 8.2 s [6.1, 10.3] (seed 13).
  - Averaged over the four runs (per-scenario differences averaged across runs, then a scenario-cluster BCa bootstrap): multi-agent flights −6.5 s [−7.4, −5.5], single-agent flights −9.4 s [−12.9, −6.0], score +0.15 [+0.07, +0.24], violations unchanged. Multi-agent failures fall in every run (−2.5, −9, −3, −6), on average −5 [−10.5, +0.5] per 1000 scenarios.
  - With the run as the unit (one-sample t-test over 4 runs), the effects on flight time (multi-agent p = 0.005, single-agent p = 0.03) and score (p = 0.02) also exclude 0. Command: `python -m scripts.seed_repeat_pooled --seeds 11 12 13`.
  - At the 1185M step the repeats have 12 [2, 23], 7.5 [−2.5, +18.5] and 5 [−4.5, +14.5] more multi-agent failures than our 1185M checkpoint. Our geodesic stage was stopped while we monitored seed 42.
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

# Final-fortnight experiments (pre-registered 2026-11-15, before any result)

Baseline: the submitted policy, `runs/ft_rev/final.zip` (1,370,310,656 steps).
Constraints kept: the solution stays RL-based; the policy is never trained to enter a
restricted area; seed 42 (reporting) and seed 777 (held-out) are never trained or mined on.

## Candidates

| Test | Candidate | What changes |
| --- | --- | --- |
| 1 | `soup4` | Uniform weight average of the four ft_rev ends (submitted, repeats 11, 12, 13), which all descend from `submission/geo_init` (1.07B). One network. |
| 1 | `ens4` | The same four policies, action probabilities averaged per head at decision time (`fast_eval --ensemble`). Whether a decision-time ensemble counts as RL-only is the user's call. |
| 4 | `ft_hard` | 150M steps from the submitted policy with prioritized replay of mined hard scenarios: SA timeouts and slowest 5%, MA scenarios with a timeout, mined with the submitted policy on seeds 3001-3020; scenarios with any goal inside a restricted area are excluded. Replay probabilities 0.15 SA / 0.3 MA (as ft_pool). |
| 5 | `ft_time` | 150M steps from the submitted policy with a time cost of 0.02 per second instead of 0.01. |
| control | `ft_cont` | 150M steps from the submitted policy, nothing changed. Attributes any gain of ft_hard / ft_time to the change rather than to more training. |

All three training stages use ft_rev's arguments, training seed 20, entropy 0.0015 -> 0.001,
and are judged on `final.zip` (no best-checkpoint selection).

Not candidates (report only): test 2, the submitted policy on denser traffic (20 and 30
aircraft, 8 restricted areas; not scored); test 3, fly-direct with BlueSky's MVP conflict
resolution as a classical reference row.

## Evaluation

Every candidate is evaluated with `atc_agent.fast_eval` (1000 SA episodes, 1000 MA scenarios)
on seed 42 and seed 777, and paired against the submitted policy per scenario
(`scripts.seed_repeat_pooled`-style: scenario-cluster BCa bootstrap, 95%).

Primary outcome: MA failures per 1000 scenarios. Secondary: the selection score
(100 goal - 10 violations - 0.01 flight time, equal SA/MA weight), violations per aircraft,
flight time.

## Decision rule (fixed now)

A candidate replaces the submitted policy only if all of these hold:

1. Pooled over seeds 42 and 777, either the MA-failure difference has its whole 95% CI below 0,
   or the score difference has its whole CI above 0 with an MA-failure point estimate <= 0.
2. No safety regression: pooled MA and SA violation differences have point estimates <= 0 or
   CIs that include 0, and SA failures do not rise significantly (exact McNemar, p < 0.05).
3. Confirmation on a third stream used for nothing else, seed 9001: the qualifying effect
   from 1 has the same sign and the score difference's CI does not lie entirely below 0.
4. If several candidates qualify, the one with the larger pooled score difference is taken;
   the others are reported.

If none qualifies, the submission stays as it is and all outcomes are reported as ablations.
Freeze: 2026-11-25. Official-harness re-run, video and report update follow the freeze.

## Amendments (recorded before the affected result)

- 2026-11-15, test 2: the scale-up keeps 5 restricted areas, not 8. The observation has exactly
  five obstacle tokens, so more areas would be invisible to the policy; only traffic is scaled
  (10/15/20/30 MA aircraft, 10/20/30 SA intruders, 300 scenarios per size, `atc_agent.scale_eval`).
  Test 3 adds a second classical row, MVP behind the area-free grid route (`--nav route`),
  because plain fly-direct + MVP ignores restricted areas by construction.
- 2026-11-16, tests 4/5 and control (execution only): from the first resume of ft_time on (at
  1,445M of 1,520M steps) every stage runs 13 workers x (96 SA + 8 MA slots) instead of
  ft_rev's 26 x (48 + 4). Seats (2288), rollout (2288 x 128), batches and all PPO settings are
  unchanged; only the number of simulator processes halves, because the shared PC's memory
  could no longer hold 26 (a worker holds ~1.05 GiB at either seat count, measured). The
  per-worker scenario seeds differ, which is random variation like any resume.
- 2026-11-18, control (execution only): from its resume at 1,403.7M steps, ft_cont runs
  8 workers x (156 SA + 13 MA slots): the same 2288 seats, 12:1 SA:MA slot ratio, rollout,
  batches and PPO settings. A worker holds ~1.15 GiB at 286 seats as at 176 (measured), so the
  trainer drops from ~24 to ~17 GiB and can start at 28 GiB of free commit memory; overnight the
  PC kept 29-34 GiB free, under the previous 34 GiB start condition. Decided before any
  ft_cont evaluation exists.
- 2026-11-15, test 4: the pool from seeds 3001-3020 holds 315 SA scenarios (16 timeouts + 299
  slow) and 261 MA timeout scenarios; 10 SA timeouts and 244 of 505 MA timeout scenarios have a
  goal inside a restricted area and are excluded as specified (verified: no kept scenario has a
  goal inside an area). Replay probabilities are kept. (Corrected 2026-11-16: the first version
  of this note misread the pool's kept counts as totals.)

## Results (2026-11-20; all candidates complete before the freeze)

Decision: no candidate meets rule 1, so the submission stays (`runs/ft_rev/final.zip`).
Differences are candidate minus submitted, pooled over seeds 42 + 777 (2000 SA episodes,
2000 MA scenarios), mean [95% BCa CI]; computed by `runs/diag/judge18.py` (main repo).

| Candidate | MA failures /1000 | Score | MA time [s] | SA time [s] | Rule 1 | Rule 2 |
| --- | --- | --- | --- | --- | --- | --- |
| soup4 | -1.0 [-11.0, +8.5] | +0.00 [-0.14, +0.15] | -1.6 [-3.5, +0.2] | +2.6 [-2.6, +8.2] | no | yes |
| ens4 | +0.5 [-9.0, +10.0] | +0.01 [-0.15, +0.17] | -1.3 [-3.1, +0.5] | +2.9 [-2.4, +9.1] | no | yes |
| ft_hard | -3.0 [-13.0, +6.0] | -0.07 [-0.25, +0.10] | +1.5 [-0.3, +3.4] | +10.6 [+4.3, +17.7] | no | yes |
| ft_time | +3.5 [-6.5, +14.5] | -0.02 [-0.18, +0.14] | -3.2 [-5.2, -1.2] | -3.0 [-8.7, +3.0] | no | yes |
| ft_cont | +3.0 [-7.0, +14.0] | -0.14 [-0.33, +0.04] | -0.6 [-2.5, +1.2] | +7.1 [+1.8, +13.0] | no | yes |

Not part of the decision (exploratory, not corrected for multiplicity):

- Against the equal-compute control ft_cont (`runs/diag/vs_control18.py`), the doubled time
  cost shortens flights, MA -2.6 s [-4.5, -0.7] and SA -10.1 s [-16.4, -3.8], with score
  +0.12 [+0.01, +0.25] and MA violations +0.004 [+0.000, +0.008] per aircraft. Hard replay vs
  ft_cont: MA failures -6.0 [-17.0, +4.0], MA time +2.1 s [+0.2, +4.1].
- More training alone (ft_cont) does not improve the submitted policy; SA flights get longer.
- Of the submitted policy's 136 MA aircraft failures on seeds 42 + 777 (20,000 aircraft),
  50 have a goal inside a restricted area (52 such aircraft in total), reachable only by an
  entry; no candidate changes that class. The other 86 are the addressable failures
  (`runs/diag/failure_class18.py`).
- Test 2 (`runs/scale/ft_rev.txt`, 300 scenarios per size, 5 areas): MA goal 99.3 / 98.8 /
  98.8 / 97.6 % at 10 / 15 / 20 / 30 aircraft, intrusions per aircraft 0.005 / 0.016 / 0.019 /
  0.036 against fly-direct 0.84 / 1.32 / 1.75 / 2.67; SA goal 100 / 99.3 / 99.3 % at 10 / 20 /
  30 intruders, intrusions 0.09 / 0.27 / 0.54 against 1.11 / 2.25 / 3.49.
- Test 3 (official harness, seed 42), goal / intrusions / RA events per aircraft / time / score:
  MA RL 99.39 % / 0.008 / 0.009 / 917 s / 90.0; direct + MVP 99.96 % / 0.004 / 0.804 / 859 s /
  83.3; route + MVP 99.44 % / 0.098 / 0.594 / 986 s / 82.4. SA RL 99.7 % / 0.126 / 0.012 /
  979 s / 88.5; direct + MVP 97.2 % / 0.52 / 0.84 / 1020 s / 72.7; route + MVP 95.6 % / 0.59 /
  0.61 / 1184 s / 70.5. MVP resolves pure MA conflicts at least as well as the policy but flies
  through restricted areas.

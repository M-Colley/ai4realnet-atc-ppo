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
- 2026-11-15, test 4: the pool from seeds 3001-3020 holds 305 SA scenarios (6 timeouts + 299
  slow) and 17 MA timeout scenarios; 244 of 261 MA timeout scenarios have a goal inside a
  restricted area and are excluded as specified. Replay probabilities are kept.

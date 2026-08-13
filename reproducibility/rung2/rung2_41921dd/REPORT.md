# Rung 2 reproducibility baseline

Status: **passed**. This is a diagnostic pipeline baseline, not a paper-scale
performance result.

## Immutable identities

- Training code: `41921dddeac5ef8c7bb4c9adc9b33b12516e4cea`
- Standalone evaluation code: `a15dc3a7ffb2ca1af55d7d655c70ce31b69755ee`
- Report generator code: `017f26ddc8c5a490be0501269db8e1a151082291`
- Racetrack submodule: `b95c4eff766f6367d66b310ea20cd2c9563712c0`
- Seed: 0
- Budget: 20,000 training steps per action space
- Split: Sakhir train / Austin validation / Budapest test
- Test protocol: two deterministic, arc-length-spaced spawn points

Every standalone test artifact records a clean evaluation Git state and the
SHA-256 of its selected checkpoint. The manifest checks that all four training
and evaluation revisions match their declared identities.

## Held-out Budapest results

| Action space | Mean return | Mean progress | Completion | Crash rate | Mean lateral error |
|---|---:|---:|---:|---:|---:|
| steer-speed | 47.143 | 0.1480 | 0% | 100% | 0.0964 m |
| curvature-speed | 1.992 | 0.0308 | 0% | 100% | 0.2759 m |
| lookahead-point | 45.635 | 0.1472 | 0% | 100% | 0.3177 m |
| Bézier | 40.437 | 0.1462 | 0% | 100% | 0.3998 m |

At this short budget, steer-speed has the best progress and substantially lower
lateral error. Lookahead-point and Bézier reach similar progress by different
control representations, while curvature-speed is much weaker. These are
single-seed diagnostics only; they do not support an action-space superiority
claim.

## Validity checks

- Four of four runs finished at exactly 20,000 steps.
- Four of four test files are explicitly labeled as test-split evaluations.
- Reward components sum to the recorded total within `1e-7`.
- Crash termination and crash-penalty application agree for every episode.
- All numerical metrics are finite and realized steering is nonzero.
- Raw collision impulses remain visible, but the actuator-plausibility gate
  uses nonterminal acceleration. Maximum nonterminal longitudinal acceleration
  was 8.00 m/s² across the matrix; raw collision impulses reached 406.79 m/s².
- The four generated CSV tables reproduced byte-for-byte on a second aggregate
  run.

## Recovery disclosure

The original foreground orchestrator reached its two-hour execution limit
during steer-speed's final validation. Its exact 20,000-step checkpoint was
already durable. The committed recovery command re-evaluated that checkpoint on
the original Austin and Sakhir spawn protocol without adding training steps.
Its Austin score (110.956) strictly exceeded the saved 10,000-step best score
(108.552), so the declared model-selection rule selected the 20,000-step model.
Recovery provenance is stored in the run metadata and surfaced in the manifest.

## Interpretation and next gate

Rung 2 establishes that the experiment can be reproduced, audited, recovered
after a post-checkpoint interruption, and summarized without train/test leakage.
It also shows that 20,000 steps are insufficient for workshop claims: all eight
Budapest episodes crashed and no policy completed a lap.

The next performance rung should predeclare a larger training budget, use at
least five seeds per action space, preserve the same train/validation/test split,
and report confidence intervals plus paired seed-level comparisons. No action-
space ranking should be presented as a result until completion rate becomes
nonzero and the ranking is stable across seeds.

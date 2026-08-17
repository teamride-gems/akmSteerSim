# Action-interface ensemble falsification screen

Decision: **KILL AT COMPETENCE GATE**

The five-policy screen was executed on August 17, 2026. It was designed to ask
whether disagreement among direct-steering, curvature, and lookahead policies
predicts failures of one fixed direct-action anchor better than disagreement
among three direct-action seeds. The protocol failed before that hypothesis
could be tested: three of the five fixed members were not competent.

## Fixed protocol

- Train track: Sakhir.
- Checkpoint selection: Austin and BrandsHatch only.
- Training budget: 50,000 SAC transitions per newly trained member.
- ID gate: ten deterministic, arc-length-spaced Sakhir starts.
- Minimum completion: 90%.
- No post-hoc member replacement.
- Behavior matching: lap duration and mean speed within 10% of the direct-0
  anchor, mean lateral error no more than 0.10 m above the anchor, and mean
  steering-rate ratio between 0.5 and 2.0.

## Competence result

| Member | Action interface | Terminations | Completion | Mean progress | Mean speed | Gate |
|---|---|---|---:|---:|---:|:---:|
| direct-0 | steering-speed | 10 lap complete | 100% | 1.00 | 3.91 m/s | pass |
| direct-4 | steering-speed | 10 lap complete | 100% | 1.00 | 3.75 m/s | pass |
| direct-5 | steering-speed | 10 crashes | 0% | 0.44 | 3.33 m/s | fail |
| curvature-0 | curvature-speed | 10 crashes | 0% | 0.44 | 3.47 m/s | fail |
| lookahead-0 | lookahead-point-speed | 10 timeouts | 0% | 0.92 | 3.38 m/s | fail |

Direct-5 additionally failed the lap-duration and speed bands. Curvature-0
failed completion, lap duration, and speed. Lookahead-0 failed completion,
speed, and steering-rate matching; although it reached 92% mean progress, every
episode exhausted the 120-second physical horizon.

## Valid stopping decision

OOD monitoring evaluation was not run. A heterogeneous ensemble containing two
incompetent structured-interface policies and a homogeneous ensemble containing
an incompetent direct seed would make command disagreement a proxy for policy
quality. Replacing any failed policy after observing this gate would violate the
fixed-member screen.

This result does **not** show that action-interface diversity cannot be useful
for monitoring. It shows that the present SAC training setup cannot produce the
competence-matched five-policy comparison cheaply or reliably enough to test
that claim. The correct outcome for this protocol is therefore a kill, not a
negative AUPRC result and not authorization for more ensemble runs.

## Immutable identities

- New-policy training code: `0471563efc5892e101781f42614082eba75d5568`
- Competence evaluation code: `a35964aa23d186aafa25f3a416ede8aa2d73cd4d`
- Racetrack submodule: `b95c4eff766f6367d66b310ea20cd2c9563712c0`
- Direct-0 checkpoint SHA-256: `687c51416be88bad665e73cbcd170bedf57e24afdf4ab7bfa643f7acdbb4a72c`
- Direct-4 checkpoint SHA-256: `0c77d27ec412e1bcc4ae354ae28125f1e6d7521d24cbf28f3ebba1348d089d9c`
- Direct-5 checkpoint SHA-256: `7f291f8705f5867ba5d0b5a00fb994108498ee42268be37bae6ae66b6fd7f6e9`
- Curvature-0 checkpoint SHA-256: `40b1eaa6a1d5dd8670246df0392e960373a5b6729e324e9b042c4f5d41e38e96`
- Lookahead-0 checkpoint SHA-256: `354f4f24ee38fe39c7522ad77af4eee102d853403dbf0022c0e72dbcc8af8d54`

The evaluation provenance recorded a clean worktree. The lookahead training
wall clock includes a host sleep interval and is invalid for timing comparisons;
its transition count, seed, checkpoint sequence, and final 50,000-step metadata
remain intact.

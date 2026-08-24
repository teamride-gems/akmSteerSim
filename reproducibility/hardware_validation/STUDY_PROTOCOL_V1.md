# Frozen randomized F1TENTH validation: protocol v1

## Status and purpose

This design is frozen before any physical outcome is observed. Engineering checks on stands and a low-speed synthetic pilot may verify wiring, topic mappings, logging, localization, deadman, e-stop, and command bounds, but may not use a main-study bundle or change the hypotheses, source-selection rule, outcome definitions, thresholds, exclusions, or analysis.

The study tests one narrow transport question: does the simulator's cleanest causal/downstream reversal for Innovation-Gated Steering survive the physical command, actuator, localization, and timing stack?

## Sealed sources and selection

The source corpus is the hash-sealed output-jitter command set from the independently implemented kinematic-plant study. Eligible sources must have survived source generation, have a nonterminated clean reference, receive a nonidentity timing intervention, remain within normalized steering magnitude 0.60, and have maximum packet step at most 0.45.

Within each checkpoint, sources are sorted only by suppression fraction and spawn identifier. Six evenly spaced ranks are selected. Simulated hardware-plant outcomes are forbidden selection variables. The same twelve sources are crossed with speed caps 1.0 and 1.5 m/s.

## Five blinded conditions

Every checkpoint-source-speed block contains five randomly ordered runs:

1. clean command replay A;
2. identical clean command replay B;
3. direct jittered command replay;
4. Innovation-Gated Steering;
5. the exact accepted-target/increment timing placebo.

The duplicate clean runs estimate the physical/localization repeatability floor. Gate and timing-placebo main-phase targets preserve exactly the same accepted target and increment sequence; only event timing differs. All conditions share source speed commands, capped at the block speed, and a common one-second straight ramp-in and one-second straight ramp-down. Only the 41-packet, 2.05-second main phase enters scientific path metrics.

There are 24 blocks and 120 main runs. A deterministic balanced crossover assigns opaque condition codes and balances condition position and immediate carryover as closely as the finite design permits. The operator sees codes, not semantic condition names.

## Safety and execution

The car must begin on stands. A physical e-stop and an independent safety supervisor are mandatory. Software execution is fail-closed: real commands require the exact arm phrase, fresh localization, a fresh asserted deadman, a nonasserted e-stop, a frozen schedule/bundle/site record, and an active bag/logger. Any stale safety or telemetry signal, geofence crossing, yaw-rate violation, bound violation, operator abort, or exception triggers repeated zero-speed/zero-steering stop commands.

A common final safety limiter caps speed at 1.5 m/s, steering at 0.26 rad, steering slew at 3.5 rad/s, and acceleration at 2.0 m/s². Pre-limiter study targets and sent commands are both logged. The limiter is part of the physical interface and may create post-treatment differences; it must never be bypassed to preserve a scientific contrast.

## Data and validity

Every command/telemetry record is monotonic-time stamped and SHA-256 hash chained. Each run archives its exact config, site, schedule row, condition key, command bundle, program provenance, ROS bag, completion or abort reason, start pose, command timing, and safety state.

Start pose must be within 0.04 m and 4 degrees of the frozen surveyed pose before arming. Command p95 lateness must not exceed 15 ms, maximum lateness 50 ms, telemetry gaps 100 ms, and median telemetry rate must be at least 40 Hz. Localization must be finite throughout the main phase. Technical reruns are permitted only for the four pre-motion cases listed in the configuration; the original attempt remains archived. Every post-motion abort or failure remains an outcome.

## Estimands and binding interpretation

Trajectories are expressed in the local start frame and interpolated onto the commanded main-phase time grid. The paired clean runs form their block's reference trajectory and clean-repeatability diagnostic. A post-motion failure receives the frozen 1.0 m path-error fill.

The primary specificity estimand is timing-placebo RMS error minus gate RMS error, balanced across speed-by-checkpoint cells. The primary downstream-harm estimand is gate RMS error minus direct RMS error. Both require a mean difference of at least 0.02 m, a nonnegative paired-bootstrap lower bound, and positive effects in all four speed-by-checkpoint cells. Uncertainty uses 10,000 block-paired bootstrap draws, resampling sources within cell and weighting the four cells equally.

`REPRODUCED_REVERSAL` requires valid clean repeatability and both primary conditions. Other valid results are `SPECIFICITY_ONLY`, `DOWNSTREAM_ONLY`, or `NOT_REPRODUCED`. Any failed provenance, balance, completeness, run-validity, or clean-repeatability requirement produces `INVALID`, not a scientific null.

## Scope

No outcome demonstrates better RL learning, faster laps, general action-space superiority, universal gate ordering, or robot safety. The study is a physical transport audit of one frozen diagnosis.

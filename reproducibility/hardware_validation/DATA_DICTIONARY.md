# Hardware-study data dictionary

## Immutable preparation

- `study_v1/FREEZE.json`: official pre-physical code/data hashes and amendment rule.
- `study_v1/operator_prepared/PREPARED_MANIFEST.json`: hashes for the active
  operator package and the declared-but-absent sealed key.
- `operator_prepared/operator_schedule.csv`: opaque run order visible to the operator.
- `operator_prepared/machine_schedule.json`: run-to-code mapping used by the ROS 1 runner.
- `operator_prepared/bundles/*.json`: target command streams keyed only by opaque code.
- `operator_prepared/safety_envelope.csv`: safety predictions whose condition column is replaced by opaque code.
- `study_v1/prepared/`: lead-side semantic preparation and post-lock analysis inputs; operators do not inspect this directory during collection.
- `prepared/condition_key.json`: sealed code key, intentionally absent and held by the study lead until collection is complete.

## Run archive

Each scientific run is stored as `hardware_runs/study_v1/HW###/attempt_###/`.

- `frozen_inputs/`: exact config, local site, freeze, prepared manifest, schedule row, and command bundle copied before logging/motion.
- `records.jsonl`: append-only canonical JSON records linked by `previous_record_sha256` and `record_sha256`.
- `run_manifest.json`: identity, hashes, adapter, start-pose error, completion/motion status, abort reason, packet counts, and terminal log hash.
- `validation.json`: binding per-run integrity, timing, telemetry, command, safety, bag, and eligibility checks.
- `rosbag/`: raw ROS messages for commands, odometry, safety state, and
  configured auxiliary sensors. Amendment 004 records Frank runs as rosbag1.
- `rosbag_process.log`: bag-recorder console output.
- `launch_failure.json`: preserved pre-motion adapter/logger/bag failure; only this enumerated class may be attempted again.

`motion_started` becomes true on the first positive sent speed. A failed attempt with motion is an outcome. `technical_valid` means all archival and runtime validity checks passed. `eligible_outcome` means the valid record is either complete or a preserved post-motion failure.

## Hash-chained record types

- `run_start`: adapter, opaque condition code, bundle, initial telemetry, and static hashes.
- `command`: packet/phase indices, planned and actual monotonic time, lateness, target command, sent command, and limiter flags.
- `telemetry`: receive/source time, world pose, yaw, speed, yaw rate, steering feedback/source, deadman/e-stop state/times, and optional battery voltage; command context is included after execution starts.
- `safe_stop_command`: one of twenty zero-command packets emitted on every exit.
- `run_end`: completion, motion, and abort state.

## Analysis outputs

- `result.json`: evidence class, binding verdict, invalidity reasons, clean repeatability, both primary effects/bootstrap intervals/cell means, failures, and block metrics.
- `REPORT.md`: human-readable locked summary.

Trajectories use the local frame of each run's measured start pose and the 41 planned main-phase timestamps. Clean A/B are averaged for the block reference. Candidate error is RMS Euclidean path error. Post-motion candidate failure receives the preregistered 1.0 m fill; a clean-reference failure invalidates the study. Bootstrap resampling is paired by block within the four checkpoint × speed cells and weights cells equally.

After every outcome is locked, the study lead joins runs to the lead-side
semantic schedule by `run_id` and uses `scripts/analyze_hardware_study_ros1.py`.
Archives containing only `ros1_ackermann_noetic` outcomes are classified as
`PHYSICAL_HARDWARE`; mixed-adapter evidence is invalid.

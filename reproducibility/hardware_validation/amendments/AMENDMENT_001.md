# Hardware-study amendment 001: native ROS 1 Noetic interface

**Status:** disclosed before physical outcomes; required before any motion on Frank  
**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`  
**Reason:** the post-freeze RiDE Drive audit established that Frank runs Ubuntu 20.04 and ROS 1 Noetic, while the frozen integration layer supports ROS 2 only.

## Scope

This amendment replaces no frozen scientific input. It adds a native ROS 1 transport, safety bridge, rosbag1 recording, site-capture wrapper, configuration capture, and ROS 1 operator instructions. The original `FREEZE.json` remains byte-identical and is verified before every amended preflight, pilot, or study run.

The distributable ROS 1 operator package omits
`study_v1/prepared/condition_key.json` while preserving its entry in the
byte-identical prepared manifest. Amended operator verification checks every
other prepared input and fails if the sealed key is present. The study lead
retains the hash-matching key separately and supplies it only after all 120
outcomes are locked for the frozen analysis. This packaging rule changes no
schedule, condition, command, threshold, or analysis decision.

The following remain unchanged:

- all 24 command bundles and the 120-row blinded schedule;
- source selection, conditions, packet timing, steering targets, and speed targets;
- common command limiter and runtime safety thresholds;
- validity rules, retry rules, failure handling, and locked analysis;
- the distinction between engineering pilots and scientific outcomes.

## Transport equivalence

The native adapter publishes `ackermann_msgs/AckermannDriveStamped` at the frozen 20 Hz packet rate and consumes `nav_msgs/Odometry`, `std_msgs/Bool` deadman state, and `std_msgs/Bool` latched e-stop state. Optional joint-state and battery topics retain the same telemetry fields as the frozen ROS 2 adapter. Timestamps are converted to seconds without changing the execution clock, command limiter, logging schema, or validation.

ROS 1 data are recorded with `rosbag record` into each immutable attempt directory. The required topics are identical in role to the frozen ROS 2 recording set, with `/scan` and the raw VESC command topics included when configured.

## Added fail-closed gates

1. The site record must declare ROS 1 Noetic.
2. The autonomous Ackermann mux input and fixed-world odometry topic must be confirmed with `rostopic type` and `rostopic info`.
3. The measured wheelbase must replace the previous 0.33 m assumption.
4. The current `vesc.yaml` must contain exactly one `max_acceleration`, match the site record, and be no greater than **1.5 m/s²**. This conservative physical-controller ceiling comes from the latest RiDE drivetrain notes; it does not alter the frozen target sequences.
5. The repaired steering servo, rear motor, gearbox, and slipper clutch must pass the mechanical checklist.
6. Onboard localization and independent evaluation ground truth must be named separately. If independent ground truth is unavailable, that limitation must be recorded before data collection and the corresponding paper claim narrowed.
7. rosbag1 must start and create an output file before the adapter can publish a motion command.

## Integrity and outcome policy

`AMENDMENT_001.json` hashes every amendment-controlled runtime and instruction file. Each run archives that record and all referenced files. A hash mismatch prevents launch. The original freeze continues to govern the scientific design.

No pilot result may be used in the scientific analysis. Once any scheduled attempt moves, it remains an outcome under the original retry policy. Adapter, topic, unit, sign, localization, or logging faults found after motion may not be repaired and silently rerun.


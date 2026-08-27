# RiDE F1TENTH hardware-study operator runbook

> **Frank uses ROS 1 Noetic. Do not use the ROS 2 commands below on Frank.**
> Use `ROS1_OPERATOR_RUNBOOK.md` under pre-outcome `AMENDMENT_006`. This file is
> retained as the original ROS 2 integration record.

## Stop conditions

Do not put the car on the ground unless every item below is true:

- the vehicle uses ROS 2 and accepts `ackermann_msgs/AckermannDriveStamped`;
- localized odometry is at least 40 Hz and is expressed in one fixed world frame;
- a physical motor e-stop has been tested on stands;
- a second person is the dedicated safety supervisor and holds the software deadman/e-stop controller;
- the measured obstacle-free radius is at least **6.0 m** from the marked start pose;
- the stands pilot and ground pilot both end with `technical_valid: true`;
- the exact official `FREEZE.json` verifies without amendment.

This repository's supplied hardware adapter is ROS 2. If the RiDE car is ROS 1, uses an unstamped drive message, lacks world-frame odometry, or remaps steering/speed semantics, stop and implement/test an explicit adapter before the visit. Do not improvise topic conversions during the study.

## Roles and area

The operator runs commands and resets the car to the taped pose. The safety supervisor does nothing else: they maintain line of sight, hold the deadman, and release it or assert the software e-stop at the first concern. The physical e-stop remains immediately reachable as an independent backup. No spectators enter the 12 m diameter clear area.

The frozen slow-actuator screen predicts at most 4.520 m radius. Runtime commands stop at a 5.0 m geofence, and the course must preserve at least 1.0 m beyond that boundary for braking and human reaction.

## Robot software prerequisites

Use Python 3.10 in a ROS 2 environment with these packages available:

```text
rclpy
ackermann_msgs
nav_msgs
sensor_msgs
std_msgs
std_srvs
rosbag2
rosbag2_storage_mcap
```

The non-ROS Python requirements are `numpy` and `pyyaml`. Source the robot's ROS installation and workspace before every command window. Confirm `ros2 topic list`, `ros2 topic hz <odom-topic>`, and `ros2 topic info <drive-topic>` manually.

## 1. Configure topics without changing frozen files

Copy `configs/hardware_site_template.yaml` to an ignored file named `local_hardware_site_draft.yaml`. Edit only the draft with the real drive, odometry, joystick, safety, joint-state, and bag topic names; the joystick button indices; frame names; ROS domain; wheelbase; and bag storage settings. Never edit the tracked template or official freeze.

Start the normal vehicle, localization, joystick, and Ackermann/VESC nodes. In a separate terminal, start the fail-closed software safety bridge using the draft:

```powershell
python scripts/ros2_hardware_safety_bridge.py --site local_hardware_site_draft.yaml
```

It starts with the software e-stop latched. With both configured buttons released and fresh joystick messages arriving, reset it:

```powershell
ros2 service call /hardware_study/reset_software_estop std_srvs/srv/Trigger "{}"
```

Test that joystick silence and deadman release both publish `false` on the deadman topic, and that the software e-stop button latches `true` until the service is deliberately called again.

## 2. Capture the marked start pose

Keep the car stationary at the taped start pose. Capture at least three seconds of localization and write the final ignored site record:

```powershell
python scripts/capture_hardware_site.py `
  --template local_hardware_site_draft.yaml `
  --output local_hardware_site.yaml `
  --site-id UMD_SITE_ID `
  --robot-id RIDE_CAR_ID `
  --course-id COURSE_ID `
  --clear-radius-m MEASURED_RADIUS `
  --localization-system SYSTEM_NAME
```

The capture refuses overwrite, motion over 0.03 m/s, fewer than 40 samples, or localization dispersion over 0.02 m. Restart the bridge with `local_hardware_site.yaml`, reset its latch, and keep this site file unchanged for the entire study.

## 3. Live preflight on stands

With wheels securely off the ground, test the physical e-stop using the robot's normal low-speed commissioning/teleoperation procedure. Confirm it cuts propulsion independently of this study software. Clear the full ground course and complete the printed checklist.

Hold the software deadman and run:

```powershell
python scripts/hardware_preflight.py `
  --adapter ros2 `
  --site local_hardware_site.yaml `
  --output hardware_runs/preflight/robot_preflight.json `
  --wheels-on-stands-verified `
  --physical-estop-tested `
  --course-cleared `
  --localization-checked `
  --zero-command-test "ZERO COMMAND TEST ON STANDS"
```

This verifies every frozen hash, the 6 m course margin, start pose, stationary state, deadman/e-stop freshness, measured speed, localization rate/gaps, and a two-second zero-command path. A passed record is valid for at most 12 hours and only while the site file is byte-identical.

## 4. Engineering qualification only

First run the straight 0.20 m/s sequence while still on stands:

```powershell
python scripts/run_hardware_engineering_pilot.py `
  --mode stands --adapter ros2 `
  --site local_hardware_site.yaml `
  --preflight hardware_runs/preflight/robot_preflight.json
```

Require `completed: true` and `technical_valid: true`. Check steering direction, speed sign, odometry sign/frame, stop behavior, deadman release, bag contents, and command timestamps. Do not change frozen main-study code to make the pilot look better; correct only a proven integration defect through a disclosed pre-outcome amendment.

Place the car on the taped ground pose, clear the area, hold the deadman, and run the unrelated 0.50 m/s, ±0.05 rad S-turn:

```powershell
python scripts/run_hardware_engineering_pilot.py `
  --mode ground --adapter ros2 `
  --site local_hardware_site.yaml `
  --preflight hardware_runs/preflight/robot_preflight.json
```

Again require both success fields. Pilot data are permanently marked engineering-only and cannot enter the paper analysis.

## 5. Main blinded study

Keep `prepared/condition_key.json` away from the operator until collection is complete. Use only `prepared/operator_schedule.csv`. The runner independently enforces the next row, refuses overwrite, and permits a repeat only after an enumerated pre-motion technical failure.

For the first row the command form is:

```powershell
python scripts/run_hardware_study.py `
  --run-id HW001 --adapter ros2 `
  --site local_hardware_site.yaml `
  --preflight hardware_runs/preflight/robot_preflight.json
```

Replace only the run ID and opaque code using the next operator-schedule row. Before every command:

1. return the car to the taped pose and heading;
2. announce run ID and code to the supervisor;
3. verify the area is empty and physical e-stop reachable;
4. reset the software e-stop only if its cause has been understood;
5. hold the deadman, execute exactly once, then release it after the stop packet train.

`eligible_outcome: true` is the gate for moving to the next row. A completed run and a valid post-motion abort are both outcomes. Never repeat a run that moved. Re-running the same command is accepted only for a frozen pre-motion technical reason and preserves the first attempt.

After every five-run block, copy the entire `hardware_runs/study_v1` directory to two independent storage locations. Do not rename, edit, or hand-repair JSONL, manifests, validations, or bags.

## 6. Locked analysis

Only after all 120 rows have an eligible outcome, unblind and run:

```powershell
python scripts/analyze_hardware_study.py `
  --runs hardware_runs/study_v1 `
  --output hardware_runs/study_v1_analysis
```

The only valid scientific labels are `REPRODUCED_REVERSAL`, `SPECIFICITY_ONLY`, `DOWNSTREAM_ONLY`, and `NOT_REPRODUCED`. `INVALID` means the physical result cannot support a paper claim. Mock output is always labeled `ENGINEERING_MOCK_ONLY`.

## Emergency behavior

Release the software deadman first and use the physical e-stop whenever motion is unsafe. Do not wait for the program. The runner also stops on stale telemetry/safety messages, software e-stop, measured speed over 1.8 m/s, yaw rate over 3.5 rad/s, a 5.0 m radius crossing, timing failure, exception, or normal exit, and emits twenty zero-speed/zero-steering packets.

After any emergency, photograph the vehicle and course, preserve all files, record what happened in the session checklist, and determine whether motion had started. Do not decide rerun eligibility from memory; let the archived manifest and runner enforce it.

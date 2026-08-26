# Frank ROS 1 Noetic hardware-study runbook

This runbook implements `AMENDMENT_005`. The original frozen protocol and operator rules still apply. Use these commands on Frank instead of the ROS 2 commands in `OPERATOR_RUNBOOK.md`.

## 1. Prepare a local site record

Copy `configs/hardware_site_ros1_template.yaml` to the ignored file `local_hardware_site_ros1_draft.yaml`. The command topic, Cartographer TF chain, wheel-odometry topic, joystick topic, autonomous-heartbeat topic, and stop index are prefilled from the 2026-08-25 captures and Amendment 005. Resolve the remaining placeholders using live inspection or measurement. The active VESC configuration must be set to and captured at the selected 2.0 m/s² acceleration ceiling.

Confirm the command interfaces before starting the bridge:

```bash
rostopic type /vesc/high_level/ackermann_cmd_mux/input/nav_0
rostopic info /vesc/high_level/ackermann_cmd_mux/input/nav_0
rostopic type /vesc/odom
rostopic hz /vesc/odom
rostopic type /tf
rostopic hz /tf
rostopic type /vesc/joy
```

The required types are `ackermann_msgs/AckermannDriveStamped`, `nav_msgs/Odometry`, `tf2_msgs/TFMessage`, and `sensor_msgs/Joy`. The adapter composes `cartographer_map -> cartographer_odom -> base_link` for fixed-world pose and uses `/vesc/odom` for speed and yaw rate.

Capture the live interface and controller files:

```bash
python3 scripts/capture_ros1_configuration.py \
  --site local_hardware_site_ros1_draft.yaml \
  --vesc-config ~/racecar_ws/src/f1tenth_system/racecar/racecar/config/racecar-v2/vesc.yaml \
  --joy-config ~/racecar_ws/src/f1tenth_system/racecar/racecar/config/racecar-v2/joy_teleop.yaml \
  --launch-file ~/racecar_ws/src/f1tenth_system/racecar/racecar/launch/teleop.launch \
  --output hardware_runs/configuration_capture/session_001
```

Add every other controller, mux, or Cartographer file actually loaded for the
session with a repeatable `--loaded-file PATH` argument. The capture refuses an
unknown topic type, a missing file, multiple/missing VESC acceleration values,
or any disagreement among `vesc.yaml`, the site record, and the active `/vesc`
ROS parameters. All three must report exactly 2.0 m/s².

## 2. Start and test the ROS 1 safety bridge

```bash
python3 scripts/ros1_hardware_safety_bridge.py \
  --site local_hardware_site_ros1_draft.yaml
```

The software e-stop starts latched. The bridge publishes a zero command to
Frank's low-level safety mux unless the joystick connection and the active
runner's heartbeat are both fresh. With the joystick connected and every
button released, reset the latch:

```bash
rosservice call /hardware_study/reset_software_estop
```

Keep these four terminals visible during the stands-only bridge check:

```bash
rostopic echo /hardware_study/deadman
rostopic echo /hardware_study/estop
rostopic echo /hardware_study/run_active
rostopic echo /vesc/low_level/ackermann_cmd_mux/active
```

With Frank securely on stands, run the bounded heartbeat test from a fifth
terminal:

```bash
python3 scripts/test_ros1_safety_heartbeat.py \
  --site local_hardware_site_ros1_draft.yaml \
  --stands-confirm "FRANK ON STANDS - TEST HEARTBEAT"
```

Verify that the low-level zero-command override is selected before the test;
the runner heartbeat becomes true, the override releases, and the authorization
topic becomes true after the one-second mux-clearance interval; then the test
ends automatically, the heartbeat becomes false, and the zero-command override
returns on the next bridge cycle. Repeat the bounded test and press joystick button
index 6 while it is active. The e-stop and zero-command override must remain
latched after the test finishes, until every button is released and the reset
service is called again. Also verify that disconnecting the joystick restores
the override no later than 0.30 seconds after the last joystick message. Button
index 5 is not held and has no safety
authorization role under Amendment 005.

The bounded test must finish and print that authorization was withdrawn before
starting preflight or a pilot. The bridge rejects multiple heartbeat publishers,
so a leftover test process cannot authorize a real run.

## 3. Capture the stationary start pose

After completing the mechanical checklist, place the car on the taped start pose and run:

```bash
python3 scripts/capture_hardware_site_ros1.py \
  --template local_hardware_site_ros1_draft.yaml \
  --output local_hardware_site_ros1.yaml \
  --site-id UMD_SITE_ID \
  --robot-id frank \
  --course-id COURSE_ID \
  --operator OPERATOR_NAME \
  --safety-supervisor SUPERVISOR_NAME \
  --clear-radius-m MEASURED_RADIUS \
  --localization-system cartographer_tf
```

This reuses the frozen stationary-capture requirements: at least 40 samples, speed no greater than 0.03 m/s, and no more than 0.02 m positional dispersion.

## 4. Signed live preflight

Restart the bridge with the final site file, reset its latch, put the car securely on stands, and run. The runner supplies its own heartbeat; nobody holds a button. The safety supervisor keeps the connected joystick in hand with button 6 immediately reachable:

```bash
python3 scripts/ros1_hardware_preflight.py \
  --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --output hardware_runs/preflight/robot_preflight_ros1.json \
  --operator-sign "OPERATOR NAME" \
  --supervisor-sign "SUPERVISOR NAME" \
  --wheels-on-stands-verified \
  --physical-estop-tested \
  --course-cleared \
  --localization-checked \
  --zero-command-test "ZERO COMMAND TEST ON STANDS"
```

A passing record is valid for at most 12 hours and only while the site file is byte-identical.

## 5. Engineering qualification

Run the 0.20 m/s stands pilot first. The runner supplies its own heartbeat, and
the safety supervisor presses button 6 at the first reason to stop:

```bash
python3 scripts/run_hardware_engineering_pilot_ros1.py \
  --mode stands --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --preflight hardware_runs/preflight/robot_preflight_ros1.json \
  --arm "AREA CLEAR - ESTOP READY" \
  --operator-confirmation "RUN ENGINEERING PILOT ON STANDS"
```

Inspect the bag and confirm steering/speed signs, mux ownership, stop behavior,
timestamps, localization, autonomous authorization, and e-stop behavior. Only after it passes,
run the 0.50 m/s ground pilot with the same press-button-6-to-stop rule:

```bash
python3 scripts/run_hardware_engineering_pilot_ros1.py \
  --mode ground --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --preflight hardware_runs/preflight/robot_preflight_ros1.json \
  --arm "AREA CLEAR - ESTOP READY" \
  --operator-confirmation "RUN ENGINEERING PILOT ON GROUND"
```

Both archives are permanently engineering-only.

## 6. Main blinded study

Begin only after both pilots pass. This package prospectively records that the current study uses onboard Cartographer localization without independent trajectory ground truth; paper claims must not describe Cartographer trajectories as independently validated ground truth.

Use only
`study_v1/operator_prepared/operator_schedule.csv` during collection. The ROS 1
runners default to the opaque operator package, so the schedule, packet bundle,
validation file, and run manifest expose only the run's code. Do not inspect
the lead-side `study_v1/prepared` directory or compare bundles before all 120
outcomes are locked. The unblinding key is deliberately absent, and the tools
fail closed if it is added to the operator package.

```bash
python3 scripts/run_hardware_study_ros1.py \
  --run-id HW001 --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --preflight hardware_runs/preflight/robot_preflight_ros1.json \
  --arm "AREA CLEAR - ESTOP READY" \
  --operator-confirmation "RUN HW001 CODE C"
```

Use the next opaque code from the operator schedule. The amended runner still enforces schedule order, refuses overwrite, records rosbag1 before motion, archives the original freeze and amendment, and applies the original post-motion no-rerun rule.

For every main run, nobody holds an authorization button. The safety supervisor
keeps the connected joystick and physical motor e-stop immediately reachable
and presses button 6 at the first concern. Do not touch other joystick controls
during autonomous execution. Runner exit, heartbeat loss, joystick loss, or an
e-stop press automatically restores the zero-command override. A stopped
attempt that already moved remains an outcome under the frozen rerun rule.
After pressing button 6, do not call the reset service until the runner has
exited and `/hardware_study/run_active` is false.

## 7. Outcome lock and analysis

After collection, return the complete `hardware_runs/study_v1/` tree to the
study lead and lock its hashes before unblinding. The study lead—not the robot
operator—then runs `scripts/analyze_hardware_study_ros1.py`. That wrapper
preserves the frozen analysis while correctly classifying archives from the
native Noetic adapter as physical hardware evidence. Do not run or inspect the
semantic analysis during data collection.

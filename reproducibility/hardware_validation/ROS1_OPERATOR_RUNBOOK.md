# Frank ROS 1 Noetic hardware-study runbook

This runbook implements `AMENDMENT_001`. The original frozen protocol and operator rules still apply. Use these commands on Frank instead of the ROS 2 commands in `OPERATOR_RUNBOOK.md`.

## 1. Prepare a local site record

Copy `configs/hardware_site_ros1_template.yaml` to the ignored file `local_hardware_site_ros1_draft.yaml`. Resolve every placeholder using measured values and live ROS inspection. In particular, do not guess the autonomous mux input, odometry topic, wheelbase, joystick buttons, or VESC acceleration setting.

Confirm the command interfaces before starting the bridge:

```bash
rostopic type AUTONOMOUS_ACKERMANN_TOPIC
rostopic info AUTONOMOUS_ACKERMANN_TOPIC
rostopic type FIXED_WORLD_ODOMETRY_TOPIC
rostopic hz FIXED_WORLD_ODOMETRY_TOPIC
```

The required types are `ackermann_msgs/AckermannDriveStamped` and `nav_msgs/Odometry`.

Capture the live interface and controller files:

```bash
python3 scripts/capture_ros1_configuration.py \
  --site local_hardware_site_ros1_draft.yaml \
  --vesc-config ~/racecar_ws/src/f1tenth_system/racecar/racecar/config/racecar-v2/vesc.yaml \
  --joy-config ~/racecar_ws/src/f1tenth_system/racecar/racecar/config/racecar-v2/joy_teleop.yaml \
  --launch-file ~/racecar_ws/src/f1tenth_system/racecar/racecar/launch/teleop.launch \
  --output hardware_runs/configuration_capture/session_001
```

The capture refuses an unknown topic type, missing controller file, multiple/missing VESC acceleration values, a value above 1.5 m/s², or disagreement between `vesc.yaml` and the site record.

## 2. Start and test the ROS 1 safety bridge

```bash
python3 scripts/ros1_hardware_safety_bridge.py \
  --site local_hardware_site_ros1_draft.yaml
```

The software e-stop starts latched. With a fresh joystick and both configured buttons released:

```bash
rosservice call /hardware_study/reset_software_estop
```

On stands, verify all four behaviors: deadman press publishes true, release publishes false, joystick disconnection publishes false within the configured stale interval, and the e-stop remains latched until a deliberate reset. Independently test the physical motor e-stop.

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
  --localization-system SYSTEM_NAME
```

This reuses the frozen stationary-capture requirements: at least 40 samples, speed no greater than 0.03 m/s, and no more than 0.02 m positional dispersion.

## 4. Signed live preflight

Restart the bridge with the final site file, reset its latch, put the car securely on stands, and run:

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

Run the 0.20 m/s stands pilot first:

```bash
python3 scripts/run_hardware_engineering_pilot_ros1.py \
  --mode stands --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --preflight hardware_runs/preflight/robot_preflight_ros1.json \
  --arm "AREA CLEAR - ESTOP READY" \
  --operator-confirmation "RUN ENGINEERING PILOT ON STANDS"
```

Inspect the bag and confirm steering/speed signs, mux ownership, stop behavior, timestamps, localization, deadman, and e-stop behavior. Only after it passes, run the 0.50 m/s ground pilot:

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

Begin only after both pilots pass and the independent evaluation system has been synchronized or its absence prospectively documented.

The unblinding file `study_v1/prepared/condition_key.json` is deliberately not
distributed with this branch. The study lead retains it separately until all
120 outcomes are locked. Do not copy it onto Frank: the amended preflight,
pilot, and study runners require it to be absent.

```bash
python3 scripts/run_hardware_study_ros1.py \
  --run-id HW001 --adapter ros1 \
  --site local_hardware_site_ros1.yaml \
  --preflight hardware_runs/preflight/robot_preflight_ros1.json \
  --arm "AREA CLEAR - ESTOP READY" \
  --operator-confirmation "RUN HW001 CODE C"
```

Use the next opaque code from the operator schedule. The amended runner still enforces schedule order, refuses overwrite, records rosbag1 before motion, archives the original freeze and amendment, and applies the original post-motion no-rerun rule.


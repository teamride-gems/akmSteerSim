# Mechanical-team handoff for Frank experiments

Scientific collection is paused. This is the entry point for Frank interface
qualification and engineering pilots only; it does not authorize a scientific
run. The ROS 1 runbook and signed acceptance checklist remain binding for the
qualification work they cover.

Start with the shorter repository entry page, `FRANK_START_HERE.md`, then use
this handoff for prerequisites and the binding runbook for commands.

## Get the operator package

For a new checkout on Frank:

```bash
git clone --branch ride/project-leadership-handoff-20260901 --single-branch \
  https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim
git submodule update --init --recursive
```

For an existing checkout, preserve any local robot configuration first, then
fetch and switch to `ride/project-leadership-handoff-20260901`. Keep local site
records and collected data outside Git; both are ignored by this branch.

## Files to use

The historical `study_v1/operator_prepared` schedule remains in the repository
for provenance. Do not use it during current qualification work and do not
start any of its 120 scheduled runs. The scientific runner is disabled.

- `configs/hardware_site_ros1_template.yaml` — copy to the ignored
  `local_hardware_site_ros1_draft.yaml` and fill only measured or live-verified
  values.
- `reproducibility/hardware_validation/MECHANICAL_ACCEPTANCE_CHECKLIST.md` —
  print and complete before ground motion.
- `reproducibility/hardware_validation/ROS1_OPERATOR_RUNBOOK.md` — exact setup,
  preflight, pilot, and study commands.
- `reproducibility/hardware_validation/STUDY_PROTOCOL_V1.md` — historical
  frozen protocol; retained for provenance, not current authorization.
- `reproducibility/hardware_validation/amendments/AMENDMENT_006.md` — current
  authorization status; the ROS 1 safety interface remains documented in
  `AMENDMENT_005.md`.

## Robot-side prerequisites

Use the AGX Xavier with Ubuntu 20.04, ROS Noetic, Python 3, the existing
`racecar_ws`, and the robot's normal VESC/Ackermann drivers. In every terminal:

```bash
source /opt/ros/noetic/setup.bash
source ~/racecar_ws/devel/setup.bash
```

Confirm the required ROS packages before proceeding:

```bash
python3 -c "import rospy, yaml, numpy; from ackermann_msgs.msg import AckermannDriveStamped; from nav_msgs.msg import Odometry; from sensor_msgs.msg import Joy; from std_msgs.msg import Bool; from std_srvs.srv import Trigger; from tf2_msgs.msg import TFMessage"
which rosbag
rospack find racecar
```

If an import is missing, install the matching Noetic package through the team's
normal ROS dependency process. Do not upgrade Frank's operating system, ROS
distribution, workspace, VESC firmware, or controller package during a
qualification session.

## Remaining qualification inputs

The 2026-08-25 stationary capture resolved the command, mux, Cartographer,
wheel-odometry, joystick, LiDAR, IMU, and VESC topic interfaces. Before the
pilot, resolve only the remaining fail-closed fields in the site template:

1. Set and capture `max_acceleration: 2.0` in the `vesc.yaml` Frank actually
   loads. The previously copied 2.5 m/s² value will fail configuration capture.
2. Record the current axle-center-to-axle-center wheelbase measurement. The
   approximate 0.250 m configuration value remains provisional until then.
3. Record the surveyed course radius and taped start pose.

## Required order

1. Complete the required wheelbase/course measurements with propulsion disabled.
2. Fill the draft site file and capture the live ROS/configuration record using
   `scripts/capture_ros1_configuration.py`.
3. Start `scripts/ros1_hardware_safety_bridge.py`; verify the low-level
   zero-command override, runner-heartbeat timeout, joystick timeout, and
   latched software e-stop using `scripts/test_ros1_safety_heartbeat.py`.
4. Capture the taped start pose with `scripts/capture_hardware_site_ros1.py`.
5. Run the signed `scripts/ros1_hardware_preflight.py` on stands.
6. Pass the 0.20 m/s stands pilot.
7. Pass the 0.50 m/s ground pilot in the surveyed clear area.
8. Preserve and review both engineering-only bags. Do not authorize `HW001`;
   the historical scientific collection is paused pending a new prospective
   protocol.

No button is held during autonomous execution. Each approved runner supplies
its own short-timeout heartbeat. The safety supervisor keeps the connected
joystick in hand and presses index 6 to latch the software e-stop. Runner exit,
heartbeat loss, joystick loss, or button 6 restores the zero-command override.
Do not touch other joystick controls while the autonomous runner is active, and
do not reset an e-stop until the runner has exited and its heartbeat is false.

## Immediate stop conditions

Stop for grinding, clutch slip, intermittent propulsion, steering binding,
unexpected mux ownership, reversed signs, stale localization, missing bag
topics, battery brownout, deadman/e-stop failure, any person entering the clear
area, or any unresolved checklist item. Engineering troubleshooting may resume
after the cause is documented. No scientific scheduled runs are currently
authorized.

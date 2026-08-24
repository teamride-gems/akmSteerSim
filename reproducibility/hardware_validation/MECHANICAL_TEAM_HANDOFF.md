# Mechanical-team handoff for Frank experiments

This is the entry point for preparing Frank. Do not begin a scientific run from
this page alone: the ROS 1 runbook and signed acceptance checklist remain
binding.

## Get the operator package

For a new checkout on Frank:

```bash
git clone --branch ride/frank-experiment-package-20260821 --single-branch \
  https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim
git submodule update --init --recursive
```

For an existing checkout, preserve any local robot configuration first, then
fetch and switch to `ride/frank-experiment-package-20260821`. Keep local site
records and collected data outside Git; both are ignored by this branch.

## Files to use

This operator package intentionally excludes
`study_v1/prepared/condition_key.json`. The study lead retains that sealed file
until all 120 scheduled outcomes are locked. Do not request, recreate, or add
the key to Frank's checkout; the ROS 1 tools fail closed if it is present.

- `configs/hardware_site_ros1_template.yaml` — copy to the ignored
  `local_hardware_site_ros1_draft.yaml` and fill only measured or live-verified
  values.
- `reproducibility/hardware_validation/MECHANICAL_ACCEPTANCE_CHECKLIST.md` —
  print and complete before ground motion.
- `reproducibility/hardware_validation/ROS1_OPERATOR_RUNBOOK.md` — exact setup,
  preflight, pilot, and study commands.
- `reproducibility/hardware_validation/STUDY_PROTOCOL_V1.md` — frozen scientific
  and rerun rules.
- `reproducibility/hardware_validation/amendments/AMENDMENT_001.md` — ROS 1
  compatibility and physical-controller gates.

## Robot-side prerequisites

Use the AGX Xavier with Ubuntu 20.04, ROS Noetic, Python 3, the existing
`racecar_ws`, and the robot's normal VESC/Ackermann drivers. In every terminal:

```bash
source /opt/ros/noetic/setup.bash
source ~/racecar_ws/devel/setup.bash
```

Confirm the required ROS packages before proceeding:

```bash
python3 -c "import rospy, yaml, numpy; from ackermann_msgs.msg import AckermannDriveStamped; from nav_msgs.msg import Odometry; from sensor_msgs.msg import Joy; from std_msgs.msg import Bool; from std_srvs.srv import Trigger"
which rosbag
rospack find racecar
```

If an import is missing, install the matching Noetic package through the team's
normal ROS dependency process. Do not upgrade Frank's operating system, ROS
distribution, workspace, VESC firmware, or controller package during study
collection.

## Information the mechanical team must supply

1. Current `vesc.yaml`, `joy_teleop.yaml`, and every launch file used.
2. The autonomous `AckermannDriveStamped` mux input and its priority relative to
   teleoperation.
3. A fixed-world `nav_msgs/Odometry` topic running at least 40 Hz.
4. Measured wheelbase, mass, steering limits, turn radii, steering latency,
   speed latency, and stopping distances.
5. Written confirmation that the servo, rear motor, gearbox, slipper clutch,
   batteries, physical motor e-stop, and teleoperation override pass the
   acceptance checklist.
6. The onboard localization system and the independent evaluation/ground-truth
   system, including update rate and accuracy.

## Required order

1. Complete mechanical inspection and measurements with propulsion disabled.
2. Fill the draft site file and capture the live ROS/configuration record using
   `scripts/capture_ros1_configuration.py`.
3. Start `scripts/ros1_hardware_safety_bridge.py`; verify deadman timeout,
   latched software e-stop, and independent physical e-stop on stands.
4. Capture the taped start pose with `scripts/capture_hardware_site_ros1.py`.
5. Run the signed `scripts/ros1_hardware_preflight.py` on stands.
6. Pass the 0.20 m/s stands pilot.
7. Pass the 0.50 m/s ground pilot in the surveyed clear area.
8. Preserve and review both engineering-only bags before authorizing `HW001`.

## Immediate stop conditions

Stop for grinding, clutch slip, intermittent propulsion, steering binding,
unexpected mux ownership, reversed signs, stale localization, missing bag
topics, battery brownout, deadman/e-stop failure, any person entering the clear
area, or any unresolved checklist item. Engineering troubleshooting may resume
after the cause is documented; a scheduled run that moved remains an outcome
under the frozen no-silent-rerun policy.

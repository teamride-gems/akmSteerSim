# Frank live interface confirmation: 2026-08-25

This pre-motion record supplements the stationary bag inspection. It contains
live interface output and mechanical-lead configuration decisions; it is not a
pilot or scientific outcome.

## Confirmed decisions

- `/vesc/joy` button index 6 is unused and reserved for the experiment's
  software stop input.
- The experiment controller acceleration ceiling is 2.0 m/s². Frank's active
  `vesc.yaml` must be set to and captured at 2.0 m/s² before preflight; the
  previously copied 2.5 m/s² value does not pass configuration capture.
- The reported minimum turn radius is approximately 3.3 ft (1.01 m) in one
  direction and slightly smaller in the other. These approximate values are
  planning information, not a replacement for the protocol's measured-radius
  calibration.

## Relevant live topics

The live ROS master reported the following experiment-relevant interfaces:

- Cartographer and sensing: `/cartographer_map`, `/constraint_list`,
  `/diagnostics`, `/imu`, `/landmark_poses_list`, `/laser_status`, `/scan`,
  `/scan_matched_points2`, `/submap_list`, `/tf`, `/tf_static`, and
  `/trajectory_node_list`.
- VESC sensing: `/vesc/odom`, `/vesc/sensors/core`, and
  `/vesc/sensors/servo_position_command`.
- VESC commands: `/vesc/commands/motor/brake`,
  `/vesc/commands/motor/current`, `/vesc/commands/motor/duty_cycle`,
  `/vesc/commands/motor/position`, `/vesc/commands/motor/speed`,
  `/vesc/commands/motor/unsmoothed_speed`, `/vesc/commands/servo/position`, and
  `/vesc/commands/servo/unsmoothed_position`.
- High-level mux: `/vesc/high_level/ackermann_cmd_mux/active`, input topics
  `default` and `nav_0` through `nav_3`, and `/output`.
- Low-level mux: `/vesc/low_level/ackermann_cmd_mux/active`, inputs
  `/navigation`, `/safety`, and `/teleop`, and `/output`.
- Joystick: `/vesc/joy` and `/vesc/joy/set_feedback`.

`/dev/null`, parameter-description/update topics, bond topics, and legacy mux
aliases are intentionally excluded from the required experiment bag because
they do not contribute commands, outcomes, safety state, or diagnostic evidence.

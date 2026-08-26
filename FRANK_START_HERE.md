# Frank experiment start page

Use branch `ride/frank-experiment-package-20260821`. This branch is the
published operator package for the first Frank experiment round. The binding
commands are in
`reproducibility/hardware_validation/ROS1_OPERATOR_RUNBOOK.md`; do not run the
main schedule from this page alone.

## What is already resolved

The package is configured for Frank's ROS 1 Noetic interfaces, Cartographer TF
localization, wheel odometry, `/vesc/joy`, the high-level navigation mux input,
and the low-level zero-command safety override. The scientific schedule has 120
fixed runs. The operator-facing schedule, command bundles, and run logs use
only opaque condition codes.

## What must be measured on site

Before motion, copy `configs/hardware_site_ros1_template.yaml` to
`local_hardware_site_ros1_draft.yaml`, then supply:

- the axle-center-to-axle-center wheelbase;
- the surveyed clear course radius;
- the taped start pose captured by the provided stationary tool; and
- an active VESC `max_acceleration` of exactly 2.0 m/s².

The configuration capture checks both the supplied `vesc.yaml` and the active
`/vesc` ROS parameters. A file that says 2.0 while the running controller uses
another value is rejected.

## Operator rule that matters throughout motion testing

The safety supervisor must hold `/vesc/joy` button index 5, the right bumper,
by itself throughout every preflight motion check, pilot, and scientific run.
Pressing any other joystick button at the same time, releasing the bumper, or
losing joystick messages withdraws authorization and restores the zero-command
override. Button index 6 latches the software e-stop.

## Required order

1. Complete the site draft and controller/interface capture.
2. Start and verify the safety bridge on stands.
3. Capture the taped start pose into `local_hardware_site_ros1.yaml`.
4. Complete the signed live preflight.
5. Pass and inspect the 0.20 m/s stands pilot.
6. Pass and inspect the 0.50 m/s ground pilot.
7. Only then start `HW001` and continue in the printed operator schedule order.

Use only
`reproducibility/hardware_validation/study_v1/operator_prepared/operator_schedule.csv`
during collection. Do not inspect the lead-side `study_v1/prepared` directory
or compare packet bundles until every outcome is locked. Return the complete
ignored `hardware_runs/` directory, both local site YAML files, and the signed
acceptance checklist to the study lead after collection.

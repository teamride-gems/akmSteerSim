# Frank experiment start page

> Scientific collection is paused. Do not run `HW001` or the 120-run schedule.
> This package is retained for general ROS 1 interface qualification, safety,
> localization, command-path, and logging work. The frozen study replays
> offline command sequences and is not the project's intended closed-loop RL
> policy-transfer experiment. Project leads should begin with
> `BRIAN_HANDOFF.md`.

The historical operator package is preserved on branch
`ride/frank-experiment-package-20260821`. The binding
commands are in
`reproducibility/hardware_validation/ROS1_OPERATOR_RUNBOOK.md`; do not run the
main schedule. Use the runbook only for the explicitly permitted interface
qualification and engineering-pilot steps recorded in `BRIAN_HANDOFF.md`.

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

## Autonomous safety behavior

Nobody holds a button during an authorized qualification run. The preflight
and engineering-pilot runners publish a short-timeout authorization heartbeat
while they are actively in control. The scientific-study runner is disabled.
The safety supervisor keeps the connected joystick in hand and presses
button index 6 to stop. Button 6 latches the software e-stop; runner exit,
runner-heartbeat loss, or joystick-message loss also restores the low-level
zero-command override automatically. Do not touch the other joystick controls
during autonomous execution because Frank's normal teleoperation stack remains
connected to the command mux. After pressing button 6, do not reset the latch
until the runner has exited and `/hardware_study/run_active` is false.

## Required order

1. Complete the site draft and controller/interface capture.
2. Start and verify the autonomous heartbeat safety bridge on stands.
3. Capture the taped start pose into `local_hardware_site_ros1.yaml`.
4. Complete the signed live preflight.
5. Pass and inspect the 0.20 m/s stands pilot.
6. Pass and inspect the 0.50 m/s ground pilot.
7. Stop and return the artifacts to the project lead. `HW001` is not currently
   authorized.

Return the complete ignored `hardware_runs/` directory, both local site YAML
files, the configuration capture, and the signed acceptance checklist to the
project lead after qualification. The historical blinded schedule remains in
the repository for provenance, not execution.

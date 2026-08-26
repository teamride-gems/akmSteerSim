# Hardware-study amendment 005: autonomous heartbeat safety authorization

**Status:** disclosed after software release review and before any physical
motion outcome

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_004.json`

**Reason:** the continuous hold-to-run button introduced by Amendment 004 was
unnecessary for autonomous execution and created an avoidable human-interruption
path. A runner-owned, short-timeout authorization heartbeat provides autonomous
operation while retaining fail-closed behavior.

## Scope

This amendment changes only the ROS 1 motion-authorization mechanism and its
operator verification:

- preflight, pilot, and scientific-run adapters publish
  `/hardware_study/run_active` only while their main execution path is alive;
- the bridge releases the low-level zero-command override only when both the
  runner heartbeat and the joystick stream are fresh and the software e-stop is
  not latched, and it rejects zero or multiple heartbeat publishers;
- runner exit publishes false, and runner crash, heartbeat loss, or joystick
  loss restores the override no later than 0.30 seconds after the last valid
  message at the
  configured 20 Hz bridge rate;
- joystick button index 6 remains the press-to-stop control and latches the
  software e-stop; no button is held during autonomous execution;
- stationary site capture cannot publish authorization; and
- a bounded stands-only heartbeat utility verifies mux release, timeout, and
  e-stop behavior without issuing a drive command.

The one-second mux-clearance interval remains: the stop override ceases only
after raw authorization is present, and the runner receives authorization only
after the mux has had one second to release the prior stop input. The physical
motor e-stop remains the independent backup.

All scientific command values, schedule order, opaque condition assignment,
packet timing, target speed and steering, outcome definitions, thresholds,
retry rules, prepared inputs, and frozen numerical analysis remain unchanged.
No motion pilot or main-study outcome was observed before this amendment.

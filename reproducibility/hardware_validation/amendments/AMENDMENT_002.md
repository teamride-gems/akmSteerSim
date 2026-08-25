# Hardware-study amendment 002: Frank Cartographer TF integration

**Status:** disclosed after a stationary interface capture and before any physical motion outcome

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_001.json`

**Reason:** Frank's 2026-08-25 stationary bag established that the fixed-world
Cartographer pose is published as a TF chain, not as the
`nav_msgs/Odometry` interface assumed by amendment 001.

## Evidence and scope

The reviewed bag is identified in `FRANK_STATIONARY_BAG_20260825.md`. The car
was stationary, every recorded motion command was zero, and the capture is not
a pilot or scientific outcome. No action-space performance result was observed
before this amendment.

This amendment changes only the robot integration layer:

- fixed-world pose is composed from
  `cartographer_map -> cartographer_odom -> base_link` on `/tf`;
- speed and yaw rate continue to come from `/vesc/odom`;
- the joystick source is corrected from `/joy` to `/vesc/joy`;
- the safety bridge independently drives the confirmed low-level safety-mux
  input with a zero command whenever deadman authorization is absent;
- deadman status remains false for a one-second mux-clearance interval before
  the experiment runner may arm;
- the confirmed autonomous input and relevant logging topics are prefilled;
- configuration capture validates the source joystick and localization TF
  topics before the locally generated safety topics exist;
- onboard Cartographer is prospectively documented as localization, not
  independent trajectory ground truth.

All command bundles, schedules, randomization, packet timing, targets, limits,
outcome definitions, thresholds, retry rules, and locked analyses remain
unchanged.

## Remaining fail-closed gates

The active VESC `max_acceleration` must be captured from the configuration that
Frank actually loads and must not exceed 1.5 m/s². The physical control mapped
to joystick button index 6 must be confirmed before it is assigned as the
software e-stop. The approximately 0.250 m configured wheelbase does not clear
the existing physical-measurement gate by itself.

No motion pilot or main-study run is authorized while any of these gates remain
unresolved.

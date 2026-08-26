# Hardware-study amendment 004: operator-package release audit

**Status:** disclosed after stationary interface inspection and before any
physical motion outcome

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_003.json`

**Reason:** a final pre-motion release audit found that the ROS 1 runner still
read semantic condition names from lead-side preparation, the frozen analysis
recognized only the original ROS 2 adapter as physical evidence, the ROS 1 site
capture could discard the independent-ground-truth declaration, the safety
bridge did not reject simultaneous joystick buttons, and configuration capture
did not confirm the active controller parameter tree.

## Scope

This amendment changes only blinding enforcement, integration validation,
safety authorization, evidence labeling, and operator documentation:

- the ROS 1 preflight, pilots, and study runner default to a hash-verified
  operator package whose schedule, bundles, envelope, and run archives contain
  opaque codes rather than semantic condition names;
- the safety bridge authorizes deadman only while button index 5 is held by
  itself; any simultaneous button, release, or stale joystick input restores
  the low-level zero-command override;
- configuration capture verifies that `vesc.yaml`, the site record, and the
  active `/vesc` ROS parameter tree all contain the same 2.0 m/s² acceleration
  ceiling and archives any additional loaded controller files supplied by the
  operator;
- ROS 1 stationary capture preserves the prospectively declared
  independent-ground-truth field; and
- the post-lock ROS 1 analysis wrapper classifies only a pure set of
  `ros1_ackermann_noetic` outcomes as physical hardware evidence and rejects
  mixed-adapter evidence.

All scientific command values, schedule order, condition-code assignment,
packet timing, target speed and steering, outcome definitions, thresholds,
retry rules, and frozen numerical analysis remain unchanged. No motion pilot
or main-study outcome was observed before this amendment.

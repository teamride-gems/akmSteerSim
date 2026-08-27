# Hardware-study amendment 006: streamlined operator prompts

**Status:** disclosed before physical outcomes

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_005.json`

## Scope

This amendment removes typed arm and run-confirmation phrases from the pilot and
study runners, and removes the typed stands confirmation from the bounded
heartbeat utility. The utility remains bounded and must be used only with the
vehicle securely on stands.

Runtime safety behavior is unchanged: the bridge keeps the zero-command
override active unless the runner heartbeat and joystick stream are fresh, and
button index 6 latches the software e-stop. Preflight remains required for a
real run and continues to report concrete configuration and safety failures,
such as insufficient surveyed clear radius or an invalid button-6 mapping.

All scientific command values, schedule order, opaque condition assignment,
packet timing, target speed and steering, outcome definitions, thresholds,
retry rules, prepared inputs, and frozen numerical analysis remain unchanged.

# Hardware-study amendment 003: live Frank limits and topic confirmation

**Status:** disclosed after live interface inspection and before any physical motion outcome

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_002.json`

**Reason:** the mechanical lead confirmed the experiment stop-button index,
selected a 2.0 m/s² controller acceleration ceiling, and supplied the live ROS
topic list after amendment 002 was frozen.

## Scope

This amendment changes only pre-motion robot configuration and logging:

- `/vesc/joy` index 6 is assigned as the experiment software-stop input;
- the VESC controller acceleration ceiling becomes 2.0 m/s², matching the
  frozen common sent-command acceleration limit;
- configuration capture still requires exactly one active `max_acceleration`
  value and exact agreement with the site record, so the previously copied
  2.5 m/s² value remains invalid;
- the required bag list adds the live VESC motor-command channels, mux-active
  indicators, diagnostics, and laser status.

The reported approximately 1.01 m minimum turn radius is retained only as
planning information. It does not replace measured left/right radius or
wheelbase calibration.

All scientific command bundles, schedules, conditions, packet timing, steering
targets, speed targets, outcomes, thresholds, retry rules, and locked analyses
remain unchanged. No motion pilot or main-study outcome was observed before
this amendment.

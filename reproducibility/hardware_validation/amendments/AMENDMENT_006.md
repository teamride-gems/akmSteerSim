# Hardware-study amendment 006: pause scientific collection and complete handoff

**Status:** prospective authorization change after engineering-only interface
captures and before any physical scientific outcome

**Base freeze:** `reproducibility/hardware_validation/study_v1/FREEZE.json`

**Prior amendment:** `AMENDMENT_005.json`

## Reason

The frozen 120-run schedule replays action-dependent command sequences. It is
useful for actuator and interface characterization, but it does not execute a
policy closed loop on Frank and therefore does not directly test the project's
intended question about transfer induced by action-space representations.
Continuing it as the main scientific study would risk collecting a complete,
well-controlled dataset for the wrong construct.

## Authorization and handoff changes

- `scripts/run_hardware_study_ros1.py` now refuses to start every scientific
  run, including `HW001`.
- Configuration capture, stationary safety checks, signed preflight, and
  explicitly labeled engineering pilots remain available for reusable robot
  qualification.
- Existing schedules, prepared inputs, opaque condition assignments, and
  analysis code are retained unchanged for provenance. They are not current
  authorization to collect scientific outcomes.
- Current robot-facing instructions point to the published leadership-handoff
  branch and consistently distinguish qualification from scientific
  authorization.
- Older readiness and acceptance documents are explicitly historical or
  qualification-only; passing them cannot authorize `HW001`.
- Project status and continuation context are recorded in durable repository
  documents and the linked Drive index.
- Restarting scientific collection requires a new prospective protocol,
  competent closed-loop policies, matched controls, review, and a later
  amendment created before observing the new study's outcomes.

## Evidence available before this amendment

The August 26 lab session produced engineering-only interface captures. The
aborted preflight bag contained only zero commands. The crawl capture showed
healthy sensing and localization interfaces but also contained neutral command,
motor, and servo streams. Neither capture is an outcome from the frozen
scientific schedule. The analysis is recorded in
`reproducibility/hardware_validation/LAB_SESSION_20260826.md`.

No threshold, retry rule, scientific command, prepared study input, or prior
record has been retroactively altered. This amendment changes future
authorization and makes that change fail closed in the runner.

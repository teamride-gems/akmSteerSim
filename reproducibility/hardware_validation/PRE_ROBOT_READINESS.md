# Pre-robot readiness record

**Status: SUPERSEDED FOR FRANK BY ROS 1 AMENDMENT 005**

The original official freeze remains intact at
`study_v1/FREEZE.json` (`FROZEN_BEFORE_PHYSICAL_OUTCOMES`, SHA-256
`38A58B34F57D5CE2056B1863B98E24A7D4AB8FFC1D6429F509DA09CC54E48A02`).
Its original ROS 2 integration assumptions are historical and must not be used
to launch Frank.

Frank's current operator path is:

1. repository root `FRANK_START_HERE.md`;
2. `MECHANICAL_TEAM_HANDOFF.md`;
3. `MECHANICAL_ACCEPTANCE_CHECKLIST.md`; and
4. `ROS1_OPERATOR_RUNBOOK.md` under `AMENDMENT_006`.

The package now includes Frank's native ROS 1 Noetic adapter, Cartographer TF
composition, rosbag1 logging, an opaque-code operator schedule and command set,
a fail-closed runner-heartbeat/joystick/mux safety bridge, active-controller parameter capture,
live preflight, two engineering pilots, and a ROS 1-aware locked-analysis
wrapper. The current branch is the intentionally published experiment branch;
the earlier warning about an uncommitted `ablation` working tree is obsolete.

Physical collection is authorized only after the remaining site facts are
captured: measured wheelbase, surveyed course radius, taped start pose, and an
active VESC acceleration setting of exactly 2.0 m/s². Both engineering pilots
must pass before `HW001`. Offline or mock results remain pipeline evidence and
must not be reported as physical outcomes.

# Pre-robot readiness record

**Status: HISTORICAL; SCIENTIFIC COLLECTION PAUSED BY AMENDMENT 006**

The original official freeze remains intact at
`study_v1/FREEZE.json` (`FROZEN_BEFORE_PHYSICAL_OUTCOMES`, SHA-256
`38A58B34F57D5CE2056B1863B98E24A7D4AB8FFC1D6429F509DA09CC54E48A02`).
Its original ROS 2 integration assumptions are historical and must not be used
to launch Frank.

Frank's current operator path is:

1. repository root `FRANK_START_HERE.md`;
2. `MECHANICAL_TEAM_HANDOFF.md`;
3. `MECHANICAL_ACCEPTANCE_CHECKLIST.md`; and
4. `ROS1_OPERATOR_RUNBOOK.md` under the current `AMENDMENT_006` status.

The package now includes Frank's native ROS 1 Noetic adapter, Cartographer TF
composition, rosbag1 logging, an opaque-code operator schedule and command set,
a fail-closed runner-heartbeat/joystick/mux safety bridge, active-controller parameter capture,
signed live preflight, two engineering pilots, and a ROS 1-aware locked-analysis
wrapper. The current branch is the intentionally published experiment branch;
the earlier warning about an uncommitted `ablation` working tree is obsolete.

The remaining site facts and both engineering pilots are qualification work
only. They do not authorize `HW001` or any scientific collection. A new
closed-loop protocol must be reviewed, frozen, and authorized prospectively
before scientific physical outcomes are collected. Offline, mock, and
engineering results remain pipeline evidence and must not be reported as
binding physical-study outcomes.

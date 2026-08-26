# Pre-robot readiness record

**Status: SUPERSEDED FOR FRANK BY ROS 1 AMENDMENT 001**

The RiDE Drive audit subsequently confirmed that Frank uses ROS 1 Noetic. For
Frank, use `amendments/AMENDMENT_003.md`, `ROS1_OPERATOR_RUNBOOK.md`, and
`MECHANICAL_ACCEPTANCE_CHECKLIST.md`. The ROS 2 readiness record below remains
historical evidence of the pre-amendment state and must not be used to launch
Frank.

This means the offline scientific, execution, safety-software, logging, validation, and analysis package is complete. It does **not** authorize immediate ground execution. The site-specific ROS topics, captured start pose, physical course, physical e-stop, vehicle behavior, and human safety roles can only be qualified with the RiDE car present.

## Official freeze

- Freeze: `reproducibility/hardware_validation/study_v1/FREEZE.json`
- Freeze SHA-256: `38A58B34F57D5CE2056B1863B98E24A7D4AB8FFC1D6429F509DA09CC54E48A02`
- Status in record: `FROZEN_BEFORE_PHYSICAL_OUTCOMES`
- Frozen at: `2026-08-21T06:42:28.744148+00:00`
- Amendment rule: never overwrite; preserve and disclose any numbered amendment before inspecting additional physical outcomes.

## Completed offline gates

- 12 deterministic safety-screened source trajectories selected without outcome-based selection.
- 24 immutable source × speed command bundles and 120 blinded crossover runs.
- Exact duplicate clean commands and exact gate/placebo accepted-target and accepted-increment matching.
- Position and immediate-carryover counts each bounded between 4 and 6.
- Conservative slow-actuator envelope: maximum predicted radius 4.520 m and lateral displacement 0.894 m.
- 93 deterministic unit/regression tests passed.
- Repository validity preflight passed.
- Installed F1TENTH physics integration smoke passed at 100 Hz plant / 20 Hz control.
- Official-freeze mock live preflight passed.
- Synthetic stands and ground pilot paths passed strict end-to-end mock validation.
- Exact candidate freeze completed 120/120 blinded mock runs with 120/120 technically valid, eligible outcomes.
- Locked 10,000-draw, four-cell-weighted bootstrap analysis completed without interactive choices.

The mock analysis produced `ENGINEERING_MOCK_ONLY / NOT_REPRODUCED`: clean repeat median/p95 0/0 m; placebo-minus-gate 0.0087 m (95% bootstrap 0.0024–0.0169); gate-minus-direct 0.0015 m (0.0003–0.0031). This is a pipeline qualification and must not be reported as physical evidence or used to revise the frozen thresholds.

## Remaining gates that require the physical car/site

1. Confirm the vehicle is ROS 2 and accepts stamped Ackermann drive messages with the assumed steering/speed signs and units. A ROS 1 or nonstandard interface requires a tested, disclosed adapter before motion.
2. Create ignored `local_hardware_site_draft.yaml`, configure actual topics/buttons/frames, and capture immutable `local_hardware_site.yaml` from stationary localization.
3. Survey and clear at least a 6.0 m radius around the marked start pose.
4. Assign a dedicated safety supervisor; test the independent physical motor e-stop and software deadman/latched e-stop on stands.
5. Obtain a signed, passed live preflight less than 12 hours before collection.
6. Pass the 0.20 m/s stands pilot and 0.50 m/s ground S-turn pilot with plausible direction, localization, stopping, and ROS bag data.
7. Only then begin `HW001`, following the printed operator checklist and preserving schedule order.

## Repository handoff warning

The working tree contains this hardware package plus earlier research changes that are not committed on the current `ablation` branch. The official freeze protects file contents, but the exact snapshot still needs an intentional reviewed commit/push before it can be reliably checked out on the robot computer. Do not copy an arbitrary older branch to the car.

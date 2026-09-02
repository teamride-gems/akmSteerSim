# RiDE RL action-spaces project: Brian handoff

This is the authoritative starting point for project leadership after
2026-09-01. The project is not currently collecting scientific hardware
outcomes. Preserve that status until a new closed-loop experiment is designed,
piloted, and frozen.

## Source-of-truth order

1. This GitHub branch is authoritative for code, executable protocols, tests,
   and durable result summaries.
2. The RiDE Google Drive handoff folder is authoritative for large robot bags,
   model binaries, team-facing planning material, and the cross-link index:
   <https://drive.google.com/drive/folders/15wQ75uWR9kpIhexPR9BLsYyU3oN_R_9G>
3. Informal messages and meeting notes are context only. They cannot override
   a committed protocol or durable result report.

Repository: <https://github.com/teamride-gems/akmSteerSim>

Drive `00 START HERE` index:
<https://docs.google.com/document/d/1ONnym9thjiRQZhfAKamTnqvwLU9wGF9-HCeULk6soI8/edit>

Handoff branch: `ride/project-leadership-handoff-20260901`

Historical Frank operator branch: `ride/frank-experiment-package-20260821`

## Executive status

The repository provides a functional F1TENTH/SAC research stack with four
action interfaces: direct steering-speed, curvature-speed, lookahead point,
and Bezier. It includes reproducibility controls, deterministic tests, policy
evaluation, hardware safety tooling, ROS 1 transport, Cartographer pose
capture, rosbag logging, and two completed negative-result ladders.

There is no active paper hypothesis with both a passed simulation gate and an
authorized physical protocol.

The previously prepared 120-run Frank protocol is paused before scientific
physical outcomes. It replays short action sequences generated offline by two
trained direct-action policies. It does not run a policy onboard, consume live
LiDAR observations, complete laps, compare trained action spaces, or establish
closed-loop sim-to-real policy transfer. Do not run `HW001` or present that
protocol as the project's current direction.

The safety and data infrastructure from that package remains useful for future
closed-loop work.

## What has been established

### Simulator and validity infrastructure

- Python 3.10, Stable-Baselines3 SAC, and `f110-gym` are the supported stack.
- Policy and plant timing, reward integration, collision accounting, map
  loading, action constraints, and provenance have deterministic tests.
- The four action interfaces share observations, reward, vehicle limits, and
  plant dynamics. Their decoders are implemented in
  `utils/action_spaces_utils.py` and exercised through
  `envs/f1tenth_sb3_env.py`.
- `scripts/preflight.py` is the general repository preflight. The editable
  F1TENTH integration smoke must also pass locally.

### Completed research ladders

- The closing-the-action-interface ladder failed its preregistered gates. Only
  two of five direct-policy seeds met the required completion level, and
  decoder-state activation was below the fixed relevance floor.
- The action-interface ensemble study failed its competence gate. Direct seeds
  0 and 4 were competent on the fixed Sakhir screen; direct seed 5, curvature
  seed 0, and lookahead seed 0 were not. The OOD disagreement hypothesis was
  correctly not evaluated.
- The Rung 2 four-action-space baseline is reproducible diagnostic evidence,
  not a competent policy comparison. All four diagnostic policies failed both
  held-out test episodes.

Do not compare action spaces using incompetent policies or reuse a failed
ladder without a substantively new hypothesis and prospective gates.

### Frank hardware interface

The current ROS 1 integration has confirmed interfaces for:

- `ackermann_msgs/AckermannDriveStamped` command routing through Frank's mux;
- `/vesc/odom`, `/vesc/joy`, VESC state, servo command, IMU, LiDAR, and TF;
- Cartographer's `cartographer_map -> cartographer_odom -> base_link` chain;
- a fail-closed autonomous runner heartbeat;
- joystick button index 6 as a latched software e-stop;
- rosbag1 recording and configuration capture.

The 2026-08-26 bags show expected sensor rates, Cartographer TF output, VESC
feedback with no faults, and correct automatic return to the Safety mux after
the heartbeat ended. Cartographer reported motion while wheel odometry stayed
fixed, so localization accuracy and stationary drift are not established by
these bags. They contain no nonzero speed or steering commands, so powered
motion, steering response, the button-6 path, and joystick-loss stop remain
unverified. See
`reproducibility/hardware_validation/LAB_SESSION_20260826.md`.

Only the aborted-preflight bag is currently present in the Drive handoff. The
larger ground-crawl bag was analyzed earlier but is no longer on the local
machine; Brian should recover the original from the lab team.

## Current model artifacts

Checkpoint binaries are intentionally excluded from Git. The Drive handoff
contains `akmSteerSim_selected_models_20260902.zip` (SHA-256
`9D8A208B3B1FDC7E2725754567D0FB20DDBC1840F6F00A0902AD7111792E3AFF`), a
hash-manifested archive with:

- the two competent direct-policy checkpoints, `gate0_direct_s0` and
  `gate0_direct_s4`;
- the four Rung 2 diagnostic checkpoints for reproducibility only;
- each selected run's resolved configuration, metadata, and evaluation files.

The direct checkpoints are the only currently demonstrated competent policies,
but they were trained for the simulator observation contract. They are not yet
authorized for live execution on Frank.

## Recommended next phase

The most aligned next direction is a closed-loop action-interface transfer
study. A defensible version asks whether an action representation changes
sim-to-real degradation through a specified mechanism, such as command
bandwidth, steering slew, actuator saturation, or sensitivity to delay.

Before choosing the final paper claim:

1. Finish general Frank qualification: button-6 stop, joystick-loss stop,
   nonzero stands commands, low-speed ground commands, steering sign and scale,
   timing, localization, and bag completeness.
2. Implement a ROS 1 observation adapter that exactly documents how live scan,
   speed, realized steering, yaw rate, heading error, lateral error, and
   acceleration map into the simulator's 28-element observation.
3. Resolve the physical centerline dependency. The existing Drive document
   `Map to Centerline - Nik` is a design note, not a verified implementation.
4. Implement live checkpoint inference and action decoding behind the existing
   safety bridge. Start with a zero-motion shadow mode, then stands-only
   inference, then a bounded low-speed ground pilot.
5. Train enough seeds for each candidate action space to pass a prospective,
   identical competence gate. Select checkpoints without physical outcome
   information.
6. Use matched direct-action controls to separate geometric inductive bias from
   simple command smoothing or dimensionality.
7. Run a small blinded physical pilot. Freeze a main protocol only after the
   observation, inference, decoder, actuator, localization, and logging path is
   demonstrated end to end.

The final study should quantify both simulation performance and physical
performance, report the sim-to-real drop, and test the proposed mechanism with
command-level diagnostics. Do not use the robot merely to replay simulator
actions if the claim is policy transfer.

## Brian's first-week checklist

1. Clone this branch with its submodule and run the complete preflight.
2. Read the negative-result reports before scheduling new training.
3. Download and verify the model-artifact archive from the Drive handoff.
4. Obtain the complete local site YAMLs, configuration-capture folder,
   preflight output, pilot folders, and terminal output from the lab team. They
   were requested but are not currently in this repository.
5. Complete only the general hardware qualification checks listed above. Do
   not start the 120-run schedule.
6. Write a one-page prospective candidate hypothesis with its mechanism,
   comparison, competence gate, physical task, and kill criteria before adding
   a large experiment matrix.
7. Review that candidate against current literature before committing robot or
   training time.

## Essential commands

```powershell
git clone --branch ride/project-leadership-handoff-20260901 --recurse-submodules `
  https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim

cd ..
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
git checkout 4fdb9c7e6fb7c701290f4dc18377d07c1681724f
cd ..\akmSteerSim

py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.39"
python -m pip install -r requirements_min.txt
python -m pip install -e ..\f1tenth_gym

python scripts\preflight.py
```

The verified simulator revision is
`4fdb9c7e6fb7c701290f4dc18377d07c1681724f`. Adjust the editable
`f1tenth_gym` path if the clone is not adjacent to this repository. On Frank,
use ROS 1 Noetic and the hardware runbooks only for interface qualification
until a new scientific protocol is approved.

## Navigation map

- `README.MD`: codebase architecture, setup, and historical ladders.
- `PROJECT_CONTEXT.md`: compact orientation for future project work.
- `reproducibility/rung2/rung2_41921dd/REPORT.md`: four-space diagnostic
  baseline.
- `reproducibility/action_ensemble_screen/REPORT.md`: failed competence gate.
- `FRANK_START_HERE.md`: retained Frank interface package, now explicitly
  paused for scientific collection.
- `reproducibility/hardware_validation/ROS1_OPERATOR_RUNBOOK.md`: exact ROS 1
  interface and qualification commands.
- `reproducibility/hardware_validation/LAB_SESSION_20260826.md`: first returned
  bag analysis and remaining hardware checks.
- `reproducibility/hardware_validation/STUDY_PROTOCOL_V1.md`: historical frozen
  replay study; do not treat it as the live-policy-transfer plan.

## Handoff rules

- Commit code, configs, protocols, and compact result summaries to GitHub.
- Keep large bags and model binaries in the Drive handoff, each with hashes and
  a short provenance note.
- Never make Drive and GitHub independent copies of executable code.
- Preserve failed and abandoned protocols; label them instead of deleting or
  rewriting their historical claims.
- Freeze hypotheses and selection rules before looking at their binding
  outcomes.
- Keep test tracks out of checkpoint selection.
- Require competence before interpreting action-space comparisons.
- Treat every physical safety change as a new qualification requirement.

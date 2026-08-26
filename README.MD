# akmSteerSim: F1TENTH + SAC action-space experiments

This repository studies how policy action representations affect reinforcement
learning for Ackermann-steered racing. It provides direct steering, curvature,
lookahead-point, and Bezier action decoders while keeping the simulator,
observation model, actuator limits, and reward definition shared.

## Canonical pipeline

```text
configs/{vehicle,sac}.yaml
  -> scripts/sweep.py
  -> scripts/run_one_experiment.py
  -> rl/train.py
  -> rl/common.py
  -> envs/F1TenthSACEnv
  -> utils/{action_spaces,state_processing,reward}.py
  -> f110_gym F110Env
```

Periodic validation selects `best_validation_model.zip`. Test tracks are not
evaluated unless `--eval_after_train` is requested or `rl/eval.py` is run
explicitly after training.

## Repository layout

```text
assets/f1tenth_racetracks/  Track maps and centerlines (Git submodule)
configs/                    Vehicle/environment and SAC configurations
envs/                       Gymnasium wrapper around F1TENTH
rl/                         Training, validation, and held-out evaluation
scripts/                    Sweep, analysis, rollout, and plotting entry points
tests/                      Deterministic regressions and F1TENTH smoke test
utils/                      Action decoding, geometry, state, reward, normalization
checkpoints/                Generated model artifacts (ignored for new files)
runs/                       Generated TensorBoard logs (ignored for new files)
```

The legacy root-level `run_experiment.py` is not the canonical experiment
runner. Use `scripts/sweep.py` or `scripts/run_one_experiment.py`.

## Environment contract

The normalized observation has 28 elements with the default configuration:

```text
[v, a_long, steering, yaw_rate, heading_error, lateral_error, a_lat,
 21 lidar-sector minima]
```

The steering feature is the realized F1TENTH vehicle state, not the requested
command. The LiDAR configuration distinguishes the simulator's native input
field of view (`input_fov_deg`) from an optional narrower policy field of view
(`fov_deg`).

Available normalized policy action spaces are:

| Name | Dimensions | Decoded command |
|---|---:|---|
| `steer_speed` | 2 | steering angle, speed |
| `curvature_speed` | 2 | curvature converted to steering, speed |
| `lookahead_point` | 3 | local lookahead point converted by pure pursuit, speed |
| `bezier` | 5 | Bezier control points converted by pure pursuit, speed |

All decoded commands pass through the same steering-rate, acceleration, and
absolute command constraints, including the first command after reset. The
default plant integrates at 100 Hz while the policy and stateful decoder act at
20 Hz (`action_repeat: 5`); each constrained command is held across the five
plant integrations.

Episodes end on collision, one environment-measured closed-loop lap, or the
physical time horizon. `max_episode_seconds` is converted using the policy
control timestep, and environment construction fails if the horizon cannot
reach the configured lap threshold even at maximum speed.

The running reward is integrated in physical time:

```text
w_progress * centerline_arc_delta
+ w_a_long * (a_long / ref_a_long)^2 * dt
+ w_a_lat  * (a_lat  / ref_a_lat)^2  * dt
+ w_time * dt
+ one-time collision penalty
```

This makes the running objective invariant to simulator/control frequency for
the same physical trajectory. SAC discounting is also specified in physical
time (`discount_half_life_seconds`) and resolved against the policy timestep.

## Action-interface ensemble study

The workshop hypothesis was tested with a fail-closed five-policy screen: does
structured diversity from direct steering, curvature, and lookahead interfaces
predict failures of one fixed direct-action anchor better than an equally sized
direct-action seed ensemble and a temporal state-risk model?

The member identities, competence thresholds, OOD conditions, metrics, and
pass/kill rules are fixed in `configs/action_ensemble_screen.yaml`. The screen
reuses the two competent Gate 0 checkpoints and trains exactly one additional
direct policy, one curvature policy, and one lookahead policy. It stops before
OOD evaluation if any member fails completion or behavior matching; weak
members may not be replaced after seeing results.

```powershell
python scripts\run_action_ensemble_study.py
```

The two shift families are LiDAR dropout and steering-command delay. The direct
anchor generates every evaluated trajectory while all five policies run in
shadow mode on the same observation. Native actions are decoded to normalized
physical steering-speed commands before disagreement is measured. The temporal
risk baseline uses the same observation history, thresholds are calibrated at
a fixed frame-level false-positive rate, and confidence intervals resample
whole episodes rather than correlated timesteps.

The August 17, 2026 screen **failed at the competence gate**. Direct seeds 0
and 4 completed 10/10 fixed Sakhir laps, while direct seed 5 and curvature seed
0 crashed on 10/10, and lookahead seed 0 timed out on 10/10. OOD disagreement
was therefore not evaluated: doing so would confound interface diversity with
policy quality. The durable negative-result record is in
`reproducibility/action_ensemble_screen/REPORT.md`; detailed regenerated output
remains under ignored `experiments/action_ensemble_screen/`.

## Historical closing-the-action-interface ladder

The earlier stateful-decoder direction was killed by its preregistered gates:
only two of five direct SAC seeds reached 90% completion, and the steering
limiter activated on 4.74% of successful transitions versus the fixed 20%
floor. The retained audit procedure was:

1. **Gate 0 — direct SAC functionality.** Train five `steer_speed` seeds using
   `configs/gate0_sac.yaml`. Select checkpoints only on the declared validation
   tracks. On 50 fixed, arc-length-spaced Sakhir starts, at least four of five
   seeds must achieve at least 90% collision-free lap completion. The decision
   script reports per-seed outcomes, IQM, and stratified bootstrap intervals.
2. **Gate 1 — decoder-state relevance.** From successful frozen-policy laps,
   fit episode-split models `m: o -> h`, `f: (o,a) -> c`, and
   `g: (o,a,h) -> c`, then run standardized caliper matching on reachable
   histories. All thresholds in `configs/decoder_state_gate.yaml` were fixed
   before the audit. Recoverable or irrelevant register state, rare limiter
   activation, or a failed matched-history contrast kills the paper direction.

The environment logs the full interface transition on every step: policy
observation, raw action, previous constrained command, decoded pre-limit
command, executed command, realized command, next observation, limiter flags,
correction magnitudes, and commanded-versus-realized gaps.

Gate 0 evaluation:

```powershell
python scripts\evaluate_gate0.py `
  checkpoints\gate0_direct_s0 checkpoints\gate0_direct_s1 `
  checkpoints\gate0_direct_s2 checkpoints\gate0_direct_s3 `
  checkpoints\gate0_direct_s4
```

Gate 1 nontraining audit:

```powershell
python scripts\audit_decoder_state.py `
  --gate0_result experiments\gate0\gate0_result.json

# Only if the successful-lap activation floor passes:
python scripts\collect_decoder_audit.py `
  --run_dir checkpoints\gate0_direct_s0 `
  --tracks Sakhir --episodes_per_track 25

python scripts\audit_decoder_state.py `
  experiments\gate1\decoder_audit_dataset.npz
```

The joint Gate 0/Gate 1 criterion did not pass, so the proposed 12-run
Direct/Curvature × {shuffled-register, actual-register} screen is not
authorized and must not be revived without a substantively new hypothesis.

## Setup

Python 3.10 is the supported runtime.

```powershell
git clone https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim
git submodule update --init --recursive

py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.39"
python -m pip install -r requirements_min.txt
python -m pip install -e ..\f1tenth_gym
```

Adjust the editable F1TENTH path to its local clone. The wrapper redirects
Numba's cache into ignored `.numba_cache/`; this avoids import stalls when the
editable simulator source is outside a writable workspace.

## Validity preflight

Run both checks before training:

```powershell
python -m unittest discover -s tests -v
python tests\f110_integration_smoke.py
```

The unit suite covers collision semantics and penalty, realized steering,
LiDAR FOV, closed-loop lap completion, feasible horizons, first-action slew
limits, reward timestep invariance, and every bundled track. The integration
smoke checks the installed F1TENTH observation schema and a real physics step.

The same checks, plus Git artifact-hygiene validation, are available through:

```powershell
python scripts\preflight.py
```

CI runs the deterministic portion on every push and pull request. The installed
F1TENTH smoke remains a required local preflight because it exercises the
editable simulator, map loading, Numba runtime, and real physics step.

## Frank hardware experiment package

Frank's ROS 1 Noetic deployment starts with `FRANK_START_HERE.md`, followed by
`reproducibility/hardware_validation/MECHANICAL_TEAM_HANDOFF.md`. The binding
steps are in `ROS1_OPERATOR_RUNBOOK.md`, and the physical sign-off is
`MECHANICAL_ACCEPTANCE_CHECKLIST.md`. The package includes the frozen blinded
schedule, command bundles, safety/preflight tools, native ROS 1 runner, and
locked analysis. Generated robot data stay under ignored `hardware_runs/`.

The ROS 1 runners use `study_v1/operator_prepared`, whose schedules, bundles,
and archives expose only opaque condition codes. The condition key is excluded
and retained separately by the study lead until collection is complete.

## Reproducibility baseline

`configs/sac_debug.yaml` defines the Rung 2 pipeline baseline: 20,000 training
steps, seed 0, Sakhir training, Austin validation, Budapest test, and two fixed
evaluation spawns. It is diagnostic evidence, not a paper-scale comparison.

From a clean committed tree, one command trains and evaluates all four action
spaces, validates their artifacts and numerical diagnostics, and regenerates
the aggregate tables:

```powershell
python scripts\run_repro_baseline.py --device cpu
```

Each run records:

- exact Git commit, branch, dirty state, remote, and track-submodule commit;
- standalone-evaluation Git provenance and the selected checkpoint's SHA-256;
- Python, package, OS, CPU, CUDA, cuDNN, and GPU information;
- resolved vehicle and SAC configurations;
- initialization, running, completion, or failure status;
- training/evaluation wall time and final timestep;
- reward-component, collision-penalty, realized-steering, acceleration, and
  termination diagnostics. Raw collision impulses are reported separately
  from nonterminal acceleration; only nonterminal acceleration is used as the
  actuator-plausibility gate.

The generated report lives under `reproducibility/rung2/<run-prefix>/`. It can
be regenerated without retraining using:

```powershell
python scripts\run_repro_baseline.py `
  --aggregate-only `
  --run-prefix <run-prefix>
```

For recovered or post-hoc reports, the manifest records separate immutable
training, standalone-evaluation, and reporting commits. The first two are
inferred from the run artifacts and must each be identical across all four
conditions, or aggregation fails closed.

If orchestration is interrupted after an exact 20,000-step checkpoint is
written but before its final validation finishes, recover without adding any
training steps:

```powershell
python scripts\finalize_interrupted_run.py `
  --run-id <run-id> `
  --device cpu `
  --eval-test
```

The recovery command verifies the checkpoint timestep, reproduces the fixed
Austin validation and Sakhir diagnostic spawns, re-applies the declared model-
selection score, records separate recovery provenance, and only then evaluates
the selected model on Budapest.

The root-level `run_experiment.py` is intentionally retired and fails closed.
It must not be restored as an experiment path because it used obsolete and
potentially stale checkpoint/metric associations.

## Training and evaluation

Inspect a full four-space sweep without launching jobs:

```powershell
python scripts\sweep.py --include_bezier --seeds 0 --no-ablation --dry-run
```

Run one condition and evaluate the selected validation checkpoint on the held-
out test split:

```powershell
python scripts\run_one_experiment.py `
  --action_space steer_speed `
  --obs_regime full `
  --seed 0 `
  --eval_after_train
```

Run the configured multi-seed sweep:

```powershell
python scripts\sweep.py --include_bezier --eval_after_train
```

Outputs are written to `runs/` and `checkpoints/<run_id>/`. Each modern run
stores its complete vehicle and SAC configuration in `run_meta.json`.

## Rollout visualization

```powershell
python scripts\run_trained_policy.py `
  --model checkpoints\<run_id>\eval_results\best_validation_model.zip `
  --vehicle_cfg configs\vehicle.yaml `
  --track Sakhir `
  --steps 12000 `
  --record --record_path rollouts\policy.npz

python scripts\replay_rollout_html.py `
  --rollout rollouts\policy.npz `
  --out rollouts\policy.html `
  --lidar
```

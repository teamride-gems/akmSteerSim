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
absolute command constraints, including the first command after reset.

Episodes end on collision, one environment-measured closed-loop lap, or the
physical time horizon. `max_episode_seconds` is converted using the detected
simulator timestep, and environment construction fails if the horizon cannot
reach the configured lap threshold even at maximum speed.

The running reward is integrated in physical time:

```text
w_progress * centerline_arc_delta
+ w_a_long * (a_long / ref_a_long)^2 * dt
+ w_a_lat  * (a_lat  / ref_a_lat)^2  * dt
+ w_time * dt
+ one-time collision penalty
```

This makes the episode objective invariant to simulator/control frequency for
the same physical trajectory.

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

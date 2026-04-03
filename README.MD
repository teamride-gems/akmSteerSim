# akmSteerSim (F1TENTH + SB3 SAC) — Contributor README

This repository provides a reinforcement-learning training and evaluation stack built on top of the `f110_gym` simulator, using Stable-Baselines3 SAC.

It contains:

- A **Gym-style environment wrapper** around `f110_gym`  
  (`envs/f1tenth_sb3_env.py`)
- Scripts to **sanity-check observations/actions** and verify **LiDAR**  
  (`scripts/simple_drive_vis.py`)
- Tools to **record rollouts** to `.npz` and **replay** them in a self-contained **HTML viewer**  
  (`scripts/replay_rollout_html.py`)
- A training pipeline for **SAC (Stable-Baselines3)** with optional **multi-map training schedules**  
  (`rl/train.py`)


---

## Table of Contents

- [Repo Layout](#repo-layout)
- [Core Concepts](#core-concepts)
  - [Maps & Naming Conventions](#maps--naming-conventions)
  - [Observation Vector](#observation-vector)
  - [Actions](#actions)
  - [Rewards](#rewards)
- [Setup](#setup)
  - [WSL2 Notes (Rendering)](#wsl2-notes-rendering)
  - [Dependency Compatibility (IMPORTANT)](#dependency-compatibility-important)
- [Quickstart](#quickstart)
- [Scripts](#scripts)
- [Training](#training)
- [How to Run a Saved Model](#how-to-run-a-saved-model)
- [Troubleshooting](#troubleshooting)
- [Contributing Guidelines](#contributing-guidelines)
- [Recommended Command Cheatsheet](#recommended-command-cheatsheet)

---
## Repo Layout

akmSteerSim/
├─ assets/
│  └─ f1tenth_racetracks/
│     └─ <TrackName>/
│        ├─ <TrackName>_map.yaml
│        ├─ <TrackName>_map.png
│        └─ <TrackName>_centerline.csv
│
├─ configs/
│  ├─ vehicle.yaml
│  └─ sac.yaml
│
├─ drivers/
│  └─ driver_spec.py
│
├─ envs/
│  └─ f1tenth_sb3_env.py
│
├─ metrics/
│  └─ metrics.csv
│
├─ rl/
│  ├─ train.py
│  ├─ eval.py
│  └─ metrics_logger.py
│
├─ scripts/
│  ├─ simple_drive_vis.py
│  ├─ replay_rollout_html.py
│  ├─ metrics_logger.py
│  ├─ run_eval.py
│  ├─ run_trained_policy.py
│  └─ random_pose_gen.py
│
├─ utils/
│  ├─ geometry.py
│  ├─ normalization.py
│  ├─ reward.py
│  ├─ state_processing.py
│  └─ action_spaces_utils.py
│
├─ runs/         # TensorBoard logs (created at runtime)
├─ checkpoints/  # model weights (created at runtime)
└─ rollouts/     # recorded npz + HTML replays (created at runtime)


---

## Core Concepts

### Maps & Naming Conventions

This repo uses a consistent naming convention:

- Folder per track: `assets/f1tenth_racetracks/<TrackName>/`
- Map YAML:          `<TrackName>_map.yaml`
- Map PNG:           `<TrackName>_map.png`
- Centerline CSV:    `<TrackName>_centerline.csv`

Orginal tracks from https://github.com/f1tenth/f1tenth_racetracks#


### Observation Vector

The environment wrapper returns a fixed observation vector including kinematics + centerline errors + LiDAR sectors.

Typical layout

[0]  v                 (speed)  
[1]  a_long            (longitudinal accel)  
[2]  steering          (steering state)  
[3]  yaw_rate          (angular velocity about z)  
[4]  heading_error     (rad)  relative to centerline tangent  
[5]  lateral_error     (m)    signed distance to centerline  
[6]  a_lat             (lateral accel)  
[7] lidar_sectors     (normalized)  

### Actions

The wrapper expects a 2D continuous action:

action[0]: speed_norm in [-1, 1] (mapped to m/s by your action mapper)

action[1]: steer_norm in [-1, 1] (mapped to steering radians)

### Rewards

From configs/vehicle.yaml, reward config often looks like:

reward:  
  progress_gain: 1.0  
  collision_penalty: -10.0  
  offtrack_penalty: -5.0  

### Setup

cd ~/akmSteerSim  
python3 -m venv .venv  
source .venv/bin/activate  
python -m pip install -U pip  
pip install -r requirements.txt  


### f110_gym (simulator) is used from a git clone

This repo depends on the `f110_gym` simulator package (`f110_gym` / `f110_gym.envs.F110Env`).
In our setup, `f110_gym` is not installed from PyPI — it is pulled via `git clone` and then made importable.

Example clone location:
- `/home/<you>/f1tenth_gym/`
- Python package lives at: `/home/<you>/f1tenth_gym/gym/f110_gym/`

After cloning, install it into the venv (recommended):  

source .venv/bin/activate  
pip install -e /path/to/f1tenth_gym/gym



### WSL2 Notes (Rendering)

Rendering via pyglet/OpenGL can be flaky on WSL2 depending on WSLg and GPU/OpenGL forwarding.

This repo supports a headless workflow:

record rollouts to .npz

replay in a self-contained HTML viewer


### Dependency Compatibility (IMPORTANT)

This stack is sensitive to NumPy constraints:

f110-gym expects NumPy <= 1.22.x

numba requires NumPy >= 1.22

avoid NumPy 2.x for this simulator stack unless everything is rebuilt

Recommended pinned set (Python 3.10):

pip install --no-cache-dir --only-binary=:all:
  "numpy==1.22.0"
  "llvmlite==0.39.1" "numba==0.56.4"
  "pandas==1.5.3" "matplotlib==3.6.3"

pip install "stable-baselines3==2.7.0" tqdm rich pyyaml

### Quickstart
A) Verify sim + LiDAR (headless record + HTML replay)

Record:

mkdir -p rollouts  
python scripts/simple_drive_vis.py
  --vehicle_cfg configs/vehicle.yaml
  --record --record_path rollouts/heuristic_run.npz


Replay:

python scripts/replay_rollout_html.py
  --rollout rollouts/heuristic_run.npz 
  --out rollouts/heuristic_run.html
  --stride 1 --lidar


Serve & open:

cd rollouts  
python -m http.server 8000 --bind 0.0.0.0


Open in Windows browser:

http://localhost:8000/heuristic_run.html

### Scripts 

Purpose:

Runs a short rollout using the env wrapper.  
Can render (if supported) or run headless.  
Can print LiDAR stats and record .npz rollouts for replay.

Common usage:

Headless + record:

python scripts/simple_drive_vis.py
  --vehicle_cfg configs/vehicle.yaml
  --record --record_path rollouts/sakhir_test.npz


Render (if it works):

python scripts/simple_drive_vis.py --vehicle_cfg configs/vehicle.yaml


Outputs:

rollouts/<name>.npz containing: pose, action, e_head, e_lat, optional lidar_sectors, and the track_centerline_csv used.


scripts/replay_rollout_html.py

Purpose:

Creates a self-contained HTML replay viewer from .npz.


Slider + play/pause; shows pose + action + errors + optional LiDAR bars.

Usage:

python scripts/replay_rollout_html.py
  --rollout rollouts/sakhir_test.npz
  --out rollouts/sakhir_test.html
  --stride 1
  --lidar


Open the replay:

cd rollouts
python -m http.server 8000 --bind 0.0.0.0


Then open:

http://localhost:8000/sakhir_test.html

### Training 

rl/train.py

python rl/train.py --vehicle_cfg configs/vehicle.yaml --sac_cfg configs/sac.yaml


Train SAC on the F1TENTH env wrapper.

Log to TensorBoard (runs/).

Save weights to checkpoints (checkpoints/).

Support multi-map training schedules (from configs/sac.yaml)

### Tensor Board Command 

tensorboard --logdir runs --bind_all --port 6006


### How to Run a Saved Model

Recommended workflow:

run the model headless and record a rollout

replay in HTML

If you have a script like scripts/run_trained_policy.py:


  python scripts/run_trained_policy.py
    --model checkpoints/20260120-233406/sac_final
    --vehicle_cfg configs/vehicle.yaml
    --track Sakhir
    --steps 5000
    --record --record_path rollouts/sac_sakhir.npz


Replay:

python scripts/replay_rollout_html.py
  --rollout rollouts/sac_sakhir.npz
  --out rollouts/sac_sakhir.html
  --lidar

Troubleshooting
Rendering window doesn’t show (WSL2)

Use headless + record + HTML replay.

HTML replay is blank

Fixes:

Serve via python -m http.server instead of opening the file directly

Reduce size with --stride 4 or --stride 10

Verify rollout pose data is sane:

python - <<'PY'

import numpy as np

d=np.load("rollouts/sakhir_run.npz", allow_pickle=True)

pose=d["pose"]

print("pose shape:", pose.shape)

print("nan count:", np.isnan(pose).sum())

print("min x,y:", np.nanmin(pose[:,0]), np.nanmin(pose[:,1]))

print("max x,y:", np.nanmax(pose[:,0]), np.nanmax(pose[:,1]))

PY

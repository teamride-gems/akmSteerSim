# akmSteerSim (F1TENTH + SB3 SAC) — Contributor README

## Overview

This repository is designed to study how **action space design affects reinforcement learning performance** in Ackermann-steered systems.

It enables controlled comparisons between different action parameterizations (e.g., direct steering, curvature, trajectory-based representations) while keeping the environment and reward structure fixed.

---

## Repo Layout

akmSteerSim /
├─ assets /
│  └─ f1tenth_racetracks /
├─ configs /
│  ├─ vehicle.yaml
│  └─ sac.yaml
├─ envs /
│  └─ f1tenth_sb3_env.py
├─ rl /
│  └─ train.py
├─ scripts /
│  ├─ simple_drive_vis.py
│  └─ replay_rollout_html.py
├─ runs /
├─ checkpoints /
└─ rollouts /

---

## Core Concepts

### Maps

Tracks follow this structure:

- assets/f1tenth_racetracks/<TrackName>/
- <TrackName>_map.yaml
- <TrackName>_map.png
- <TrackName>_centerline.csv

These files define the geometry and reference trajectory used by the simulator.

---

### Observation Vector

The environment returns a fixed observation vector combining:

- Vehicle kinematics (speed, steering, acceleration)
- Track-relative errors (heading + lateral deviation)
- Processed LiDAR sectors

Typical layout:

[0] v  
[1] a_long  
[2] steering  
[3] yaw_rate  
[4] heading_error  
[5] lateral_error  
[6] a_lat  
[7] lidar_sectors  

This structure is critical for reproducibility across experiments.

---

### Actions

The policy outputs normalized actions:

action[0]: speed_norm ∈ [-1, 1]  
action[1]: steer_norm ∈ [-1, 1]  

These are NOT directly executed.

Instead, the pipeline is:

policy → action space → (steering_angle, speed) → constraints → simulator

Key idea:

- The policy operates in a normalized space
- Different action parameterizations map this into physical commands

This repository supports multiple representations (curvature, lookahead, etc.), all of which ultimately produce:

(steering_angle, speed)

This abstraction is central to the research goal.

---

### Rewards

Defined in configs/vehicle.yaml:

reward:
  progress_gain: 1.0
  collision_penalty: -10.0
  offtrack_penalty: -5.0

These shape behavior:

- Encourage forward progress
- Penalize unsafe states

---

## Setup

### Requirements

- Python 3.10
- Git

---

### Installation

git clone https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.39"

pip install -r requirements_min.txt

---

### Install Simulator

cd ..
git clone https://github.com/f1tenth/f1tenth_gym.git
pip install -e f1tenth_gym
cd akmSteerSim

---

### Install Tracks

cd assets
git clone https://github.com/f1tenth/f1tenth_racetracks.git
cd ..

---

### Verify

python -c "import f110_gym; print('ok')"

Expected:
ok

---

## Quickstart (Minimal Path)

pip install -r requirements_min.txt

python scripts/simple_drive_vis.py \
  --vehicle_cfg configs/vehicle.yaml \
  --record --record_path rollouts/test.npz

python scripts/replay_rollout_html.py \
  --rollout rollouts/test.npz \
  --out rollouts/test.html

cd rollouts && python -m http.server 8000

Open:
http://localhost:8000/test.html

Expected:

- rollouts/test.npz created
- HTML viewer renders trajectory + LiDAR

---

## Getting Ready (Dev Context)

Before contributing, ensure you can:

- Run a rollout
- Replay HTML
- Train SAC briefly

Key files:

- envs/f1tenth_sb3_env.py
- rl/train.py
- scripts/simple_drive_vis.py

Core pipeline:

policy → action space → (steering_angle, speed) → constraints → simulator

Understanding this pipeline is essential for modifying action spaces or training logic.

---

## Core Workflows

### Rollout + Replay

python scripts/simple_drive_vis.py \
  --vehicle_cfg configs/vehicle.yaml \
  --record --record_path rollouts/run.npz

python scripts/replay_rollout_html.py \
  --rollout rollouts/run.npz \
  --out rollouts/run.html \
  --lidar

---

### Training

python rl/train.py \
  --vehicle_cfg configs/vehicle.yaml \
  --sac_cfg configs/sac.yaml

Outputs:

- runs/ (TensorBoard logs)
- checkpoints/ (saved models)

---

### Run a Trained Model

python scripts/run_trained_policy.py \
  --model checkpoints/<run>/sac_final \
  --vehicle_cfg configs/vehicle.yaml \
  --track Sakhir \
  --steps 5000 \
  --record --record_path rollouts/policy.npz

---

## Troubleshooting

### Rendering Issues (WSL2)

Use headless mode:

- Record rollouts
- Replay using HTML viewer

---

### HTML Replay Issues

Fixes:

- Use python -m http.server
- Reduce size with --stride
- Validate rollout data:

python - <<'PY'
import numpy as np
d=np.load("rollouts/run.npz")
print(d["pose"].shape)
PY

---

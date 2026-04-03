# akmSteerSim (F1TENTH + SB3 SAC) — Contributor README

## Overview

This repository is designed to study how **action space design affects reinforcement learning performance** in Ackermann-steered systems.

It enables controlled comparisons between different action parameterizations (e.g., direct steering, curvature, trajectory-based representations) while keeping the environment and reward structure fixed.

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
│ └─ f1tenth_racetracks/  
│ └─ <TrackName>/  
│ ├─ <TrackName>_map.yaml  
│ ├─ <TrackName>_map.png  
│ └─ <TrackName>_centerline.csv  
├─ configs/  
│ ├─ vehicle.yaml  
│ └─ sac.yaml  
├─ envs/  
│ └─ f1tenth_sb3_env.py  
├─ rl/  
│ └─ train.py  
├─ scripts/  
│ ├─ simple_drive_vis.py  
│ └─ replay_rollout_html.py  
├─ runs/ # TensorBoard logs (created at runtime)  
├─ checkpoints/ # model weights (created at runtime)  
└─ rollouts/ # recorded npz + html replays (created at runtime)  

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
---

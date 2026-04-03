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

```text
akmSteerSim/
├─ assets/                  # Static assets for simulation (maps, track data)
│  └─ f1tenth_racetracks/  # F1TENTH track dataset
│     └─ <TrackName>/      # Individual track directory
│        ├─ <TrackName>_map.yaml       # Map metadata (resolution, origin)
│        ├─ <TrackName>_map.png        # Occupancy grid map
│        └─ <TrackName>_centerline.csv # Track centerline (used for reward/eval)
│
├─ configs/                # Experiment and system configuration files
│  ├─ vehicle.yaml         # Vehicle parameters (wheelbase, limits, constraints)
│  └─ sac.yaml             # SAC hyperparameters and training settings
│
├─ drivers/                # Policy/driver abstractions
│  └─ driver_spec.py       # Interface defining driver behavior
│
├─ envs/                   # RL environments (Gym-compatible)
│  └─ f1tenth_sb3_env.py   # F1TENTH environment for training and evaluation
│
├─ metrics/                # Stored evaluation outputs
│  └─ metrics.csv          # Aggregated performance metrics across runs
│
├─ rl/                     # Core RL pipeline (training + evaluation logic)
│  ├─ train.py             # Training loop (SAC)
│  └─ eval.py              # Policy evaluation logic
│
├─ scripts/                # Entry points for experiments, visualization, and utilities
│  ├─ simple_drive_vis.py      # Visualize agent behavior in the environment
│  ├─ replay_rollout_html.py   # Generate HTML replay from rollout data
│  ├─ metrics_logger.py        # Logging/export utilities for experiment metrics
│  ├─ run_eval.py              # Run evaluation on trained checkpoints
│  ├─ run_trained_policy.py    # Execute a trained policy rollout
│  └─ random_pose_gen.py       # Generate randomized initial conditions
│
├─ utils/                  # Core utilities (shared across env, RL, and scripts)
│  ├─ geometry.py              # Geometric computations (curvature, transforms)
│  ├─ normalization.py         # Input/output normalization
│  ├─ reward.py                # Reward function definitions
│  ├─ state_processing.py      # State representation construction
│  └─ action_spaces_utils.py   # Action space definitions and mappings
│
├─ runs/                   # TensorBoard logs (created at runtime)
├─ checkpoints/            # Saved model weights (created at runtime)
└─ rollouts/               # Recorded trajectories + HTML replays (created at runtime)
```

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

## Setup (Cross-Platform)

### Requirements

- Python 3.10 (strongly recommended)
- Git

---

### Step 1: Clone

```bash
git clone https://github.com/teamride-gems/akmSteerSim.git
cd akmSteerSim
```

---

### Step 2: Virtual Environment

#### Windows
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

---

### Step 3: Compatibility Fix

```bash
python -m pip install --upgrade "pip<24.1" "setuptools<66" "wheel<0.39"
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements_min.txt
```

---

### Step 5: Install f110_gym

```bash
cd ..
git clone https://github.com/f1tenth/f1tenth_gym.git
pip install -e f1tenth_gym
cd akmSteerSim
```

---

### Step 6: Install Track Assets

```bash
cd assets
git clone https://github.com/f1tenth/f1tenth_racetracks.git
cd ..
```

---

### Step 7: Verify Installation

```bash
python -c "import f110_gym; print('ok')"
python -c "import numpy; print(numpy.__version__)"
```

Expected:
```
ok
1.22.0
```

---

### Step 8: Sanity Check

```bash
python scripts/simple_drive_vis.py --vehicle_cfg configs/vehicle.yaml --record --record_path rollouts/heuristic_run.npz
```

---

### Step 9: Replay

```bash
python scripts/replay_rollout_html.py --rollout rollouts/heuristic_run.npz --out rollouts/heuristic_run.html
cd rollouts
python -m http.server 8000
```

Open:
http://localhost:8000/heuristic_run.html

---

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

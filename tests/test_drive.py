# tests/test_drive.py
import numpy as np
import os
import sys

# Add project root (…/akmSteerSim) to sys.path so `envs`, `utils`, etc. can be imported
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.f1tenth_sb3_env import F1TenthSACEnv
# other imports below…

def test_handcrafted_driver_runs():
    env = F1TenthSACEnv()
    obs, info = env.reset()

    done = False
    steps = 0
    while not done and steps < 500:
        # simple policy: drive forward, steer away from closest obstacle
        lidar = obs[7:]          # last 21 entries
        idx_min = np.argmin(lidar)
        steer_norm = (idx_min - 10) / 10.0   # crude “turn away”
        steer_norm = np.clip(steer_norm, -1.0, 1.0)
        action = np.array([0.6, -steer_norm], dtype=float)  # 60% speed
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1

    env.close()
    assert steps > 10

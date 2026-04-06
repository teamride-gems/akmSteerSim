"""
State vector construction for F1TenthSACEnv.

State layout (must match STATE_IDX_* constants in f1tenth_sb3_env.py):
  [0] v        — longitudinal speed (m/s)
  [1] a_long   — longitudinal acceleration (m/s²)
  [2] d        — steering angle (rad), physical servo state (action-space-independent)
  [3] r        — yaw rate (rad/s)
  [4] e_head   — heading error to centerline (rad)
  [5] e_lat    — lateral error to centerline (m, signed)
  [6] a_lat    — lateral acceleration (m/s²)
  [7:] lidar   — per-sector minimum distances (m)
"""

import warnings

import numpy as np

from .geometry import project_to_centerline

N_SCALARS = 7


def lidar_to_sectors(scan, cfg):
    clip_min = cfg["lidar"]["clip_min_m"]
    clip_max = cfg["lidar"]["clip_max_m"]
    sectors = cfg["lidar"]["sectors"]

    scan = np.asarray(scan, dtype=float)
    scan = np.clip(scan, clip_min, clip_max)

    fov = cfg["lidar"]["fov_deg"]
    n = scan.size
    mid = n // 2
    half = int(n * (fov / 360.0) / 2)
    window = scan[mid - half : mid + half]

    if window.size < sectors:
        warnings.warn(
            f"Lidar scan has {window.size} beams in FOV window but "
            f"{sectors} sectors requested. Some sectors will be empty.",
            stacklevel=2,
        )

    splits = np.array_split(window, sectors)
    mins = []
    q = cfg["lidar"].get("outlier_quantile", 0.995)
    for seg in splits:
        if seg.size == 0:
            mins.append(clip_max)
            continue
        hi = np.quantile(seg, q)
        seg = seg[seg <= hi]
        mins.append(np.min(seg) if seg.size else clip_max)
    return np.asarray(mins, dtype=float)


def make_state(obs_raw, centerline, cfg):
    x, y, yaw = obs_raw["pose"]
    v = float(obs_raw["speed"])
    d = float(obs_raw["steer"])
    r = float(obs_raw.get("yaw_rate", 0.0))
    a_long = float(obs_raw.get("a_long", 0.0))
    a_lat = float(obs_raw.get("a_lat", 0.0))

    e_lat, e_head = project_to_centerline(np.array([x, y, yaw]), centerline)

    lidar_mins = lidar_to_sectors(obs_raw["scan"], cfg)

    state = np.concatenate([
        [v, a_long, d, r, e_head, e_lat, a_lat],
        lidar_mins,
    ])
    return state
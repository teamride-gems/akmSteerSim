import numpy as np
import pandas as pd
from hashlib import sha256
from typing import List, Dict, Any
import math

def deterministic_hash(*args) -> int:
    h = sha256()
    for a in args:
        h.update(str(a).encode())
    return int(h.hexdigest(), 16) % (2**31 - 1)

class Centerline:
    def __init__(self, centerline_csv: str):
        df = pd.read_csv(centerline_csv)
        # Accept columns 'x','y' or first two cols
        if 'x' in df.columns and 'y' in df.columns:
            self.x = df['x'].to_numpy()
            self.y = df['y'].to_numpy()
        else:
            self.x = df.iloc[:,0].to_numpy()
            self.y = df.iloc[:,1].to_numpy()
        # compute arc-length s and tangents
        dx = np.diff(self.x); dy = np.diff(self.y)
        ds = np.sqrt(dx*dx + dy*dy)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.length = float(self.s[-1])
        # headings (tangent) computed by derivative (center difference)
        # we compute heading per point via finite difference
        headings = np.arctan2(np.concatenate([dy, dy[-1:]]), np.concatenate([dx, dx[-1:]]))
        self.heading = headings

    def sample(self, s_query: float):
        # clamp
        s_query = max(0.0, min(s_query, self.length))
        x = np.interp(s_query, self.s, self.x)
        y = np.interp(s_query, self.s, self.y)
        yaw = np.interp(s_query, self.s, self.heading)
        # ensure yaw continuity
        return float(x), float(y), float(yaw)

def offset_perp(x, y, yaw, lateral):
    # offset by lateral (m) to the left (positive)
    x_off = x - lateral * math.sin(yaw)
    y_off = y + lateral * math.cos(yaw)
    return x_off, y_off

def generate_start_poses_from_centerline(centerline_csv: str,
                                         n_poses: int,
                                         seed: int,
                                         min_spacing_frac: float = 0.02,
                                         yaw_jitter_rad: float = 0.087,
                                         lateral_jitter_m: float = 0.05) -> List[Dict[str, Any]]:
    """
    Generate n_poses deterministic start poses given a centerline csv.
    Returns list of dicts with keys: pose_id, s, x, y, yaw, seed_for_pose
    """
    cl = Centerline(centerline_csv)
    L = cl.length
    min_spacing_m = max(0.01, min_spacing_frac * L)

    rng = np.random.RandomState(seed)
    s_list = []
    # systematic sampling with jitter to avoid clustering
    attempts = 0
    max_attempts = n_poses * 100
    while len(s_list) < n_poses and attempts < max_attempts:
        s = rng.uniform(0.0, L)
        if all(abs(s - s0) >= min_spacing_m for s0 in s_list):
            s_list.append(s)
        attempts += 1
    # fallback grid spacing if insufficient
    if len(s_list) < n_poses:
        base = np.linspace(0, L, num=n_poses+2)[1:-1]  # skip 0 and L
        for s in base:
            if len(s_list) < n_poses:
                s_list.append(float(s))

    poses = []
    for i, s in enumerate(s_list):
        x, y, yaw = cl.sample(s)
        yaw += float(rng.normal(0.0, yaw_jitter_rad))
        lateral = float(rng.uniform(-lateral_jitter_m, lateral_jitter_m))
        x_off, y_off = offset_perp(x, y, yaw, lateral)
        pose_seed = deterministic_hash(seed, int(s*1000), i)
        poses.append({
            "pose_id": i,
            "s": float(s),
            "x": float(x_off),
            "y": float(y_off),
            "yaw": float(yaw),
            "pose_seed": int(pose_seed)
        })
    return poses
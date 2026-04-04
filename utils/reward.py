"""
Reward function for F1TenthSACEnv.

Design principles for the action-space comparison paper:
  - Action-space-agnostic: no steering angle, steering rate, or
    action-dependent terms. Smoothness differences between action spaces
    are emergent, not trained — this strengthens the paper's claims.
  - Progress-dominant: at typical speeds, the progress term outweighs
    acceleration penalties by ~30x. The penalties provide a gentle
    smoothness nudge without overriding the "drive fast" objective.
  - Lateral error (e_lat) is computed but NOT used in the reward.
    Centerline tracking emerges from the heading-aligned progress term.

Reward components:
  progress:   w_progress * v * cos(e_head)
  a_long_pen: w_a_long * (a_long / ref_a_long)^2
  a_lat_pen:  w_a_lat  * (a_lat  / ref_a_lat)^2
  time_pen:   w_time   (constant per step)
  crash_pen:  crash_penalty (on collision)
"""

import numpy as np
from typing import Dict, Optional, Tuple

from utils.geometry import project_to_centerline


def compute_reward(
    obs_raw: Dict,
    centerline: np.ndarray,
    cfg: Dict,
    e_lat: Optional[float] = None,
    e_head: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the scalar reward and a detailed breakdown dict.

    Args:
        obs_raw: raw observation dict from the simulator
        centerline: Nx2+ track centerline array
        cfg: vehicle/environment config dict
        e_lat: pre-computed lateral error (if None, computed here)
        e_head: pre-computed heading error (if None, computed here)

    Returns:
        total: scalar reward
        terms: dict with reward decomposition and raw quantities
    """
    rw = cfg.get("reward", {})

    # --- weights ---
    w_progress = float(rw.get("w_progress", 1.0))
    w_a_long   = float(rw.get("w_a_long", -0.1))
    w_a_lat    = float(rw.get("w_a_lat", -0.1))
    w_time     = float(rw.get("w_time", -0.01))
    crash_pen  = float(rw.get("crash_penalty", -10.0))

    # --- reference scales for normalization ---
    ref_a_long = float(rw.get("ref_a_long", 5.0))
    ref_a_lat  = float(rw.get("ref_a_lat", 8.0))

    # --- raw quantities ---
    v      = float(obs_raw.get("speed", 0.0))
    pose   = obs_raw.get("pose", [0.0, 0.0, 0.0])
    a_long = float(obs_raw.get("a_long", 0.0))
    a_lat  = float(obs_raw.get("a_lat", 0.0))
    crash  = bool(obs_raw.get("crash", False))

    # --- centerline projection (reuse if pre-computed) ---
    if e_lat is None or e_head is None:
        e_lat, e_head = project_to_centerline(pose, centerline)

    # --- reward components ---
    r_progress = w_progress * v * np.cos(e_head)
    r_a_long   = w_a_long * (a_long / max(1e-6, ref_a_long)) ** 2
    r_a_lat    = w_a_lat  * (a_lat  / max(1e-6, ref_a_lat)) ** 2
    r_time     = w_time
    r_crash    = crash_pen if crash else 0.0

    total = r_progress + r_a_long + r_a_lat + r_time + r_crash

    # --- terms: everything needed for paper analysis ---
    terms = {
        # reward decomposition
        "total":      float(total),
        "progress":   float(r_progress),
        "a_long_pen": float(r_a_long),
        "a_lat_pen":  float(r_a_lat),
        "time_pen":   float(r_time),
        "crash_pen":  float(r_crash),
        # raw quantities for post-hoc analysis
        "speed":         float(v),
        "a_long":        float(a_long),
        "a_lat":         float(a_lat),
        "heading_error": float(e_head),
        "lateral_error": float(e_lat),
    }

    return float(total), terms
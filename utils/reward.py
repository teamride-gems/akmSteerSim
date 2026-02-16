import numpy as np
from typing import Dict, Tuple

from utils.geometry import project_to_centerline

def compute_reward(obs_raw, centerline, cfg):
    """
    Reward = forward progress
             + accel smoothness penalties
             + time penalty
             + crash penalty

    All weights come from cfg["reward"].
    """
    rw = cfg.get("reward", {})
    w_progress = float(rw.get("w_progress", 1.0))
    w_a_long   = float(rw.get("w_a_long", -0.1))
    w_a_lat    = float(rw.get("w_a_lat", -0.1))
    w_time     = float(rw.get("w_time", -0.01))
    crash_pen  = float(rw.get("crash_penalty", -10.0))

    ref_a_long = float(rw.get("ref_a_long", 5.0))
    ref_a_lat  = float(rw.get("ref_a_lat", 8.0))

    v = float(obs_raw.get("speed", 0.0))
    pose = obs_raw.get("pose", [0.0, 0.0, 0.0])
    _, e_head = project_to_centerline(pose, centerline)

    a_long = float(obs_raw.get("a_long", 0.0))
    a_lat  = float(obs_raw.get("a_lat", 0.0))
    crash  = bool(obs_raw.get("crash", False))
    r_progress = w_progress * v * np.cos(e_head)

    r_along = w_a_long * (a_long / max(1e-6, ref_a_long))**2
    r_alat  = w_a_lat  * (a_lat  / max(1e-6, ref_a_lat))**2

    r_time = w_time

    r_crash = crash_pen if crash else 0.0

    r = r_progress + r_along + r_alat + r_time + r_crash

    return float(r)



# for if we want to policy for sim to real to really avoid walls and stuff
def compute_reward2(obs_raw: Dict, centerline: np.ndarray, cfg: Dict) -> Tuple[float, Dict]:
    rw = cfg.get("reward", {})

    v = float(obs_raw.get("speed", 0.0))
    pose = obs_raw.get("pose", [0.0, 0.0, 0.0])
    _, e_head = project_to_centerline(pose, centerline)

    crash = bool(obs_raw.get("crash", False))

    offtrack = bool(obs_raw.get("offtrack", False))
    if "on_track" in obs_raw:
        offtrack = not bool(obs_raw["on_track"])

    # v_max from config
    v_max = float(cfg.get("v_max", cfg.get("vehicle", {}).get("max_speed_mps", 1.0)))
    v_max = max(1e-6, v_max)

    # clearance from lidar
    clearance = None
    if "lidar_sectors" in obs_raw:
        arr = np.asarray(obs_raw["lidar_sectors"], dtype=float).ravel()
        clearance = float(np.min(arr)) if arr.size else None
    elif "scan" in obs_raw:
        arr = np.asarray(obs_raw["scan"], dtype=float).ravel()
        clearance = float(np.min(arr)) if arr.size else None
    if clearance is None:
        clearance = float(rw.get("min_clear_m", 1.0))

    # weights/params
    w_progress = float(rw.get("w_progress", 1.0))
    w_speed = float(rw.get("w_speed", 0.0))
    speed_power = float(rw.get("speed_power", 1.0))

    w_heading = float(rw.get("w_heading", -1.0))
    crash_penalty = float(rw.get("crash_penalty", -50.0))
    time_penalty = float(rw.get("time_penalty", -0.01))

    w_accel_long = float(rw.get("w_accel_long", 0.0))
    w_accel_lat = float(rw.get("w_accel_lat", 0.0))
    w_steer_rate = float(rw.get("w_steer_rate", 0.0))

    w_clearance = float(rw.get("w_clearance", 0.0))
    min_clear_m = float(rw.get("min_clear_m", 0.6))
    crash_clear_m = float(rw.get("crash_clear_m", 0.25))

    progress_if_offtrack = float(rw.get("progress_if_offtrack", 0.0))

    # 1) progress along track direction
    r_progress_raw = v * float(np.cos(e_head))
    if offtrack:
        r_progress_raw *= progress_if_offtrack
    r_progress = w_progress * r_progress_raw

    # 2) speed bonus (current speed only)
    # normalize to [0,1], then optionally curve it
    v_norm = float(np.clip(v / v_max, 0.0, 1.0))
    r_speed_raw = v_norm ** max(1e-6, speed_power)
    r_speed = w_speed * r_speed_raw

    # 3) heading penalty (magnitude)
    r_heading = w_heading * (-abs(float(e_head)))

    # 4) clearance shaping from lidar
    if clearance >= min_clear_m:
        r_clear_raw = +1.0
    else:
        if clearance <= crash_clear_m:
            r_clear_raw = -2.0
        else:
            frac = (clearance - crash_clear_m) / max(1e-6, (min_clear_m - crash_clear_m))
            r_clear_raw = float(-2.0 + 3.0 * frac)  # [-2, +1]
    r_clear = w_clearance * r_clear_raw

    # 5) smoothness penalties
    a_long = float(obs_raw.get("a_long", 0.0))
    a_lat = float(obs_raw.get("a_lat", 0.0))
    steer_rate = float(obs_raw.get("steer_rate", 0.0))
    r_smooth = -(abs(a_long) * w_accel_long + abs(a_lat) * w_accel_lat + abs(steer_rate) * w_steer_rate)


    r_crash = crash_penalty if crash else 0.0

    total = r_progress + r_speed + r_heading + r_clear + r_smooth + time_penalty + r_crash

    terms = {
        "progress": float(r_progress),
        "speed": float(r_speed),
        "heading": float(r_heading),
        "clearance": float(r_clear),
        "smooth": float(r_smooth),
        "time": float(time_penalty),
        "crash": float(r_crash),
        "v": float(v),
        "v_norm": float(v_norm),
        "e_head": float(e_head),
        "clear_min": float(clearance),
    }
    return float(total), terms

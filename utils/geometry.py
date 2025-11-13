import numpy as np
from typing import Tuple


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


def project_to_centerline(pose: np.ndarray, centerline: np.ndarray) -> Tuple[float, float]:
    """
    Project a pose (x, y, yaw) onto a polyline centerline.
    
    Args:
        pose: [x, y, yaw] where yaw is the heading in radians
        centerline: Nx2 array of (x, y) points forming the track centerline
    
    Returns:
        e_lat: lateral error (positive = right of centerline)
        e_head: heading error in radians (positive = turning right relative to track)
    """
    x, y, yaw = pose
    
    dists = np.sqrt((centerline[:, 0] - x)**2 + (centerline[:, 1] - y)**2)
    idx = np.argmin(dists)
    
    if idx < len(centerline) - 1:
        p1 = centerline[idx]
        p2 = centerline[idx + 1]
    elif idx > 0:
        p1 = centerline[idx - 1]
        p2 = centerline[idx]
    else:
        e_lat = dists[idx]
        e_head = 0.0
        return e_lat, e_head
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    track_heading = np.arctan2(dy, dx)
    

    v_to_point = np.array([x - p1[0], y - p1[1]])
    v_segment = np.array([dx, dy])
    segment_length = np.sqrt(dx**2 + dy**2)
    
    if segment_length > 1e-6:
        e_lat = np.cross(v_segment, v_to_point) / segment_length
    else:
        e_lat = dists[idx]
    
    e_head = yaw - track_heading
    e_head = (e_head + np.pi) % (2 * np.pi) - np.pi
    
    return float(e_lat), float(e_head)



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




def make_info(obs_raw):
    return {
    "crash": bool(obs_raw.get("crash", False))
    }
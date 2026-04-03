import numpy as np
from typing import Tuple


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
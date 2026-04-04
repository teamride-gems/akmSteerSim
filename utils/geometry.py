"""
Geometric utilities for centerline projection.

The primary function `project_to_centerline` computes lateral and heading
error by finding the true closest point on the polyline, not just the
nearest vertex. This avoids discontinuous jumps at vertex boundaries
that would appear as artifacts in steering TV and lateral error metrics.
"""

import numpy as np
from typing import Tuple


def project_to_centerline(pose: np.ndarray, centerline: np.ndarray) -> Tuple[float, float]:
    """
    Project a pose (x, y, yaw) onto a polyline centerline.

    Finds the true closest point across all segments of the polyline,
    including the closing segment (last -> first) for closed tracks.
    Fully vectorized — cost is O(N) with numpy, same asymptotic
    complexity as the nearest-vertex approach.

    Args:
        pose: [x, y, yaw] where yaw is the heading in radians
        centerline: Nx2+ array of (x, y, ...) points forming the track centerline

    Returns:
        e_lat: signed lateral error (cross-track distance).
               Sign follows the cross product convention of the coordinate frame.
        e_head: heading error in radians, wrapped to [-pi, pi].
               Positive means the vehicle is rotated clockwise relative to
               the track tangent direction.
    """
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    n = centerline.shape[0]

    if n < 2:
        dist = np.sqrt((centerline[0, 0] - x) ** 2 + (centerline[0, 1] - y) ** 2)
        return float(dist), 0.0

    # Detect closed track: last point close to first relative to segment spacing
    closing_dist_sq = (
        (centerline[-1, 0] - centerline[0, 0]) ** 2 +
        (centerline[-1, 1] - centerline[0, 1]) ** 2
    )
    diffs = np.diff(centerline[:, :2], axis=0)
    mean_seg_len_sq = float(np.mean(diffs[:, 0] ** 2 + diffs[:, 1] ** 2))
    is_closed = closing_dist_sq < mean_seg_len_sq * 9.0  # 3x average segment length

    # Build segment start/end arrays
    if is_closed:
        # Include closing segment: last -> first
        starts = centerline[:, :2]                           # (N, 2)
        ends = np.roll(centerline[:, :2], -1, axis=0)       # (N, 2)
    else:
        starts = centerline[:-1, :2]                         # (N-1, 2)
        ends = centerline[1:, :2]                            # (N-1, 2)

    # Vectorized projection onto all segments
    seg = ends - starts                                      # segment vectors
    seg_len_sq = seg[:, 0] ** 2 + seg[:, 1] ** 2            # |seg|^2

    # Vector from segment start to query point
    to_pt = np.array([x, y]) - starts                        # (M, 2)

    # Projection parameter t, clamped to [0, 1]
    # Avoid division by zero for degenerate segments
    safe_len_sq = np.maximum(seg_len_sq, 1e-12)
    t = (to_pt[:, 0] * seg[:, 0] + to_pt[:, 1] * seg[:, 1]) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)

    # Closest points on each segment
    closest = starts + t[:, None] * seg                      # (M, 2)

    # Squared distances to closest points
    dx = x - closest[:, 0]
    dy = y - closest[:, 1]
    dist_sq = dx * dx + dy * dy

    # Best segment
    best = int(np.argmin(dist_sq))

    # Track heading from best segment
    seg_dx = float(seg[best, 0])
    seg_dy = float(seg[best, 1])
    track_heading = np.arctan2(seg_dy, seg_dx)

    # Signed lateral error via cross product: seg x to_point / |seg|
    seg_len = np.sqrt(float(seg_len_sq[best]))
    if seg_len > 1e-6:
        e_lat = (seg_dx * float(to_pt[best, 1]) - seg_dy * float(to_pt[best, 0])) / seg_len
    else:
        e_lat = float(np.sqrt(dist_sq[best]))

    # Heading error, wrapped to [-pi, pi]
    e_head = yaw - track_heading
    e_head = (e_head + np.pi) % (2 * np.pi) - np.pi

    return float(e_lat), float(e_head)
import numpy as np


def _wrap_pi(a: float) -> float:
    """
    Wrap angle 'a' into [-pi, pi].
    """
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a


def project_to_centerline(pose: np.ndarray, centerline: np.ndarray):
    """
    Project the car pose onto the track centerline.

    Parameters
    ----------
    pose : array-like, shape (3,)
        [x, y, yaw] of the car in world frame.
    centerline : array-like, shape (N,2) or (N,3)
        Track waypoints [x, y] or [x, y, theta].

    Returns
    -------
    e_lat : float
        Signed lateral error (left/right of track).
    e_head : float
        Heading error (car yaw - track heading) wrapped to [-pi, pi].
    """
    x, y, yaw = pose
    xy = centerline[:, :2]

    # Find closest waypoint
    diffs = xy - np.array([x, y])
    dists = np.einsum("ij,ij->i", diffs, diffs)
    idx = int(np.argmin(dists))

    # Track heading at closest point
    if centerline.shape[1] >= 3:
        track_theta = centerline[idx, 2]
    else:
        # finite-difference tangent from neighbors
        i0 = max(0, idx - 1)
        i1 = min(len(xy) - 1, idx + 1)
        vec = xy[i1] - xy[i0]
        track_theta = np.arctan2(vec[1], vec[0])

    # Tangent & normal vectors
    t = np.array([np.cos(track_theta), np.sin(track_theta)])
    n = np.array([-t[1], t[0]])

    # Lateral error is projection onto normal
    e_lat = float(np.dot(diffs[idx], n))
    e_head = float(_wrap_pi(yaw - track_theta))
    return e_lat, e_head

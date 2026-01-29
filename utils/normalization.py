import numpy as np


class StateNormalizer:
    """
    Normalizes the state vector into roughly [-1, 1] ranges
    using physically meaningful reference values.

    Expected state layout (from make_state):
        [0] v        : forward speed (m/s)
        [1] a_long   : longitudinal acceleration (m/s^2)
        [2] d        : steering angle (rad)
        [3] r        : yaw rate (rad/s)
        [4] e_head   : heading error (rad)
        [5] e_lat    : lateral error (m)
        [6] a_lat    : lateral acceleration (m/s^2)
        [7:] lidar   : lidar sector minimum distances (m)
    """

    def __init__(self, cfg):
        # --- Velocity ---
        self.v_max = float(cfg["v_max"])

        # --- Steering (RADIAN SAFE) ---
        # Prefer radians if provided (most F1TENTH configs do this)
        if "delta_max" in cfg:
            self.d_max = float(cfg["delta_max"])   # radians
        else:
            # Fallback: degrees -> radians
            self.d_max = np.deg2rad(float(cfg["steer_max_deg"]))

        # --- Accelerations (convert g -> m/s^2) ---
        self.a_long_ref = float(cfg["aref_long_g"]) * 9.81
        self.a_lat_ref  = float(cfg["aref_lat_g"])  * 9.81

        # --- Errors ---
        self.e_head_max = np.pi          # radians
        self.e_lat_max  = 2.0            # meters (track-width scale)

        # --- Lidar ---
        self.lidar_max = float(cfg["lidar"]["clip_max_m"])

    def __call__(self, state):
        """
        Normalize a raw state vector.
        """
        s = np.asarray(state, dtype=float).copy()

        # Speed
        s[0] /= max(1e-6, self.v_max)

        # Longitudinal acceleration
        s[1] /= max(1e-6, self.a_long_ref)

        # Steering angle
        s[2] /= max(1e-6, self.d_max)

        # Yaw rate (soft normalization)
        s[3] /= 10.0

        # Heading error
        s[4] /= self.e_head_max

        # Lateral error
        s[5] /= self.e_lat_max

        # Lateral acceleration
        s[6] /= max(1e-6, self.a_lat_ref)

        # Lidar sectors
        s[7:] /= max(1e-6, self.lidar_max)

        return s

    def normalize(self, state):
        """
        Alias for __call__ (for readability / compatibility).
        """
        return self(state)

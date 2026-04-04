"""
State vector normalization for F1TenthSACEnv.

Normalizes each feature to roughly [-1, 1] using physically
meaningful reference values derived from the vehicle config.

Expected state layout (from state_processing.make_state):
    [0] v        : forward speed (m/s)
    [1] a_long   : longitudinal acceleration (m/s²)
    [2] d        : steering angle (rad)
    [3] r        : yaw rate (rad/s)
    [4] e_head   : heading error (rad)
    [5] e_lat    : lateral error (m)
    [6] a_lat    : lateral acceleration (m/s²)
    [7:] lidar   : lidar sector minimum distances (m)
"""

import numpy as np


class StateNormalizer:

    def __init__(self, cfg):
        # --- Speed ---
        self.v_max = float(cfg.get("v_max", cfg.get("vehicle", {}).get("max_speed_mps", 5.0)))

        # --- Steering angle ---
        delta_max = cfg.get("delta_max", None)
        if delta_max is None:
            delta_max = cfg.get("vehicle", {}).get("max_steer_rad", 0.4189)
        self.d_max = float(delta_max)

        # --- Accelerations ---
        # Reference values for normalization. These are in m/s², NOT g-units.
        reward_cfg = cfg.get("reward", {})
        self.a_long_ref = float(reward_cfg.get("ref_a_long", 5.0))
        self.a_lat_ref = float(reward_cfg.get("ref_a_lat", 8.0))

        # --- Yaw rate ---
        # Derived from Ackermann kinematics: r_max = v_max * tan(delta_max) / wheelbase
        wheelbase = float(cfg.get("vehicle", {}).get("wheelbase", cfg.get("wheelbase", 0.33)))
        self.r_max = self.v_max * np.tan(self.d_max) / max(wheelbase, 1e-6)

        # --- Errors ---
        self.e_head_max = np.pi     # radians (maximum possible heading error)
        self.e_lat_max = 2.0        # meters (approximate half-track-width)

        # --- Lidar ---
        self.lidar_max = float(cfg.get("lidar", {}).get("clip_max_m", 10.0))

    def __call__(self, state):
        s = np.asarray(state, dtype=float).copy()

        s[0] /= max(1e-6, self.v_max)         # speed -> [0, 1]
        s[1] /= max(1e-6, self.a_long_ref)     # a_long -> ~[-1, 1]
        s[2] /= max(1e-6, self.d_max)          # steering -> [-1, 1]
        s[3] /= max(1e-6, self.r_max)          # yaw rate -> ~[-1, 1]
        s[4] /= self.e_head_max                # heading error -> [-1, 1]
        s[5] /= self.e_lat_max                 # lateral error -> ~[-1, 1]
        s[6] /= max(1e-6, self.a_lat_ref)      # a_lat -> ~[-1, 1]
        s[7:] /= max(1e-6, self.lidar_max)     # lidar -> [~0, 1]

        return s

    def normalize(self, state):
        """Alias for __call__."""
        return self(state)
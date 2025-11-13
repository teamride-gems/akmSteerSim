import numpy as np


class StateNormalizer:
    def __init__(self, cfg):
        self.v_max = cfg["v_max"]
        self.d_max = np.deg2rad(cfg["steer_max_deg"])
        self.a_long_ref = cfg["aref_long_g"] * 9.81
        self.a_lat_ref = cfg["aref_lat_g"] * 9.81
        self.e_head_max = np.pi
        self.e_lat_max = 2.0
        self.lidar_max = cfg["lidar"]["clip_max_m"]


    def __call__(self, state):
        s = state.copy().astype(float)
        s[0] = s[0] / max(1e-6, self.v_max)
        s[1] = s[1] / max(1e-6, self.a_long_ref) 
        s[2] = s[2] / max(1e-6, self.d_max) 
        s[3] = s[3] / 10.0 
        s[4] = s[4] / self.e_head_max 
        s[5] = s[5] / self.e_lat_max 
        s[6] = s[6] / max(1e-6, self.a_lat_ref) 
        s[7:] = s[7:] / self.lidar_max 
        return s

    def normalize(self, state):
        """Alias for __call__ to support both calling conventions."""
        return self(state)
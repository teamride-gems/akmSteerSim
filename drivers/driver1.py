import numpy as np


class ActionMapper:
    def __init__(self, cfg):
        self.v_min = cfg["v_min"]
        self.v_max = cfg["v_max"]
        self.d_min = np.deg2rad(cfg["steer_min_deg"])
        self.d_max = np.deg2rad(cfg["steer_max_deg"])


    def __call__(self, a_norm):
        a = np.clip(np.asarray(a_norm, dtype=float), -1.0, 1.0)
        v = self.v_min + (a[0] + 1.0) * 0.5 * (self.v_max - self.v_min)
        d = self.d_min + (a[1] + 1.0) * 0.5 * (self.d_max - self.d_min)
        return float(v), float(d)
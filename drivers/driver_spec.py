import numpy as np

class ActionMapper:
    def __init__(self, cfg):
        root = cfg
        vehicle = root.get("vehicle", {})
        sim = root.get("sim", {})

        # speed limits (prefer explicit v_min/v_max; fallback to vehicle block; default v_min=0)
        self.v_min = float(root.get("v_min", vehicle.get("min_speed_mps", 0.0)))
        self.v_max = float(root.get("v_max", vehicle.get("max_speed_mps")))

        if self.v_max is None:
            raise KeyError("Need v_max (or vehicle.max_speed_mps) in config")

        # steering limits
        # prefer delta_max; fallback to vehicle.max_steer_rad; fallback to steer_max_deg
        delta_max = root.get("delta_max", None)
        if delta_max is None:
            delta_max = vehicle.get("max_steer_rad", None)
        if delta_max is None and "steer_max_deg" in root:
            delta_max = np.deg2rad(float(root["steer_max_deg"]))

        if delta_max is None:
            raise KeyError("Need steering limit: delta_max OR vehicle.max_steer_rad OR steer_max_deg")

        self.delta_max = float(delta_max)
        self.delta_min = float(root.get("delta_min", -self.delta_max))

    def __call__(self, a_norm):
        a = np.clip(np.asarray(a_norm, dtype=float), -1.0, 1.0)
        v = self.v_min + (a[0] + 1.0) * 0.5 * (self.v_max - self.v_min)
        d = self.delta_min + (a[1] + 1.0) * 0.5 * (self.delta_max - self.delta_min)
        return float(v), float(d)

    # optional compatibility shim if your env calls map_action()
    def map_action(self, action):
        return self(action)

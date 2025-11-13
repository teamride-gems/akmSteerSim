# drivers/driver_spec.py
from dataclasses import dataclass
import numpy as np

@dataclass
class ActionBounds:
    steer_min: float = -0.4189   # ~-24 deg
    steer_max: float =  0.4189   # ~+24 deg
    speed_min: float = 0.0
    speed_max: float = 6.0

class ActionMapper:
    def __init__(self, cfg_or_bounds):
        """Initialize with either a config dict or ActionBounds object."""
        if isinstance(cfg_or_bounds, dict):
            # Extract from config
            self.bounds = ActionBounds(
                steer_min=-cfg_or_bounds.get("delta_max", 0.4189),
                steer_max=cfg_or_bounds.get("delta_max", 0.4189),
                speed_min=cfg_or_bounds.get("vehicle", {}).get("min_speed_mps", 0.0),
                speed_max=cfg_or_bounds.get("v_max", 6.0)
            )
        else:
            self.bounds = cfg_or_bounds or ActionBounds()

    def _denorm(self, x: float, lo: float, hi: float) -> float:
        x = float(np.clip(x, -1.0, 1.0))
        return lo + (x + 1.0) * 0.5 * (hi - lo)

    def map_action(self, action):
        """Map normalized action [-1,1]^2 to (v, delta)."""
        a = np.asarray(action, dtype=np.float32).flatten()
        if a.shape[0] < 2:
            raise ValueError(f"Action must be [steer_norm, speed_norm], got shape {a.shape}")
        steer = self._denorm(a[0], self.bounds.steer_min, self.bounds.steer_max)
        speed = self._denorm(a[1], self.bounds.speed_min, self.bounds.speed_max)
        return float(speed), float(steer)  # Return (v, delta)

    def __call__(self, action) -> dict:
        """Legacy interface returning dict."""
        speed, steer = self.map_action(action)
        return {"steer": steer, "speed": speed}

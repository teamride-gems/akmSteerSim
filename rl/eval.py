import numpy as np
from .geometry import project_to_centerline


# r = v*cos(e_head) + λ_lat*(a_lat/aref_lat)^2 + λ_long*(a_long/aref_long)^2 + λ_time + λ_crash*(crash)


def compute_reward(obs_raw, centerline, cfg):
v = float(obs_raw["speed"])
_, e_head = project_to_centerline(obs_raw["pose"], centerline)


a_long = float(obs_raw.get("a_long", 0.0))
a_lat = float(obs_raw.get("a_lat", 0.0))
crash = bool(obs_raw.get("crash", False))


lam = cfg["lambda"]
aref_x = cfg["aref_long_g"] * 9.81
aref_y = cfg["aref_lat_g"] * 9.81


r_align = v * np.cos(e_head)
r_along = lam["a_long"] * (a_long / max(1e-6, aref_x))**2
r_alat = lam["a_lat"] * (a_lat / max(1e-6, aref_y))**2
r_time = lam["time"]
r_crash = lam["crash"] if crash else 0.0


r = r_align + r_along + r_alat + r_time + r_crash
terms = {
"align": float(r_align),
"a_long": float(r_along),
"a_lat": float(r_alat),
"time": float(r_time),
"crash": float(crash),
}
return float(r), terms
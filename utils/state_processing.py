import numpy as np
from .geometry import project_to_centerline


# Split ±105° into N sectors and take min distance per sector (after clipping & outlier removal)


def lidar_to_sectors(scan, cfg):
clip_min = cfg["lidar"]["clip_min_m"]
clip_max = cfg["lidar"]["clip_max_m"]
sectors = cfg["lidar"]["sectors"]


scan = np.asarray(scan, dtype=float)
scan = np.clip(scan, clip_min, clip_max)


# take the middle ±FOV and split into equal sectors
fov = cfg["lidar"]["fov_deg"]
n = scan.size
mid = n // 2
half = int(n * (fov / 360.0) / 2)
window = scan[mid - half : mid + half]


splits = np.array_split(window, sectors)
mins = []
q = cfg["lidar"].get("outlier_quantile", 0.995)
for seg in splits:
if seg.size == 0:
mins.append(clip_max)
continue
# Drop extreme outliers (e.g., stray max returns)
hi = np.quantile(seg, q)
seg = seg[seg <= hi]
mins.append(np.min(seg) if seg.size else clip_max)
return np.asarray(mins, dtype=float)




def make_state(obs_raw, centerline, cfg):
# Required raw inputs from sim: pose(x,y,yaw), speed, steer, yaw_rate, a_long, a_lat, scan
x, y, yaw = obs_raw["pose"]
v = float(obs_raw["speed"]) # forward speed
d = float(obs_raw["steer"]) # steering angle
r = float(obs_raw.get("yaw_rate", 0.0))
a_long = float(obs_raw.get("a_long", 0.0))
a_lat = float(obs_raw.get("a_lat", 0.0))


# Centerline errors
e_lat, e_head = project_to_centerline(np.array([x, y, yaw]), centerline)


# LiDAR → 21-mins
lidar_mins = lidar_to_sectors(obs_raw["scan"], cfg)


# Final state vector (order fixed)
state = np.concatenate([
[v, a_long, d, r, e_head, e_lat, a_lat],
lidar_mins,
])
return state




def make_info(obs_raw):
return {
"crash": bool(obs_raw.get("crash", False))
}
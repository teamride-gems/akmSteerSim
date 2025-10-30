import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml


from utils.state_processing import make_state, make_info, lidar_to_sectors
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from drivers.driver_spec import ActionMapper


# NOTE: We assume you have a racecar sim class from f1tenth_gym, e.g., RacecarEnv
# If the API differs, adapt in reset()/step() where noted.


class F1TenthSACEnv(gym.Env):
metadata = {"render_modes": ["human"], "render_fps": 30}


def __init__(self, vehicle_cfg: str, track_centerline_csv: str, render_mode=None):
super().__init__()
self.cfg = yaml.safe_load(Path(vehicle_cfg).read_text())
self.centerline = np.loadtxt(track_centerline_csv, delimiter=",", ndmin=2)


# Observation = [v, a_long, delta, yaw_rate, e_head, e_lat, a_lat, 21 lidar mins]
self.n_lidar = self.cfg["lidar"]["sectors"]
self.obs_dim = 7 + self.n_lidar


# Actions are normalized in [-1, 1]^2 then mapped to (v, δ)
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)


self.mapper = ActionMapper(self.cfg)
self.normalizer = StateNormalizer(self.cfg)


# ---- hook up the underlying f1tenth_gym here ----
# self.sim = RacecarEnv(map="Spielberg", body_dynamics="kinematic")
self.sim = None # <-- replace with real sim instance
self.render_mode = render_mode
self.t = 0


def reset(self, seed=None, options=None):
super().reset(seed=seed)
self.t = 0
# obs_raw = self.sim.reset() # (pose, speed, scan, etc.)
obs_raw = {
"pose": np.zeros(3), # x,y,yaw
"speed": 0.0,
"scan": np.ones(1080), # placeholder
"steer": 0.0,
"yaw_rate": 0.0,
"a_long": 0.0,
"a_lat": 0.0,
"crash": False,
}
state = make_state(obs_raw, self.centerline, self.cfg)
obs = self.normalizer(state)
info = make_info(obs_raw)
return obs.astype(np.float32), info


def step(self, action):
self.t += 1
v_cmd, delta_cmd = self.mapper(action)
# obs_raw_next, reward_native, terminated, truncated, info_sim = self.sim.step([delta_cmd, v_cmd])
# ---- placeholder mock (replace with sim outputs) ----
obs_raw_next = {
"pose": np.zeros(3),
"speed": float(v_cmd),
"scan": np.ones(1080),
"steer": float(delta_cmd),
"yaw_rate": 0.0,
"a_long": 0.0,
"a_lat": 0.0,
"crash": False,
}
reward, terms = compute_reward(obs_raw_next, self.centerline, self.cfg)
obs = self.normalizer(make_state(obs_raw_next, self.centerline, self.cfg)).astype(np.float32)


terminated = bool(terms["crash"]) # end episode on crash
truncated = False
info = {"native_terms": terms}
return obs, float(reward), terminated, truncated, info


def render(self):
pass


def close(self):
pass
import gymnasium as gym
import gym as legacy_gym  # used only for the F110Env
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml
import gym as legacy_gym
import f110_gym 
from utils.state_processing import make_state, make_info
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from drivers.driver1 import ActionMapper


class F1TenthSACEnv(gym.Env):
    def __init__(
        self,
        vehicle_cfg: str = "configs/vehicle.yaml",
        track_centerline_csv: str = "tracks/test-track-1/YasMarina_centerline.csv",
        map_yaml: str = "tracks/test-track-1/YasMarina_map.yaml",
        render_mode=None,
    ):
        super().__init__()
        self.cfg_path = Path(vehicle_cfg)
        self.centerline_path = Path(track_centerline_csv)
        self.map_yaml_path = Path(map_yaml)

        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.centerline = np.loadtxt(self.centerline_path, delimiter=",", ndmin=2)

        # Observation = [7 core state vars + N lidar sectors]
        self.n_lidar = self.cfg["lidar"]["sectors"]
        self.obs_dim = 7 + self.n_lidar

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.mapper = ActionMapper(self.cfg)
        self.normalizer = StateNormalizer(self.cfg)

        # ---- create the underlying F110Env (from f110-gym) ----
        # Use the *base* path (no extension); f110-gym will append ".png"
        map_base = self.map_yaml_path.with_suffix("")  # drops ".yaml" -> "…/YasMarina_map"

        self.sim = legacy_gym.make(
            "f110-v0",
            map=str(map_base),
            map_ext=".png",
            num_agents=1,
        )

        # Default starting pose: (x, y, yaw)
        self._start_poses = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        self.render_mode = render_mode
        self.t = 0

        self._last_v = 0.0
        self._last_yaw_rate = 0.0



    # ---------- helpers to convert F110Env obs -> our obs_raw ----------

    def _obs_to_raw(self, obs_dict):
        """
        F110Env obs is a dict with batched values (num_agents, ...).
        We pull out ego index 0 and put into our obs_raw format.
        """
        # index of “our” car
        idx = 0

        # lidar scan: shape (num_agents, num_beams)
        scan = np.asarray(obs_dict["scans"][idx], dtype=float)

        x = float(obs_dict["poses_x"][idx])
        y = float(obs_dict["poses_y"][idx])
        yaw = float(obs_dict["poses_theta"][idx])

        # longitudinal velocity approx: x-velocity of ego
        v = float(obs_dict["linear_vels_x"][idx])
        yaw_rate = float(obs_dict["ang_vels_z"][idx])

        # collisions is 0/1 per agent
        crash = bool(obs_dict["collisions"][idx] > 0.5)

        # we’re not computing true accel yet → 0 for now
        a_long = 0.0
        a_lat = 0.0

        # steering is not directly given by F110Env; start with 0
        steer = 0.0

        obs_raw = {
            "pose": np.array([x, y, yaw], dtype=float),
            "speed": v,
            "scan": scan,
            "steer": steer,
            "yaw_rate": yaw_rate,
            "a_long": a_long,
            "a_lat": a_lat,
            "crash": crash,
        }
        return obs_raw

    # ------------------- Gymnasium API -------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # F110Env reset wants poses: (num_agents, 3)
        res = self.sim.reset(self._start_poses.copy())

        # Some versions return obs directly, others (obs, *stuff)
        if isinstance(res, dict):
            obs_sim = res
        elif isinstance(res, tuple) or isinstance(res, list):
            obs_sim = res[0]
        else:
            # Fallback: assume it's already an obs dict-like
            obs_sim = res

        obs_raw = self._obs_to_raw(obs_sim)
        state = make_state(obs_raw, self.centerline, self.cfg)
        obs = self.normalizer(state)
        info = make_info(obs_raw)
        return obs.astype(np.float32), info



    def step(self, action):
        self.t += 1

        # RL action -> (v_cmd, delta_cmd)
        v_cmd, delta_cmd = self.mapper(action)

        # F110Env expects shape (num_agents, 2) = [[steer, speed]]
        act_sim = np.array([[delta_cmd, v_cmd]], dtype=np.float32)

        obs_sim, _reward_native, done, info_sim = self.sim.step(act_sim)
        obs_raw_next = self._obs_to_raw(obs_sim)

        reward, terms = compute_reward(obs_raw_next, self.centerline, self.cfg)
        obs = self.normalizer(
            make_state(obs_raw_next, self.centerline, self.cfg)
        ).astype(np.float32)

        terminated = bool(done or terms.get("crash", False))
        truncated = False
        info = {"native_terms": terms, "sim_info": info_sim}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # delegate to sim’s renderer
        self.sim.render(mode="human")

    def close(self):
        self.sim.close()

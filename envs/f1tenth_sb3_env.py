import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml

from utils.state_processing import make_state, make_info, lidar_to_sectors
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from drivers.driver_spec import ActionMapper


class F1TenthSACEnv(gym.Env):
    """
    Gymnasium wrapper that:
      - Spawns the f1tenth/f110 simulator (class import preferred; falls back to gym ID)
      - Maps normalized actions in [-1,1]^2 -> (v, δ) then to sim's expected order ([δ, v] for F110)
      - Builds your state vector: [v, a_long, δ, yaw_rate, e_head, e_lat, a_lat, lidar(21 mins)]
      - Computes reward r = v*cos(e_head) + penalties (a_long^2, a_lat^2, time) + crash penalty
    """
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
        self.render_mode = render_mode
        self._api = None
        self._dt = 1.0 / 30.0  # sane default; will try to infer if the sim exposes it
        self._last_for_rates = None  # store last (t, x, y, yaw, v) to finite-diff rates

        # Preferred: direct class import (works regardless of Gym vs Gymnasium registry)
        self.sim = None
        try:
            from f110_gym.envs import F110Env  # type: ignore
            # Common kwargs; tweak if your fork needs different names
            self.sim = F110Env(map="columbia_simple", num_agents=1)
            self._api = "class"
        except Exception:
            # Fallback: use the registered ID (this is registered under classic gym)
            try:
                import gym as gym_legacy  # classic gym, not gymnasium
                self.sim = gym_legacy.make("f110-v0")
                self._api = "id"
            except Exception as e:
                raise RuntimeError(
                    "Could not create F110 sim via class or env ID. "
                    "Check that f1tenth_gym is installed (pip install -e ./f1tenth_gym) "
                    "and that 'f110_gym' is importable."
                ) from e

        # Try to learn dt if the sim exposes it (nice-to-have)
        for key in ("dt", "_dt", "time_step", "timestep"):
            if hasattr(self.sim, key) and isinstance(getattr(self.sim, key), (float, int)):
                val = float(getattr(self.sim, key))
                if val > 0:
                    self._dt = val
                    break

    # ------------- sim <-> wrapper glue helpers -------------

    def _pack_obs_dict(self, d):
        """
        Convert a heterogeneous dict from various F110 forks to a unified raw-obs dict
        our pipeline expects. Missing signals are filled later via finite differences.
        """
        # pose
        if "pose" in d and len(np.asarray(d["pose"]).ravel()) >= 3:
            pose = np.asarray(d["pose"], dtype=float).ravel()
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        else:
            x = float(d.get("x", 0.0))
            y = float(d.get("y", 0.0))
            yaw = float(d.get("yaw", d.get("theta", 0.0)))

        # speed & steer
        v = float(d.get("speed", d.get("v", 0.0)))
        steer = float(d.get("steer", d.get("delta", 0.0)))

        # lidar
        scan = d.get("scan", d.get("lidar", None))
        if scan is None:
            # some forks use 'ranges'
            scan = d.get("ranges", np.ones(1080, dtype=float))
        scan = np.asarray(scan, dtype=float).ravel()

        return {
            "pose": np.array([x, y, yaw], dtype=float),
            "speed": v,
            "scan": scan,
            "steer": steer,
            "yaw_rate": float(d.get("yaw_rate", d.get("r", 0.0))),
            "a_long": float(d.get("a_long", d.get("ax", 0.0))),
            "a_lat": float(d.get("a_lat", d.get("ay", 0.0))),
            "crash": bool(d.get("crash", d.get("done", False))),
        }

    def _extract_obs(self, sim_obs):
        """
        Handle common F110 observation shapes:
          - dict with keys ('scan', 'x','y','theta'/'yaw', 'speed'/'v', 'delta'/'steer', ...)
          - tuple/list where index 0 is a dict
          - np.ndarray (rare): we only pull lidar from it and leave pose zeros
        """
        if isinstance(sim_obs, dict):
            return self._pack_obs_dict(sim_obs)

        if isinstance(sim_obs, (list, tuple)) and len(sim_obs) > 0 and isinstance(sim_obs[0], dict):
            return self._pack_obs_dict(sim_obs[0])

        if isinstance(sim_obs, (list, tuple)) and len(sim_obs) > 0 and isinstance(sim_obs[0], (np.ndarray, list)):
            # Heuristic: [scan, ...] with no pose — keep zeros for pose/speeds
            scan = np.asarray(sim_obs[0], dtype=float).ravel()
            return {
                "pose": np.zeros(3, dtype=float),
                "speed": 0.0,
                "scan": scan,
                "steer": 0.0,
                "yaw_rate": 0.0,
                "a_long": 0.0,
                "a_lat": 0.0,
                "crash": False,
            }

        if isinstance(sim_obs, np.ndarray):
            # Only lidar known
            scan = np.asarray(sim_obs, dtype=float).ravel()
            return {
                "pose": np.zeros(3, dtype=float),
                "speed": 0.0,
                "scan": scan,
                "steer": 0.0,
                "yaw_rate": 0.0,
                "a_long": 0.0,
                "a_lat": 0.0,
                "crash": False,
            }

        # Fallback: zeros
        return {
            "pose": np.zeros(3, dtype=float),
            "speed": 0.0,
            "scan": np.ones(1080, dtype=float),
            "steer": 0.0,
            "yaw_rate": 0.0,
            "a_long": 0.0,
            "a_lat": 0.0,
            "crash": False,
        }

    def _finite_difference_kin(self, obs_raw):
        """
        Fill yaw_rate, a_long, a_lat from last state if missing or zero.
        """
        x, y, yaw = obs_raw["pose"]
        v = obs_raw["speed"]
        now = getattr(self.sim, "t", None)
        t_now = float(now) if isinstance(now, (float, int)) else None

        if self._last_for_rates is None:
            self._last_for_rates = {"t": t_now, "x": x, "y": y, "yaw": yaw, "v": v}
            return obs_raw

        t_prev = self._last_for_rates["t"]
        if t_now is None or t_prev is None:
            dt = self._dt
        else:
            dt = max(1e-3, float(t_now) - float(t_prev))

        dx = x - self._last_for_rates["x"]
        dy = y - self._last_for_rates["y"]
        dyaw = (yaw - self._last_for_rates["yaw"] + np.pi) % (2 * np.pi) - np.pi
        dv = v - self._last_for_rates["v"]_

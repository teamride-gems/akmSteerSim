import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml
import os

from utils.state_processing import make_state
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from drivers.driver_spec import ActionMapper


class F1TenthSACEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vehicle_cfg, track_centerline_csv, render_mode=None, *args, **kwargs):
        if isinstance(vehicle_cfg, dict):
            cfg = vehicle_cfg
        else:
            p = Path(vehicle_cfg)
            if not p.exists():
                raise FileNotFoundError(f"vehicle_cfg path not found: {p}")
            cfg = yaml.safe_load(p.read_text())

        if not cfg:
            raise ValueError("vehicle_cfg is empty/invalid")

        self.cfg = cfg

        cl_path = Path(track_centerline_csv)
        if not cl_path.exists():
            raise FileNotFoundError(f"Centerline CSV not found at {cl_path}")
        self.centerline = np.loadtxt(cl_path, delimiter=",", ndmin=2)

        self.render_mode = render_mode

        # lidar sector count for your state vector
        self.n_lidar = int(self.cfg["lidar"]["sectors"])
        self.obs_dim = 7 + self.n_lidar

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.mapper = ActionMapper(self.cfg)
        self.normalizer = StateNormalizer(self.cfg)

        self._dt = 1.0 / 30.0
        self._last_for_rates = None

        # OPTIONAL: reduce raw scan beams before sectorization (CPU saver)
        # e.g. cfg:
        # lidar:
        #   sectors: 21
        #   raw_beams: 360
        self._raw_beams = int(self.cfg.get("lidar", {}).get("raw_beams", 0))  # 0 = no downsample

        sim_cfg = self.cfg.get("sim", {})
        raw_map_name = sim_cfg.get("map_name", "Sakhir")
        track_name = str(raw_map_name).replace("_map", "").strip()
        map_dir = Path(sim_cfg.get("map_dir", f"assets/f1tenth_racetracks/{track_name}"))
        if not map_dir.exists():
            raise FileNotFoundError(f"Map directory not found: {map_dir}")

        map_stem = f"{track_name}_map"  # <Track>_map.yaml/png
        map_path_no_ext = map_dir / map_stem
        map_yaml_check = map_dir / f"{map_stem}.yaml"
        if not map_yaml_check.exists():
            raise FileNotFoundError(f"Map file not found: {map_yaml_check}")

        try:
            from f110_gym.envs import F110Env  # type: ignore
            self.sim = F110Env(map=str(map_path_no_ext), num_agents=1)
        except Exception as e:
            raise RuntimeError("Could not create F110 sim. Check f110_gym + map paths.") from e

        # best-effort dt inference
        for key in ("dt", "_dt", "time_step", "timestep"):
            if hasattr(self.sim, key) and isinstance(getattr(self.sim, key), (float, int)):
                val = float(getattr(self.sim, key))
                if val > 0:
                    self._dt = val
                    break

        self._step_i = 0

    def _downsample_scan(self, scan: np.ndarray) -> np.ndarray:
        """Optionally downsample raw scan beams to cfg lidar.raw_beams (uniform stride)."""
        scan = np.asarray(scan, dtype=float).ravel()
        if self._raw_beams and self._raw_beams > 0 and scan.size > self._raw_beams:
            stride = max(1, int(np.floor(scan.size / self._raw_beams)))
            scan = scan[::stride]
        return scan

    def _pack_obs_dict(self, d):
        """Support f110_gym keys (arrays per agent)."""

        def first_scalar(x, default=0.0):
            if x is None:
                return float(default)
            a = np.asarray(x)
            if a.size == 0:
                return float(default)
            return float(a.ravel()[0])

        # pose
        if "pose" in d and np.asarray(d["pose"]).size >= 3:
            pose = np.asarray(d["pose"], dtype=float).ravel()
            x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        elif "poses_x" in d and "poses_y" in d:
            x = first_scalar(d.get("poses_x"), 0.0)
            y = first_scalar(d.get("poses_y"), 0.0)
            yaw = first_scalar(d.get("poses_theta", 0.0), 0.0)
        else:
            x = float(d.get("x", 0.0))
            y = float(d.get("y", 0.0))
            yaw = float(d.get("yaw", d.get("theta", 0.0)))

        # speed
        v = None
        for k in ("speed", "v", "linear_vels_x", "linear_vel_x", "vels_x", "vel_x", "speeds"):
            if k in d:
                v = first_scalar(d.get(k), 0.0)
                break
        if v is None:
            v = 0.0

        # steering
        steer = None
        for k in ("steer", "delta", "steering_delta", "steering_deltas", "deltas"):
            if k in d:
                steer = first_scalar(d.get(k), 0.0)
                break
        if steer is None:
            steer = 0.0

        # lidar scan
        scan = None
        if "scans" in d:
            scans = np.asarray(d["scans"], dtype=float)
            scan = scans[0] if scans.ndim >= 2 else scans
        elif "scan" in d:
            scan = np.asarray(d["scan"], dtype=float)
        elif "lidar" in d:
            scan = np.asarray(d["lidar"], dtype=float)
        elif "ranges" in d:
            scan = np.asarray(d["ranges"], dtype=float)

        if scan is None:
            scan = np.ones(1080, dtype=float)

        scan = self._downsample_scan(scan)

        crash = bool(d.get("crash", d.get("done", False)))
        yr = d.get("yaw_rate", d.get("r", d.get("ang_vels_z", 0.0)))
        return {
            "pose": np.array([x, y, yaw], dtype=float),
            "speed": float(v),
            "scan": scan,
            "steer": float(steer),
            "yaw_rate": first_scalar(yr, 0.0),
            "a_long": float(d.get("a_long", d.get("ax", 0.0))),
            "a_lat": float(d.get("a_lat", d.get("ay", 0.0))),
            "crash": crash,
        }

    def _extract_obs(self, sim_obs):
        if isinstance(sim_obs, dict):
            return self._pack_obs_dict(sim_obs)
        if isinstance(sim_obs, (list, tuple)) and len(sim_obs) > 0 and isinstance(sim_obs[0], dict):
            return self._pack_obs_dict(sim_obs[0])
        # fallback
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
        x, y, yaw = obs_raw["pose"]
        v = obs_raw["speed"]
        now = getattr(self.sim, "t", None)
        t_now = float(now) if isinstance(now, (float, int)) else None

        if self._last_for_rates is None:
            self._last_for_rates = {"t": t_now, "x": x, "y": y, "yaw": yaw, "v": v}
            return obs_raw

        t_prev = self._last_for_rates["t"]
        dt = self._dt if (t_now is None or t_prev is None) else max(1e-3, float(t_now) - float(t_prev))

        dyaw = (yaw - self._last_for_rates["yaw"] + np.pi) % (2 * np.pi) - np.pi
        dv = v - self._last_for_rates["v"]

        if abs(obs_raw.get("yaw_rate", 0.0)) < 1e-6:
            obs_raw["yaw_rate"] = dyaw / dt
        if abs(obs_raw.get("a_long", 0.0)) < 1e-6:
            obs_raw["a_long"] = dv / dt
        if abs(obs_raw.get("a_lat", 0.0)) < 1e-6 and v > 0.1:
            obs_raw["a_lat"] = v * obs_raw["yaw_rate"]

        self._last_for_rates = {"t": t_now, "x": x, "y": y, "yaw": yaw, "v": v}
        return obs_raw
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_i = 0
        self._last_for_rates = None
        options = options or {}

        N = self.centerline.shape[0]

        # Fixed spawn for eval if provided
        if "spawn_index" in options:
            i = int(options["spawn_index"])
            i = max(1, min(i, N - 2))
        else:
            # Default behavior: random spawn on centerline
            i = int(self.np_random.integers(1, N - 1))

        x = float(self.centerline[i, 0])
        y = float(self.centerline[i, 1])
        dx = float(self.centerline[i + 1, 0] - self.centerline[i, 0])
        dy = float(self.centerline[i + 1, 1] - self.centerline[i, 1])
        theta = float(np.arctan2(dy, dx))

        poses = np.array([[x, y, theta]], dtype=np.float32)

        sim_obs, _, _, _ = self.sim.reset(poses=poses)
        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        state = make_state(obs_raw, self.centerline, self.cfg)
        state_norm = self.normalizer.normalize(state)

        info = {
            "crash": bool(obs_raw.get("crash", False)),
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),
            "spawn_index": int(i),
        }
        return state_norm.astype(np.float32), info

    def step(self, action):
        self._step_i += 1

        # normalized action -> (v, delta)
        v_target, delta_target = self.mapper.map_action(action)

        sim_action = np.array([[delta_target, v_target]], dtype=float)
        sim_obs, _, done, _ = self.sim.step(sim_action)

        # handle done as array/bool safely
        sim_done = bool(np.asarray(done).ravel()[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)

        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        state = make_state(obs_raw, self.centerline, self.cfg)
        state_norm = self.normalizer.normalize(state)

        reward, reward_terms = compute_reward(obs_raw, self.centerline, self.cfg)

        crash = bool(obs_raw.get("crash", False))
        terminated = bool(crash or sim_done)
        truncated = False

        info = {
            "crash": crash,
            "sim_done": sim_done,
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),
            "steer": float(obs_raw.get("steer", 0.0)),
            "reward_terms": reward_terms,
        }

        return state_norm.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        """WSL2 can hang in pyglet/OpenGL. Try, but fail-open."""
        if self.render_mode != "human":
            return
        try:
            # prefer renderer.flip if present (sometimes less likely to hang)
            r = getattr(self.sim, "renderer", None)
            if r is not None and hasattr(r, "flip"):
                r.flip()
            else:
                self.sim.render()
        except Exception as e:
            if not hasattr(self, "_render_failed"):
                self._render_failed = True
                print("RENDER DISABLED:", repr(e))
            self.render_mode = None

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass

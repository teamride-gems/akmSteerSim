import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml

from utils.state_processing import make_state
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from utils.geometry import project_to_centerline
from utils.action_spaces_utils import (
    get_policy_dim,
    get_action_space_spec,
    raw_action_to_command,
    refresh_action_space_bounds,
    get_speed_bounds,
    get_steering_bounds,
)


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

        # precompute cumulative arc length for progress tracking
        diffs = np.diff(self.centerline[:, :2], axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._cl_cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._track_length = float(self._cl_cumlen[-1])

        self.render_mode = render_mode

        # ---- episode horizon ----
        self._max_steps = int(cfg.get("max_episode_steps", 3000))

        # ---- reset perturbation ----
        reset_cfg = cfg.get("reset", {})
        self._reset_lat_noise = float(reset_cfg.get("lateral_noise_m", 0.0))
        self._reset_head_noise = float(reset_cfg.get("heading_noise_rad", 0.0))

        # ---- observation ablation ----
        # when True, e_head and e_lat are zeroed out in the state vector
        self._ablate_geometry = bool(cfg.get("ablate_centerline_features", False))

        # ---- action space setup ----
        self.action_space_name = str(cfg.get("action_space", "steer_speed"))
        self._robot_config = self._build_robot_config(cfg)
        refresh_action_space_bounds(self._robot_config)

        spec = get_action_space_spec(self.action_space_name)
        self._policy_dim = spec.policy_dim

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_dim,),
            dtype=np.float32,
        )

        self._prev_command = None

        # ---- observation space ----
        self.n_lidar = int(self.cfg["lidar"]["sectors"])
        self.obs_dim = 7 + self.n_lidar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.normalizer = StateNormalizer(self.cfg)

        self._dt = 1.0 / 30.0
        self._last_for_rates = None

        self._raw_beams = int(self.cfg.get("lidar", {}).get("raw_beams", 0))

        # ---- simulator setup ----
        sim_cfg = self.cfg.get("sim", {})
        raw_map_name = sim_cfg.get("map_name", "Sakhir")
        track_name = str(raw_map_name).replace("_map", "").strip()
        map_dir = Path(sim_cfg.get("map_dir", f"assets/f1tenth_racetracks/{track_name}"))
        if not map_dir.exists():
            raise FileNotFoundError(f"Map directory not found: {map_dir}")

        map_stem = f"{track_name}_map"
        map_path_no_ext = map_dir / map_stem
        map_yaml_check = map_dir / f"{map_stem}.yaml"
        if not map_yaml_check.exists():
            raise FileNotFoundError(f"Map file not found: {map_yaml_check}")

        try:
            from f110_gym.envs import F110Env
            self.sim = F110Env(map=str(map_path_no_ext), num_agents=1)
        except Exception as e:
            raise RuntimeError("Could not create F110 sim. Check f110_gym + map paths.") from e

        for key in ("dt", "_dt", "time_step", "timestep"):
            if hasattr(self.sim, key) and isinstance(getattr(self.sim, key), (float, int)):
                val = float(getattr(self.sim, key))
                if val > 0:
                    self._dt = val
                    break

        # ---- per-episode tracking ----
        self._step_i = 0
        self._start_progress = 0.0
        self._prev_progress = 0.0
        self._prev_steer_cmd = 0.0

        print(
            f"[F1TenthSACEnv] action_space={self.action_space_name} "
            f"policy_dim={self._policy_dim} "
            f"obs_dim={self.obs_dim} "
            f"max_steps={self._max_steps} "
            f"ablate_geometry={self._ablate_geometry}"
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_robot_config(cfg: dict) -> dict:
        vehicle = cfg.get("vehicle", {})
        rc = {}

        rc["min_speed"] = float(
            cfg.get("v_min", vehicle.get("min_speed_mps", 0.0))
        )
        rc["max_speed"] = float(
            cfg.get("v_max", vehicle.get("max_speed_mps", 5.0))
        )

        delta_max = cfg.get("delta_max", None)
        if delta_max is None:
            delta_max = vehicle.get("max_steer_rad", None)
        if delta_max is None and "steer_max_deg" in cfg:
            delta_max = np.deg2rad(float(cfg["steer_max_deg"]))
        if delta_max is None:
            delta_max = 0.4189
        delta_max = float(delta_max)
        delta_min = float(cfg.get("delta_min", -delta_max))
        rc["min_steering_angle"] = delta_min
        rc["max_steering_angle"] = delta_max

        wb = vehicle.get("wheelbase", vehicle.get("wheelbase_m",
             cfg.get("wheelbase", cfg.get("wheelbase_m", 0.33))))
        rc["wheelbase"] = float(wb)

        if "max_steering_rate" in vehicle:
            rc["max_steering_rate"] = float(vehicle["max_steering_rate"])
        if "max_acceleration" in vehicle:
            rc["max_acceleration"] = float(vehicle["max_acceleration"])

        for key in (
            "lookahead_min_x", "lookahead_max_x", "lookahead_max_abs_y",
            "bezier_min_x", "bezier_max_x", "bezier_max_abs_y",
            "bezier_end_x", "bezier_min_dx",
            "bezier_num_samples", "bezier_lookahead_distance",
        ):
            if key in cfg:
                rc[key] = float(cfg[key])

        return rc

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _downsample_scan(self, scan: np.ndarray) -> np.ndarray:
        scan = np.asarray(scan, dtype=float).ravel()
        if self._raw_beams and self._raw_beams > 0 and scan.size > self._raw_beams:
            stride = max(1, int(np.floor(scan.size / self._raw_beams)))
            scan = scan[::stride]
        return scan

    def _pack_obs_dict(self, d):
        def first_scalar(x, default=0.0):
            if x is None:
                return float(default)
            a = np.asarray(x)
            if a.size == 0:
                return float(default)
            return float(a.ravel()[0])

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

        v = None
        for k in ("speed", "v", "linear_vels_x", "linear_vel_x", "vels_x", "vel_x", "speeds"):
            if k in d:
                v = first_scalar(d.get(k), 0.0)
                break
        if v is None:
            v = 0.0

        steer = None
        for k in ("steer", "delta", "steering_delta", "steering_deltas", "deltas"):
            if k in d:
                steer = first_scalar(d.get(k), 0.0)
                break
        if steer is None:
            steer = 0.0

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

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _centerline_progress(self, pose) -> float:
        """Return arc-length progress along centerline (handles wrap)."""
        xy = np.array([pose[0], pose[1]])
        dists = np.linalg.norm(self.centerline[:, :2] - xy, axis=1)
        idx = int(np.argmin(dists))
        return float(self._cl_cumlen[idx])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_i = 0
        self._last_for_rates = None
        self._prev_command = None
        self._prev_steer_cmd = 0.0

        # random spawn on centerline
        N = self.centerline.shape[0]
        i = int(self.np_random.integers(1, N - 1))
        x = float(self.centerline[i, 0])
        y = float(self.centerline[i, 1])
        dx = float(self.centerline[i + 1, 0] - self.centerline[i, 0])
        dy = float(self.centerline[i + 1, 1] - self.centerline[i, 1])
        theta = float(np.arctan2(dy, dx))

        # optional reset perturbation
        if self._reset_lat_noise > 0:
            lat_offset = float(self.np_random.uniform(
                -self._reset_lat_noise, self._reset_lat_noise
            ))
            # perpendicular to centerline tangent
            perp_x = -np.sin(theta)
            perp_y = np.cos(theta)
            x += lat_offset * perp_x
            y += lat_offset * perp_y

        if self._reset_head_noise > 0:
            theta += float(self.np_random.uniform(
                -self._reset_head_noise, self._reset_head_noise
            ))

        poses = np.array([[x, y, theta]], dtype=np.float32)

        sim_obs, _, _, _ = self.sim.reset(poses=poses)
        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        state = make_state(obs_raw, self.centerline, self.cfg)
        if self._ablate_geometry:
            state = self._zero_geometry_features(state)
        state_norm = self.normalizer.normalize(state)

        # initialize progress tracking
        self._start_progress = self._centerline_progress(obs_raw["pose"])
        self._prev_progress = self._start_progress

        info = {
            "crash": bool(obs_raw.get("crash", False)),
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),
        }
        return state_norm.astype(np.float32), info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        self._step_i += 1

        # ---- action pipeline ----
        command = raw_action_to_command(
            self.action_space_name,
            action,
            self._robot_config,
            prev_command=self._prev_command,
            dt=self._dt,
            apply_final_constraints=True,
        )

        steering_angle = float(command["steering_angle"])
        speed = float(command["speed"])

        # constraint analysis: compare pre vs post constraint
        pre_steer = float(command.get("pre_constraint_steering", steering_angle))
        pre_speed = float(command.get("pre_constraint_speed", speed))
        steer_clipped = abs(pre_steer - steering_angle) > 1e-6
        speed_clipped = abs(pre_speed - speed) > 1e-6

        # steering rate
        steer_rate = (steering_angle - self._prev_steer_cmd) / self._dt
        self._prev_steer_cmd = steering_angle
        self._prev_command = command

        # ---- simulator step ----
        sim_action = np.array([[steering_angle, speed]], dtype=float)
        sim_obs, _, done, _ = self.sim.step(sim_action)

        sim_done = bool(np.asarray(done).ravel()[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)

        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        state = make_state(obs_raw, self.centerline, self.cfg)
        if self._ablate_geometry:
            state = self._zero_geometry_features(state)
        state_norm = self.normalizer.normalize(state)

        reward, reward_terms = compute_reward(obs_raw, self.centerline, self.cfg)

        # ---- termination logic ----
        crash = bool(obs_raw.get("crash", False))
        terminated = bool(crash or sim_done)
        truncated = (self._step_i >= self._max_steps) and not terminated

        # ---- progress tracking ----
        current_progress = self._centerline_progress(obs_raw["pose"])
        # handle wrap-around
        delta_progress = current_progress - self._prev_progress
        if delta_progress < -self._track_length / 2:
            delta_progress += self._track_length
        elif delta_progress > self._track_length / 2:
            delta_progress -= self._track_length
        self._prev_progress = current_progress

        total_progress = current_progress - self._start_progress
        if total_progress < -self._track_length / 2:
            total_progress += self._track_length

        # ---- centerline errors for logging (always computed, even if ablated from state) ----
        e_lat, e_head = project_to_centerline(obs_raw["pose"], self.centerline)

        # ---- termination reason ----
        if crash:
            term_reason = "crash"
        elif sim_done and not crash:
            term_reason = "sim_done"
        elif truncated:
            term_reason = "timeout"
        else:
            term_reason = "running"

        # ---- info dict: everything needed for paper analysis ----
        info = {
            # episode metadata
            "step": self._step_i,
            "term_reason": term_reason,
            "action_space": self.action_space_name,

            # vehicle state
            "crash": crash,
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),

            # commands
            "steer_cmd": steering_angle,
            "speed_cmd": speed,
            "steer_rate": float(steer_rate),

            # constraint analysis
            "pre_constraint_steer": pre_steer,
            "pre_constraint_speed": pre_speed,
            "steer_clipped": steer_clipped,
            "speed_clipped": speed_clipped,
            "steer_clip_mag": abs(pre_steer - steering_angle),
            "speed_clip_mag": abs(pre_speed - speed),

            # tracking quality
            "lateral_error": float(e_lat),
            "heading_error": float(e_head),
            "min_lidar": float(np.min(obs_raw["scan"])),

            # progress
            "delta_progress": float(delta_progress),
            "total_progress": float(total_progress),
            "normalized_progress": float(total_progress / self._track_length),

            # reward decomposition
            "reward_breakdown": reward_terms,
        }

        return state_norm.astype(np.float32), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation ablation
    # ------------------------------------------------------------------

    def _zero_geometry_features(self, state: np.ndarray) -> np.ndarray:
        """
        Zero out e_head (index 4) and e_lat (index 5) in the state vector.
        State layout: [v, a_long, d, r, e_head, e_lat, a_lat, lidar...]
        """
        state = state.copy()
        state[4] = 0.0  # e_head
        state[5] = 0.0  # e_lat
        return state

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return
        try:
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
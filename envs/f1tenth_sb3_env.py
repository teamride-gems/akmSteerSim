import os
import sys
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
import yaml

from utils.state_processing import make_state
from utils.reward import compute_reward
from utils.normalization import StateNormalizer
from utils.action_spaces_utils import (
    get_policy_dim,
    get_action_space_spec,
    raw_action_to_command,
    refresh_action_space_bounds,
)


STATE_IDX_V = 0
STATE_IDX_A_LONG = 1
STATE_IDX_DELTA = 2
STATE_IDX_R = 3
STATE_IDX_E_HEAD = 4
STATE_IDX_E_LAT = 5
STATE_IDX_A_LAT = 6
STATE_N_SCALARS = 7

# Ablation modes:
#   "exteroceptive" — zero centerline-derived features (e_head, e_lat) only.
#                     Proprioceptive kinematics (r, a_lat) are preserved.
#   "all_geometry"  — zero e_head, e_lat AND r, a_lat.
# The default is "all_geometry" to prevent leaking track geometry
# through a_lat ≈ v * yaw_rate. See audit item M3.
ABLATION_MODE = "all_geometry"


class F1TenthSACEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, vehicle_cfg, track_centerline_csv, render_mode=None):
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
        if self.centerline.ndim != 2 or self.centerline.shape[0] < 2 or self.centerline.shape[1] < 2:
            raise ValueError(
                f"Centerline must be an array of shape (N, >=2); got {self.centerline.shape}"
            )

        # Treat every centerline as a closed loop, including the final segment
        # from the last waypoint back to the first.  F1TENTH track CSVs do not
        # repeat the first waypoint at the end.
        points = self.centerline[:, :2]
        diffs = np.roll(points, -1, axis=0) - points
        seg_lengths = np.linalg.norm(diffs, axis=1)
        if np.any(seg_lengths <= 0.0):
            warnings.warn(
                "Centerline contains zero-length segments. Progress projection may be unstable.",
                stacklevel=2,
            )
        self._cl_seg_lengths = seg_lengths
        self._cl_cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._cl_starts = points
        self._cl_segments = np.roll(points, -1, axis=0) - points
        self._cl_safe_len_sq = np.maximum(
            np.sum(self._cl_segments * self._cl_segments, axis=1), 1e-12
        )
        self._track_length = float(self._cl_cumlen[-1])
        if self._track_length <= 0.0:
            raise ValueError("Track length must be positive.")

        self.render_mode = render_mode
        self._lap_completion_fraction = float(cfg.get("lap_completion_fraction", 1.0))
        if not 0.0 < self._lap_completion_fraction <= 1.0:
            raise ValueError(
                "lap_completion_fraction must be in (0, 1]; got "
                f"{self._lap_completion_fraction}"
            )
        self._configured_max_steps = cfg.get("max_episode_steps")
        self._max_episode_seconds = cfg.get("max_episode_seconds")
        if self._configured_max_steps is not None and self._max_episode_seconds is not None:
            raise ValueError(
                "Configure only one of max_episode_steps or max_episode_seconds, not both."
            )

        reset_cfg = cfg.get("reset", {})
        self._reset_lat_noise = float(reset_cfg.get("lateral_noise_m", 0.0))
        self._reset_head_noise = float(reset_cfg.get("heading_noise_rad", 0.0))

        self._ablate_geometry = bool(cfg.get("ablate_centerline_features", False))

        self.action_space_name = str(cfg.get("action_space", "steer_speed"))
        self._robot_config = self._build_robot_config(cfg)
        refresh_action_space_bounds(self._robot_config)

        spec = get_action_space_spec(self.action_space_name)
        self._policy_dim = spec.policy_dim
        expected_policy_dim = int(get_policy_dim(self.action_space_name))
        if self._policy_dim != expected_policy_dim:
            raise ValueError(
                f"Policy dim mismatch for action space '{self.action_space_name}': "
                f"spec={self._policy_dim}, get_policy_dim={expected_policy_dim}"
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_dim,),
            dtype=np.float32,
        )

        self._prev_command = None
        self._last_policy_observation = None

        self.n_lidar = int(self.cfg["lidar"]["sectors"])
        self.obs_dim = STATE_N_SCALARS + self.n_lidar
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.normalizer = StateNormalizer(self.cfg)

        self._sim_dt = 1.0 / 30.0
        self._action_repeat = int(cfg.get("action_repeat", 1))
        if self._action_repeat <= 0:
            raise ValueError("action_repeat must be a positive integer.")
        self._last_for_rates = None
        self._raw_beams = int(self.cfg.get("lidar", {}).get("raw_beams", 0))
        self._sim_obs_schema_validated = False

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

        # The simulator is commonly installed editable outside this repository.
        # Its @njit(cache=True) decorators otherwise try to create cache files
        # beside that read-only source tree, which can hang at import time.
        numba_cache_dir = Path(
            sim_cfg.get(
                "numba_cache_dir",
                Path(__file__).resolve().parents[1] / ".numba_cache",
            )
        ).resolve()
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
        if "numba" in sys.modules:
            import numba
            if not numba.config.CACHE_DIR:
                numba.config.CACHE_DIR = str(numba_cache_dir)

        try:
            from f110_gym.envs import F110Env
            self.sim = F110Env(map=str(map_path_no_ext), num_agents=1)
        except Exception as e:
            raise RuntimeError("Could not create F110 sim. Check f110_gym + map paths.") from e

        dt_found = False
        for key in ("dt", "_dt", "time_step", "timestep"):
            if hasattr(self.sim, key) and isinstance(getattr(self.sim, key), (float, int)):
                val = float(getattr(self.sim, key))
                if val > 0:
                    self._sim_dt = val
                    dt_found = True
                    break
        if not dt_found:
            warnings.warn(
                f"Could not detect sim timestep from F110Env attributes. "
                f"Falling back to dt={self._sim_dt:.4f}s "
                f"(1/{1.0 / self._sim_dt:.0f} Hz). "
                f"Steering rate and finite-difference kinematics may be inaccurate.",
                stacklevel=2,
            )

        self._dt = self._sim_dt * self._action_repeat

        if self._max_episode_seconds is not None:
            max_episode_seconds = float(self._max_episode_seconds)
            if max_episode_seconds <= 0.0:
                raise ValueError("max_episode_seconds must be positive.")
            self._max_steps = int(np.ceil(max_episode_seconds / self._dt))
        elif self._configured_max_steps is not None:
            self._max_steps = int(self._configured_max_steps)
        else:
            self._max_steps = int(np.ceil(120.0 / self._dt))
        if self._max_steps <= 0:
            raise ValueError("Episode horizon must contain at least one step.")
        self._validate_episode_horizon()

        self._step_i = 0
        self._cumulative_progress = 0.0       # FIX (M7): cumulative delta tracking
        self._prev_progress = 0.0
        self._prev_steer_cmd = 0.0

        self._validate_state_layout()

        print(
            f"[F1TenthSACEnv] action_space={self.action_space_name} "
            f"policy_dim={self._policy_dim} "
            f"obs_dim={self.obs_dim} "
            f"control_dt={self._dt:.4f}s "
            f"sim_dt={self._sim_dt:.4f}s "
            f"action_repeat={self._action_repeat} "
            f"max_steps={self._max_steps} "
            f"lap_fraction={self._lap_completion_fraction:.3f} "
            f"ablate_geometry={self._ablate_geometry}"
        )

    @property
    def track_length(self) -> float:
        return self._track_length

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def simulator_dt(self) -> float:
        return self._sim_dt

    @property
    def action_repeat(self) -> int:
        return self._action_repeat

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    @property
    def lap_completion_fraction(self) -> float:
        return self._lap_completion_fraction

    def _validate_episode_horizon(self) -> None:
        """Reject configurations that cannot complete a lap even at v_max."""
        max_speed = float(self._robot_config["max_speed"])
        if max_speed <= 0.0:
            raise ValueError("Maximum vehicle speed must be positive.")
        minimum_steps = int(np.ceil(
            self._lap_completion_fraction * self._track_length / (max_speed * self._dt)
        ))
        if self._max_steps < minimum_steps:
            configured_seconds = self._max_steps * self._dt
            minimum_seconds = minimum_steps * self._dt
            raise ValueError(
                "Episode horizon is physically incapable of reaching the lap-completion "
                f"threshold: configured={self._max_steps} steps ({configured_seconds:.2f}s), "
                f"minimum={minimum_steps} steps ({minimum_seconds:.2f}s) at "
                f"v_max={max_speed:.2f}m/s on a {self._track_length:.2f}m track."
            )

    @staticmethod
    def _build_robot_config(cfg: dict) -> dict:
        vehicle = cfg.get("vehicle", {})
        rc = {}

        rc["min_speed"] = float(cfg.get("v_min", vehicle.get("min_speed_mps", 0.0)))
        rc["max_speed"] = float(cfg.get("v_max", vehicle.get("max_speed_mps", 5.0)))

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

        wb = vehicle.get(
            "wheelbase",
            vehicle.get("wheelbase_m", cfg.get("wheelbase", cfg.get("wheelbase_m", 0.33))),
        )
        rc["wheelbase"] = float(wb)

        if "max_steering_rate" in vehicle:
            rc["max_steering_rate"] = float(vehicle["max_steering_rate"])
        if "max_acceleration" in vehicle:
            rc["max_acceleration"] = float(vehicle["max_acceleration"])

        for key in (
            "lookahead_min_x",
            "lookahead_max_x",
            "lookahead_max_abs_y",
            "bezier_min_x",
            "bezier_max_x",
            "bezier_max_abs_y",
            "bezier_end_x",
            "bezier_min_dx",
            "bezier_num_samples",
            "bezier_lookahead_distance",
        ):
            if key in cfg:
                rc[key] = float(cfg[key])

        return rc

    def _validate_state_layout(self):
        dummy_obs = {
            "pose": np.zeros(3),
            "speed": 0.0,
            "scan": np.ones(1080),
            "steer": 0.0,
            "yaw_rate": 0.0,
            "a_long": 0.0,
            "a_lat": 0.0,
            "crash": False,
        }
        state = make_state(dummy_obs, self.centerline, self.cfg)
        if state.shape[0] != self.obs_dim:
            raise ValueError(
                f"make_state returned {state.shape[0]} dims but obs_dim={self.obs_dim} "
                f"(STATE_N_SCALARS={STATE_N_SCALARS} + n_lidar={self.n_lidar}). "
                f"State layout constants may be out of sync with make_state()."
            )

    def _downsample_scan(self, scan: np.ndarray) -> np.ndarray:
        scan = np.asarray(scan, dtype=float).ravel()
        if scan.size == 0:
            raise ValueError("Simulator returned an empty lidar scan.")
        if not np.all(np.isfinite(scan)):
            raise ValueError("Simulator returned a non-finite lidar scan.")
        if self._raw_beams and self._raw_beams > 0 and scan.size > self._raw_beams:
            stride = max(1, int(np.floor(scan.size / self._raw_beams)))
            scan = scan[::stride]
        return scan

    def _read_sim_steering(self, ego_idx: int = 0):
        """Read the realized front-wheel steering state from F1TENTH internals.

        The installed F1TENTH observation omits steering even though the
        simulator state stores it at index 2.  Failing loudly is preferable to
        silently substituting zero, because steering is part of the policy
        observation used by every experiment.
        """
        try:
            simulator = getattr(self.sim, "sim")
            agents = getattr(simulator, "agents")
            state = np.asarray(agents[int(ego_idx)].state, dtype=float).ravel()
            if state.size > 2 and np.isfinite(state[2]):
                return float(state[2])
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return None

    def _pack_obs_dict(self, d):
        def first_scalar(x, default=0.0):
            if x is None:
                return float(default)
            a = np.asarray(x)
            if a.size == 0:
                return float(default)
            return float(a.ravel()[0])

        ego_idx = int(first_scalar(d.get("ego_idx"), 0.0))

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
        for k in (
            "steer",
            "delta",
            "steering_angle",
            "steering_angles",
            "steering_delta",
            "steering_deltas",
            "deltas",
        ):
            if k in d:
                steer = first_scalar(d.get(k), 0.0)
                break
        if steer is None:
            steer = self._read_sim_steering(ego_idx)
        if steer is None:
            raise KeyError(
                "Simulator observation does not expose realized steering and the "
                "F1TENTH internal steering state could not be read."
            )

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
            raise KeyError(
                f"Simulator observation is missing lidar data. Available keys: {sorted(d.keys())}"
            )

        scan = self._downsample_scan(scan)

        collision_value = d.get("collisions", d.get("collision", 0.0))
        collision_values = np.asarray(collision_value).ravel()
        if collision_values.size:
            collision_idx = int(np.clip(ego_idx, 0, collision_values.size - 1))
            collision = bool(collision_values[collision_idx])
        else:
            collision = False
        crash_value = d.get("crash", False)
        crash = bool(first_scalar(crash_value, 0.0)) or collision
        yr = d.get("yaw_rate", d.get("r", d.get("ang_vels_z", 0.0)))
        packed = {
            "pose": np.array([x, y, yaw], dtype=float),
            "speed": float(v),
            "scan": scan,
            "steer": float(steer),
            "yaw_rate": first_scalar(yr, 0.0),
            "a_long": float(d.get("a_long", d.get("ax", 0.0))),
            "a_lat": float(d.get("a_lat", d.get("ay", 0.0))),
            "crash": crash,
        }
        if not np.all(np.isfinite(packed["pose"])):
            raise ValueError(f"Simulator returned non-finite pose: {packed['pose']}")
        return packed

    def _extract_obs(self, sim_obs):
        if isinstance(sim_obs, dict):
            obs = self._pack_obs_dict(sim_obs)
        elif isinstance(sim_obs, (list, tuple)) and len(sim_obs) > 0 and isinstance(sim_obs[0], dict):
            obs = self._pack_obs_dict(sim_obs[0])
        else:
            raise TypeError(
                f"Unrecognized simulator observation type: {type(sim_obs).__name__}. "
                "Expected dict or sequence whose first element is a dict. "
                f"Raw value preview: {repr(sim_obs)[:300]}"
            )

        if not self._sim_obs_schema_validated:
            self._sim_obs_schema_validated = True
            required = {"pose", "speed", "scan", "steer", "yaw_rate", "a_long", "a_lat", "crash"}
            missing = [k for k in required if k not in obs]
            if missing:
                raise KeyError(f"Packed simulator observation missing required fields: {missing}")
        return obs

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

    def _project_centerline_state(self, pose):
        """Return arc progress, lateral error, and heading error in one pass."""
        xy = np.asarray(pose[:2], dtype=float)
        to_point = xy - self._cl_starts
        t = np.sum(to_point * self._cl_segments, axis=1) / self._cl_safe_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = self._cl_starts + t[:, None] * self._cl_segments
        dist_sq = np.sum((xy - closest) ** 2, axis=1)
        best = int(np.argmin(dist_sq))

        segment = self._cl_segments[best]
        segment_length = float(self._cl_seg_lengths[best])
        if segment_length > 1e-6:
            e_lat = (
                float(segment[0]) * float(to_point[best, 1])
                - float(segment[1]) * float(to_point[best, 0])
            ) / segment_length
        else:
            e_lat = float(np.sqrt(dist_sq[best]))

        track_heading = float(np.arctan2(segment[1], segment[0]))
        e_head = float((float(pose[2]) - track_heading + np.pi) % (2 * np.pi) - np.pi)
        arc_progress = float(
            self._cl_cumlen[best] + float(t[best]) * self._cl_seg_lengths[best]
        )
        return arc_progress, float(e_lat), e_head

    def _projected_arc_progress(self, pose) -> float:
        return self._project_centerline_state(pose)[0]

    def _spawn_pose(self, idx: int):
        N = self.centerline.shape[0]
        idx = int(np.clip(idx, 1, N - 2))

        x = float(self.centerline[idx, 0])
        y = float(self.centerline[idx, 1])
        dx = float(self.centerline[idx + 1, 0] - self.centerline[idx, 0])
        dy = float(self.centerline[idx + 1, 1] - self.centerline[idx, 1])
        theta = float(np.arctan2(dy, dx))
        return x, y, theta

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_i = 0
        self._last_for_rates = None
        self._prev_command = None
        self._last_policy_observation = None
        self._prev_steer_cmd = 0.0
        self._cumulative_progress = 0.0       # FIX (M7)

        options = options or {}
        N = self.centerline.shape[0]

        if "spawn_index" in options:
            spawn_idx = int(np.clip(options["spawn_index"], 1, N - 2))
        else:
            spawn_idx = int(self.np_random.integers(1, N - 1))

        x, y, theta = self._spawn_pose(spawn_idx)

        if "spawn_index" not in options:
            if self._reset_lat_noise > 0:
                lat_offset = float(self.np_random.uniform(-self._reset_lat_noise, self._reset_lat_noise))
                perp_x = -np.sin(theta)
                perp_y = np.cos(theta)
                x += lat_offset * perp_x
                y += lat_offset * perp_y

            if self._reset_head_noise > 0:
                theta += float(self.np_random.uniform(-self._reset_head_noise, self._reset_head_noise))

        poses = np.array([[x, y, theta]], dtype=np.float32)
        sim_obs, _, _, _ = self.sim.reset(poses=poses)
        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        # Seed the slew-rate limiter and steering-rate metric from the actual
        # reset state so the very first policy action obeys the same physical
        # constraints as every later action.
        self._prev_command = {
            "steering_angle": float(obs_raw["steer"]),
            "speed": float(obs_raw["speed"]),
        }
        self._prev_steer_cmd = float(obs_raw["steer"])

        arc_progress, e_lat, e_head = self._project_centerline_state(obs_raw["pose"])
        state = make_state(
            obs_raw,
            self.centerline,
            self.cfg,
            e_lat=e_lat,
            e_head=e_head,
        )
        if self._ablate_geometry:
            state = self._zero_geometry_features(state)
        state_norm = self.normalizer.normalize(state)

        self._prev_progress = arc_progress
        policy_observation = state_norm.astype(np.float32)
        self._last_policy_observation = policy_observation.copy()

        info = {
            "crash": bool(obs_raw.get("crash", False)),
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),
            "spawn_index": spawn_idx,
            "lateral_error": float(e_lat),
            "heading_error": float(e_head),
        }
        return policy_observation, info

    def step(self, action):
        self._step_i += 1

        if self._last_policy_observation is None:
            raise RuntimeError("step() called before reset().")
        policy_observation = self._last_policy_observation.copy()
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        previous_command = np.array(
            [
                float(self._prev_command["steering_angle"]),
                float(self._prev_command["speed"]),
            ],
            dtype=np.float32,
        )

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

        pre_steer = float(command.get("pre_constraint_steering", steering_angle))
        pre_speed = float(command.get("pre_constraint_speed", speed))
        steer_clipped = abs(pre_steer - steering_angle) > 1e-6
        speed_clipped = abs(pre_speed - speed) > 1e-6

        steer_rate = (steering_angle - self._prev_steer_cmd) / self._dt
        self._prev_steer_cmd = steering_angle
        self._prev_command = command

        sim_action = np.array([[steering_angle, speed]], dtype=float)
        sim_done = False
        internal_steps = 0
        for _ in range(self._action_repeat):
            sim_obs, _, done, _ = self.sim.step(sim_action)
            internal_steps += 1
            sim_done = (
                bool(np.asarray(done).ravel()[0])
                if isinstance(done, (list, tuple, np.ndarray))
                else bool(done)
            )
            if sim_done:
                break

        obs_raw = self._extract_obs(sim_obs)
        obs_raw = self._finite_difference_kin(obs_raw)

        current_progress, e_lat, e_head = self._project_centerline_state(obs_raw["pose"])
        state = make_state(
            obs_raw,
            self.centerline,
            self.cfg,
            e_lat=e_lat,
            e_head=e_head,
        )
        if self._ablate_geometry:
            state = self._zero_geometry_features(state)
        state_norm = self.normalizer.normalize(state)

        crash = bool(obs_raw.get("crash", False))

        # Closed-loop cumulative arc progress makes completion independent of
        # F1TENTH's two-return start-zone toggle semantics.
        delta_progress = current_progress - self._prev_progress
        if delta_progress < -self._track_length / 2:
            delta_progress += self._track_length
        elif delta_progress > self._track_length / 2:
            delta_progress -= self._track_length
        self._prev_progress = current_progress
        self._cumulative_progress += delta_progress

        reward, reward_terms = compute_reward(
            obs_raw,
            self.centerline,
            self.cfg,
            e_lat=e_lat,
            e_head=e_head,
            dt=internal_steps * self._sim_dt,
            delta_progress=delta_progress,
        )

        lap_complete = bool(
            self._cumulative_progress
            >= self._lap_completion_fraction * self._track_length
        )
        terminated = bool(crash or lap_complete or sim_done)
        truncated = (self._step_i >= self._max_steps) and not terminated

        if crash:
            term_reason = "crash"
        elif lap_complete:
            term_reason = "lap_complete"
        elif sim_done and not crash:
            term_reason = "sim_done"
        elif truncated:
            term_reason = "timeout"
        else:
            term_reason = "running"

        next_policy_observation = state_norm.astype(np.float32)
        self._last_policy_observation = next_policy_observation.copy()
        realized_command = np.array(
            [float(obs_raw["steer"]), float(obs_raw["speed"])], dtype=np.float32
        )
        executed_command = np.array([steering_angle, speed], dtype=np.float32)
        pre_constraint_command = np.array([pre_steer, pre_speed], dtype=np.float32)

        info = {
            "step": self._step_i,
            "internal_sim_steps": internal_steps,
            "elapsed_seconds": internal_steps * self._sim_dt,
            "term_reason": term_reason,
            "action_space": self.action_space_name,
            "crash": crash,
            "lap_complete": lap_complete,
            "pose": obs_raw["pose"].copy(),
            "speed": float(obs_raw["speed"]),
            "realized_steer": float(obs_raw["steer"]),
            "a_long": float(obs_raw["a_long"]),
            "a_lat": float(obs_raw["a_lat"]),
            "steer_cmd": steering_angle,
            "speed_cmd": speed,
            "steer_rate": float(steer_rate),
            "pre_constraint_steer": pre_steer,
            "pre_constraint_speed": pre_speed,
            "steer_clipped": steer_clipped,
            "speed_clipped": speed_clipped,
            "steer_clip_mag": abs(pre_steer - steering_angle),
            "speed_clip_mag": abs(pre_speed - speed),
            # Complete action-interface transition for the Gate 0/1 audit.
            "policy_observation": policy_observation,
            "raw_action": raw_action,
            "previous_command": previous_command,
            "pre_constraint_command": pre_constraint_command,
            "executed_command": executed_command,
            "realized_command": realized_command,
            "next_policy_observation": next_policy_observation,
            "limiter_active": bool(steer_clipped or speed_clipped),
            "steer_command_realized_gap": abs(steering_angle - float(obs_raw["steer"])),
            "speed_command_realized_gap": abs(speed - float(obs_raw["speed"])),
            "lateral_error": float(e_lat),
            "heading_error": float(e_head),
            "min_lidar": float(np.min(obs_raw["scan"])),
            "delta_progress": float(delta_progress),
            "total_progress": float(self._cumulative_progress),
            "normalized_progress": float(np.clip(
                self._cumulative_progress / self._track_length, 0.0, 1.0
            )),
            "reward_breakdown": reward_terms,
        }

        return next_policy_observation, float(reward), terminated, truncated, info

    def _zero_geometry_features(self, state: np.ndarray) -> np.ndarray:
        """Zero geometry-leaking features in the ablated observation regime.

        FIX (M3): also zeros yaw_rate (r) and lateral acceleration (a_lat),
        which are proprioceptive proxies for track curvature (a_lat ≈ v * r).
        Controlled by the module-level ABLATION_MODE constant.
        """
        state = state.copy()
        # Always zero centerline-derived features
        state[STATE_IDX_E_HEAD] = 0.0
        state[STATE_IDX_E_LAT] = 0.0
        # Also zero kinematic proxies for geometry
        if ABLATION_MODE == "all_geometry":
            state[STATE_IDX_R] = 0.0
            state[STATE_IDX_A_LAT] = 0.0
        return state

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

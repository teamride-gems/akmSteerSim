"""Regression tests for the environment validity fixes.

These tests use a deterministic in-memory F1TENTH stand-in so they exercise
the complete wrapper without paying the real simulator's JIT startup cost.
The separate integration smoke test covers the installed F1TENTH package.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from envs.f1tenth_sb3_env import F1TenthSACEnv
from utils.reward import compute_reward
from utils.state_processing import lidar_to_sectors


SQUARE_CENTERLINE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)


class _FakeAgent:
    def __init__(self):
        # F1TENTH state: x, y, steer, velocity, yaw, yaw rate, slip.
        self.state = np.zeros(7, dtype=float)


class _FakeSimulator:
    def __init__(self):
        self.agents = [_FakeAgent()]


class _FakeF110Env:
    timestep = 0.1

    def __init__(self, **_kwargs):
        self.sim = _FakeSimulator()
        self.force_collision = False
        self.force_done = False
        self.pose_sequence = []
        self.current_time = 0.0

    def _obs(self):
        state = self.sim.agents[0].state
        return {
            "ego_idx": 0,
            "scans": [np.linspace(1.0, 8.0, 8)],
            "poses_x": [state[0]],
            "poses_y": [state[1]],
            "poses_theta": [state[4]],
            "linear_vels_x": [state[3]],
            "linear_vels_y": [0.0],
            "ang_vels_z": [state[5]],
            "collisions": [float(self.force_collision)],
            "lap_counts": [0.0],
        }

    def reset(self, poses):
        state = self.sim.agents[0].state
        state[:] = 0.0
        state[0], state[1], state[4] = poses[0]
        self.current_time = 0.0
        return self._obs(), self.timestep, False, {}

    def step(self, action):
        state = self.sim.agents[0].state
        state[2] = float(action[0, 0])
        state[3] = float(action[0, 1])
        if self.pose_sequence:
            x, y, yaw = self.pose_sequence.pop(0)
            state[0], state[1], state[4] = x, y, yaw
        self.current_time += self.timestep
        return self._obs(), self.timestep, bool(self.force_done or self.force_collision), {}

    def close(self):
        pass


def _vehicle_config(map_dir: Path, **overrides):
    cfg = {
        "action_space": "steer_speed",
        "max_episode_seconds": 5.0,
        "lap_completion_fraction": 1.0,
        "v_min": 0.0,
        "v_max": 10.0,
        "delta_max": 0.4,
        "vehicle": {
            "wheelbase": 0.33,
            "max_steer_rad": 0.4,
            "min_speed_mps": 0.0,
            "max_speed_mps": 10.0,
            "max_steering_rate": 1.0,
            "max_acceleration": 2.0,
        },
        "lidar": {
            "sectors": 4,
            "fov_deg": 270.0,
            "input_fov_deg": 270.0,
            "clip_min_m": 0.05,
            "clip_max_m": 10.0,
            "outlier_quantile": 1.0,
        },
        "sim": {"map_dir": str(map_dir), "map_name": "Unit_map"},
        "reward": {
            "w_progress": 1.0,
            "w_a_long": -0.1,
            "w_a_lat": -0.1,
            "w_time": -0.01,
            "crash_penalty": -10.0,
            "ref_a_long": 5.0,
            "ref_a_lat": 8.0,
        },
    }
    cfg.update(overrides)
    return cfg


class ValiditySprintTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.centerline_path = root / "Unit_centerline.csv"
        np.savetxt(self.centerline_path, SQUARE_CENTERLINE, delimiter=",")
        (root / "Unit_map.yaml").write_text("image: Unit_map.png\n", encoding="utf-8")

        fake_package = types.ModuleType("f110_gym")
        fake_envs = types.ModuleType("f110_gym.envs")
        fake_envs.F110Env = _FakeF110Env
        fake_package.envs = fake_envs
        self.module_patch = patch.dict(
            sys.modules,
            {"f110_gym": fake_package, "f110_gym.envs": fake_envs},
        )
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        self.temp_dir.cleanup()

    def make_env(self, **overrides):
        cfg = _vehicle_config(Path(self.temp_dir.name), **overrides)
        return F1TenthSACEnv(cfg, self.centerline_path)

    def test_collision_array_drives_crash_penalty_and_termination(self):
        env = self.make_env()
        env.reset(options={"spawn_index": 1})
        env.sim.force_collision = True

        _, reward, terminated, truncated, info = env.step(np.zeros(2))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["crash"])
        self.assertEqual(info["term_reason"], "crash")
        self.assertEqual(info["reward_breakdown"]["crash_pen"], -10.0)
        self.assertLess(reward, -9.0)

    def test_realized_steering_is_read_from_f1tenth_internal_state(self):
        env = self.make_env()
        env.sim.sim.agents[0].state[2] = 0.123

        packed = env._pack_obs_dict(env.sim._obs())

        self.assertAlmostEqual(packed["steer"], 0.123)

    def test_native_270_degree_lidar_scan_is_not_cropped_again(self):
        cfg = _vehicle_config(Path(self.temp_dir.name))
        sectors = lidar_to_sectors(np.arange(1.0, 9.0), cfg)

        np.testing.assert_allclose(sectors, [1.0, 3.0, 5.0, 7.0])

    def test_infeasible_episode_horizon_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "physically incapable"):
            self.make_env(max_episode_seconds=0.3)

    def test_closed_loop_progress_terminates_after_one_lap(self):
        env = self.make_env()
        env.reset(options={"spawn_index": 1})
        env.sim.pose_sequence = [
            (1.0, 1.0, np.pi),
            (0.0, 1.0, -np.pi / 2),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, np.pi / 2),
        ]

        result = None
        for _ in range(4):
            result = env.step(np.zeros(2))
        _, _, terminated, truncated, info = result

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["term_reason"], "lap_complete")
        self.assertAlmostEqual(info["normalized_progress"], 1.0)

    def test_first_action_obeys_steering_and_acceleration_limits(self):
        env = self.make_env()
        env.reset(options={"spawn_index": 1})

        _, _, _, _, info = env.step(np.ones(2))

        self.assertLessEqual(abs(info["steer_rate"]), 1.0 + 1e-9)
        self.assertAlmostEqual(info["steer_cmd"], 0.1)
        self.assertAlmostEqual(info["speed_cmd"], 0.2)
        self.assertTrue(info["steer_clipped"])
        self.assertTrue(info["speed_clipped"])

    def test_running_reward_is_timestep_invariant(self):
        cfg = _vehicle_config(Path(self.temp_dir.name))
        obs = {
            "pose": np.array([0.0, 0.0, 0.0]),
            "speed": 2.0,
            "a_long": 1.0,
            "a_lat": 2.0,
            "crash": False,
        }

        def total_for(dt, steps):
            reward, _ = compute_reward(
                obs,
                SQUARE_CENTERLINE,
                cfg,
                e_lat=0.0,
                e_head=0.0,
                dt=dt,
                delta_progress=obs["speed"] * dt,
            )
            return reward * steps

        self.assertAlmostEqual(total_for(0.01, 100), total_for(0.02, 50))

    def test_configured_horizon_can_reach_every_bundled_track(self):
        cfg = yaml.safe_load(Path("configs/vehicle.yaml").read_text(encoding="utf-8"))
        available_distance = float(cfg["v_max"]) * float(cfg["max_episode_seconds"])
        completion_fraction = float(cfg["lap_completion_fraction"])
        track_root = Path("assets/f1tenth_racetracks")

        too_long = []
        centerline_paths = list(track_root.glob("*/*_centerline.csv"))
        self.assertGreater(len(centerline_paths), 0, "No bundled track centerlines found.")
        for centerline_path in centerline_paths:
            points = np.loadtxt(centerline_path, delimiter=",", ndmin=2)[:, :2]
            closed_diffs = np.roll(points, -1, axis=0) - points
            required_distance = completion_fraction * float(
                np.sum(np.linalg.norm(closed_diffs, axis=1))
            )
            if required_distance > available_distance:
                too_long.append(centerline_path.parent.name)

        self.assertFalse(too_long, f"Episode horizon cannot complete tracks: {too_long}")


if __name__ == "__main__":
    unittest.main()

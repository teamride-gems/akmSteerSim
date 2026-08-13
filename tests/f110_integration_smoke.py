"""Smoke-test the validity contract against the installed F1TENTH simulator.

This is intentionally not named ``test_*.py`` because importing F1TENTH can
trigger a long Numba compilation on a cold machine. Run it explicitly before
launching an experiment:

    python tests/f110_integration_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.common import make_env_for_track


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "vehicle.yaml").read_text(encoding="utf-8"))
    env = make_env_for_track(cfg, "Sakhir")
    try:
        obs, reset_info = env.reset(seed=0, options={"spawn_index": 1})
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        assert env.dt == 0.01, f"Unexpected F1TENTH timestep: {env.dt}"
        assert env.max_episode_steps == 12_000
        assert not reset_info["crash"]

        previous_realized_steer = env._read_sim_steering(0)
        assert previous_realized_steer is not None

        obs, reward, terminated, truncated, first_info = env.step(
            np.ones(env.action_space.shape, dtype=np.float32)
        )

        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert abs(first_info["steer_rate"]) <= cfg["vehicle"]["max_steering_rate"] + 1e-9
        assert first_info["speed_cmd"] <= cfg["vehicle"]["max_acceleration"] * env.dt + 1e-9

        # The installed simulator intentionally buffers two steering commands.
        # Step far enough for the first constrained command to reach dynamics,
        # then verify that our observation reads the realized state rather than
        # the requested command.
        info = first_info
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(
                np.ones(env.action_space.shape, dtype=np.float32)
            )
        realized_steer = env._read_sim_steering(0)
        assert realized_steer is not None
        assert realized_steer != previous_realized_steer
        assert not np.isclose(realized_steer, info["steer_cmd"])
        assert np.isfinite(info["reward_breakdown"]["total"])
        assert len(obs) == 7 + cfg["lidar"]["sectors"]

        raw_obs = env.sim.current_obs
        assert "collisions" in raw_obs, f"Unexpected simulator schema: {sorted(raw_obs)}"
        assert "scans" in raw_obs

        print(
            "F1TENTH integration smoke passed: "
            f"dt={env.dt:.3f}s, horizon={env.max_episode_steps} steps, "
            f"first_steer_cmd={first_info['steer_cmd']:.4f}rad, "
            f"realized_steer={realized_steer:.4f}rad"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

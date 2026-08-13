#!/usr/bin/env python3
"""
Run a heuristic policy in F1TenthSACEnv for visualization and debugging.

WSL2 note: real-time pyglet rendering can hang. This script:
- runs headless by default
- can record rollouts to .npz (replay with replay_rollout_html.py)
- can attempt rendering if you pass --render

Supports all action spaces defined in action_spaces_utils:
  steer_speed, curvature_speed, lookahead_point, bezier
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import time
import numpy as np
import yaml

from utils.action_spaces_utils import get_policy_dim
from rl.common import normalize_track_name, make_env_for_track
from scripts.metrics_logger import EpisodeLogger, StepLogger


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Heuristic policies per action space
# ---------------------------------------------------------------------------

_LAST_STEER = 0.0


def _compute_steer_and_speed(info: dict):
    """
    Shared PD logic using info dict values (physical units, not normalized).
    Returns (steer_cmd, speed_cmd) both in [-1, 1] as raw policy outputs.
    """
    global _LAST_STEER

    e_head = float(info.get("heading_error", 0.0))
    e_lat = float(info.get("lateral_error", 0.0))

    k_head = 0.9
    k_lat = 0.25

    steer_cmd = -(k_head * e_head + k_lat * e_lat)
    steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))

    alpha = 0.25
    steer = (1.0 - alpha) * _LAST_STEER + alpha * steer_cmd
    steer = float(np.clip(steer, -1.0, 1.0))
    _LAST_STEER = steer

    # slow down when deviating
    speed = 0.4
    speed *= float(np.clip(1.0 - 2.5 * abs(e_head) - 0.8 * abs(e_lat), 0.1, 1.0))
    speed = float(np.clip(speed, -1.0, 1.0))

    return steer, speed


def heuristic_steer_speed(info: dict) -> np.ndarray:
    steer, speed = _compute_steer_and_speed(info)
    return np.array([steer, speed], dtype=np.float32)


def heuristic_curvature_speed(info: dict) -> np.ndarray:
    steer, speed = _compute_steer_and_speed(info)
    return np.array([steer, speed], dtype=np.float32)


def heuristic_lookahead_point(info: dict) -> np.ndarray:
    steer, speed = _compute_steer_and_speed(info)
    return np.array([0.0, steer, speed], dtype=np.float32)


def heuristic_bezier(info: dict) -> np.ndarray:
    steer, speed = _compute_steer_and_speed(info)
    return np.array([0.0, steer * 0.5, 0.0, steer * 0.3, speed], dtype=np.float32)


HEURISTIC_POLICIES = {
    "steer_speed": heuristic_steer_speed,
    "curvature_speed": heuristic_curvature_speed,
    "lookahead_point": heuristic_lookahead_point,
    "bezier": heuristic_bezier,
}


def main():
    global _LAST_STEER

    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--track", default=None, help="Track name (default: from vehicle config)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--render", action="store_true", help="Attempt realtime rendering")
    ap.add_argument("--render_every", type=int, default=5)
    ap.add_argument("--record", action="store_true", help="Save rollout to .npz for replay")
    ap.add_argument("--record_path", default="rollouts/rollout.npz")
    ap.add_argument("--lidar_print_every", type=int, default=0, help="0 disables lidar prints")
    args = ap.parse_args()

    cfg_path = ROOT / args.vehicle_cfg
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    vehicle_cfg = load_yaml(str(cfg_path))

    action_space_name = vehicle_cfg.get("action_space", "steer_speed")
    policy_dim = get_policy_dim(action_space_name)

    if action_space_name not in HEURISTIC_POLICIES:
        raise ValueError(
            f"No heuristic policy for '{action_space_name}'. "
            f"Available: {list(HEURISTIC_POLICIES.keys())}"
        )

    heuristic_fn = HEURISTIC_POLICIES[action_space_name]

    # Resolve track
    if args.track:
        track = normalize_track_name(args.track)
    else:
        raw_map = vehicle_cfg.get("sim", {}).get("map_name", "Sakhir_map")
        track = normalize_track_name(raw_map)

    print(f"action_space: {action_space_name} (policy_dim={policy_dim})")
    print(f"track: {track}")
    print(f"render: {args.render}")
    print(f"record: {args.record}")

    render_mode = "human" if args.render else None
    env = make_env_for_track(vehicle_cfg, track, render_mode=render_mode)

    obs, info = env.reset()

    # Loggers
    ep_logger = EpisodeLogger("metrics/episodes.csv")
    step_logger = StepLogger("metrics/trajectory.csv")

    # Recording buffers
    rec_poses = []
    rec_actions = []
    rec_steer_cmds = []
    rec_speed_cmds = []
    rec_steer_rates = []
    rec_e_lats = []
    rec_e_heads = []
    rec_speeds = []
    rec_rewards = []
    rec_ep_ids = []

    total_reward = 0.0
    episode_id = 0
    ep_len = 0

    for t in range(args.steps):
        # Heuristic uses info dict (physical values), not normalized obs
        action = heuristic_fn(info)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        ep_len += 1

        # Per-step logging
        step_logger.log(episode_id=episode_id, info=info, reward=float(reward))

        # Debug lidar
        if args.lidar_print_every > 0 and (t % args.lidar_print_every == 0):
            act_str = ",".join(f"{a:+.2f}" for a in action)
            print(
                f"t={t:5d} act=[{act_str}] "
                f"e_head={info.get('heading_error', 0.0):+.3f} "
                f"e_lat={info.get('lateral_error', 0.0):+.3f} "
                f"speed={info.get('speed', 0.0):.2f}"
            )

        # Recording
        if args.record:
            pose = info.get("pose", [0.0, 0.0, 0.0])
            rec_poses.append([float(pose[0]), float(pose[1]), float(pose[2])])
            rec_actions.append([float(a) for a in np.asarray(action).ravel()])
            rec_steer_cmds.append(float(info.get("steer_cmd", 0.0)))
            rec_speed_cmds.append(float(info.get("speed_cmd", 0.0)))
            rec_steer_rates.append(float(info.get("steer_rate", 0.0)))
            rec_e_lats.append(float(info.get("lateral_error", 0.0)))
            rec_e_heads.append(float(info.get("heading_error", 0.0)))
            rec_speeds.append(float(info.get("speed", 0.0)))
            rec_rewards.append(float(reward))
            rec_ep_ids.append(episode_id)

        # Render
        if args.render and (t % args.render_every == 0):
            try:
                env.render()
            except Exception as e:
                print(f"RENDER ERROR: {repr(e)}")
                args.render = False
            if args.sleep > 0:
                time.sleep(args.sleep)

        if terminated or truncated:
            term_reason = info.get("term_reason", "unknown")
            progress = info.get("normalized_progress", 0.0)

            ep_logger.log(
                episode_id=episode_id,
                action_space=action_space_name,
                track=track,
                reward=total_reward,
                length=ep_len,
                term_reason=term_reason,
                normalized_progress=progress,
            )

            print(
                f"[episode {episode_id}] t={t} reward={total_reward:.1f} "
                f"progress={progress:.3f} term={term_reason}"
            )

            episode_id += 1
            total_reward = 0.0
            ep_len = 0
            _LAST_STEER = 0.0
            obs, info = env.reset()

    # Cleanup
    ep_logger.close()
    step_logger.close()
    try:
        env.close()
    except Exception:
        pass

    if args.record:
        out_path = Path(args.record_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_path,
            pose=np.array(rec_poses, dtype=float),
            action=np.array(rec_actions, dtype=float),
            steer_cmd=np.array(rec_steer_cmds, dtype=float),
            speed_cmd=np.array(rec_speed_cmds, dtype=float),
            steer_rate=np.array(rec_steer_rates, dtype=float),
            lateral_error=np.array(rec_e_lats, dtype=float),
            heading_error=np.array(rec_e_heads, dtype=float),
            speed=np.array(rec_speeds, dtype=float),
            reward=np.array(rec_rewards, dtype=float),
            episode_id=np.array(rec_ep_ids, dtype=int),
            track=track,
            action_space=action_space_name,
        )
        print(f"Saved rollout: {out_path}")
        print(f"Replay: python scripts/replay_rollout_html.py --rollout {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
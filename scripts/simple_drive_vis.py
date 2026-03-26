#!/usr/bin/env python3
"""
Run a rollout in your F1TenthSACEnv.

WSL2 note: real-time pyglet rendering can hang. So this script:
- runs headless by default
- can record rollouts to .npz
- can still attempt rendering if you pass --render
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import importlib
import inspect
import time
import numpy as np
import yaml
from metrics_logger import LapMetricsLogger


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_env_class(module_name: str):
    mod = importlib.import_module(module_name)
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != module_name:
            continue
        if hasattr(obj, "step") and hasattr(obj, "reset"):
            candidates.append(obj)
    if not candidates:
        raise RuntimeError(f"No env-like class with step/reset found in {module_name}")
    return candidates[0]


_LAST_STEER = 0.0


def heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """Gentle centerline PD + smoothing."""
    global _LAST_STEER
    obs = np.asarray(obs, dtype=float)

    e_head = float(obs[4])
    e_lat = float(obs[5])

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

    return np.array([speed, steer], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--sleep", type=float, default=0.0)

    # Rendering (off by default for WSL2)
    ap.add_argument("--render", action="store_true", help="Attempt realtime rendering (may hang on WSL2)")
    ap.add_argument("--render_every", type=int, default=5)

    # Recording
    ap.add_argument("--record", action="store_true", help="Save rollout to .npz for replay")
    ap.add_argument("--record_path", default="rollouts/rollout.npz")

    # Debug prints
    ap.add_argument("--lidar_print_every", type=int, default=0, help="0 disables lidar prints")

    args = ap.parse_args()

    cfg_path = Path(args.vehicle_cfg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    vehicle_cfg = load_yaml(str(cfg_path))

    env_cls = find_env_class("envs.f1tenth_sb3_env")

    # Resolve track centerline from racetracks convention
    sim_cfg = vehicle_cfg.get("sim", {})
    raw_map_name = sim_cfg.get("map_name", "Sakhir")
    track_name = str(raw_map_name).replace("_map", "").strip()
    map_dir = Path(sim_cfg.get("map_dir", f"assets/f1tenth_racetracks/{track_name}"))
    track_centerline_csv = map_dir / f"{track_name}_centerline.csv"
    if not track_centerline_csv.exists():
        raise FileNotFoundError(f"Centerline file not found: {track_centerline_csv}")

    print("map_name:", raw_map_name)
    print("track_name:", track_name)
    print("map_dir:", map_dir)
    print("centerline:", track_centerline_csv)

    env = env_cls(vehicle_cfg=vehicle_cfg, track_centerline_csv=str(track_centerline_csv),
                  render_mode=("human" if args.render else None))

    obs, info = env.reset()
    
    lap_id = 0
    lap_start_time = time.time()
    logger = LapMetricsLogger("metrics/lap_metrics.csv")
    # time step edits - nik
    logger.enable_step_log("metrics/timestep_metrics.csv")

    # recorder buffers
    rec = {
        "t": [],
        "pose": [],
        "speed": [],
        "action": [],
        "reward": [],
        "terminated": [],
        "sim_done": [],
        "crash": [],
        "e_head": [],
        "e_lat": [],
        "lidar_sectors": [],
        "track_centerline_csv": str(track_centerline_csv),
    }

    total_reward = 0.0
    episode = 0

    for t in range(args.steps):
        action = heuristic_policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        # time step edits - nik
        t_sec = time.time() - lap_start_time
        speed_mps = float(info.get("speed", np.nan))
        if np.isfinite(speed_mps):
            logger.log_step_speed(t_sec=t_sec, speed_mps=speed_mps, lap_id=lap_id)

        # debug lidar
        if args.lidar_print_every > 0 and (t % args.lidar_print_every == 0):
            lidar = np.asarray(obs[7:], dtype=float)
            print(
                f"t={t:5d} act=[{action[0]:+.2f},{action[1]:+.2f}] "
                f"e_head={float(obs[4]):+.3f} e_lat={float(obs[5]):+.3f} "
                f"lidar(min/mean/max/std)=({lidar.min():.3f},{lidar.mean():.3f},{lidar.max():.3f},{lidar.std():.3f})"
            )

        # record
        if args.record:
            pose = np.asarray(info.get("pose", [np.nan, np.nan, np.nan]), dtype=float)
            rec["t"].append(t)
            rec["pose"].append(pose)
            rec["speed"].append(float(info.get("speed", np.nan)))
            rec["action"].append(np.asarray(action, dtype=float))
            rec["reward"].append(float(reward))
            rec["terminated"].append(bool(terminated or truncated))
            rec["sim_done"].append(bool(info.get("sim_done", False)))
            rec["crash"].append(bool(info.get("crash", False)))
            rec["e_head"].append(float(obs[4]))
            rec["e_lat"].append(float(obs[5]))
            rec["lidar_sectors"].append(np.asarray(obs[7:], dtype=float))

        # render (optional)
        if args.render and (t % args.render_every == 0):
            env.render()
            if args.sleep > 0:
                time.sleep(args.sleep)

        if terminated or truncated:
            episode += 1
            print(f"[episode {episode}] t={t} ep_return={total_reward:.3f} crash={bool(info.get('crash', False))} sim_done={bool(info.get('sim_done', False))}")
            
            # Get info
            lap_time_sec = time.time() - lap_start_time
            if bool(info.get("crash",    False)):
                lap_status = "CRASH"
            elif bool(info.get("sim_done", False)):
                lap_status = "SUCCESS"
            else:
                lap_status = "TIMEOUT"
            lap_progress = float(info.get("lap_progress", 0.0)) # Still needs to be implemented

            # Log metrics
            logger.log_lap(
                lap_id         = lap_id,
                policy_id      = "heurisitc run",
                action_space_id= "N/A",
                track_id       = track_name,
                lap_status     = lap_status,
                lap_time_sec   = lap_time_sec,
                lap_progress   = lap_progress,
            )

            # Reset lap
            lap_id += 1
            lap_start_time = time.time()
            # time step edits - nik
            logger.reset_step()
            total_reward = 0.0
            obs, info = env.reset()

    # time step edits - nik
    # TIMEOUT
    logger.log_lap(
        lap_id         = lap_id,
        policy_id      = "heurisitc run",
        action_space_id= "N/A",
        track_id       = track_name,
        lap_status     = "TIMEOUT",
        lap_time_sec   = time.time() - lap_start_time,
        lap_progress   = 0.0,
    )
    logger.close()
    env.close()

    if args.record:
        out_path = Path(args.record_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # stack arrays
        np.savez_compressed(
            out_path,
            t=np.asarray(rec["t"], dtype=int),
            pose=np.asarray(rec["pose"], dtype=float),
            speed=np.asarray(rec["speed"], dtype=float),
            action=np.asarray(rec["action"], dtype=float),
            reward=np.asarray(rec["reward"], dtype=float),
            terminated=np.asarray(rec["terminated"], dtype=bool),
            sim_done=np.asarray(rec["sim_done"], dtype=bool),
            crash=np.asarray(rec["crash"], dtype=bool),
            e_head=np.asarray(rec["e_head"], dtype=float),
            e_lat=np.asarray(rec["e_lat"], dtype=float),
            lidar_sectors=np.asarray(rec["lidar_sectors"], dtype=float),
            track_centerline_csv=rec["track_centerline_csv"],
        )
        print(f"Saved rollout: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

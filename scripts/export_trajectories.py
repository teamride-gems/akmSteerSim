#!/usr/bin/env python3
"""
Export per-step trajectory data from all trained checkpoints.

For each run in checkpoints/, loads the best model, runs one episode
per eval track, and saves the per-step trajectory as a .npz file.
These trajectories are used by plot_paper_figures.py for steering
profile figures.

Usage:
  python scripts/export_trajectories.py
  python scripts/export_trajectories.py --checkpoints_dir checkpoints --output rollouts/
  python scripts/export_trajectories.py --steps_per_episode 500
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from rl.common import normalize_track_name, make_env_for_track, arc_length_spawn_indices


def export_trajectory(
    model,
    vehicle_cfg: Dict[str, Any],
    track: str,
    spawn_idx: int,
    max_steps: int,
    seed: int = 1000,
) -> Dict[str, np.ndarray]:
    """Run one episode, capturing per-step data for trajectory plots."""
    env = make_env_for_track(vehicle_cfg, track, render_mode=None)

    poses = []
    steer_cmds = []
    speed_cmds = []
    steer_rates = []
    lat_errors = []
    head_errors = []
    speeds = []
    rewards = []
    steps = []

    try:
        obs, info = env.reset(seed=seed, options={"spawn_index": spawn_idx})
        done = False
        step = 0

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pose = info.get("pose", [0.0, 0.0, 0.0])
            poses.append([float(pose[0]), float(pose[1]), float(pose[2])])
            steer_cmds.append(float(info.get("steer_cmd", 0.0)))
            speed_cmds.append(float(info.get("speed_cmd", 0.0)))
            steer_rates.append(float(info.get("steer_rate", 0.0)))
            lat_errors.append(float(info.get("lateral_error", 0.0)))
            head_errors.append(float(info.get("heading_error", 0.0)))
            speeds.append(float(info.get("speed", 0.0)))
            rewards.append(float(reward))
            steps.append(step)

            step += 1
            done = bool(terminated or truncated)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return {
        "pose": np.array(poses, dtype=float),
        "steer_cmd": np.array(steer_cmds, dtype=float),
        "speed_cmd": np.array(speed_cmds, dtype=float),
        "steer_rate": np.array(steer_rates, dtype=float),
        "lateral_error": np.array(lat_errors, dtype=float),
        "heading_error": np.array(head_errors, dtype=float),
        "speed": np.array(speeds, dtype=float),
        "reward": np.array(rewards, dtype=float),
        "step": np.array(steps, dtype=int),
        "term_reason": info.get("term_reason", "unknown"),
        "normalized_progress": float(info.get("normalized_progress", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Export trajectories from trained checkpoints")
    ap.add_argument("--checkpoints_dir", default="checkpoints")
    ap.add_argument("--output", default="rollouts")
    ap.add_argument("--steps_per_episode", type=int, default=3000,
                     help="Max steps per trajectory episode")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tracks", default=None,
                     help="Override: comma-separated tracks (default: from run_meta)")
    args = ap.parse_args()

    ckpt_root = ROOT / args.checkpoints_dir
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_root.exists():
        print(f"No checkpoints directory at {ckpt_root}")
        return

    run_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir()])
    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        action_space = meta.get("action_space", "unknown")
        obs_regime = "ablated" if meta.get("ablate_geometry", False) else "full"
        seed = meta.get("seed", 0)
        run_id = run_dir.name

        # Only export full-observation runs (ablated are for the ablation table, not figures)
        if obs_regime == "ablated":
            continue

        # Find best model
        checkpoint = run_dir / "eval_results" / "best_model.zip"
        if not checkpoint.exists():
            checkpoint = run_dir / "sac_final.zip"
        if not checkpoint.exists():
            print(f"  {run_id}: no checkpoint found, skipping")
            continue

        # Determine tracks
        if args.tracks:
            tracks = [normalize_track_name(t) for t in args.tracks.split(",")]
        else:
            tracks = [normalize_track_name(t) for t in meta.get("eval_tracks", [])]
        if not tracks:
            train_track = meta.get("train_track", "Sakhir")
            tracks = [normalize_track_name(train_track)]

        vehicle_cfg = meta["vehicle_cfg"]

        print(f"\n{run_id}: {action_space} seed={seed}")
        print(f"  checkpoint: {checkpoint.name}")
        print(f"  tracks: {tracks}")

        model = SAC.load(str(checkpoint), device=args.device)

        for track in tracks:
            # Spawn at the first arc-length position
            env_temp = make_env_for_track(vehicle_cfg, track, render_mode=None)
            spawn_indices = arc_length_spawn_indices(env_temp.centerline, 1)
            try:
                env_temp.close()
            except Exception:
                pass

            traj = export_trajectory(
                model, vehicle_cfg, track,
                spawn_idx=spawn_indices[0],
                max_steps=args.steps_per_episode,
                seed=1000 + seed,
            )

            out_path = output_dir / f"{run_id}_{track}.npz"
            np.savez_compressed(
                out_path,
                **traj,
                track=track,
                action_space=action_space,
                seed=seed,
                run_id=run_id,
            )
            print(f"  {track}: {len(traj['step'])} steps, "
                  f"progress={traj['normalized_progress']:.3f}, "
                  f"term={traj['term_reason']} -> {out_path.name}")

    print(f"\nAll trajectories saved to {output_dir}/")


if __name__ == "__main__":
    main()
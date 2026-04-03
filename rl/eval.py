#!/usr/bin/env python3
"""
Standalone evaluation of a trained checkpoint.

Loads a saved SAC model, runs deterministic episodes on specified tracks,
and outputs full paper-relevant metrics as JSON and printed summary.

Usage:
  # eval on training track
  python rl/eval.py --checkpoint checkpoints/steer_speed_full_s0/sac_final.zip

  # eval on specific tracks
  python rl/eval.py --checkpoint checkpoints/steer_speed_full_s0/sac_final.zip \
                    --tracks Sakhir,Austin,Budapest

  # use run_meta.json to reconstruct config automatically
  python rl/eval.py --checkpoint checkpoints/steer_speed_full_s0/sac_final.zip --from_meta
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from envs.f1tenth_sb3_env import F1TenthSACEnv


# ----------------------------
# Track helpers (same as train.py)
# ----------------------------

def normalize_track_name(track: str) -> str:
    return str(track).replace("_map", "").strip()


def resolve_map_dir(track: str) -> Path:
    t = normalize_track_name(track)
    return ROOT / "assets" / "f1tenth_racetracks" / t


def resolve_centerline_csv(track: str) -> Path:
    t = normalize_track_name(track)
    return resolve_map_dir(t) / f"{t}_centerline.csv"


def make_env_for_track(vehicle_cfg: Dict[str, Any], track: str):
    track = normalize_track_name(track)
    track_dir = resolve_map_dir(track)
    cl = resolve_centerline_csv(track)

    if not track_dir.exists():
        raise FileNotFoundError(f"Track folder not found: {track_dir}")
    if not cl.exists():
        raise FileNotFoundError(f"Centerline CSV not found: {cl}")

    cfg = deepcopy(vehicle_cfg)
    cfg.setdefault("sim", {})
    cfg["sim"]["map_name"] = f"{track}_map"
    cfg["sim"]["map_dir"] = str(track_dir)
    cfg["sim"]["track_name"] = track

    return F1TenthSACEnv(vehicle_cfg=cfg, track_centerline_csv=str(cl), render_mode=None)


# ----------------------------
# Episode runner
# ----------------------------

def run_episode(model, env, seed: int, spawn_idx: int, deterministic: bool = True) -> Dict[str, Any]:
    """Run one episode, collect all metrics."""
    obs, info = env.reset(seed=seed, options={"spawn_index": spawn_idx})

    ep_reward = 0.0
    ep_len = 0
    lat_errors = []
    head_errors = []
    speeds = []
    abs_steer_rates = []
    steer_cmds = []
    min_lidars = []
    steer_clips = 0
    speed_clips = 0
    steer_clip_mags = []
    speed_clip_mags = []

    done = False
    last_info = info

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += float(reward)
        ep_len += 1

        lat_errors.append(abs(float(info.get("lateral_error", 0.0))))
        head_errors.append(abs(float(info.get("heading_error", 0.0))))
        speeds.append(float(info.get("speed", 0.0)))
        abs_steer_rates.append(abs(float(info.get("steer_rate", 0.0))))
        steer_cmds.append(float(info.get("steer_cmd", 0.0)))
        min_lidars.append(float(info.get("min_lidar", 10.0)))

        if info.get("steer_clipped", False):
            steer_clips += 1
        if info.get("speed_clipped", False):
            speed_clips += 1
        steer_clip_mags.append(float(info.get("steer_clip_mag", 0.0)))
        speed_clip_mags.append(float(info.get("speed_clip_mag", 0.0)))

        last_info = info
        done = bool(terminated or truncated)

    steer_arr = np.array(steer_cmds)
    steer_tv = float(np.sum(np.abs(np.diff(steer_arr)))) if len(steer_arr) > 1 else 0.0
    n = max(1, ep_len)

    return {
        "reward": ep_reward,
        "length": ep_len,
        "term_reason": last_info.get("term_reason", "unknown"),
        "normalized_progress": float(last_info.get("normalized_progress", 0.0)),
        "mean_lateral_error": float(np.mean(lat_errors)) if lat_errors else 0.0,
        "max_lateral_error": float(np.max(lat_errors)) if lat_errors else 0.0,
        "mean_heading_error": float(np.mean(head_errors)) if head_errors else 0.0,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_abs_steer_rate": float(np.mean(abs_steer_rates)) if abs_steer_rates else 0.0,
        "steer_tv": steer_tv,
        "steer_clip_frac": steer_clips / n,
        "speed_clip_frac": speed_clips / n,
        "mean_steer_clip_mag": float(np.mean(steer_clip_mags)) if steer_clip_mags else 0.0,
        "mean_speed_clip_mag": float(np.mean(speed_clip_mags)) if speed_clip_mags else 0.0,
        "min_lidar": float(np.min(min_lidars)) if min_lidars else 0.0,
    }


def eval_on_track(model, vehicle_cfg: Dict, track: str, n_episodes: int, deterministic: bool = True) -> List[Dict]:
    """Run n_episodes on a track with fixed, evenly-spaced spawn points."""
    env = make_env_for_track(vehicle_cfg, track)
    results = []

    try:
        n_points = int(env.centerline.shape[0])
        if n_points <= 3:
            spawn_indices = [1] * n_episodes
        else:
            spawn_indices = np.linspace(1, n_points - 2, num=n_episodes, dtype=int).tolist()

        for ep_idx, spawn_idx in enumerate(spawn_indices):
            result = run_episode(model, env, seed=1000 + ep_idx, spawn_idx=spawn_idx, deterministic=deterministic)
            result["track"] = track
            result["episode"] = ep_idx
            result["spawn_idx"] = int(spawn_idx)
            results.append(result)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return results


# ----------------------------
# Summary
# ----------------------------

def summarize(episodes: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics from a list of episode results."""
    n = len(episodes)
    if n == 0:
        return {}

    return {
        "n_episodes": n,
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "std_reward": float(np.std([e["reward"] for e in episodes])),
        "mean_progress": float(np.mean([e["normalized_progress"] for e in episodes])),
        "completion_rate": sum(1 for e in episodes if e["normalized_progress"] >= 0.95) / n,
        "crash_rate": sum(1 for e in episodes if e["term_reason"] == "crash") / n,
        "timeout_rate": sum(1 for e in episodes if e["term_reason"] == "timeout") / n,
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "mean_lateral_error": float(np.mean([e["mean_lateral_error"] for e in episodes])),
        "mean_heading_error": float(np.mean([e["mean_heading_error"] for e in episodes])),
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
        "mean_steer_rate": float(np.mean([e["mean_abs_steer_rate"] for e in episodes])),
        "mean_steer_tv": float(np.mean([e["steer_tv"] for e in episodes])),
        "steer_clip_frac": float(np.mean([e["steer_clip_frac"] for e in episodes])),
        "speed_clip_frac": float(np.mean([e["speed_clip_frac"] for e in episodes])),
    }


def print_summary(track: str, summary: Dict[str, float]):
    print(f"\n--- {track} ({summary.get('n_episodes', 0)} episodes) ---")
    print(f"  Return:         {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Progress:       {summary['mean_progress']:.3f}")
    print(f"  Completion:     {summary['completion_rate']:.1%}")
    print(f"  Crash rate:     {summary['crash_rate']:.1%}")
    print(f"  Lat error:      {summary['mean_lateral_error']:.4f} m")
    print(f"  Head error:     {summary['mean_heading_error']:.4f} rad")
    print(f"  Speed:          {summary['mean_speed']:.2f} m/s")
    print(f"  Steer rate:     {summary['mean_steer_rate']:.4f} rad/s")
    print(f"  Steer TV:       {summary['mean_steer_tv']:.2f}")
    print(f"  Steer clipped:  {summary['steer_clip_frac']:.1%}")
    print(f"  Speed clipped:  {summary['speed_clip_frac']:.1%}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to saved model (.zip)")
    ap.add_argument("--vehicle_cfg", default=None, help="Vehicle config yaml (auto-detected from run_meta if --from_meta)")
    ap.add_argument("--tracks", default=None, help="Comma-separated track names")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--from_meta", action="store_true", help="Load config from run_meta.json in checkpoint dir")
    ap.add_argument("--output", default=None, help="Save results JSON to this path")
    ap.add_argument("--deterministic", type=bool, default=True)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # try with .zip extension
        if ckpt_path.with_suffix(".zip").exists():
            ckpt_path = ckpt_path.with_suffix(".zip")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- load config ---
    if args.from_meta:
        # look for run_meta.json in the checkpoint's parent directory
        meta_path = ckpt_path.parent / "run_meta.json"
        if not meta_path.exists():
            meta_path = ckpt_path.parent.parent / "run_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run_meta.json not found near {ckpt_path}")

        with open(meta_path) as f:
            meta = json.load(f)
        vehicle_cfg = meta["vehicle_cfg"]
        default_tracks = meta.get("eval_tracks", [])
    elif args.vehicle_cfg:
        cfg_path = ROOT / args.vehicle_cfg
        if not cfg_path.exists():
            raise FileNotFoundError(f"Vehicle config not found: {cfg_path}")
        vehicle_cfg = yaml.safe_load(cfg_path.read_text())
        default_tracks = []
    else:
        raise ValueError("Provide --vehicle_cfg or use --from_meta")

    # --- tracks ---
    if args.tracks:
        tracks = [normalize_track_name(t) for t in args.tracks.split(",") if t.strip()]
    elif default_tracks:
        tracks = [normalize_track_name(t) for t in default_tracks]
    else:
        raw_map = vehicle_cfg.get("sim", {}).get("map_name", "Sakhir_map")
        tracks = [normalize_track_name(raw_map)]

    # --- load model ---
    print(f"Loading checkpoint: {ckpt_path}")
    model = SAC.load(str(ckpt_path))

    print(f"Action space: {vehicle_cfg.get('action_space', 'steer_speed')}")
    print(f"Ablated: {vehicle_cfg.get('ablate_centerline_features', False)}")
    print(f"Tracks: {tracks}")
    print(f"Episodes per track: {args.n_episodes}")

    # --- evaluate ---
    all_results = {}
    all_episodes = []

    for track in tracks:
        episodes = eval_on_track(model, vehicle_cfg, track, args.n_episodes, args.deterministic)
        all_results[track] = {
            "episodes": episodes,
            "summary": summarize(episodes),
        }
        all_episodes.extend(episodes)
        print_summary(track, all_results[track]["summary"])

    # overall
    if len(tracks) > 1:
        print_summary("OVERALL", summarize(all_episodes))

    # --- save ---
    output_path = args.output
    if output_path is None:
        output_path = str(ckpt_path.parent / "eval_standalone.json")

    # make episodes JSON-serializable (no numpy)
    serializable = {}
    for track, data in all_results.items():
        serializable[track] = {
            "summary": data["summary"],
            "episodes": data["episodes"],
        }

    Path(output_path).write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
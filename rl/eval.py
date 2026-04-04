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
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from rl.common import (
    normalize_track_name,
    make_env_for_track,
    arc_length_spawn_indices,
    EpisodeResult,
    run_eval_episode,
    summarize_episodes,
)


# ----------------------------
# Eval runner
# ----------------------------

def eval_on_track(
    model,
    vehicle_cfg: Dict[str, Any],
    track: str,
    n_episodes: int,
    deterministic: bool = True,
) -> List[EpisodeResult]:
    """
    Run n_episodes on a track with arc-length-normalized spawn points.
    Returns a list of EpisodeResult dataclasses.
    """
    env = make_env_for_track(vehicle_cfg, track, render_mode=None)
    results = []

    try:
        spawn_indices = arc_length_spawn_indices(env.centerline, n_episodes)

        for ep_idx, spawn_idx in enumerate(spawn_indices):
            result = run_eval_episode(
                model, env,
                seed=1000 + ep_idx,
                spawn_idx=spawn_idx,
                deterministic=deterministic,
            )
            results.append(result)
    finally:
        try:
            env.close()
        except Exception:
            pass

    return results


# ----------------------------
# Display
# ----------------------------

def print_summary(label: str, summary: Dict[str, float]):
    print(f"\n--- {label} ({summary.get('n_episodes', 0)} episodes) ---")
    print(f"  Return:          {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Progress:        {summary['mean_progress']:.3f}")
    print(f"  Completion:      {summary['completion_rate']:.1%}")
    print(f"  Crash rate:      {summary['crash_rate']:.1%}")
    print(f"  Lat error:       {summary['mean_lateral_error']:.4f} ± {summary['std_lateral_error']:.4f} m")
    print(f"  Head error:      {summary['mean_heading_error']:.4f} rad")
    print(f"  Speed:           {summary['mean_speed']:.2f} m/s")
    print(f"  Steer rate:      {summary['mean_steer_rate']:.4f} rad/s")
    print(f"  Steer TV:        {summary['mean_steer_tv']:.2f}  ({summary['mean_steer_tv_per_step']:.4f}/step)")
    print(f"  Steer clipped:   {summary['steer_clip_frac']:.1%}")
    print(f"  Speed clipped:   {summary['speed_clip_frac']:.1%}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to saved model (.zip)")
    ap.add_argument("--vehicle_cfg", default=None, help="Vehicle config yaml (auto-detected with --from_meta)")
    ap.add_argument("--tracks", default=None, help="Comma-separated track names")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--from_meta", action="store_true", help="Load config from run_meta.json in checkpoint dir")
    ap.add_argument("--output", default=None, help="Save results JSON to this path")
    ap.add_argument("--device", default="auto", help="Device for model inference")
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic policy (default: deterministic)")
    args = ap.parse_args()

    deterministic = not args.stochastic

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        if ckpt_path.with_suffix(".zip").exists():
            ckpt_path = ckpt_path.with_suffix(".zip")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- load config ---
    train_track = None
    if args.from_meta:
        meta_path = ckpt_path.parent / "run_meta.json"
        if not meta_path.exists():
            meta_path = ckpt_path.parent.parent / "run_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run_meta.json not found near {ckpt_path}")

        with open(meta_path) as f:
            meta = json.load(f)
        vehicle_cfg = meta["vehicle_cfg"]
        default_tracks = meta.get("eval_tracks", [])
        train_track = meta.get("train_track", None)
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

    if train_track:
        train_track = normalize_track_name(train_track)

    # --- load model ---
    print(f"Loading checkpoint: {ckpt_path}")
    model = SAC.load(str(ckpt_path), device=args.device)

    action_space = vehicle_cfg.get("action_space", "steer_speed")
    ablated = vehicle_cfg.get("ablate_centerline_features", False)
    print(f"Action space: {action_space}")
    print(f"Ablated: {ablated}")
    print(f"Tracks: {tracks}")
    print(f"Episodes per track: {args.n_episodes}")
    print(f"Deterministic: {deterministic}")

    # --- evaluate ---
    all_results = {}
    all_episodes = []
    total_start = time.time()

    for track in tracks:
        track_start = time.time()
        episodes = eval_on_track(model, vehicle_cfg, track, args.n_episodes, deterministic)
        track_elapsed = time.time() - track_start

        summary = summarize_episodes(episodes)
        summary["wall_clock_seconds"] = track_elapsed

        # Label as train or heldout
        is_train = (train_track is not None and track == train_track)
        label = f"{track} [TRAIN]" if is_train else track

        all_results[track] = {
            "episodes": [vars(e) for e in episodes],
            "summary": summary,
            "is_train_track": is_train,
        }
        all_episodes.extend(episodes)
        print_summary(label, summary)

    total_elapsed = time.time() - total_start

    # overall
    if len(tracks) > 1:
        overall = summarize_episodes(all_episodes)
        overall["wall_clock_seconds"] = total_elapsed
        print_summary("OVERALL", overall)

        # heldout-only summary (excluding training track)
        heldout_episodes = [
            e for track, data in all_results.items()
            for e in [EpisodeResult(**ep) for ep in data["episodes"]]
            if not data["is_train_track"]
        ]
        if heldout_episodes and len(heldout_episodes) < len(all_episodes):
            heldout_summary = summarize_episodes(heldout_episodes)
            print_summary("HELDOUT ONLY", heldout_summary)

    # --- save ---
    output_path = args.output or str(ckpt_path.parent / "eval_standalone.json")

    output_data = {
        "checkpoint": str(ckpt_path),
        "action_space": action_space,
        "ablated": ablated,
        "deterministic": deterministic,
        "n_episodes_per_track": args.n_episodes,
        "train_track": train_track,
        "wall_clock_seconds": total_elapsed,
        "tracks": all_results,
    }

    if len(tracks) > 1:
        output_data["overall_summary"] = summarize_episodes(all_episodes)

    Path(output_path).write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Standalone evaluation with explicit split semantics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.common import (
    normalize_track_name,
    make_env_for_track,
    arc_length_spawn_indices,
    run_eval_episode,
    summarize_episodes,
)


def eval_on_track(model, vehicle_cfg: Dict[str, Any], track: str, n_episodes: int, deterministic: bool = True):
    env = make_env_for_track(vehicle_cfg, track, render_mode=None)
    results = []
    try:
        spawn_indices = arc_length_spawn_indices(env.centerline, n_episodes)
        for ep_idx, spawn_idx in enumerate(spawn_indices):
            results.append(
                run_eval_episode(
                    model,
                    env,
                    seed=1000 + ep_idx,
                    spawn_idx=spawn_idx,
                    deterministic=deterministic,
                )
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
    return results


def print_summary(label: str, summary: Dict[str, float]) -> None:
    print(f"\n--- {label} ({summary.get('n_episodes', 0)} episodes) ---")
    print(f"  Return:          {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Progress:        {summary['mean_progress']:.3f}")
    print(f"  Completion:      {summary['completion_rate']:.1%}")
    print(f"  Crash rate:      {summary['crash_rate']:.1%}")
    print(f"  Lat error:       {summary['mean_lateral_error']:.4f} ± {summary['std_lateral_error']:.4f} m")
    print(f"  Head error:      {summary['mean_heading_error']:.4f} rad")
    print(f"  Speed:           {summary['mean_speed']:.2f} m/s")
    print(f"  Steer rate:      {summary['mean_steer_rate']:.4f} rad/s")
    print(f"  Steer TV:        {summary['mean_steer_tv']:.2f} ({summary['mean_steer_tv_per_step']:.4f}/step)")
    print(f"  Steer clipped:   {summary['steer_clip_frac']:.1%}")
    print(f"  Speed clipped:   {summary['speed_clip_frac']:.1%}")


def _normalize_tracks(raw: List[str]) -> List[str]:
    return [normalize_track_name(t) for t in raw if str(t).strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to saved model (.zip)")
    ap.add_argument("--vehicle_cfg", default=None)
    ap.add_argument("--tracks", default=None, help="Comma-separated custom tracks")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--from_meta", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument(
        "--evaluation_split",
        choices=["auto", "train", "validation", "test", "all", "custom"],
        default="auto",
    )
    args = ap.parse_args()

    deterministic = not args.stochastic
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists() and ckpt_path.with_suffix(".zip").exists():
        ckpt_path = ckpt_path.with_suffix(".zip")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_track = None
    validation_tracks: List[str] = []
    test_tracks: List[str] = []

    if args.from_meta:
        meta_path = ckpt_path.parent / "run_meta.json"
        if not meta_path.exists():
            meta_path = ckpt_path.parent.parent / "run_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"run_meta.json not found near {ckpt_path}")

        meta = json.loads(meta_path.read_text())
        vehicle_cfg = meta["vehicle_cfg"]
        train_track = normalize_track_name(meta.get("train_track")) if meta.get("train_track") else None
        validation_tracks = _normalize_tracks(meta.get("validation_tracks", meta.get("eval_tracks", [])))
        test_tracks = _normalize_tracks(meta.get("test_tracks", []))
    elif args.vehicle_cfg:
        cfg_path = ROOT / args.vehicle_cfg
        if not cfg_path.exists():
            raise FileNotFoundError(f"Vehicle config not found: {cfg_path}")
        vehicle_cfg = yaml.safe_load(cfg_path.read_text())
    else:
        raise ValueError("Provide --vehicle_cfg or use --from_meta")

    if args.evaluation_split == "custom":
        if not args.tracks:
            raise ValueError("--evaluation_split custom requires --tracks")
        tracks = _normalize_tracks(args.tracks.split(","))
        track_groups = {track: "custom" for track in tracks}
    elif args.evaluation_split == "train":
        tracks = [train_track] if train_track else []
        track_groups = {track: "train" for track in tracks}
    elif args.evaluation_split == "validation":
        tracks = list(validation_tracks)
        track_groups = {track: "validation" for track in tracks}
    elif args.evaluation_split == "test":
        tracks = list(test_tracks)
        track_groups = {track: "test" for track in tracks}
    elif args.evaluation_split == "all":
        tracks = []
        if train_track:
            tracks.append(train_track)
        tracks.extend(validation_tracks)
        tracks.extend([t for t in test_tracks if t not in tracks])
        track_groups = {}
        if train_track:
            track_groups[train_track] = "train"
        for track in validation_tracks:
            track_groups[track] = "validation"
        for track in test_tracks:
            track_groups[track] = "test"
    else:
        if args.tracks:
            tracks = _normalize_tracks(args.tracks.split(","))
            track_groups = {track: "custom" for track in tracks}
        elif test_tracks:
            tracks = list(test_tracks)
            track_groups = {track: "test" for track in tracks}
        elif validation_tracks:
            tracks = list(validation_tracks)
            track_groups = {track: "validation" for track in tracks}
        elif train_track:
            tracks = [train_track]
            track_groups = {train_track: "train"}
        else:
            raw_map = vehicle_cfg.get("sim", {}).get("map_name", "Sakhir_map")
            track = normalize_track_name(raw_map)
            tracks = [track]
            track_groups = {track: "custom"}

    if not tracks:
        raise ValueError("No tracks resolved for evaluation.")

    print(f"Loading checkpoint: {ckpt_path}")
    model = SAC.load(str(ckpt_path), device=args.device)
    action_space = vehicle_cfg.get("action_space", "steer_speed")
    ablated = vehicle_cfg.get("ablate_centerline_features", False)

    print(f"Action space: {action_space}")
    print(f"Ablated: {ablated}")
    print(f"Evaluation split: {args.evaluation_split}")
    print(f"Tracks: {tracks}")
    print(f"Episodes per track: {args.n_episodes}")
    print(f"Deterministic: {deterministic}")

    all_results: Dict[str, Any] = {}
    split_episodes: Dict[str, list] = {"train": [], "validation": [], "test": [], "custom": []}
    all_episodes = []
    total_start = time.time()

    for track in tracks:
        track_start = time.time()
        episodes = eval_on_track(model, vehicle_cfg, track, args.n_episodes, deterministic)
        elapsed = time.time() - track_start
        group = track_groups.get(track, "custom")
        summary = summarize_episodes(episodes)
        summary["wall_clock_seconds"] = elapsed
        all_results[track] = {
            "episodes": [vars(e) for e in episodes],
            "summary": summary,
            "track_group": group,
        }
        split_episodes.setdefault(group, []).extend(episodes)
        all_episodes.extend(episodes)
        print_summary(f"{track} [{group.upper()}]", summary)

    total_elapsed = time.time() - total_start
    overall_summary = summarize_episodes(all_episodes)
    overall_summary["wall_clock_seconds"] = total_elapsed
    if len(all_episodes) > 0 and len(tracks) > 1:
        print_summary("OVERALL", overall_summary)

    output_data: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "action_space": action_space,
        "ablated": ablated,
        "deterministic": deterministic,
        "evaluation_split": args.evaluation_split,
        "n_episodes_per_track": args.n_episodes,
        "train_track": train_track,
        "validation_tracks": validation_tracks,
        "test_tracks": test_tracks,
        "wall_clock_seconds": total_elapsed,
        "tracks": all_results,
        "overall_summary": overall_summary,
    }

    for split_name in ["train", "validation", "test", "custom"]:
        if split_episodes.get(split_name):
            output_data[f"{split_name}_summary"] = summarize_episodes(split_episodes[split_name])

    output_path = args.output
    if output_path is None:
        split_suffix = args.evaluation_split if args.evaluation_split != "auto" else (
            "test" if test_tracks else "validation" if validation_tracks else "train"
        )
        output_path = str(ckpt_path.parent / f"eval_standalone_{split_suffix}.json")

    Path(output_path).write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect reachable action-interface transitions from a frozen SAC policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.common import arc_length_spawn_indices, make_env_for_track, normalize_track_name
from utils.provenance import collect_provenance, utc_now_iso, write_json


TRACE_FIELDS = (
    "policy_observation",
    "raw_action",
    "previous_command",
    "pre_constraint_command",
    "executed_command",
    "realized_command",
    "next_policy_observation",
)


def _episode(model, env, seed: int, spawn_idx: int, deterministic: bool) -> Dict:
    observation, _ = env.reset(seed=seed, options={"spawn_index": spawn_idx})
    transitions: Dict[str, List] = {field: [] for field in TRACE_FIELDS}
    transitions.update(
        {
            "steer_limiter_active": [],
            "speed_limiter_active": [],
            "steer_correction": [],
            "speed_correction": [],
            "normalized_progress": [],
            "lateral_error": [],
            "heading_error": [],
        }
    )

    done = False
    final_info = {}
    while not done:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, _, terminated, truncated, info = env.step(action)
        for field in TRACE_FIELDS:
            if field not in info:
                raise KeyError(f"Instrumented environment did not provide '{field}'.")
            transitions[field].append(np.asarray(info[field], dtype=np.float32))
        transitions["steer_limiter_active"].append(bool(info["steer_clipped"]))
        transitions["speed_limiter_active"].append(bool(info["speed_clipped"]))
        transitions["steer_correction"].append(float(info["steer_clip_mag"]))
        transitions["speed_correction"].append(float(info["speed_clip_mag"]))
        transitions["normalized_progress"].append(float(info["normalized_progress"]))
        transitions["lateral_error"].append(float(info["lateral_error"]))
        transitions["heading_error"].append(float(info["heading_error"]))
        done = bool(terminated or truncated)
        final_info = info

    return {
        "transitions": transitions,
        "term_reason": final_info.get("term_reason", "unknown"),
        "final_progress": float(final_info.get("normalized_progress", 0.0)),
        "spawn_idx": int(spawn_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best_validation")
    parser.add_argument("--tracks", default="Sakhir")
    parser.add_argument("--episodes_per_track", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default="experiments/gate1/decoder_audit_dataset.npz")
    args = parser.parse_args()

    if args.episodes_per_track <= 0:
        raise ValueError("episodes_per_track must be positive.")
    run_dir = Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_dir}")
    vehicle_cfg = yaml.safe_load((run_dir / "resolved_vehicle.yaml").read_text(encoding="utf-8"))

    if args.checkpoint == "best_validation":
        checkpoint = run_dir / "eval_results" / "best_validation_model.zip"
    elif args.checkpoint == "final":
        checkpoint = run_dir / "sac_final.zip"
    else:
        checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    tracks = [normalize_track_name(x) for x in args.tracks.split(",") if x.strip()]
    all_arrays: Dict[str, List[np.ndarray]] = {field: [] for field in TRACE_FIELDS}
    for field in (
        "steer_limiter_active",
        "speed_limiter_active",
        "steer_correction",
        "speed_correction",
        "normalized_progress",
        "lateral_error",
        "heading_error",
        "episode_id",
        "track_id",
        "episode_complete",
    ):
        all_arrays[field] = []
    episode_summaries = []
    episode_id = 0

    for track_id, track in enumerate(tracks):
        env = make_env_for_track(vehicle_cfg, track, render_mode=None)
        model = SAC.load(str(checkpoint), env=env, device=args.device)
        spawn_indices = arc_length_spawn_indices(env.centerline, args.episodes_per_track)
        try:
            for local_episode, spawn_idx in enumerate(spawn_indices):
                result = _episode(
                    model,
                    env,
                    seed=args.seed + 1000 * track_id + local_episode,
                    spawn_idx=spawn_idx,
                    deterministic=not args.stochastic,
                )
                transitions = result.pop("transitions")
                n = len(transitions["policy_observation"])
                completed = result["term_reason"] == "lap_complete"
                for field in TRACE_FIELDS:
                    all_arrays[field].append(np.asarray(transitions[field]))
                for field in (
                    "steer_limiter_active",
                    "speed_limiter_active",
                    "steer_correction",
                    "speed_correction",
                    "normalized_progress",
                    "lateral_error",
                    "heading_error",
                ):
                    all_arrays[field].append(np.asarray(transitions[field]))
                all_arrays["episode_id"].append(np.full(n, episode_id, dtype=np.int32))
                all_arrays["track_id"].append(np.full(n, track_id, dtype=np.int16))
                all_arrays["episode_complete"].append(np.full(n, completed, dtype=bool))
                episode_summaries.append(
                    {"episode_id": episode_id, "track": track, "length": n, **result}
                )
                print(
                    f"episode={episode_id} track={track} length={n} "
                    f"progress={result['final_progress']:.3f} term={result['term_reason']}"
                )
                episode_id += 1
        finally:
            env.close()

    packed = {field: np.concatenate(chunks, axis=0) for field, chunks in all_arrays.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "dataset": str(output.resolve()),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "action_space": meta["action_space"],
        "tracks": tracks,
        "deterministic": not args.stochastic,
        "n_episodes": episode_id,
        "n_transitions": int(packed["episode_id"].size),
        "completion_rate": float(np.mean([e["term_reason"] == "lap_complete" for e in episode_summaries])),
        "steer_limiter_activation_fraction": float(np.mean(packed["steer_limiter_active"])),
        "speed_limiter_activation_fraction": float(np.mean(packed["speed_limiter_active"])),
        "episodes": episode_summaries,
        "provenance": collect_provenance(ROOT),
    }
    manifest_path = output.with_suffix(".json")
    write_json(manifest_path, manifest)
    print(f"Saved {manifest['n_transitions']} transitions: {output.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()

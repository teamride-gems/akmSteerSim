#!/usr/bin/env python3
"""Recover an interrupted post-20k validation without resuming training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.common import (
    arc_length_spawn_indices,
    make_env_for_track,
    model_selection_score,
    normalize_track_name,
    run_eval_episode,
    summarize_episodes,
)
from utils.provenance import collect_provenance, utc_now_iso, write_json


def evaluate_track(
    model: SAC, vehicle_cfg: Dict[str, Any], track: str, n_episodes: int
) -> List[Any]:
    env = make_env_for_track(vehicle_cfg, track, render_mode=None)
    try:
        spawn_indices = arc_length_spawn_indices(env.centerline, n_episodes)
        return [
            run_eval_episode(
                model,
                env,
                seed=1000 + episode_index,
                spawn_idx=spawn_index,
                deterministic=True,
            )
            for episode_index, spawn_index in enumerate(spawn_indices)
        ]
    finally:
        env.close()


def previous_best_score(results_dir: Path) -> float:
    scores = []
    for path in results_dir.glob("eval_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        score = data.get("validation_model_selection_score")
        if score is not None:
            scores.append(float(score))
    return max(scores, default=-float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a run interrupted during its final validation callback"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "checkpoints" / args.run_id
    meta_path = run_dir / "run_meta.json"
    checkpoint = run_dir / "sac_20000_steps.zip"
    results_dir = run_dir / "eval_results"
    if not meta_path.exists() or not checkpoint.exists():
        raise SystemExit("Recovery requires run_meta.json and sac_20000_steps.zip")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    configured_steps = int(meta.get("sac_cfg", {}).get("train_steps", -1))
    if configured_steps != 20_000:
        raise SystemExit(f"Refusing recovery for configured train_steps={configured_steps}")
    if meta.get("status") == "complete":
        raise SystemExit("Run is already complete; no training finalization is needed")

    model = SAC.load(str(checkpoint), device=args.device)
    if int(model.num_timesteps) != configured_steps:
        raise SystemExit(
            f"Checkpoint timestep {model.num_timesteps} != configured {configured_steps}"
        )

    validation_tracks = [normalize_track_name(t) for t in meta["validation_tracks"]]
    n_episodes = int(meta["n_eval_episodes"])
    start = time.time()
    validation_results = {
        track: evaluate_track(model, meta["vehicle_cfg"], track, n_episodes)
        for track in validation_tracks
    }
    validation_flat = [
        episode for episodes in validation_results.values() for episode in episodes
    ]
    validation_summary = summarize_episodes(validation_flat)
    score = model_selection_score(validation_summary)

    train_track = normalize_track_name(meta["train_track"])
    train_results = evaluate_track(model, meta["vehicle_cfg"], train_track, n_episodes)
    elapsed = time.time() - start
    prior_score = previous_best_score(results_dir)
    snapshot = {
        "timestep": configured_steps,
        "wall_clock_hours": None,
        "validation_tracks": {
            track: [vars(episode) for episode in episodes]
            for track, episodes in validation_results.items()
        },
        "validation_summary": validation_summary,
        "validation_model_selection_score": score,
        "train_track": train_track,
        "train_summary": summarize_episodes(train_results),
        "recovered_after_interruption": True,
        "recovery_eval_seconds": elapsed,
    }
    write_json(results_dir / "eval_000020000.json", snapshot)

    best_path = results_dir / "best_validation_model.zip"
    selected_20k = score > prior_score
    if selected_20k:
        shutil.copy2(checkpoint, best_path)
    if not best_path.exists():
        raise SystemExit("No best validation model is available after recovery")
    shutil.copy2(checkpoint, run_dir / "sac_final.zip")

    meta["status"] = "complete"
    meta["training_completed_at_utc"] = utc_now_iso()
    meta["training_wall_clock_seconds"] = None
    meta["final_num_timesteps"] = configured_steps
    meta["artifacts"] = {
        "final_model": "sac_final.zip",
        "best_validation_model": "eval_results/best_validation_model.zip",
        "replay_buffer": None,
    }
    meta["interruption_recovery"] = {
        "reason": "orchestrator_timeout_during_final_validation",
        "source_checkpoint": "sac_20000_steps.zip",
        "checkpoint_num_timesteps": int(model.num_timesteps),
        "recovered_at_utc": utc_now_iso(),
        "recovery_provenance": collect_provenance(ROOT),
        "validation_score": score,
        "previous_best_score": prior_score,
        "selected_20k_checkpoint": selected_20k,
        "training_wall_clock_seconds_unavailable": True,
    }
    write_json(meta_path, meta)

    if args.eval_test:
        output_path = run_dir / "eval_standalone_test.json"
        command = [
            sys.executable,
            str(ROOT / "rl" / "eval.py"),
            "--checkpoint", str(best_path),
            "--from_meta",
            "--evaluation_split", "test",
            "--n_episodes", str(n_episodes),
            "--device", args.device,
            "--output", str(output_path),
        ]
        subprocess.run(command, cwd=str(ROOT), check=True)

    print(
        f"Recovered {args.run_id}: validation_score={score:.6f}, "
        f"selected_20k={selected_20k}"
    )


if __name__ == "__main__":
    main()

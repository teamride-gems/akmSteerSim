#!/usr/bin/env python3
"""
Aggregate experiment results into paper-ready tables and CSVs.

Reads eval JSON files from checkpoints/<run_id>/eval_results/ and produces:
  - summary CSV with per-condition means and standard errors
  - per-seed CSV for statistical analysis
  - printed table for quick inspection

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --checkpoints_dir checkpoints --output results/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_run_id(run_id: str) -> Dict[str, str]:
    """
    Parse run_id format: {action_space}_{obs_regime}_s{seed}
    e.g. curvature_speed_full_s3 -> action_space=curvature_speed, obs=full, seed=3
    """
    parts = run_id.rsplit("_s", 1)
    if len(parts) != 2:
        return {"action_space": run_id, "obs_regime": "unknown", "seed": "0"}

    seed = parts[1]
    prefix = parts[0]

    if prefix.endswith("_ablated"):
        obs_regime = "ablated"
        action_space = prefix[: -len("_ablated")]
    elif prefix.endswith("_full"):
        obs_regime = "full"
        action_space = prefix[: -len("_full")]
    else:
        obs_regime = "unknown"
        action_space = prefix

    return {"action_space": action_space, "obs_regime": obs_regime, "seed": seed}


def load_final_eval(eval_dir: Path) -> Dict[str, Any]:
    """Load the last (highest timestep) eval JSON from a run's eval_results/."""
    jsons = sorted(eval_dir.glob("eval_*.json"))
    if not jsons:
        return {}
    # last file = highest timestep
    with open(jsons[-1]) as f:
        return json.load(f)


def extract_episode_metrics(eval_data: Dict[str, Any]) -> List[Dict[str, float]]:
    """Flatten per-track episode results into a list of metric dicts."""
    episodes = []
    tracks = eval_data.get("tracks", {})
    for track_name, eps in tracks.items():
        for ep in eps:
            ep["track"] = track_name
            episodes.append(ep)
    return episodes


def aggregate_run(run_dir: Path) -> Dict[str, Any]:
    """Aggregate a single run into summary metrics."""
    eval_dir = run_dir / "eval_results"
    if not eval_dir.exists():
        return {}

    # parse run identity
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        action_space = meta.get("action_space", "unknown")
        obs_regime = "ablated" if meta.get("ablate_geometry", False) else "full"
        seed = meta.get("seed", 0)
    else:
        parsed = parse_run_id(run_dir.name)
        action_space = parsed["action_space"]
        obs_regime = parsed["obs_regime"]
        seed = parsed["seed"]

    eval_data = load_final_eval(eval_dir)
    if not eval_data:
        return {}

    episodes = extract_episode_metrics(eval_data)
    if not episodes:
        return {}

    n = len(episodes)
    return {
        "action_space": action_space,
        "obs_regime": obs_regime,
        "seed": int(seed),
        "n_episodes": n,
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "std_reward": float(np.std([e["reward"] for e in episodes])),
        "mean_progress": float(np.mean([e["normalized_progress"] for e in episodes])),
        "completion_rate": sum(1 for e in episodes if e.get("normalized_progress", 0) >= 0.95) / n,
        "crash_rate": sum(1 for e in episodes if e.get("term_reason") == "crash") / n,
        "mean_lateral_error": float(np.mean([e.get("mean_lateral_error", 0) for e in episodes])),
        "mean_heading_error": float(np.mean([e.get("mean_heading_error", 0) for e in episodes])),
        "mean_speed": float(np.mean([e.get("mean_speed", 0) for e in episodes])),
        "mean_steer_rate": float(np.mean([e.get("mean_abs_steer_rate", 0) for e in episodes])),
        "mean_steer_tv": float(np.mean([e.get("steer_tv", 0) for e in episodes])),
        "steer_clip_frac": float(np.mean([e.get("steer_clip_frac", 0) for e in episodes])),
        "speed_clip_frac": float(np.mean([e.get("speed_clip_frac", 0) for e in episodes])),
    }


def build_summary_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Group by (action_space, obs_regime) and compute mean ± stderr across seeds."""
    group_cols = ["action_space", "obs_regime"]
    metric_cols = [
        "mean_reward", "mean_progress", "completion_rate", "crash_rate",
        "mean_lateral_error", "mean_heading_error", "mean_speed",
        "mean_steer_rate", "mean_steer_tv", "steer_clip_frac", "speed_clip_frac",
    ]

    rows = []
    for key, group in per_seed_df.groupby(group_cols):
        action_space, obs_regime = key
        row = {"action_space": action_space, "obs_regime": obs_regime, "n_seeds": len(group)}
        for col in metric_cols:
            vals = group[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_se"] = float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def print_paper_table(summary_df: pd.DataFrame):
    """Print a formatted table suitable for paper results section."""
    print("\n" + "=" * 100)
    print("PAPER RESULTS TABLE")
    print("=" * 100)

    key_metrics = [
        ("mean_reward", "Return", ".1f"),
        ("mean_progress", "Progress", ".3f"),
        ("completion_rate", "Compl%", ".2f"),
        ("crash_rate", "Crash%", ".2f"),
        ("mean_lateral_error", "LatErr(m)", ".4f"),
        ("mean_steer_rate", "SteerRate", ".4f"),
        ("steer_clip_frac", "SteerClip%", ".3f"),
    ]

    # header
    header = f"{'Action Space':<22} {'Obs':<8}"
    for _, label, _ in key_metrics:
        header += f" {label:>14}"
    print(header)
    print("-" * len(header))

    for _, row in summary_df.sort_values(["obs_regime", "action_space"]).iterrows():
        line = f"{row['action_space']:<22} {row['obs_regime']:<8}"
        for col, _, fmt in key_metrics:
            mean = row.get(f"{col}_mean", float("nan"))
            se = row.get(f"{col}_se", float("nan"))
            if np.isnan(mean):
                line += f" {'N/A':>14}"
            elif np.isnan(se):
                line += f" {format(mean, fmt):>14}"
            else:
                line += f" {format(mean, fmt) + '±' + format(se, fmt):>14}"
        print(line)

    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", default="checkpoints", help="Root checkpoints directory")
    ap.add_argument("--output", default="results", help="Output directory for CSVs")
    args = ap.parse_args()

    ckpt_root = ROOT / args.checkpoints_dir
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_root.exists():
        print(f"No checkpoints directory found at {ckpt_root}")
        return

    # collect all runs
    run_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir()])
    print(f"Found {len(run_dirs)} run directories in {ckpt_root}")

    per_seed_rows = []
    for run_dir in run_dirs:
        row = aggregate_run(run_dir)
        if row:
            per_seed_rows.append(row)

    if not per_seed_rows:
        print("No valid results found.")
        return

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = output_dir / "per_seed_results.csv"
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"Per-seed results: {per_seed_path} ({len(per_seed_df)} runs)")

    summary_df = build_summary_table(per_seed_df)
    summary_path = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table: {summary_path}")

    print_paper_table(summary_df)


if __name__ == "__main__":
    main()
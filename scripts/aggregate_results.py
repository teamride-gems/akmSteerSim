#!/usr/bin/env python3
"""
Aggregate experiment results into paper-ready tables and CSVs.

Reads eval JSON files from checkpoints/<run_id>/eval_results/ and produces:
  - per-seed CSV with all metrics for statistical analysis
  - summary CSV with per-condition means and standard errors across seeds
  - per-track CSV with train vs. heldout breakdown
  - printed table for quick inspection

Usage:
  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --checkpoints_dir checkpoints --output results/
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------
# Result loading
# ----------------------------

def load_run_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def load_eval_data(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load evaluation results, preferring standalone eval over training snapshots.

    Priority:
      1. eval_standalone.json (from rl/eval.py — clean final eval)
      2. Last eval_*.json snapshot (from training callback)
    """
    eval_dir = run_dir / "eval_results"

    # Prefer standalone eval if available
    standalone = run_dir / "eval_standalone.json"
    if standalone.exists():
        with open(standalone) as f:
            data = json.load(f)
        # Standalone format has tracks nested under "tracks" key
        if "tracks" in data:
            return data
        # Might be a flat format — wrap it
        return {"tracks": data}

    # Fall back to training eval snapshots
    if not eval_dir.exists():
        return None
    jsons = sorted(eval_dir.glob("eval_*.json"))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def extract_episodes(eval_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten per-track episode results into a list, tagging each with track name."""
    episodes = []
    tracks = eval_data.get("tracks", {})
    for track_name, track_data in tracks.items():
        # Handle both formats: list of episodes directly, or dict with "episodes" key
        if isinstance(track_data, list):
            eps = track_data
        elif isinstance(track_data, dict) and "episodes" in track_data:
            eps = track_data["episodes"]
        else:
            continue
        for ep in eps:
            ep_copy = dict(ep)
            ep_copy["track"] = track_name
            episodes.append(ep_copy)
    return episodes


# ----------------------------
# Per-run aggregation
# ----------------------------

METRIC_KEYS = [
    "mean_reward", "std_reward",
    "mean_progress", "completion_rate", "crash_rate",
    "mean_lateral_error", "std_lateral_error", "max_lateral_error",
    "mean_heading_error",
    "mean_speed",
    "mean_steer_rate", "mean_steer_tv", "mean_steer_tv_per_step",
    "steer_clip_frac", "speed_clip_frac",
    "mean_ep_len",
]


def compute_metrics(episodes: List[Dict], label: str = "") -> Dict[str, Any]:
    """Compute aggregate metrics from a list of episode dicts."""
    n = len(episodes)
    if n == 0:
        return {}

    def _safe_mean(key, default=0.0):
        return float(np.mean([e.get(key, default) for e in episodes]))

    def _safe_std(key, default=0.0):
        return float(np.std([e.get(key, default) for e in episodes]))

    result = {
        "n_episodes": n,
        "mean_reward": _safe_mean("reward"),
        "std_reward": _safe_std("reward"),
        "mean_progress": _safe_mean("normalized_progress"),
        "completion_rate": sum(1 for e in episodes if e.get("normalized_progress", 0) >= 0.95) / n,
        "crash_rate": sum(1 for e in episodes if e.get("term_reason") == "crash") / n,
        "mean_lateral_error": _safe_mean("mean_lateral_error"),
        "std_lateral_error": _safe_std("mean_lateral_error"),
        "max_lateral_error": float(np.max([e.get("max_lateral_error", 0) for e in episodes])),
        "mean_heading_error": _safe_mean("mean_heading_error"),
        "mean_speed": _safe_mean("mean_speed"),
        "mean_steer_rate": _safe_mean("mean_abs_steer_rate"),
        "mean_steer_tv": _safe_mean("steer_tv"),
        "mean_steer_tv_per_step": _safe_mean("steer_tv_per_step"),
        "steer_clip_frac": _safe_mean("steer_clip_frac"),
        "speed_clip_frac": _safe_mean("speed_clip_frac"),
        "mean_ep_len": _safe_mean("length"),
    }
    return result


def aggregate_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate a single run into summary metrics with train/heldout split."""
    meta = load_run_meta(run_dir)
    eval_data = load_eval_data(run_dir)

    if eval_data is None:
        return None

    # Identity from metadata
    if meta:
        action_space = meta.get("action_space", "unknown")
        obs_regime = "ablated" if meta.get("ablate_geometry", False) else "full"
        seed = meta.get("seed", 0)
        train_track = meta.get("train_track", None)
    else:
        # Fallback: parse from directory name
        parsed = parse_run_id(run_dir.name)
        action_space = parsed["action_space"]
        obs_regime = parsed["obs_regime"]
        seed = int(parsed["seed"])
        train_track = None

    episodes = extract_episodes(eval_data)
    if not episodes:
        return None

    # Overall metrics
    overall = compute_metrics(episodes)
    overall.update({
        "action_space": action_space,
        "obs_regime": obs_regime,
        "seed": seed,
        "run_id": run_dir.name,
    })

    # Train vs. heldout split
    if train_track:
        train_eps = [e for e in episodes if e.get("track") == train_track]
        heldout_eps = [e for e in episodes if e.get("track") != train_track]

        if train_eps:
            train_metrics = compute_metrics(train_eps)
            for k, v in train_metrics.items():
                overall[f"train_{k}"] = v

        if heldout_eps:
            heldout_metrics = compute_metrics(heldout_eps)
            for k, v in heldout_metrics.items():
                overall[f"heldout_{k}"] = v

    return overall


def parse_run_id(run_id: str) -> Dict[str, str]:
    """Fallback parser for run_id format: {action_space}_{obs_regime}_s{seed}"""
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


# ----------------------------
# Cross-seed summary
# ----------------------------

def build_summary_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Group by (action_space, obs_regime) and compute mean ± stderr across seeds."""
    group_cols = ["action_space", "obs_regime"]

    # Aggregate all numeric columns
    metric_cols = [c for c in per_seed_df.columns
                   if c not in group_cols + ["seed", "run_id", "n_episodes"]
                   and pd.api.types.is_numeric_dtype(per_seed_df[c])]

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


# ----------------------------
# Display
# ----------------------------

def _fmt_cell(mean: float, se: float, fmt: str) -> str:
    if np.isnan(mean):
        return "N/A"
    if np.isnan(se):
        return format(mean, fmt)
    return f"{format(mean, fmt)}±{format(se, fmt)}"


def print_paper_table(summary_df: pd.DataFrame, prefix: str = ""):
    """Print a formatted table suitable for paper results section."""
    key_metrics = [
        ("mean_reward", "Return", ".1f"),
        ("mean_progress", "Progress", ".3f"),
        ("completion_rate", "Compl%", ".2f"),
        ("crash_rate", "Crash%", ".2f"),
        ("mean_lateral_error", "LatErr(m)", ".4f"),
        ("mean_steer_tv_per_step", "TV/step", ".4f"),
        ("mean_steer_rate", "SteerRate", ".4f"),
        ("steer_clip_frac", "SteerClip%", ".3f"),
    ]

    title = f"{prefix}RESULTS TABLE" if prefix else "RESULTS TABLE"
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")

    header = f"{'Action Space':<22} {'Obs':<8} {'Seeds':>5}"
    for _, label, _ in key_metrics:
        header += f" {label:>16}"
    print(header)
    print("-" * len(header))

    for _, row in summary_df.sort_values(["obs_regime", "action_space"]).iterrows():
        line = f"{row['action_space']:<22} {row['obs_regime']:<8} {int(row['n_seeds']):>5}"
        for col, _, fmt in key_metrics:
            mean = row.get(f"{prefix}{col}_mean", float("nan"))
            se = row.get(f"{prefix}{col}_se", float("nan"))
            line += f" {_fmt_cell(mean, se, fmt):>16}"
        print(line)

    print("=" * 110)


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

    # Print overall table
    print_paper_table(summary_df)

    # Print train-track-only table if available
    if any(c.startswith("train_") for c in summary_df.columns):
        print_paper_table(summary_df, prefix="train_")

    # Print heldout-only table if available
    if any(c.startswith("heldout_") for c in summary_df.columns):
        print_paper_table(summary_df, prefix="heldout_")


if __name__ == "__main__":
    main()
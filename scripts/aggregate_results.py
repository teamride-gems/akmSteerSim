#!/usr/bin/env python3
"""
Aggregate standalone evaluation results into analysis-friendly CSVs.

Expected inputs per run directory:
  - run_meta.json
  - eval_standalone_test.json           (preferred)
  - eval_standalone_validation.json     (fallback — tagged with warning)
  - eval_standalone.json                (fallback — tagged with warning)

FIX (M1 from audit): Records which eval file was actually loaded and
warns when test data is missing. Columns prefixed with "test_" are only
populated from actual test-split eval files.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METRIC_KEYS = [
    "mean_reward",
    "std_reward",
    "mean_progress",
    "completion_rate",
    "crash_rate",
    "mean_lateral_error",
    "std_lateral_error",
    "max_lateral_error",
    "mean_heading_error",
    "mean_speed",
    "mean_steer_rate",
    "mean_steer_tv",
    "mean_steer_tv_per_step",
    "steer_clip_frac",
    "speed_clip_frac",
    "mean_reward_progress",
    "mean_reward_a_long_pen",
    "mean_reward_a_lat_pen",
    "mean_reward_time_pen",
    "mean_reward_crash_pen",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_run_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = run_dir / "run_meta.json"
    return load_json(meta_path) if meta_path.exists() else None


def load_eval_data(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Returns (eval_data, source_tag) where source_tag indicates which file was loaded."""
    candidates = [
        (run_dir / "eval_standalone_test.json", "test"),
        (run_dir / "eval_standalone_validation.json", "validation_fallback"),
        (run_dir / "eval_standalone.json", "generic_fallback"),
    ]
    for path, tag in candidates:
        if path.exists():
            return load_json(path), tag

    eval_dir = run_dir / "eval_results"
    if eval_dir.exists():
        snapshots = sorted(eval_dir.glob("eval_*.json"))
        if snapshots:
            return load_json(snapshots[-1]), "training_snapshot_fallback"

    return None, "missing"


def _prefix_summary(row: Dict[str, Any], prefix: str, summary: Dict[str, Any]) -> None:
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            row[f"{prefix}_{key}"] = value


def aggregate_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    meta = load_run_meta(run_dir)
    eval_data, eval_source = load_eval_data(run_dir)
    if eval_data is None:
        return None

    # FIX (M1): Warn when test eval data is not available
    if eval_source != "test":
        warnings.warn(
            f"Run '{run_dir.name}': No test eval found, using '{eval_source}'. "
            f"Columns prefixed with 'test_' will be empty for this run. "
            f"Re-run eval.py with --evaluation_split test to fix.",
            stacklevel=2,
        )

    row: Dict[str, Any] = {
        "run_id": run_dir.name,
        "eval_source": eval_source,
    }

    if meta:
        row["action_space"] = meta.get("action_space", "unknown")
        row["obs_regime"] = "ablated" if meta.get("ablate_geometry", False) else "full"
        row["seed"] = int(meta.get("seed", 0))
        row["train_track"] = meta.get("train_track")
        row["selection_metric"] = meta.get("selection_metric")
        row["selected_checkpoint_name"] = meta.get("selected_checkpoint_name")
        row["total_params"] = meta.get("total_params")
        row["trainable_params"] = meta.get("trainable_params")
        row["action_dim"] = meta.get("action_dim")
    else:
        row["action_space"] = "unknown"
        row["obs_regime"] = "unknown"
        row["seed"] = 0

    if "overall_summary" in eval_data:
        _prefix_summary(row, "overall", eval_data["overall_summary"])

    for split_name in ("train", "validation", "test", "custom"):
        summary = eval_data.get(f"{split_name}_summary")
        if summary:
            # FIX (M1): Only populate test_ columns from actual test data
            if split_name == "test" and eval_source != "test":
                # Don't populate test_ columns from non-test sources
                continue
            _prefix_summary(row, split_name, summary)

    # Backfill overall_* from per-track data if needed
    if not any(key.startswith("overall_") for key in row.keys()) and "tracks" in eval_data:
        episodes: List[Dict[str, Any]] = []
        for track, track_data in eval_data["tracks"].items():
            for ep in track_data.get("episodes", []):
                ep_copy = dict(ep)
                ep_copy["track"] = track
                episodes.append(ep_copy)

        if episodes:
            row["overall_n_episodes"] = len(episodes)
            row["overall_mean_reward"] = float(np.mean([e.get("reward", 0.0) for e in episodes]))
            row["overall_std_reward"] = float(np.std([e.get("reward", 0.0) for e in episodes]))
            row["overall_mean_progress"] = float(np.mean([e.get("normalized_progress", 0.0) for e in episodes]))
            row["overall_completion_rate"] = float(np.mean([e.get("normalized_progress", 0.0) >= 0.95 for e in episodes]))
            row["overall_crash_rate"] = float(np.mean([e.get("term_reason") == "crash" for e in episodes]))

    return row


def build_summary_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["action_space", "obs_regime"]
    exclude_cols = group_cols + [
        "seed", "run_id", "train_track", "selection_metric",
        "selected_checkpoint_name", "eval_source",
    ]
    metric_cols = [
        c for c in per_seed_df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(per_seed_df[c])
    ]

    rows: List[Dict[str, Any]] = []
    for (action_space, obs_regime), group in per_seed_df.groupby(group_cols):
        row: Dict[str, Any] = {
            "action_space": action_space,
            "obs_regime": obs_regime,
            "n_seeds": int(len(group)),
        }
        for col in metric_cols:
            vals = group[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_track_table(checkpoints_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in checkpoints_dir.iterdir() if p.is_dir()):
        meta = load_run_meta(run_dir)
        eval_data, eval_source = load_eval_data(run_dir)
        if eval_data is None or "tracks" not in eval_data:
            continue
        action_space = meta.get("action_space", "unknown") if meta else "unknown"
        obs_regime = "ablated" if meta and meta.get("ablate_geometry", False) else "full"
        seed = int(meta.get("seed", 0)) if meta else 0

        for track, track_data in eval_data["tracks"].items():
            summary = track_data.get("summary", {})
            row = {
                "run_id": run_dir.name,
                "action_space": action_space,
                "obs_regime": obs_regime,
                "seed": seed,
                "track": track,
                "track_group": track_data.get("track_group", "unknown"),
                "eval_source": eval_source,
            }
            row.update(summary)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate patched experiment results")
    ap.add_argument("--checkpoints_dir", default="checkpoints")
    ap.add_argument("--output", default="results")
    ap.add_argument("--require_test", action="store_true",
                    help="Skip runs without proper test eval data")
    args = ap.parse_args()

    checkpoints_dir = ROOT / args.checkpoints_dir
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for run_dir in sorted(p for p in checkpoints_dir.iterdir() if p.is_dir()):
        row = aggregate_run(run_dir)
        if row is None:
            continue
        if args.require_test and row.get("eval_source") != "test":
            print(f"SKIPPING {run_dir.name}: eval_source={row.get('eval_source')} (--require_test)")
            skipped += 1
            continue
        rows.append(row)

    if not rows:
        raise SystemExit(f"No evaluation results found. ({skipped} skipped by --require_test)")

    per_seed_df = pd.DataFrame(rows).sort_values(["action_space", "obs_regime", "seed", "run_id"])
    summary_df = build_summary_table(per_seed_df)
    per_track_df = build_per_track_table(checkpoints_dir)

    per_seed_path = output_dir / "per_seed_results.csv"
    summary_path = output_dir / "summary_table.csv"
    per_track_path = output_dir / "per_track_results.csv"

    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    per_track_df.to_csv(per_track_path, index=False)

    print(f"Wrote: {per_seed_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_track_path}")

    # Report eval source distribution
    if "eval_source" in per_seed_df.columns:
        source_counts = per_seed_df["eval_source"].value_counts()
        print(f"\nEval source distribution:")
        for src, count in source_counts.items():
            print(f"  {src}: {count}")
        non_test = (per_seed_df["eval_source"] != "test").sum()
        if non_test > 0:
            print(f"\n⚠ WARNING: {non_test} runs lack proper test eval data!")

    display_cols = [
        c for c in [
            "action_space",
            "obs_regime",
            "n_seeds",
            "test_mean_progress_mean",
            "test_completion_rate_mean",
            "test_crash_rate_mean",
            "test_mean_lateral_error_mean",
        ]
        if c in summary_df.columns
    ]
    if display_cols:
        print("\nSummary preview:")
        print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
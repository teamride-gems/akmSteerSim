#!/usr/bin/env python3
"""
Generate paper figures from experiment results.

Produces:
  Figure 1: Learning curves across action spaces (mean ± stderr across seeds)
  Figure 2: Observation ablation comparison
  Figure 3: Mechanistic analysis (smoothness, constraint saturation)

Usage:
  python scripts/plot_paper_figures.py
  python scripts/plot_paper_figures.py --results results/ --output figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# try importing matplotlib, fail gracefully
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install with: pip install matplotlib")


# consistent styling
ACTION_SPACE_COLORS = {
    "steer_speed": "#d62728",
    "curvature_speed": "#2ca02c",
    "lookahead_point": "#1f77b4",
    "bezier": "#9467bd",
}

ACTION_SPACE_LABELS = {
    "steer_speed": "Direct (δ, v)",
    "curvature_speed": "Curvature (κ, v)",
    "lookahead_point": "Lookahead (x, y, v)",
    "bezier": "Bézier",
}


def load_training_curves(checkpoints_dir: Path) -> pd.DataFrame:
    """
    Load eval metrics over training from eval JSON snapshots.
    Returns DataFrame with columns: action_space, obs_regime, seed, timestep, + metrics
    """
    rows = []
    for run_dir in sorted(checkpoints_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        action_space = meta.get("action_space", "unknown")
        obs_regime = "ablated" if meta.get("ablate_geometry", False) else "full"
        seed = meta.get("seed", 0)

        eval_dir = run_dir / "eval_results"
        if not eval_dir.exists():
            continue

        for eval_json in sorted(eval_dir.glob("eval_*.json")):
            with open(eval_json) as f:
                data = json.load(f)

            timestep = data.get("timestep", 0)
            episodes = []
            for track_eps in data.get("tracks", {}).values():
                episodes.extend(track_eps)

            if not episodes:
                continue

            rows.append({
                "action_space": action_space,
                "obs_regime": obs_regime,
                "seed": seed,
                "timestep": timestep,
                "mean_reward": np.mean([e["reward"] for e in episodes]),
                "mean_progress": np.mean([e["normalized_progress"] for e in episodes]),
                "crash_rate": np.mean([1 if e.get("term_reason") == "crash" else 0 for e in episodes]),
                "mean_lateral_error": np.mean([e.get("mean_lateral_error", 0) for e in episodes]),
                "mean_steer_rate": np.mean([e.get("mean_abs_steer_rate", 0) for e in episodes]),
                "steer_clip_frac": np.mean([e.get("steer_clip_frac", 0) for e in episodes]),
                "mean_steer_tv": np.mean([e.get("steer_tv", 0) for e in episodes]),
            })

    return pd.DataFrame(rows)


def plot_learning_curves(df: pd.DataFrame, output_dir: Path, metric: str = "mean_reward", ylabel: str = "Mean Return"):
    """Figure: learning curves for full-observation runs, mean ± stderr across seeds."""
    if not HAS_MPL:
        return

    full_df = df[df["obs_regime"] == "full"]
    if full_df.empty:
        print(f"No full-observation data for {metric}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for action_space in full_df["action_space"].unique():
        sub = full_df[full_df["action_space"] == action_space]
        grouped = sub.groupby("timestep")[metric]
        mean = grouped.mean()
        se = grouped.std() / np.sqrt(grouped.count())

        color = ACTION_SPACE_COLORS.get(action_space, "gray")
        label = ACTION_SPACE_LABELS.get(action_space, action_space)

        ax.plot(mean.index, mean.values, color=color, label=label, linewidth=2)
        ax.fill_between(mean.index, (mean - se).values, (mean + se).values, color=color, alpha=0.15)

    ax.set_xlabel("Training Timesteps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / f"learning_curves_{metric}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ablation_comparison(df: pd.DataFrame, output_dir: Path):
    """Figure: bar chart comparing full vs ablated observation for each action space."""
    if not HAS_MPL:
        return

    # use final timestep per run
    final = df.loc[df.groupby(["action_space", "obs_regime", "seed"])["timestep"].idxmax()]

    if final.empty:
        print("No data for ablation comparison")
        return

    action_spaces = sorted(final["action_space"].unique())
    obs_regimes = ["full", "ablated"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("mean_reward", "Mean Return"),
        ("mean_progress", "Normalized Progress"),
        ("mean_lateral_error", "Mean Lateral Error (m)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(action_spaces))
        width = 0.35

        for i, obs in enumerate(obs_regimes):
            means = []
            errs = []
            for action_space in action_spaces:
                sub = final[(final["action_space"] == action_space) & (final["obs_regime"] == obs)]
                vals = sub[metric].values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            offset = -width / 2 + i * width
            label = "Full obs" if obs == "full" else "Ablated (no e_head, e_lat)"
            color = "#1f77b4" if obs == "full" else "#ff7f0e"
            ax.bar(x + offset, means, width, yerr=errs, label=label, color=color, alpha=0.8, capsize=3)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ACTION_SPACE_LABELS.get(a, a) for a in action_spaces], fontsize=9, rotation=15)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "ablation_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mechanistic_analysis(df: pd.DataFrame, output_dir: Path):
    """Figure: smoothness and constraint analysis across action spaces."""
    if not HAS_MPL:
        return

    full_df = df[df["obs_regime"] == "full"]
    final = full_df.loc[full_df.groupby(["action_space", "seed"])["timestep"].idxmax()]

    if final.empty:
        print("No data for mechanistic analysis")
        return

    action_spaces = sorted(final["action_space"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = [
        ("mean_steer_rate", "Mean |Steering Rate| (rad/s)"),
        ("steer_clip_frac", "Steering Clipping Fraction"),
        ("mean_steer_tv", "Steering Total Variation"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means = []
        errs = []
        colors = []
        labels = []

        for action_space in action_spaces:
            sub = final[final["action_space"] == action_space]
            vals = sub[metric].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            errs.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            colors.append(ACTION_SPACE_COLORS.get(action_space, "gray"))
            labels.append(ACTION_SPACE_LABELS.get(action_space, action_space))

        x = np.arange(len(action_spaces))
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.8, capsize=4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "mechanistic_analysis.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", default="checkpoints")
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()

    ckpt_dir = ROOT / args.checkpoints_dir
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_dir.exists():
        print(f"No checkpoints directory at {ckpt_dir}")
        return

    print("Loading training curves...")
    df = load_training_curves(ckpt_dir)

    if df.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df)} eval snapshots across {df['seed'].nunique()} seeds")
    print(f"Action spaces: {sorted(df['action_space'].unique())}")
    print(f"Obs regimes: {sorted(df['obs_regime'].unique())}")

    print("\nGenerating figures...")
    plot_learning_curves(df, output_dir, metric="mean_reward", ylabel="Mean Return")
    plot_learning_curves(df, output_dir, metric="mean_progress", ylabel="Normalized Progress")
    plot_ablation_comparison(df, output_dir)
    plot_mechanistic_analysis(df, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
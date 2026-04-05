#!/usr/bin/env python3
"""
Generate all paper figures from experiment results.

Reads from:
  - checkpoints/*/eval_results/eval_*.json  (training curves)
  - results/per_seed_results.csv            (final performance, from aggregate_results.py)
  - rollouts/*.npz                          (steering profiles, from export_trajectories.py)

Produces:
  Figure 1: Learning curves (reward, progress, smoothness) — overall and heldout
  Figure 2: Final performance comparison bar chart
  Figure 3: Train vs. heldout generalization
  Figure 4: Observation ablation comparison
  Figure 5: Mechanistic analysis (smoothness, constraint saturation)
  Figure 6: Steering profiles (per-step commands from trajectory data)
  Figure 7: Reward decomposition
  Figure 8: Wall-clock training time comparison

Usage:
  python scripts/plot_paper_figures.py
  python scripts/plot_paper_figures.py --checkpoints_dir checkpoints --output figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install with: pip install matplotlib")


# ----------------------------
# Publication-quality defaults
# ----------------------------

def _setup_rcparams():
    """Configure matplotlib for publication-quality output."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        # Fonts: Type 1 / TrueType (required by most venues)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ----------------------------
# Consistent styling
# ----------------------------

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

def _color(action_space: str) -> str:
    return ACTION_SPACE_COLORS.get(action_space, "gray")

def _label(action_space: str) -> str:
    return ACTION_SPACE_LABELS.get(action_space, action_space)

def _has_data(df) -> bool:
    """Safe check for non-empty DataFrame, handling None."""
    return isinstance(df, pd.DataFrame) and not df.empty


# ----------------------------
# Data loading
# ----------------------------

def load_training_curves(checkpoints_dir: Path) -> pd.DataFrame:
    """Load eval metrics over training from eval JSON snapshots."""
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
        train_track = meta.get("train_track", None)

        eval_dir = run_dir / "eval_results"
        if not eval_dir.exists():
            continue

        for eval_json in sorted(eval_dir.glob("eval_*.json")):
            with open(eval_json) as f:
                data = json.load(f)

            timestep = data.get("timestep", 0)
            wall_hours = data.get("wall_clock_hours", None)

            all_episodes = []
            for track_eps in data.get("tracks", {}).values():
                if isinstance(track_eps, list):
                    all_episodes.extend(track_eps)

            if not all_episodes:
                continue

            heldout_episodes = []
            train_episodes = []
            if train_track:
                for track_name, track_eps in data.get("tracks", {}).items():
                    if isinstance(track_eps, list):
                        if track_name == train_track:
                            train_episodes.extend(track_eps)
                        else:
                            heldout_episodes.extend(track_eps)

            def _ep_metrics(episodes, prefix=""):
                if not episodes:
                    return {}
                n = len(episodes)
                return {
                    f"{prefix}mean_reward": np.mean([e["reward"] for e in episodes]),
                    f"{prefix}mean_progress": np.mean([e["normalized_progress"] for e in episodes]),
                    f"{prefix}crash_rate": sum(1 for e in episodes if e.get("term_reason") == "crash") / n,
                    f"{prefix}mean_lateral_error": np.mean([e.get("mean_lateral_error", 0) for e in episodes]),
                    f"{prefix}mean_steer_rate": np.mean([e.get("mean_abs_steer_rate", 0) for e in episodes]),
                    f"{prefix}steer_clip_frac": np.mean([e.get("steer_clip_frac", 0) for e in episodes]),
                    f"{prefix}mean_steer_tv": np.mean([e.get("steer_tv", 0) for e in episodes]),
                    f"{prefix}mean_steer_tv_per_step": np.mean([e.get("steer_tv_per_step", 0) for e in episodes]),
                    f"{prefix}completion_rate": sum(1 for e in episodes if e.get("normalized_progress", 0) >= 0.95) / n,
                    f"{prefix}mean_reward_progress": np.mean([e.get("mean_reward_progress", 0) for e in episodes]),
                    f"{prefix}mean_reward_a_long_pen": np.mean([e.get("mean_reward_a_long_pen", 0) for e in episodes]),
                    f"{prefix}mean_reward_a_lat_pen": np.mean([e.get("mean_reward_a_lat_pen", 0) for e in episodes]),
                }

            row = {
                "action_space": action_space,
                "obs_regime": obs_regime,
                "seed": seed,
                "timestep": timestep,
                "wall_clock_hours": wall_hours,
            }
            row.update(_ep_metrics(all_episodes))
            row.update(_ep_metrics(heldout_episodes, prefix="heldout_"))
            row.update(_ep_metrics(train_episodes, prefix="train_"))
            rows.append(row)

    return pd.DataFrame(rows)


def load_per_seed_csv(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load the per-seed CSV from aggregate_results.py."""
    path = results_dir / "per_seed_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _get_final(df: pd.DataFrame, full_only: bool = True) -> pd.DataFrame:
    """Extract final-timestep data, optionally filtering to full-observation runs."""
    if full_only and "obs_regime" in df.columns:
        df = df[df["obs_regime"] == "full"]
    if "timestep" in df.columns:
        return df.loc[df.groupby(["action_space", "seed"])["timestep"].idxmax()]
    return df


# ----------------------------
# Figure 1: Learning curves
# ----------------------------

def _plot_curves_panel(df: pd.DataFrame, output_dir: Path, metrics, filename: str, title_suffix: str = ""):
    """Shared logic for learning curve panels."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        for action_space in sorted(df["action_space"].unique()):
            sub = df[df["action_space"] == action_space]
            grouped = sub.groupby("timestep")[metric]
            mean = grouped.mean()
            count = grouped.count()
            se = grouped.std() / np.sqrt(count)

            ax.plot(mean.index, mean.values, color=_color(action_space),
                    label=_label(action_space), linewidth=2)
            ax.fill_between(mean.index, (mean - se).values, (mean + se).values,
                            color=_color(action_space), alpha=0.15)

        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_learning_curves(df: pd.DataFrame, output_dir: Path):
    """Learning curves: overall and heldout-only variants."""
    full_df = df[df["obs_regime"] == "full"] if "obs_regime" in df.columns else df
    if full_df.empty:
        print("No full-observation data for learning curves")
        return

    overall_metrics = [
        ("mean_reward", "Mean Return"),
        ("mean_progress", "Normalized Progress"),
        ("mean_steer_tv_per_step", "Steering TV / Step"),
    ]
    _plot_curves_panel(full_df, output_dir, overall_metrics,
                       "fig1a_learning_curves_overall.pdf")

    # Heldout-only curves (if data exists)
    heldout_metrics = [
        ("heldout_mean_reward", "Mean Return (Heldout)"),
        ("heldout_mean_progress", "Progress (Heldout)"),
        ("heldout_mean_steer_tv_per_step", "TV / Step (Heldout)"),
    ]
    if "heldout_mean_reward" in full_df.columns and full_df["heldout_mean_reward"].notna().any():
        _plot_curves_panel(full_df, output_dir, heldout_metrics,
                           "fig1b_learning_curves_heldout.pdf")


# ----------------------------
# Figure 2: Final performance comparison
# ----------------------------

def plot_final_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing all action spaces on key metrics at convergence."""
    if not HAS_MPL:
        return

    final = _get_final(df)
    if final.empty:
        print("No data for final comparison")
        return

    action_spaces = sorted(final["action_space"].unique())

    metrics = [
        ("mean_reward", "Mean Return"),
        ("mean_progress", "Progress"),
        ("completion_rate", "Completion Rate"),
        ("crash_rate", "Crash Rate"),
        ("mean_lateral_error", "Lat. Error (m)"),
        ("mean_steer_tv_per_step", "TV / Step"),
    ]
    metrics = [(m, l) for m, l in metrics if m in final.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means, errs, colors = [], [], []
        for action_space in action_spaces:
            vals = final[final["action_space"] == action_space][metric].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0)
            errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)
            colors.append(_color(action_space))

        x = np.arange(len(action_spaces))
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")

    fig.tight_layout()
    path = output_dir / "fig2_final_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 3: Train vs. heldout
# ----------------------------

def plot_train_vs_heldout(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: train-track vs. heldout-track performance."""
    if not HAS_MPL:
        return

    full_df = df[df["obs_regime"] == "full"] if "obs_regime" in df.columns else df

    metrics = [
        ("mean_reward", "Mean Return"),
        ("mean_lateral_error", "Lat. Error (m)"),
        ("mean_steer_tv_per_step", "TV / Step"),
    ]
    metrics = [(m, l) for m, l in metrics
               if f"train_{m}" in full_df.columns and f"heldout_{m}" in full_df.columns]

    if not metrics:
        print("No train/heldout split data — skipping generalization figure")
        return

    action_spaces = sorted(full_df["action_space"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(action_spaces))
        width = 0.35

        for i, (prefix, label, color) in enumerate([
            ("train_", "Train Track", "#1f77b4"),
            ("heldout_", "Heldout Tracks", "#ff7f0e"),
        ]):
            means, errs = [], []
            col = f"{prefix}{metric}"
            for action_space in action_spaces:
                vals = full_df[full_df["action_space"] == action_space][col].dropna()
                means.append(float(vals.mean()) if len(vals) > 0 else 0)
                errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)

            offset = -width / 2 + i * width
            ax.bar(x + offset, means, width, yerr=errs, label=label,
                   color=color, alpha=0.8, capsize=3)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")
        ax.legend()

    fig.tight_layout()
    path = output_dir / "fig3_train_vs_heldout.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 4: Ablation comparison
# ----------------------------

def plot_ablation_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing full vs. ablated observations per action space."""
    if not HAS_MPL:
        return

    if "timestep" in df.columns:
        final = df.loc[df.groupby(["action_space", "obs_regime", "seed"])["timestep"].idxmax()]
    else:
        final = df

    if final.empty or "obs_regime" not in final.columns or final["obs_regime"].nunique() < 2:
        print("No ablation data — skipping")
        return

    action_spaces = sorted(final["action_space"].unique())

    metrics = [
        ("mean_reward", "Mean Return"),
        ("mean_progress", "Normalized Progress"),
        ("mean_lateral_error", "Lat. Error (m)"),
    ]
    metrics = [(m, l) for m, l in metrics if m in final.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(action_spaces))
        width = 0.35

        for i, (obs, label, color) in enumerate([
            ("full", "Full observation", "#1f77b4"),
            ("ablated", "Ablated (no e_head, e_lat)", "#ff7f0e"),
        ]):
            means, errs = [], []
            for action_space in action_spaces:
                sub = final[(final["action_space"] == action_space) & (final["obs_regime"] == obs)]
                vals = sub[metric].dropna()
                means.append(float(vals.mean()) if len(vals) > 0 else 0)
                errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)

            offset = -width / 2 + i * width
            ax.bar(x + offset, means, width, yerr=errs, label=label,
                   color=color, alpha=0.8, capsize=3)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")
        ax.legend()

    fig.tight_layout()
    path = output_dir / "fig4_ablation.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 5: Mechanistic analysis
# ----------------------------

def plot_mechanistic_analysis(df: pd.DataFrame, output_dir: Path):
    """Smoothness and constraint analysis bar charts."""
    if not HAS_MPL:
        return

    final = _get_final(df)
    if final.empty:
        print("No data for mechanistic analysis")
        return

    action_spaces = sorted(final["action_space"].unique())

    metrics = [
        ("mean_steer_rate", "Mean |Steering Rate| (rad/s)"),
        ("steer_clip_frac", "Steering Clipping Fraction"),
        ("mean_steer_tv_per_step", "Steering TV / Step"),
    ]
    metrics = [(m, l) for m, l in metrics if m in final.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means, errs, colors = [], [], []
        for action_space in action_spaces:
            vals = final[final["action_space"] == action_space][metric].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0)
            errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)
            colors.append(_color(action_space))

        x = np.arange(len(action_spaces))
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")

    fig.tight_layout()
    path = output_dir / "fig5_mechanistic.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 6: Steering profiles
# ----------------------------

def plot_steering_profiles(output_dir: Path, rollout_dir: Path = None, window: int = 300):
    """
    Per-step steering command timeseries overlaid across action spaces.

    Only uses completed episodes (normalized_progress >= 0.9) to avoid
    showing truncated crash trajectories. Shows a window of `window` steps
    to highlight smoothness differences.
    """
    if not HAS_MPL:
        return

    if rollout_dir is None:
        rollout_dir = ROOT / "rollouts"
    if not rollout_dir.exists():
        print("No rollout directory found — skipping steering profiles")
        return

    npz_files = sorted(rollout_dir.glob("*.npz"))
    if not npz_files:
        print("No .npz rollouts found — skipping steering profiles")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_steer, ax_rate = axes

    plotted = set()
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        if "steer_cmd" not in data:
            continue

        action_space = str(data.get("action_space", "unknown"))

        # Only plot one rollout per action space (first seed found)
        if action_space in plotted:
            continue

        # Skip crashed/incomplete episodes
        progress = float(data.get("normalized_progress", 0.0))
        if progress < 0.9:
            continue

        steer = data["steer_cmd"][:window]
        steps = np.arange(len(steer))

        ax_steer.plot(steps, steer, color=_color(action_space),
                      label=_label(action_space), linewidth=1.5, alpha=0.9)

        # Steering rate (finite difference)
        if "steer_rate" in data:
            rate = data["steer_rate"][:window]
        else:
            rate = np.concatenate([[0.0], np.diff(steer)])
        ax_rate.plot(steps, rate, color=_color(action_space),
                     linewidth=1.2, alpha=0.8)

        plotted.add(action_space)

    if not plotted:
        # Fall back: plot whatever is available, even incomplete
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            if "steer_cmd" not in data:
                continue
            action_space = str(data.get("action_space", "unknown"))
            if action_space in plotted:
                continue
            steer = data["steer_cmd"][:window]
            ax_steer.plot(np.arange(len(steer)), steer, color=_color(action_space),
                          label=_label(action_space), linewidth=1.5, alpha=0.9)
            plotted.add(action_space)

    if not plotted:
        plt.close(fig)
        print("No usable steering profile data — skipping")
        return

    ax_steer.set_ylabel("Steering Command (rad)")
    ax_steer.legend()
    ax_rate.set_ylabel("Steering Rate (rad/s)")
    ax_rate.set_xlabel("Step")

    fig.tight_layout()
    path = output_dir / "fig6_steering_profiles.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 7: Reward decomposition
# ----------------------------

def plot_reward_decomposition(df: pd.DataFrame, output_dir: Path):
    """Bar chart showing reward component breakdown per action space."""
    if not HAS_MPL:
        return

    final = _get_final(df)
    needed = ["mean_reward_progress", "mean_reward_a_long_pen", "mean_reward_a_lat_pen"]
    if not all(c in final.columns for c in needed):
        print("No reward decomposition data — skipping")
        return

    if final.empty:
        return

    action_spaces = sorted(final["action_space"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    components = [
        ("mean_reward_progress", "Progress Reward (per step)"),
        ("mean_reward_a_long_pen", "Long. Accel Penalty (per step)"),
        ("mean_reward_a_lat_pen", "Lat. Accel Penalty (per step)"),
    ]

    for ax, (metric, ylabel) in zip(axes, components):
        means, errs, colors = [], [], []
        for action_space in action_spaces:
            vals = final[final["action_space"] == action_space][metric].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0)
            errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)
            colors.append(_color(action_space))

        x = np.arange(len(action_spaces))
        ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")

    fig.tight_layout()
    path = output_dir / "fig7_reward_decomposition.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Figure 8: Wall-clock training time
# ----------------------------

def plot_wall_clock(df: pd.DataFrame, output_dir: Path):
    """
    Bar chart of wall-clock training hours per action space.
    If one action space is structurally cheaper (e.g. 2D vs 5D policy),
    this is a practical result worth reporting.
    """
    if not HAS_MPL:
        return

    full_df = df[df["obs_regime"] == "full"] if "obs_regime" in df.columns else df

    if "wall_clock_hours" not in full_df.columns or full_df["wall_clock_hours"].isna().all():
        print("No wall-clock data — skipping")
        return

    # Take the final eval snapshot per run (max timestep = most training done)
    final = full_df.loc[full_df.groupby(["action_space", "seed"])["timestep"].idxmax()]
    final = final.dropna(subset=["wall_clock_hours"])

    if final.empty:
        print("No wall-clock data after filtering — skipping")
        return

    action_spaces = sorted(final["action_space"].unique())

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    means, errs, colors = [], [], []
    for action_space in action_spaces:
        vals = final[final["action_space"] == action_space]["wall_clock_hours"].dropna()
        means.append(float(vals.mean()) if len(vals) > 0 else 0)
        errs.append(float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0)
        colors.append(_color(action_space))

    x = np.arange(len(action_spaces))
    ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
    ax.set_ylabel("Training Time (hours)")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a) for a in action_spaces], rotation=20, ha="right")

    fig.tight_layout()
    path = output_dir / "fig8_wall_clock.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ----------------------------
# Main
# ----------------------------

def main():
    _setup_rcparams()

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_dir", default="checkpoints")
    ap.add_argument("--results_dir", default="results",
                     help="Directory with per_seed_results.csv from aggregate_results.py")
    ap.add_argument("--output", default="figures")
    ap.add_argument("--rollout_dir", default=None,
                     help="Directory with .npz rollouts for steering profiles")
    args = ap.parse_args()

    ckpt_dir = ROOT / args.checkpoints_dir
    results_dir = ROOT / args.results_dir
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training curves from eval snapshots ---
    curve_df = pd.DataFrame()
    if ckpt_dir.exists():
        print("Loading training curves...")
        curve_df = load_training_curves(ckpt_dir)
        if not curve_df.empty:
            print(f"  {len(curve_df)} snapshots, {curve_df['seed'].nunique()} seeds")
            print(f"  Action spaces: {sorted(curve_df['action_space'].unique())}")

    # --- Load final results from aggregate_results.py ---
    final_df = load_per_seed_csv(results_dir)
    if final_df is not None:
        print(f"Loaded per-seed results: {len(final_df)} runs")

    # Prefer aggregate CSV for final figures, fall back to training curves
    data_for_final = final_df if _has_data(final_df) else curve_df
    data_for_ablation = final_df if _has_data(final_df) else curve_df

    # --- Generate figures ---
    print("\nGenerating figures...")

    # Fig 1: Learning curves (from training snapshots)
    if _has_data(curve_df):
        plot_learning_curves(curve_df, output_dir)

    # Fig 2: Final performance
    if _has_data(data_for_final):
        plot_final_comparison(data_for_final, output_dir)

    # Fig 3: Train vs. heldout (requires aggregate CSV with split columns)
    if _has_data(final_df):
        plot_train_vs_heldout(final_df, output_dir)

    # Fig 4: Ablation
    if _has_data(data_for_ablation):
        plot_ablation_comparison(data_for_ablation, output_dir)

    # Fig 5: Mechanistic analysis
    if _has_data(data_for_final):
        plot_mechanistic_analysis(data_for_final, output_dir)

    # Fig 6: Steering profiles (from trajectory data)
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else None
    plot_steering_profiles(output_dir, rollout_dir=rollout_dir)

    # Fig 7: Reward decomposition
    if _has_data(data_for_final):
        plot_reward_decomposition(data_for_final, output_dir)

    # Fig 8: Wall-clock training time
    if _has_data(curve_df):
        plot_wall_clock(curve_df, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
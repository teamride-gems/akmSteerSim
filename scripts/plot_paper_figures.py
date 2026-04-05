#!/usr/bin/env python3
"""
Robust plotting script for the patched pipeline.

Defaults:
  - headline bars use TEST metrics
  - train vs test gap is explicit
  - reward decomposition includes time/crash terms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _final_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _plot_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, output_path: Path) -> None:
    if value_col not in df.columns:
        return
    plot_df = df.sort_values(["obs_regime", value_col], ascending=[True, False])
    labels = [f"{a}\n({o})" for a, o in zip(plot_df["action_space"], plot_df["obs_regime"])]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, plot_df[value_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_final_test_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_bar(
        summary_df,
        "test_mean_progress_mean",
        "Final comparison on untouched test tracks",
        "Mean normalized progress",
        output_dir / "fig_test_progress.png",
    )
    _plot_bar(
        summary_df,
        "test_completion_rate_mean",
        "Test completion rate",
        "Completion rate",
        output_dir / "fig_test_completion.png",
    )
    _plot_bar(
        summary_df,
        "test_mean_lateral_error_mean",
        "Test lateral error",
        "Mean lateral error (m)",
        output_dir / "fig_test_lateral_error.png",
    )


def plot_train_vs_test_gap(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if "train_mean_progress_mean" not in summary_df.columns or "test_mean_progress_mean" not in summary_df.columns:
        return
    plot_df = summary_df.copy()
    plot_df["generalization_gap_progress"] = plot_df["train_mean_progress_mean"] - plot_df["test_mean_progress_mean"]
    _plot_bar(
        plot_df,
        "generalization_gap_progress",
        "Generalization gap (train - test)",
        "Progress gap",
        output_dir / "fig_generalization_gap_progress.png",
    )


def plot_ablation(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if "test_mean_progress_mean" not in summary_df.columns:
        return
    _plot_bar(
        summary_df,
        "test_mean_progress_mean",
        "Ablation-sensitive final test progress",
        "Mean normalized progress",
        output_dir / "fig_ablation_test_progress.png",
    )


def plot_reward_decomposition(summary_df: pd.DataFrame, output_dir: Path) -> None:
    needed = [
        "test_mean_reward_progress_mean",
        "test_mean_reward_a_long_pen_mean",
        "test_mean_reward_a_lat_pen_mean",
        "test_mean_reward_time_pen_mean",
        "test_mean_reward_crash_pen_mean",
    ]
    if not all(c in summary_df.columns for c in needed):
        return

    for _, row in summary_df.iterrows():
        labels = ["progress", "a_long", "a_lat", "time", "crash"]
        vals = [
            row["test_mean_reward_progress_mean"],
            row["test_mean_reward_a_long_pen_mean"],
            row["test_mean_reward_a_lat_pen_mean"],
            row["test_mean_reward_time_pen_mean"],
            row["test_mean_reward_crash_pen_mean"],
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, vals)
        plt.ylabel("Mean per-step contribution")
        plt.title(f"Reward decomposition (test)\n{row['action_space']} [{row['obs_regime']}]")
        plt.tight_layout()
        stem = f"{row['action_space']}_{row['obs_regime']}_reward_decomp.png"
        plt.savefig(output_dir / stem, dpi=200)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot patched paper figures")
    ap.add_argument("--summary_csv", default="results/summary_table.csv")
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()

    summary_df = _load_csv(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_final_test_comparison(summary_df, output_dir)
    plot_train_vs_test_gap(summary_df, output_dir)
    plot_ablation(summary_df, output_dir)
    plot_reward_decomposition(summary_df, output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()

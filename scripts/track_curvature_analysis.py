#!/usr/bin/env python3
"""
Compute curvature statistics for each track in the F1TENTH dataset.

Outputs a CSV and prints a summary table so you can:
  - pick tracks that span gentle → aggressive curvature
  - stratify experimental results by track geometry
  - connect action-space performance to curvature regime

Usage:
  python scripts/track_curvature_analysis.py
  python scripts/track_curvature_analysis.py --output results/track_stats.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "assets" / "f1tenth_racetracks"


def compute_curvature_from_centerline(centerline: np.ndarray) -> np.ndarray:
    """
    Compute discrete curvature along a 2D centerline using finite differences.
    centerline: (N, 2+) array, columns 0,1 are x,y.
    Returns: (N,) curvature array (unsigned).
    """
    x = centerline[:, 0]
    y = centerline[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    denom = np.maximum(denom, 1e-10)
    curvature = np.abs(dx * ddy - dy * ddx) / denom

    return curvature


def analyze_track(track_dir: Path) -> Dict:
    """Compute all geometric statistics for one track."""
    track_name = track_dir.name
    cl_path = track_dir / f"{track_name}_centerline.csv"

    if not cl_path.exists():
        return {}

    centerline = np.loadtxt(cl_path, delimiter=",", ndmin=2)
    if centerline.shape[0] < 5:
        return {}

    # arc length
    diffs = np.diff(centerline[:, :2], axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    track_length = float(np.sum(seg_lengths))

    # curvature
    kappa = compute_curvature_from_centerline(centerline)

    # classify segments by curvature threshold
    straight_threshold = 0.5  # 1/m — radius > 2m is "straight-ish"
    tight_threshold = 2.0     # 1/m — radius < 0.5m is "tight"

    frac_straight = float(np.mean(kappa < straight_threshold))
    frac_tight = float(np.mean(kappa > tight_threshold))

    return {
        "track": track_name,
        "track_length_m": track_length,
        "n_points": int(centerline.shape[0]),
        "curvature_mean": float(np.mean(kappa)),
        "curvature_median": float(np.median(kappa)),
        "curvature_max": float(np.max(kappa)),
        "curvature_std": float(np.std(kappa)),
        "curvature_p90": float(np.percentile(kappa, 90)),
        "curvature_p99": float(np.percentile(kappa, 99)),
        "frac_straight": frac_straight,
        "frac_tight": frac_tight,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/track_stats.csv")
    args = ap.parse_args()

    if not TRACKS_DIR.exists():
        print(f"Track directory not found: {TRACKS_DIR}")
        return

    track_dirs = sorted([d for d in TRACKS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(track_dirs)} tracks in {TRACKS_DIR}\n")

    rows = []
    for td in track_dirs:
        stats = analyze_track(td)
        if stats:
            rows.append(stats)

    if not rows:
        print("No valid tracks found.")
        return

    df = pd.DataFrame(rows).sort_values("curvature_mean")

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")

    # print summary
    print(f"{'Track':<25} {'Length':>8} {'κ_mean':>8} {'κ_max':>8} {'κ_p90':>8} {'%straight':>10} {'%tight':>8}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(
            f"{row['track']:<25} "
            f"{row['track_length_m']:>7.1f}m "
            f"{row['curvature_mean']:>8.3f} "
            f"{row['curvature_max']:>8.3f} "
            f"{row['curvature_p90']:>8.3f} "
            f"{row['frac_straight']:>9.1%} "
            f"{row['frac_tight']:>7.1%}"
        )

    # recommend track selection
    print("\n--- Recommended track selection for paper ---")
    if len(df) >= 3:
        gentle = df.iloc[0]
        mid = df.iloc[len(df) // 2]
        aggressive = df.iloc[-1]
        print(f"  Gentle:     {gentle['track']} (κ_mean={gentle['curvature_mean']:.3f})")
        print(f"  Mixed:      {mid['track']} (κ_mean={mid['curvature_mean']:.3f})")
        print(f"  Aggressive: {aggressive['track']} (κ_mean={aggressive['curvature_mean']:.3f})")
    else:
        print("  (Need at least 3 tracks for stratification)")


if __name__ == "__main__":
    main()
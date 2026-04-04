#!/usr/bin/env python3
"""
Run a single action-space experiment config.

Examples:
  python scripts/run_one_experiment.py \
      --action_space steer_speed \
      --obs_regime full \
      --seed 0

  python scripts/run_one_experiment.py \
      --action_space curvature_speed \
      --obs_regime ablated \
      --seed 3 \
      --eval_after_train

This script is the unit of execution for Slurm job arrays:
one process = one config. Track and hyperparameter settings come from
sac.yaml and vehicle.yaml unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Run one experiment config")
    ap.add_argument("--action_space", required=True)
    ap.add_argument("--obs_regime", choices=["full", "ablated"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train_track", default=None,
                     help="Override training track (default: from sac.yaml)")
    ap.add_argument("--eval_tracks", default=None,
                     help="Override eval tracks (default: from sac.yaml)")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_after_train", action="store_true",
                     help="Run standalone eval.py with best model after training")
    args = ap.parse_args()

    ablate = args.obs_regime == "ablated"
    run_id = f"{args.action_space}_{args.obs_regime}_s{args.seed}"

    # --- build train command ---
    train_cmd = [
        sys.executable, str(ROOT / "rl" / "train.py"),
        "--vehicle_cfg", args.vehicle_cfg,
        "--sac_cfg", args.sac_cfg,
        "--action_space", args.action_space,
        "--seed", str(args.seed),
        "--n_eval_episodes", str(args.n_eval_episodes),
        "--device", args.device,
        "--run_id", run_id,
    ]

    # Only pass track overrides if explicitly set — otherwise train.py
    # reads from sac.yaml, keeping the config as single source of truth
    if args.train_track is not None:
        train_cmd.extend(["--train_track", args.train_track])
    if args.eval_tracks is not None:
        train_cmd.extend(["--eval_tracks", args.eval_tracks])
    if ablate:
        train_cmd.append("--ablate_geometry")

    print("=" * 70)
    print("RUNNING TRAIN")
    print(f"  run_id:       {run_id}")
    print(f"  action_space: {args.action_space}")
    print(f"  obs_regime:   {args.obs_regime}")
    print(f"  seed:         {args.seed}")
    print(f"  train_track:  {args.train_track or '(from sac.yaml)'}")
    print(f"  eval_tracks:  {args.eval_tracks or '(from sac.yaml)'}")
    print(f"  device:       {args.device}")
    print(f"  cmd: {' '.join(train_cmd)}")
    print("=" * 70)

    result = subprocess.run(train_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if not args.eval_after_train:
        return

    # --- standalone eval with best model ---
    ckpt_dir = ROOT / "checkpoints" / run_id
    checkpoint = ckpt_dir / "eval_results" / "best_model.zip"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "sac_final.zip"

    if not checkpoint.exists():
        print(f"WARNING: no checkpoint found for {run_id}, skipping eval")
        return

    # Use --from_meta so eval.py reads the exact config that training used.
    # Do NOT pass --tracks here — let --from_meta provide the track list
    # so training and standalone eval use identical tracks.
    eval_cmd = [
        sys.executable, str(ROOT / "rl" / "eval.py"),
        "--checkpoint", str(checkpoint),
        "--from_meta",
        "--n_episodes", str(args.n_eval_episodes),
        "--device", args.device,
    ]

    print("=" * 70)
    print("RUNNING EVAL")
    print(f"  checkpoint: {checkpoint}")
    print(f"  cmd: {' '.join(eval_cmd)}")
    print("=" * 70)

    result = subprocess.run(eval_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
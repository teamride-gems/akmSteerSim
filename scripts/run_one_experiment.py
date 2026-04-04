#!/usr/bin/env python3
"""
Run a single action-space experiment config.

Examples:
  python scripts/run_one_experiment.py \
      --action_space steer_speed \
      --obs_regime full \
      --seed 0 \
      --train_track Sakhir \
      --eval_tracks Sakhir,Austin \
      --vehicle_cfg configs/vehicle.yaml \
      --sac_cfg configs/sac.yaml \
      --n_eval_episodes 10 \
      --device cuda

This script is intended to be the unit of execution for Slurm job arrays:
one process = one config.
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
    ap.add_argument("--train_track", default="Sakhir")
    ap.add_argument("--eval_tracks", default="Sakhir")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_after_train", action="store_true")
    args = ap.parse_args()

    ablate = args.obs_regime == "ablated"
    run_id = f"{args.action_space}_{args.obs_regime}_s{args.seed}"

    train_cmd = [
        sys.executable, str(ROOT / "rl" / "train.py"),
        "--vehicle_cfg", args.vehicle_cfg,
        "--sac_cfg", args.sac_cfg,
        "--action_space", args.action_space,
        "--seed", str(args.seed),
        "--train_track", args.train_track,
        "--eval_tracks", args.eval_tracks,
        "--n_eval_episodes", str(args.n_eval_episodes),
        "--device", args.device,
        "--run_id", run_id,
    ]

    if ablate:
        train_cmd.append("--ablate_geometry")

    print("=" * 70)
    print("RUNNING TRAIN")
    print(f"run_id:       {run_id}")
    print(f"action_space: {args.action_space}")
    print(f"obs_regime:   {args.obs_regime}")
    print(f"seed:         {args.seed}")
    print(f"train_track:  {args.train_track}")
    print(f"eval_tracks:  {args.eval_tracks}")
    print(f"device:       {args.device}")
    print("cmd:")
    print("  " + " ".join(train_cmd))
    print("=" * 70)

    result = subprocess.run(train_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if not args.eval_after_train:
        return

    ckpt_dir = ROOT / "checkpoints" / run_id
    checkpoint = ckpt_dir / "eval_results" / "best_model.zip"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "sac_final.zip"

    if not checkpoint.exists():
        print(f"WARNING: no checkpoint found for {run_id}, skipping eval")
        return

    eval_cmd = [
        sys.executable, str(ROOT / "rl" / "eval.py"),
        "--checkpoint", str(checkpoint),
        "--from_meta",
        "--tracks", args.eval_tracks,
        "--n_episodes", str(args.n_eval_episodes),
    ]

    print("=" * 70)
    print("RUNNING EVAL")
    print("cmd:")
    print("  " + " ".join(eval_cmd))
    print("=" * 70)

    result = subprocess.run(eval_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
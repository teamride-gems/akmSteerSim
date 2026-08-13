#!/usr/bin/env python3
"""
Run a single experiment condition.

Patched semantics:
  - validation tracks are used during training
  - test tracks are evaluated only after training
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one experiment config")
    ap.add_argument("--action_space", required=True)
    ap.add_argument("--obs_regime", choices=["full", "ablated"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train_track", default=None)
    ap.add_argument("--validation_tracks", default=None)
    ap.add_argument("--test_tracks", default=None)
    ap.add_argument("--eval_tracks", default=None, help="Legacy alias for validation tracks")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval_after_train", action="store_true")
    ap.add_argument("--run_id", default=None, help="Explicit artifact directory name")
    args = ap.parse_args()

    ablate = args.obs_regime == "ablated"
    validation_tracks = args.validation_tracks or args.eval_tracks
    run_id = args.run_id or f"{args.action_space}_{args.obs_regime}_s{args.seed}"

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

    if args.train_track is not None:
        train_cmd.extend(["--train_track", args.train_track])
    if validation_tracks is not None:
        train_cmd.extend(["--validation_tracks", validation_tracks])
    if args.test_tracks is not None:
        train_cmd.extend(["--test_tracks", args.test_tracks])
    if ablate:
        train_cmd.append("--ablate_geometry")

    print("=" * 72)
    print("RUNNING TRAIN")
    print(f"  run_id:            {run_id}")
    print(f"  action_space:      {args.action_space}")
    print(f"  obs_regime:        {args.obs_regime}")
    print(f"  seed:              {args.seed}")
    print(f"  train_track:       {args.train_track or '(from config)'}")
    print(f"  validation_tracks: {validation_tracks or '(from config)'}")
    print(f"  test_tracks:       {args.test_tracks or '(from config)'}")
    print(f"  cmd: {' '.join(train_cmd)}")
    print("=" * 72)

    result = subprocess.run(train_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if not args.eval_after_train:
        return

    ckpt_dir = ROOT / "checkpoints" / run_id
    checkpoint = ckpt_dir / "eval_results" / "best_validation_model.zip"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "sac_final.zip"

    if not checkpoint.exists():
        print(f"WARNING: no checkpoint found for {run_id}; skipping standalone eval")
        return

    eval_cmd = [
        sys.executable, str(ROOT / "rl" / "eval.py"),
        "--checkpoint", str(checkpoint),
        "--from_meta",
        "--evaluation_split", "test",
        "--n_episodes", str(args.n_eval_episodes),
        "--device", args.device,
        "--output", str(ckpt_dir / "eval_standalone_test.json"),
    ]

    print("=" * 72)
    print("RUNNING FINAL TEST EVAL")
    print(f"  checkpoint: {checkpoint}")
    print(f"  cmd: {' '.join(eval_cmd)}")
    print("=" * 72)

    result = subprocess.run(eval_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

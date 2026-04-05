#!/usr/bin/env python3
"""
Experiment sweep launcher for the action-space paper.

This patched version separates:
  - train_track
  - validation_tracks  (used for periodic eval / checkpoint selection / early stop)
  - test_tracks        (never touched during training; final report only)
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_SPACES = ["steer_speed", "curvature_speed", "lookahead_point"]
EXTENDED_ACTION_SPACES = DEFAULT_ACTION_SPACES + ["bezier"]


def build_runs(
    action_spaces: List[str],
    seeds: List[int],
    ablation: bool,
    train_track: Optional[str],
    validation_tracks: Optional[str],
    test_tracks: Optional[str],
    vehicle_cfg: str,
    sac_cfg: str,
    n_eval_episodes: int,
    device: str,
    eval_after_train: bool,
) -> List[Dict]:
    obs_regimes = ["full"]
    if ablation:
        obs_regimes.append("ablated")

    runs: List[Dict] = []
    for action_space, seed, obs_regime in itertools.product(action_spaces, seeds, obs_regimes):
        run_id = f"{action_space}_{obs_regime}_s{seed}"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_one_experiment.py"),
            "--action_space", action_space,
            "--obs_regime", obs_regime,
            "--seed", str(seed),
            "--vehicle_cfg", vehicle_cfg,
            "--sac_cfg", sac_cfg,
            "--n_eval_episodes", str(n_eval_episodes),
            "--device", device,
        ]

        if train_track is not None:
            cmd.extend(["--train_track", train_track])
        if validation_tracks is not None:
            cmd.extend(["--validation_tracks", validation_tracks])
        if test_tracks is not None:
            cmd.extend(["--test_tracks", test_tracks])
        if eval_after_train:
            cmd.append("--eval_after_train")

        runs.append(
            {
                "run_id": run_id,
                "action_space": action_space,
                "obs_regime": obs_regime,
                "seed": seed,
                "cmd": cmd,
            }
        )

    return runs


def run_sequential(runs: List[Dict], dry_run: bool = False) -> None:
    failed: List[str] = []
    for i, run in enumerate(runs, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{i}/{len(runs)}] {run['run_id']}")
        print(f"{'=' * 72}")

        if dry_run:
            print("  " + " ".join(run["cmd"]))
            continue

        result = subprocess.run(run["cmd"], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"WARNING: run {run['run_id']} exited with code {result.returncode}")
            failed.append(run["run_id"])

    if failed:
        print(f"\n{len(failed)} runs failed:")
        for run_id in failed:
            print(f"  - {run_id}")


def emit_array_configs(runs: List[Dict]) -> None:
    for task_id, run in enumerate(runs):
        print(
            f"{task_id}\t{run['run_id']}\t{run['action_space']}\t"
            f"{run['obs_regime']}\t{run['seed']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch action-space experiment sweep")
    ap.add_argument(
        "--action_spaces",
        default=None,
        help=f"Comma-separated action spaces (default: {','.join(DEFAULT_ACTION_SPACES)})",
    )
    ap.add_argument("--include_bezier", action="store_true", help="Include bezier in sweep")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    ap.add_argument("--no-ablation", action="store_true", help="Skip observation ablation runs")
    ap.add_argument("--train_track", default=None, help="Override training track")
    ap.add_argument(
        "--validation_tracks",
        default=None,
        help="Comma-separated validation tracks (used during training)",
    )
    ap.add_argument(
        "--test_tracks",
        default=None,
        help="Comma-separated test tracks (never used during training)",
    )
    ap.add_argument(
        "--eval_tracks",
        default=None,
        help="Legacy alias for --validation_tracks",
    )
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval_after_train", action="store_true")
    ap.add_argument("--emit-array-configs", action="store_true")
    args = ap.parse_args()

    if args.action_spaces:
        action_spaces = [s.strip() for s in args.action_spaces.split(",") if s.strip()]
    elif args.include_bezier:
        action_spaces = EXTENDED_ACTION_SPACES
    else:
        action_spaces = DEFAULT_ACTION_SPACES

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ablation = not args.no_ablation
    validation_tracks = args.validation_tracks or args.eval_tracks

    runs = build_runs(
        action_spaces=action_spaces,
        seeds=seeds,
        ablation=ablation,
        train_track=args.train_track,
        validation_tracks=validation_tracks,
        test_tracks=args.test_tracks,
        vehicle_cfg=args.vehicle_cfg,
        sac_cfg=args.sac_cfg,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
        eval_after_train=args.eval_after_train,
    )

    print(
        f"Sweep: {len(action_spaces)} action spaces × "
        f"{2 if ablation else 1} obs regimes × {len(seeds)} seeds = {len(runs)} runs"
    )
    print(f"Action spaces: {action_spaces}")
    print(f"Seeds: {seeds}")
    print(f"Ablation: {ablation}")
    print(f"Train track: {args.train_track or '(from config)'}")
    print(f"Validation tracks: {validation_tracks or '(from config)'}")
    print(f"Test tracks: {args.test_tracks or '(from config)'}")

    if args.emit_array_configs:
        emit_array_configs(runs)
        return

    run_sequential(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

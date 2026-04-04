#!/usr/bin/env python3
"""
Experiment sweep launcher for the action-space paper.

One config can be run independently, compatible with Slurm job arrays.

Examples:
  python scripts/sweep.py
  python scripts/sweep.py --no-ablation
  python scripts/sweep.py --action_spaces steer_speed --seeds 0
  python scripts/sweep.py --emit-array-configs
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_SPACES = ["steer_speed", "curvature_speed", "lookahead_point"]
EXTENDED_ACTION_SPACES = DEFAULT_ACTION_SPACES + ["bezier"]


def build_runs(
    action_spaces: List[str],
    seeds: List[int],
    ablation: bool,
    train_track: Optional[str],
    eval_tracks: Optional[str],
    vehicle_cfg: str,
    sac_cfg: str,
    n_eval_episodes: int,
    device: str,
    eval_after_train: bool,
) -> List[Dict]:
    obs_regimes = ["full"]
    if ablation:
        obs_regimes.append("ablated")

    runs = []
    for action_space, seed, obs_regime in itertools.product(action_spaces, seeds, obs_regimes):
        run_id = f"{action_space}_{obs_regime}_s{seed}"

        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_one_experiment.py"),
            "--action_space", action_space,
            "--obs_regime", obs_regime,
            "--seed", str(seed),
            "--vehicle_cfg", vehicle_cfg,
            "--sac_cfg", sac_cfg,
            "--n_eval_episodes", str(n_eval_episodes),
            "--device", device,
        ]

        # Only pass track overrides if explicitly set — otherwise let
        # train.py read from sac.yaml so config stays the single source of truth
        if train_track is not None:
            cmd.extend(["--train_track", train_track])
        if eval_tracks is not None:
            cmd.extend(["--eval_tracks", eval_tracks])
        if eval_after_train:
            cmd.append("--eval_after_train")

        runs.append({
            "run_id": run_id,
            "action_space": action_space,
            "obs_regime": obs_regime,
            "seed": seed,
            "cmd": cmd,
        })

    return runs


def run_sequential(runs: List[Dict], dry_run: bool = False):
    n = len(runs)
    failed = []
    for i, run in enumerate(runs, start=1):
        print(f"\n{'='*70}")
        print(f"[{i}/{n}] {run['run_id']}")
        print(f"{'='*70}")

        if dry_run:
            print("  " + " ".join(run["cmd"]))
            continue

        result = subprocess.run(run["cmd"], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"WARNING: run {run['run_id']} exited with code {result.returncode}")
            failed.append(run["run_id"])

    if failed:
        print(f"\n{len(failed)} runs failed: {failed}")


def emit_array_configs(runs: List[Dict]):
    """
    Print one config per line for Slurm job arrays:
    task_id<TAB>run_id<TAB>action_space<TAB>obs_regime<TAB>seed
    """
    for task_id, run in enumerate(runs):
        print(
            f"{task_id}\t{run['run_id']}\t{run['action_space']}\t"
            f"{run['obs_regime']}\t{run['seed']}"
        )


def main():
    ap = argparse.ArgumentParser(description="Launch experiment sweep")
    ap.add_argument(
        "--action_spaces", default=None,
        help=f"Comma-separated action spaces (default: {','.join(DEFAULT_ACTION_SPACES)})"
    )
    ap.add_argument("--include_bezier", action="store_true", help="Include bezier in sweep")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    ap.add_argument("--no-ablation", action="store_true", help="Skip observation ablation runs")
    ap.add_argument("--train_track", default=None,
                     help="Override training track (default: from sac.yaml)")
    ap.add_argument("--eval_tracks", default=None,
                     help="Override eval tracks (default: from sac.yaml)")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval_after_train", action="store_true",
                     help="Run standalone eval.py after training completes")
    ap.add_argument("--emit-array-configs", action="store_true",
                     help="Print array-task mapping instead of running jobs")
    args = ap.parse_args()

    if args.action_spaces:
        action_spaces = [s.strip() for s in args.action_spaces.split(",")]
    elif args.include_bezier:
        action_spaces = EXTENDED_ACTION_SPACES
    else:
        action_spaces = DEFAULT_ACTION_SPACES

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    ablation = not args.no_ablation

    runs = build_runs(
        action_spaces=action_spaces,
        seeds=seeds,
        ablation=ablation,
        train_track=args.train_track,
        eval_tracks=args.eval_tracks,
        vehicle_cfg=args.vehicle_cfg,
        sac_cfg=args.sac_cfg,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
        eval_after_train=args.eval_after_train,
    )

    n_obs = 2 if ablation else 1
    print(f"Sweep: {len(action_spaces)} action spaces × {n_obs} obs regimes × {len(seeds)} seeds = {len(runs)} runs")
    print(f"Action spaces: {action_spaces}")
    print(f"Seeds: {seeds}")
    print(f"Ablation: {ablation}")
    print(f"Train track: {args.train_track or '(from sac.yaml)'}")
    print(f"Eval tracks: {args.eval_tracks or '(from sac.yaml)'}")

    if args.emit_array_configs:
        emit_array_configs(runs)
        return

    run_sequential(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
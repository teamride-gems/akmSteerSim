#!/usr/bin/env python3
"""
Experiment sweep launcher for the action-space paper.

Runs all combinations of:
  - action spaces (steer_speed, curvature_speed, lookahead_point, [bezier])
  - observation regimes (full, ablated)
  - seeds

Usage:
  # full sweep (3 action spaces × 2 obs regimes × 5 seeds = 30 runs)
  python scripts/sweep.py

  # just the main comparison (no ablation)
  python scripts/sweep.py --no-ablation

  # single action space, useful for debugging
  python scripts/sweep.py --action_spaces steer_speed --seeds 0

  # dry run: print commands without executing
  python scripts/sweep.py --dry-run

  # parallel: run N jobs at once (each job is one training run)
  python scripts/sweep.py --parallel 3
"""

import argparse
import subprocess
import sys
import itertools
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_SPACES = ["steer_speed", "curvature_speed", "lookahead_point"]
EXTENDED_ACTION_SPACES = DEFAULT_ACTION_SPACES + ["bezier"]


def build_commands(
    action_spaces: List[str],
    seeds: List[int],
    ablation: bool,
    train_track: str,
    eval_tracks: str,
    vehicle_cfg: str,
    sac_cfg: str,
    n_eval_episodes: int,
    device: str,
) -> List[dict]:
    """Build list of {cmd, run_id, description} for all experiment conditions."""
    obs_regimes = [False]
    if ablation:
        obs_regimes.append(True)

    runs = []
    for action_space, seed, ablate in itertools.product(action_spaces, seeds, obs_regimes):
        ablate_tag = "ablated" if ablate else "full"
        run_id = f"{action_space}_{ablate_tag}_s{seed}"

        cmd = [
            sys.executable, str(ROOT / "rl" / "train.py"),
            "--vehicle_cfg", vehicle_cfg,
            "--sac_cfg", sac_cfg,
            "--action_space", action_space,
            "--seed", str(seed),
            "--train_track", train_track,
            "--eval_tracks", eval_tracks,
            "--n_eval_episodes", str(n_eval_episodes),
            "--device", device,
            "--run_id", run_id,
        ]

        if ablate:
            cmd.append("--ablate_geometry")

        desc = f"action={action_space}  obs={ablate_tag}  seed={seed}"
        runs.append({"cmd": cmd, "run_id": run_id, "description": desc})

    return runs


def run_sequential(runs: List[dict], dry_run: bool = False):
    n = len(runs)
    for i, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{n}] {run['description']}")
        print(f"{'='*60}")

        if dry_run:
            print("  " + " ".join(run["cmd"]))
            continue

        result = subprocess.run(run["cmd"], cwd=str(ROOT))
        if result.returncode != 0:
            print(f"WARNING: run {run['run_id']} exited with code {result.returncode}")


def run_parallel(runs: List[dict], max_parallel: int, dry_run: bool = False):
    """Run up to max_parallel training jobs concurrently."""
    if dry_run:
        for run in runs:
            print(" ".join(run["cmd"]))
        return

    active = []
    remaining = list(runs)
    completed = 0
    n = len(runs)

    while remaining or active:
        # launch new jobs up to limit
        while remaining and len(active) < max_parallel:
            run = remaining.pop(0)
            print(f"[{completed + len(active) + 1}/{n}] LAUNCHING: {run['description']}")
            proc = subprocess.Popen(
                run["cmd"],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            active.append((proc, run))

        # poll active jobs
        still_active = []
        for proc, run in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, run))
            else:
                completed += 1
                status = "OK" if ret == 0 else f"FAILED (code {ret})"
                print(f"[{completed}/{n}] DONE: {run['description']} — {status}")
        active = still_active

        if active:
            import time
            time.sleep(5)

    print(f"\nAll {n} runs complete.")


def main():
    ap = argparse.ArgumentParser(description="Launch experiment sweep")
    ap.add_argument(
        "--action_spaces", default=None,
        help=f"Comma-separated action spaces (default: {','.join(DEFAULT_ACTION_SPACES)})"
    )
    ap.add_argument("--include_bezier", action="store_true", help="Include bezier in sweep")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    ap.add_argument("--no-ablation", action="store_true", help="Skip observation ablation runs")
    ap.add_argument("--train_track", default="Sakhir", help="Training track")
    ap.add_argument("--eval_tracks", default="Sakhir", help="Comma-separated eval tracks")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--n_eval_episodes", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--parallel", type=int, default=1, help="Max parallel jobs")
    args = ap.parse_args()

    if args.action_spaces:
        action_spaces = [s.strip() for s in args.action_spaces.split(",")]
    elif args.include_bezier:
        action_spaces = EXTENDED_ACTION_SPACES
    else:
        action_spaces = DEFAULT_ACTION_SPACES

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    ablation = not args.no_ablation

    runs = build_commands(
        action_spaces=action_spaces,
        seeds=seeds,
        ablation=ablation,
        train_track=args.train_track,
        eval_tracks=args.eval_tracks,
        vehicle_cfg=args.vehicle_cfg,
        sac_cfg=args.sac_cfg,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
    )

    n_obs = 2 if ablation else 1
    print(f"Sweep: {len(action_spaces)} action spaces × {n_obs} obs regimes × {len(seeds)} seeds = {len(runs)} runs")
    print(f"Action spaces: {action_spaces}")
    print(f"Seeds: {seeds}")
    print(f"Ablation: {ablation}")
    print(f"Train track: {args.train_track}")
    print(f"Eval tracks: {args.eval_tracks}")

    if args.parallel > 1:
        run_parallel(runs, args.parallel, dry_run=args.dry_run)
    else:
        run_sequential(runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
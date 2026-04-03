#!/usr/bin/env python3
"""
Full action-space comparison experiment.

Trains SAC on all four action spaces using the same sac.yaml schedule,
evaluates each final checkpoint on every val_pool track, collects
lap metrics + rollout recordings, and writes a combined summary CSV.

Usage:
    python run_experiment.py
    python run_experiment.py --sac_cfg configs/sac.yaml --device cuda
    python run_experiment.py --action_spaces steer_speed,bezier   # subset
    python run_experiment.py --eval_only    # skip training, just evaluate existing checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ALL_ACTION_SPACES = ["steer_speed", "curvature_speed", "lookahead_point", "bezier"]

EVAL_STEPS = 10000


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def save_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def run_command(cmd: List[str], label: str, log_path: Optional[Path] = None):
    """Run a subprocess, printing output live and optionally logging to file."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_file:
            log_file.write(line)

    proc.wait()
    if log_file:
        log_file.close()

    if proc.returncode != 0:
        print(f"\n*** FAILED: {label} (exit code {proc.returncode}) ***")
        return False

    print(f"\n--- Completed: {label} ---\n")
    return True


def find_latest_checkpoint(ckpt_root: Path, action_space: str) -> Optional[Path]:
    """
    Find the most recent checkpoint directory for a given action space.
    Looks for directories containing sac_final.zip under ckpt_root.
    """
    candidates = []
    for run_dir in sorted(ckpt_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        final = run_dir / "sac_final.zip"
        if not final.exists():
            final = run_dir / "sac_final"
            if not final.exists():
                continue
        # Check if this run used the right action space by looking at the
        # vehicle config we saved alongside it
        meta = run_dir / "experiment_meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                if m.get("action_space") == action_space:
                    candidates.append((run_dir, final))
            except Exception:
                pass
    if candidates:
        return candidates[0][1]  # most recent
    return None


def main():
    ap = argparse.ArgumentParser(description="Run full action-space comparison experiment")
    ap.add_argument("--vehicle_cfg", default="configs/vehicle.yaml",
                    help="Base vehicle config (action_space key will be overridden)")
    ap.add_argument("--sac_cfg", default="configs/sac.yaml")
    ap.add_argument("--device", default="auto", help="PyTorch device (auto/cpu/cuda)")
    ap.add_argument("--action_spaces", default=",".join(ALL_ACTION_SPACES),
                    help="Comma-separated list of action spaces to test")
    ap.add_argument("--eval_steps", type=int, default=EVAL_STEPS,
                    help="Steps per eval rollout")
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, only evaluate existing checkpoints")
    ap.add_argument("--train_only", action="store_true",
                    help="Skip evaluation, only train")
    args = ap.parse_args()

    action_spaces = [s.strip() for s in args.action_spaces.split(",") if s.strip()]
    for a in action_spaces:
        if a not in ALL_ACTION_SPACES:
            raise ValueError(f"Unknown action space: {a}. Valid: {ALL_ACTION_SPACES}")

    base_veh_cfg = load_yaml(ROOT / args.vehicle_cfg)
    sac_cfg = load_yaml(ROOT / args.sac_cfg)

    eval_cfg = sac_cfg.get("evaluation", {})
    val_pool = [str(t) for t in eval_cfg.get("val_pool", [])]

    # Use map_schedule tracks for eval too if val_pool is empty
    if not val_pool:
        schedule = sac_cfg.get("map_schedule", [])
        val_pool = list(set(
            str(item["track"]).replace("_map", "").strip()
            for item in schedule
        ))

    if not val_pool:
        sim_cfg = base_veh_cfg.get("sim", {})
        raw_map = sim_cfg.get("map_name", "Sakhir_map")
        val_pool = [str(raw_map).replace("_map", "").strip()]

    # Experiment directory
    exp_id = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = ROOT / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'#'*60}")
    print(f"  Action Space Comparison Experiment")
    print(f"  ID: {exp_id}")
    print(f"  Action spaces: {action_spaces}")
    print(f"  Eval tracks: {val_pool}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'#'*60}")

    # Save experiment config
    exp_meta = {
        "exp_id": exp_id,
        "action_spaces": action_spaces,
        "eval_tracks": val_pool,
        "vehicle_cfg": args.vehicle_cfg,
        "sac_cfg": args.sac_cfg,
        "eval_steps": args.eval_steps,
        "device": args.device,
    }
    (exp_dir / "experiment_config.json").write_text(json.dumps(exp_meta, indent=2))

    # Copy base configs into experiment dir for reproducibility
    shutil.copy2(ROOT / args.vehicle_cfg, exp_dir / "vehicle_base.yaml")
    shutil.copy2(ROOT / args.sac_cfg, exp_dir / "sac.yaml")

    # =====================================================================
    # Phase 1: Generate per-action-space vehicle configs
    # =====================================================================
    configs_dir = exp_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    per_as_cfg_paths = {}
    for action_space in action_spaces:
        cfg = deepcopy(base_veh_cfg)
        cfg["action_space"] = action_space
        cfg_path = configs_dir / f"vehicle_{action_space}.yaml"
        save_yaml(cfg, cfg_path)
        per_as_cfg_paths[action_space] = cfg_path

    # =====================================================================
    # Phase 2: Train
    # =====================================================================
    train_results = {}

    if not args.eval_only:
        for action_space in action_spaces:
            cfg_path = per_as_cfg_paths[action_space]
            log_path = exp_dir / "logs" / f"train_{action_space}.log"

            t0 = time.time()
            success = run_command(
                [
                    sys.executable, "rl/train.py",
                    "--vehicle_cfg", str(cfg_path),
                    "--sac_cfg", str(ROOT / args.sac_cfg),
                    "--device", args.device,
                ],
                label=f"TRAIN: {action_space}",
                log_path=log_path,
            )
            elapsed = time.time() - t0

            train_results[action_space] = {
                "success": success,
                "elapsed_sec": round(elapsed, 1),
            }

            if not success:
                print(f"\n*** Training failed for {action_space}. Check {log_path} ***")
                print("Continuing with remaining action spaces...\n")

        # Save training summary
        (exp_dir / "train_results.json").write_text(json.dumps(train_results, indent=2))

    if args.train_only:
        print("\n=== Train-only mode. Skipping evaluation. ===")
        print(f"Results in: {exp_dir}")
        return

    # =====================================================================
    # Phase 3: Find checkpoints and evaluate
    # =====================================================================
    ckpt_root = ROOT / "checkpoints"

    # Find the most recent checkpoint for each action space
    # Strategy: look at all checkpoint dirs, find sac_final, check which
    # action space was used by reading the vehicle config that was active
    checkpoint_map = {}

    # First, tag any checkpoints we just created
    if not args.eval_only:
        # The most recent dirs in checkpoints/ are from our runs
        all_ckpt_dirs = sorted(ckpt_root.iterdir(), reverse=True) if ckpt_root.exists() else []
        recent_dirs = all_ckpt_dirs[:len(action_spaces) * 2]  # generous buffer

        for run_dir in recent_dirs:
            if not run_dir.is_dir():
                continue
            final = run_dir / "sac_final.zip"
            if not final.exists():
                final = run_dir / "sac_final"
            if not final.exists():
                continue

            # Try to identify which action space this was by checking
            # TensorBoard log or by matching timestamps
            # Safest: write a meta file during training — but since we can't
            # modify train.py mid-run, we'll match by checking the vehicle
            # configs we generated
            meta_path = run_dir / "experiment_meta.json"
            if not meta_path.exists():
                # Heuristic: check phase subdirs for the vehicle config
                # that was used. For now, assign by order.
                pass

    # Let user map checkpoints if auto-detection fails
    print("\n=== Locating checkpoints ===")

    for action_space in action_spaces:
        # Try auto-detect
        found = find_latest_checkpoint(ckpt_root, action_space)
        if found:
            checkpoint_map[action_space] = found
            print(f"  {action_space}: {found}")
            continue

        # Fallback: ask user or find most recent unassigned
        if ckpt_root.exists():
            all_dirs = sorted(ckpt_root.iterdir(), reverse=True)
            for d in all_dirs:
                final = d / "sac_final.zip"
                if not final.exists():
                    final = d / "sac_final"
                if final.exists() and d.name not in [
                    str(v.parent.name) for v in checkpoint_map.values()
                ]:
                    # Check if there's a matching log
                    log = exp_dir / "logs" / f"train_{action_space}.log"
                    if log.exists():
                        checkpoint_map[action_space] = final
                        print(f"  {action_space}: {final} (matched by log)")
                        break

        if action_space not in checkpoint_map:
            print(f"  {action_space}: NOT FOUND — skipping evaluation")

    # Write checkpoint mapping
    ckpt_map_serializable = {k: str(v) for k, v in checkpoint_map.items()}
    (exp_dir / "checkpoint_map.json").write_text(json.dumps(ckpt_map_serializable, indent=2))

    # =====================================================================
    # Phase 4: Run evaluation rollouts
    # =====================================================================
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    rollout_dir = exp_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    all_eval_results = []

    for action_space in action_spaces:
        if action_space not in checkpoint_map:
            continue

        ckpt_path = checkpoint_map[action_space]
        cfg_path = per_as_cfg_paths[action_space]

        for track in val_pool:
            rollout_path = rollout_dir / f"{action_space}_{track}.npz"
            lap_csv = metrics_dir / f"laps_{action_space}_{track}.csv"
            step_csv = metrics_dir / f"steps_{action_space}_{track}.csv"
            log_path = exp_dir / "logs" / f"eval_{action_space}_{track}.log"

            # We need to modify the metrics paths per-run. Since
            # run_trained_policy.py hardcodes them, we'll set env vars
            # or just accept overwriting and rename after.
            t0 = time.time()
            success = run_command(
                [
                    sys.executable, "scripts/run_trained_policy.py",
                    "--model", str(ckpt_path),
                    "--vehicle_cfg", str(cfg_path),
                    "--track", track,
                    "--steps", str(args.eval_steps),
                    "--deterministic",
                    "--record",
                    "--record_path", str(rollout_path),
                ],
                label=f"EVAL: {action_space} on {track}",
                log_path=log_path,
            )
            elapsed = time.time() - t0

            # Move metrics files to per-run names
            default_lap = ROOT / "metrics" / "lap_metrics.csv"
            default_step = ROOT / "metrics" / "timestep_metrics.csv"
            if default_lap.exists():
                shutil.copy2(default_lap, lap_csv)
            if default_step.exists():
                shutil.copy2(default_step, step_csv)

            # Parse lap metrics if available
            eval_entry = {
                "action_space": action_space,
                "track": track,
                "eval_success": success,
                "eval_elapsed_sec": round(elapsed, 1),
                "rollout_path": str(rollout_path),
            }

            if lap_csv.exists():
                try:
                    with open(lap_csv, "r") as f:
                        reader = csv.DictReader(f)
                        laps = list(reader)

                    n_laps = len(laps)
                    n_success = sum(1 for l in laps if l.get("lap_status") == "SUCCESS")
                    n_crash = sum(1 for l in laps if l.get("lap_status") == "CRASH")
                    n_timeout = sum(1 for l in laps if l.get("lap_status") == "TIMEOUT")
                    lap_times = [
                        float(l["lap_time_sec"]) for l in laps
                        if l.get("lap_status") == "SUCCESS"
                    ]

                    eval_entry.update({
                        "n_laps": n_laps,
                        "n_success": n_success,
                        "n_crash": n_crash,
                        "n_timeout": n_timeout,
                        "success_rate": round(n_success / max(1, n_laps), 4),
                        "mean_lap_time": round(float(np.mean(lap_times)), 3) if lap_times else None,
                        "min_lap_time": round(float(np.min(lap_times)), 3) if lap_times else None,
                        "std_lap_time": round(float(np.std(lap_times)), 3) if lap_times else None,
                    })
                except Exception as e:
                    print(f"  Warning: could not parse {lap_csv}: {e}")

            all_eval_results.append(eval_entry)

    # =====================================================================
    # Phase 5: Summary
    # =====================================================================
    (exp_dir / "eval_results.json").write_text(json.dumps(all_eval_results, indent=2))

    # Write summary CSV
    summary_csv = exp_dir / "summary.csv"
    if all_eval_results:
        fieldnames = list(all_eval_results[0].keys())
        # Ensure all keys are captured
        for entry in all_eval_results:
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in all_eval_results:
                writer.writerow(entry)

    # Print summary table
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT COMPLETE: {exp_id}")
    print(f"{'#'*60}\n")

    if all_eval_results:
        # Group by action space
        by_as = {}
        for entry in all_eval_results:
            a = entry["action_space"]
            if a not in by_as:
                by_as[a] = []
            by_as[a].append(entry)

        header = f"{'Action Space':<20} {'Track':<15} {'Success%':>10} {'Laps':>6} {'Crashes':>8} {'Mean Lap(s)':>12}"
        print(header)
        print("-" * len(header))

        for action_space in action_spaces:
            entries = by_as.get(action_space, [])
            for e in entries:
                sr = e.get("success_rate")
                sr_str = f"{sr*100:.1f}%" if sr is not None else "N/A"
                n_laps = e.get("n_laps", "?")
                n_crash = e.get("n_crash", "?")
                mean_t = e.get("mean_lap_time")
                mean_str = f"{mean_t:.3f}" if mean_t is not None else "N/A"
                print(f"{action_space:<20} {e['track']:<15} {sr_str:>10} {str(n_laps):>6} {str(n_crash):>8} {mean_str:>12}")

        # Overall per action space
        print(f"\n{'--- Aggregate ---':^{len(header)}}")
        for action_space in action_spaces:
            entries = by_as.get(action_space, [])
            if not entries:
                print(f"{action_space:<20} No results")
                continue
            rates = [e["success_rate"] for e in entries if e.get("success_rate") is not None]
            times = [e["mean_lap_time"] for e in entries if e.get("mean_lap_time") is not None]
            total_crashes = sum(e.get("n_crash", 0) for e in entries)
            total_laps = sum(e.get("n_laps", 0) for e in entries)

            avg_sr = f"{np.mean(rates)*100:.1f}%" if rates else "N/A"
            avg_t = f"{np.mean(times):.3f}" if times else "N/A"
            print(f"{action_space:<20} {'(all)':<15} {avg_sr:>10} {str(total_laps):>6} {str(total_crashes):>8} {avg_t:>12}")

    print(f"\nFull results: {exp_dir}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"TensorBoard:  tensorboard --logdir runs --bind_all --port 6006")


if __name__ == "__main__":
    main()
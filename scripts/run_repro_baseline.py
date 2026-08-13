#!/usr/bin/env python3
"""Run and validate the four-condition Rung 2 reproducibility baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.provenance import write_json


ACTION_SPACES = ("steer_speed", "curvature_speed", "lookahead_point", "bezier")
ALLOWED_TERM_REASONS = {"crash", "lap_complete", "timeout", "sim_done"}
REWARD_COMPONENTS = (
    "mean_reward_progress",
    "mean_reward_a_long_pen",
    "mean_reward_a_lat_pen",
    "mean_reward_time_pen",
    "mean_reward_crash_pen",
)


def git_output(*args: str) -> str:
    return subprocess.run(
        [
            "git",
            f"--git-dir={ROOT / '.git'}",
            f"--work-tree={ROOT}",
            *args,
        ],
        cwd=str(ROOT), check=True, capture_output=True, text=True
    ).stdout.strip()


def run_and_tee(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def all_episode_records(eval_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for track_data in eval_data.get("tracks", {}).values():
        episodes.extend(track_data.get("episodes", []))
    return episodes


def finite_numbers(value: Any, path: str = "root") -> List[str]:
    failures: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(finite_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(finite_numbers(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    return failures


def validate_run(run_dir: Path, action_space: str, code_commit: str) -> Dict[str, Any]:
    failures: List[str] = []
    required_files = (
        "run_meta.json",
        "resolved_vehicle.yaml",
        "resolved_sac.yaml",
        "sac_final.zip",
        "eval_results/best_validation_model.zip",
        "eval_standalone_test.json",
    )
    for relative in required_files:
        if not (run_dir / relative).exists():
            failures.append(f"missing artifact: {relative}")

    if failures:
        return {"passed": False, "failures": failures}

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    eval_data = json.loads(
        (run_dir / "eval_standalone_test.json").read_text(encoding="utf-8")
    )
    episodes = all_episode_records(eval_data)

    if meta.get("status") != "complete":
        failures.append(f"run status is {meta.get('status')!r}, expected 'complete'")
    if meta.get("action_space") != action_space:
        failures.append("metadata action space mismatch")
    if int(meta.get("seed", -1)) != 0:
        failures.append("metadata seed is not 0")
    if int(meta.get("final_num_timesteps", -1)) != 20_000:
        failures.append("training did not finish exactly 20,000 timesteps")

    git_meta = meta.get("provenance", {}).get("git", {})
    if git_meta.get("commit") != code_commit:
        failures.append("training Git commit does not match baseline commit")
    if git_meta.get("dirty"):
        failures.append(f"training began from a dirty tree: {git_meta.get('status_porcelain')}")
    if not git_meta.get("submodules"):
        failures.append("submodule provenance is missing")
    if not meta.get("normalizer_refs"):
        failures.append("normalizer provenance is missing")
    if eval_data.get("evaluation_split") != "test":
        failures.append("standalone evaluation is not labeled as test")
    if not episodes:
        failures.append("test evaluation contains no episodes")

    nonfinite = finite_numbers(eval_data)
    if nonfinite:
        failures.append(f"non-finite evaluation metrics: {nonfinite[:5]}")

    crash_count = 0
    penalty_count = 0
    max_steer = 0.0
    max_a_long = 0.0
    max_a_lat = 0.0
    for index, episode in enumerate(episodes):
        reason = episode.get("term_reason")
        if reason not in ALLOWED_TERM_REASONS:
            failures.append(f"episode {index}: invalid termination reason {reason!r}")

        crashed = reason == "crash"
        penalized = float(episode.get("mean_reward_crash_pen", 0.0)) < 0.0
        crash_count += int(crashed)
        penalty_count += int(penalized)
        if crashed != penalized:
            failures.append(
                f"episode {index}: crash/penalty disagreement ({reason}, "
                f"{episode.get('mean_reward_crash_pen')})"
            )

        component_sum = sum(float(episode.get(key, 0.0)) for key in REWARD_COMPONENTS)
        reward_total = float(episode.get("mean_reward_total", float("nan")))
        if not math.isclose(component_sum, reward_total, rel_tol=1e-7, abs_tol=1e-7):
            failures.append(
                f"episode {index}: reward components {component_sum} != total {reward_total}"
            )

        max_steer = max(max_steer, abs(float(episode.get("max_abs_observed_steer", 0.0))))
        max_a_long = max(max_a_long, abs(float(episode.get("max_abs_a_long", 0.0))))
        max_a_lat = max(max_a_lat, abs(float(episode.get("max_abs_a_lat", 0.0))))

    if max_steer <= 1e-6:
        failures.append("realized steering remained zero throughout test evaluation")
    if max_a_long > 100.0:
        failures.append(f"longitudinal acceleration exceeded diagnostic bound: {max_a_long}")
    if max_a_lat > 100.0:
        failures.append(f"lateral acceleration exceeded diagnostic bound: {max_a_lat}")

    summary = eval_data.get("test_summary", eval_data.get("overall_summary", {}))
    return {
        "passed": not failures,
        "failures": failures,
        "run_id": run_dir.name,
        "action_space": action_space,
        "seed": 0,
        "train_steps": meta.get("final_num_timesteps"),
        "training_wall_clock_seconds": meta.get("training_wall_clock_seconds"),
        "evaluation_wall_clock_seconds": eval_data.get("wall_clock_seconds"),
        "n_test_episodes": len(episodes),
        "crash_count": crash_count,
        "crash_penalty_count": penalty_count,
        "max_abs_observed_steer": max_steer,
        "max_abs_a_long": max_a_long,
        "max_abs_a_lat": max_a_lat,
        "test_mean_reward": summary.get("mean_reward"),
        "test_mean_progress": summary.get("mean_progress"),
        "test_completion_rate": summary.get("completion_rate"),
        "test_crash_rate": summary.get("crash_rate"),
        "test_mean_lateral_error": summary.get("mean_lateral_error"),
        "total_params": meta.get("total_params"),
        "device_packages": meta.get("provenance", {}).get("packages", {}),
        "torch_runtime": meta.get("provenance", {}).get("torch_runtime", {}),
    }


def write_summary_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = [
        "run_id", "action_space", "seed", "train_steps",
        "training_wall_clock_seconds", "evaluation_wall_clock_seconds",
        "n_test_episodes", "crash_count", "crash_penalty_count",
        "max_abs_observed_steer", "max_abs_a_long", "max_abs_a_lat",
        "test_mean_reward", "test_mean_progress", "test_completion_rate",
        "test_crash_rate", "test_mean_lateral_error", "total_params", "passed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rung 2 reproducibility baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--n-eval-episodes", type=int, default=2)
    args = parser.parse_args()

    code_commit = git_output("rev-parse", "HEAD")
    run_prefix = args.run_prefix or f"rung2_{code_commit[:7]}"
    checkpoints_dir = ROOT / "checkpoints"
    report_dir = ROOT / "reproducibility" / "rung2" / run_prefix

    if not args.aggregate_only:
        status = git_output("status", "--porcelain=v1", "--untracked-files=normal")
        if status:
            raise SystemExit(
                "Rung 2 must start from a clean Git tree. Commit or remove:\n" + status
            )
        if not args.skip_preflight:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "preflight.py")],
                cwd=str(ROOT),
                check=True,
            )

        for action_space in ACTION_SPACES:
            run_id = f"{run_prefix}_{action_space}_full_s0"
            run_dir = checkpoints_dir / run_id
            meta_path = run_dir / "run_meta.json"
            eval_path = run_dir / "eval_standalone_test.json"
            if meta_path.exists() and eval_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") == "complete":
                    print(f"Skipping already complete run: {run_id}")
                    continue
            if run_dir.exists():
                raise SystemExit(
                    f"Incomplete artifact directory already exists: {run_dir}. "
                    "Move it aside before retrying."
                )

            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_one_experiment.py"),
                "--action_space", action_space,
                "--obs_regime", "full",
                "--seed", "0",
                "--vehicle_cfg", "configs/vehicle.yaml",
                "--sac_cfg", "configs/sac_debug.yaml",
                "--n_eval_episodes", str(args.n_eval_episodes),
                "--device", args.device,
                "--run_id", run_id,
                "--eval_after_train",
            ]
            run_and_tee(command, run_dir / "baseline_driver.log")

    rows = [
        validate_run(
            checkpoints_dir / f"{run_prefix}_{action_space}_full_s0",
            action_space,
            code_commit,
        )
        for action_space in ACTION_SPACES
    ]
    failures = {
        row.get("action_space", "unknown"): row.get("failures", [])
        for row in rows if not row.get("passed")
    }

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "run_prefix": run_prefix,
        "debug_config": "configs/sac_debug.yaml",
        "vehicle_config": "configs/vehicle.yaml",
        "action_spaces": list(ACTION_SPACES),
        "seed": 0,
        "device_request": args.device,
        "validation_thresholds": {
            "max_abs_a_long": 100.0,
            "max_abs_a_lat": 100.0,
            "min_nonzero_realized_steer": 1e-6,
            "reward_component_tolerance": 1e-7,
        },
        "passed": not failures,
        "failures": failures,
        "runs": rows,
    }
    write_json(report_dir / "baseline_manifest.json", manifest)
    write_summary_csv(report_dir / "baseline_summary.csv", rows)

    aggregate_command = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_results.py"),
        "--checkpoints_dir", "checkpoints",
        "--output", str(report_dir / "aggregate"),
        "--run_prefix", run_prefix,
        "--require_test",
    ]
    subprocess.run(aggregate_command, cwd=str(ROOT), check=True)

    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit("Rung 2 reproducibility gate failed.")

    print(f"Rung 2 reproducibility gate passed: {report_dir}")


if __name__ == "__main__":
    main()

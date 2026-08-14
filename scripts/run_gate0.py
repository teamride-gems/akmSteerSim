#!/usr/bin/env python3
"""Run the five formal Gate 0 seeds with bounded local parallelism."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return not result.stdout.strip()


def _run_complete(run_dir: Path) -> bool:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("status") == "complete" and int(meta.get("final_num_timesteps", -1)) == 50000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vehicle_cfg", default="configs/vehicle.yaml")
    parser.add_argument("--sac_cfg", default="configs/gate0_sac.yaml")
    parser.add_argument("--run_prefix", default="gate0_direct")
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args()

    if args.max_parallel <= 0:
        raise ValueError("max_parallel must be positive.")
    if not args.allow_dirty and not _git_clean():
        raise RuntimeError("Formal Gate 0 runs require a clean committed worktree.")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Formal Gate 0 requires exactly five distinct seeds.")

    log_dir = ROOT / "checkpoints" / "gate0_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in seeds:
        run_id = f"{args.run_prefix}_s{seed}"
        run_dir = ROOT / "checkpoints" / run_id
        if _run_complete(run_dir):
            print(f"[skip] seed={seed} already complete: {run_dir}")
            continue
        if run_dir.exists():
            raise RuntimeError(
                f"Incomplete run directory already exists: {run_dir}. "
                "Inspect it and choose a new --run_prefix; do not overwrite formal evidence."
            )
        command = [
            sys.executable,
            str(ROOT / "rl" / "train.py"),
            "--vehicle_cfg", args.vehicle_cfg,
            "--sac_cfg", args.sac_cfg,
            "--action_space", "steer_speed",
            "--seed", str(seed),
            "--run_id", run_id,
            "--device", args.device,
        ]
        pending.append((seed, run_id, command))

    active: Dict[int, Dict] = {}
    failures = []
    while pending or active:
        while pending and len(active) < args.max_parallel:
            seed, run_id, command = pending.pop(0)
            stdout_path = log_dir / f"{run_id}.stdout.log"
            stderr_path = log_dir / f"{run_id}.stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            stderr_handle = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            active[seed] = {
                "process": process,
                "run_id": run_id,
                "stdout": stdout_handle,
                "stderr": stderr_handle,
                "started": time.time(),
            }
            print(f"[start] seed={seed} pid={process.pid} run={run_id}")

        time.sleep(2.0)
        for seed, state in list(active.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            state["stdout"].close()
            state["stderr"].close()
            elapsed = time.time() - state["started"]
            if return_code == 0 and _run_complete(ROOT / "checkpoints" / state["run_id"]):
                print(f"[done] seed={seed} elapsed={elapsed / 60.0:.1f} min")
            else:
                failures.append(
                    {
                        "seed": seed,
                        "return_code": return_code,
                        "stderr": str(log_dir / f"{state['run_id']}.stderr.log"),
                    }
                )
                print(f"[failed] seed={seed} return_code={return_code}")
            del active[seed]

    if failures:
        raise RuntimeError(f"Formal Gate 0 training failures: {failures}")
    print("All five formal Gate 0 training runs completed.")
    print("Run scripts/evaluate_gate0.py on the five run directories next.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repository and runtime validity preflight."""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def tracked_artifact_violations(paths: Iterable[str]) -> List[str]:
    violations = []
    for raw in paths:
        path = raw.replace("\\", "/")
        if (
            path.startswith("checkpoints/")
            or path.startswith("runs/")
            or "/__pycache__/" in f"/{path}"
            or path.endswith((".pyc", ".pyo"))
        ):
            violations.append(raw)
    return violations


def run_checked(command: List[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repository validity preflight")
    parser.add_argument(
        "--skip-f110",
        action="store_true",
        help="Skip the installed-simulator smoke test (appropriate for lightweight CI)",
    )
    args = parser.parse_args()

    tracked = subprocess.run(
        [
            "git",
            f"--git-dir={ROOT / '.git'}",
            f"--work-tree={ROOT}",
            "ls-files",
        ],
        cwd=str(ROOT), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    violations = tracked_artifact_violations(tracked)
    if violations:
        preview = "\n".join(f"  - {path}" for path in violations[:20])
        raise SystemExit(
            f"Generated artifacts are tracked by Git ({len(violations)} files):\n{preview}"
        )

    compile_targets = [ROOT / name for name in ("envs", "rl", "scripts", "utils", "tests")]
    if not all(compileall.compile_dir(str(path), quiet=1) for path in compile_targets):
        raise SystemExit("Python compilation failed.")

    run_checked([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    if not args.skip_f110:
        run_checked([sys.executable, str(ROOT / "tests" / "f110_integration_smoke.py")])

    print("Preflight passed.")


if __name__ == "__main__":
    main()

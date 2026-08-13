"""Runtime and source provenance for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import configparser
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PACKAGE_NAMES = (
    "f110-gym",
    "gym",
    "gymnasium",
    "llvmlite",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "pyyaml",
    "scipy",
    "stable-baselines3",
    "tensorboard",
    "torch",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, args: Iterable[str], required: bool = True) -> Optional[str]:
    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if marker.lower().startswith("gitdir:"):
            git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
    try:
        result = subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                f"--work-tree={root}",
                *args,
            ],
            cwd=str(root),
            check=required,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        if required:
            raise
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=normal"])
    submodules = []
    gitmodules_path = root / ".gitmodules"
    if gitmodules_path.exists():
        parser = configparser.ConfigParser()
        parser.read(gitmodules_path, encoding="utf-8")
        for section in parser.sections():
            if not section.startswith("submodule "):
                continue
            relative_path = parser.get(section, "path")
            worktree = root / relative_path
            tree_entry = _git(
                root, ["ls-tree", "HEAD", "--", relative_path], required=False
            ) or ""
            recorded_commit = None
            if tree_entry:
                fields = tree_entry.split()
                if len(fields) >= 3:
                    recorded_commit = fields[2]
            initialized = (worktree / ".git").exists()
            working_commit = (
                _git(worktree, ["rev-parse", "HEAD"], required=False)
                if initialized else None
            )
            working_status = (
                _git(
                    worktree,
                    ["status", "--porcelain=v1", "--untracked-files=normal"],
                    required=False,
                )
                if initialized else None
            )
            submodules.append({
                "name": section.removeprefix("submodule ").strip('"'),
                "path": relative_path,
                "url": parser.get(section, "url", fallback=None),
                "recorded_commit": recorded_commit,
                "working_commit": working_commit,
                "initialized": initialized,
                "dirty": bool(working_status),
                "matches_recorded_commit": (
                    recorded_commit == working_commit
                    if recorded_commit and working_commit else False
                ),
            })
    return {
        "commit": _git(root, ["rev-parse", "HEAD"]),
        "branch": _git(root, ["branch", "--show-current"], required=False),
        "describe": _git(root, ["describe", "--always", "--dirty", "--tags"], required=False),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "remote_origin": _git(root, ["remote", "get-url", "origin"], required=False),
        "submodules": submodules,
    }


def package_versions(names: Iterable[str] = PACKAGE_NAMES) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(root: Path) -> Dict[str, Any]:
    try:
        import torch

        torch_runtime = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        torch_runtime = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "captured_at_utc": utc_now_iso(),
        "git": git_provenance(root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": package_versions(),
        "torch_runtime": torch_runtime,
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp_path.replace(path)

"""Integrity and rosbag helpers for ROS 1 hardware amendment 001."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .integrity import sha256_file, verify_paths


AMENDMENT_PATH = Path(
    "reproducibility/hardware_validation/amendments/AMENDMENT_001.json"
)
SEALED_PREPARED_PATH = "condition_key.json"


def verify_operator_paths(entries, root: Path) -> list[dict]:
    """Verify distributed inputs while enforcing operator blinding."""
    entries = list(entries)
    sealed_entries = [
        entry
        for entry in entries
        if Path(entry["path"]).as_posix() == SEALED_PREPARED_PATH
    ]
    if sealed_entries and len(sealed_entries) != 1:
        raise RuntimeError("prepared manifest must declare one sealed condition key")
    sealed_path = Path(root) / SEALED_PREPARED_PATH
    if sealed_entries and sealed_path.exists():
        raise RuntimeError(
            "condition_key.json must not be present in an operator checkout"
        )
    visible_entries = [entry for entry in entries if entry not in sealed_entries]
    return verify_paths(visible_entries, root)


def verify_operator_prepared(prepared_dir: Path) -> dict:
    """Verify the frozen prepared set without distributing its condition key."""
    manifest_path = Path(prepared_dir) / "PREPARED_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed_entries = [
        entry
        for entry in manifest["files"]
        if Path(entry["path"]).as_posix() == SEALED_PREPARED_PATH
    ]
    if len(sealed_entries) != 1:
        raise RuntimeError("prepared manifest must declare one sealed condition key")
    checks = verify_operator_paths(manifest["files"], prepared_dir)
    if not all(check["passed"] for check in checks):
        failed = [check for check in checks if not check["passed"]]
        raise RuntimeError(f"operator prepared study verification failed: {failed}")
    return manifest


def verify_ros1_amendment(root: Path, amendment_path: Path | None = None) -> dict:
    root = Path(root).resolve()
    path = (amendment_path or root / AMENDMENT_PATH).resolve()
    amendment = json.loads(path.read_text(encoding="utf-8"))
    freeze_path = root / amendment["base_freeze"]["path"]
    if sha256_file(freeze_path) != amendment["base_freeze"]["sha256"]:
        raise RuntimeError("base FREEZE.json does not match ROS 1 amendment 001")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_checks = verify_paths(freeze["files"], root)
    if not all(row["passed"] for row in freeze_checks):
        failed = [row for row in freeze_checks if not row["passed"]]
        raise RuntimeError(f"base frozen files no longer verify: {failed}")
    checks = verify_paths(amendment["files"], root)
    if not all(row["passed"] for row in checks):
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"ROS 1 amendment verification failed: {failed}")
    return amendment


def validate_ros1_site(site: dict) -> None:
    platform = site.get("platform", {})
    if platform.get("ros_version") != 1 or platform.get("ros_distro") != "noetic":
        raise RuntimeError("site must explicitly declare ROS 1 Noetic")
    if not bool(site.get("calibration", {}).get("wheelbase_measured")):
        raise RuntimeError("measured wheelbase must be recorded before ROS 1 preflight")
    controller_limit = float(
        site.get("vehicle_limits", {}).get("controller_max_acceleration_mps2", 999.0)
    )
    if controller_limit <= 0.0 or controller_limit > 1.5:
        raise RuntimeError(
            "verified VESC controller acceleration must be positive and no more than "
            "the 1.5 m/s^2 provisional Drive-documented limit"
        )
    localization = site.get("course", {}).get("localization_system")
    if not localization or "REPLACE_WITH" in str(localization):
        raise RuntimeError("localization system must be resolved before preflight")


def ros1_bag_topics(site: dict) -> list[str]:
    topics = [
        site["topics"]["drive"],
        site["topics"]["odometry"],
        site["topics"]["deadman"],
        site["topics"]["estop"],
    ]
    for key in ("joint_states", "battery_state"):
        if site["topics"].get(key):
            topics.append(site["topics"][key])
    topics.extend(site["topics"].get("additional_bag_topics", []))
    return list(dict.fromkeys(topics))


def ros1_bag_command(run_dir: Path, site: dict) -> list[str]:
    bag_dir = Path(run_dir) / "rosbag"
    bag_path = bag_dir / "record.bag"
    command = ["rosbag", "record", "-O", str(bag_path)]
    compression = str(site.get("rosbag", {}).get("compression", "none"))
    if compression == "bz2":
        command.append("--bz2")
    elif compression == "lz4":
        command.append("--lz4")
    elif compression != "none":
        raise ValueError("ROS 1 bag compression must be one of: none, bz2, lz4")
    command.extend(ros1_bag_topics(site))
    return command


def start_ros1_bag(run_dir: Path, site: dict):  # pragma: no cover - robot runtime
    bag_dir = Path(run_dir) / "rosbag"
    bag_dir.mkdir(exist_ok=False)
    log_path = Path(run_dir) / "rosbag_process.log"
    command = ros1_bag_command(run_dir, site)
    log_handle = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command, stdout=log_handle, stderr=subprocess.STDOUT, text=True
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"rosbag exited before motion; inspect {log_path} "
                f"(code {process.returncode})"
            )
        if any(item.is_file() for item in bag_dir.iterdir()):
            return process, log_handle, {
                "relative_path": "rosbag",
                "format": "rosbag1",
                "topics": ros1_bag_topics(site),
                "command": command,
            }
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=3.0)
    log_handle.close()
    raise RuntimeError("ROS 1 rosbag did not create output within 5 seconds")


def stop_ros1_bag(process, log_handle) -> None:  # pragma: no cover - robot runtime
    if process is None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=10.0)
    except (subprocess.TimeoutExpired, AttributeError):
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
    finally:
        log_handle.close()


def archive_amendment_inputs(
    root: Path, run_dir: Path, amendment_path: Path | None = None
) -> None:
    root = Path(root).resolve()
    path = (amendment_path or root / AMENDMENT_PATH).resolve()
    amendment = verify_ros1_amendment(root, path)
    archive = Path(run_dir) / "frozen_inputs" / "amendment_001"
    archive.mkdir(exist_ok=False)
    shutil.copy2(path, archive / path.name)
    for entry in amendment["files"]:
        source = root / entry["path"]
        target = archive / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify_ros1_preflight(
    path: Path, config: dict, config_path: Path, site_path: Path
) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    age = time.time() - float(result["completed_unix_s"])
    maximum_age = float(result.get("maximum_age_hours", 12.0)) * 3600.0
    checks = {
        "passed": bool(result.get("passed")),
        "fresh": 0.0 <= age <= maximum_age,
        "config_hash": result.get("config_sha256") == sha256_file(config_path),
        "site_hash": result.get("site_sha256") == sha256_file(site_path),
        "study_id": result.get("study_id") == config["study_id"],
        "adapter": result.get("adapter") == "ros1",
        "amendment": result.get("amendment_id") == "AMENDMENT_001",
    }
    if not all(checks.values()):
        raise RuntimeError(f"ROS 1 real-run preflight is not valid: {checks}")
    return result

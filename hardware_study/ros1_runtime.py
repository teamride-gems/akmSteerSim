"""Integrity and rosbag helpers for the current ROS 1 hardware amendment."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .integrity import sha256_file, verify_paths


AMENDMENT_ID = "AMENDMENT_006"
AMENDMENT_PATH = Path(
    "reproducibility/hardware_validation/amendments/AMENDMENT_006.json"
)
MAX_CONTROLLER_ACCELERATION_MPS2 = 2.0
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
    if amendment.get("amendment_id") != AMENDMENT_ID:
        raise RuntimeError(
            f"expected {AMENDMENT_ID}, found {amendment.get('amendment_id')!r}"
        )
    prior = amendment.get("prior_amendment")
    if prior:
        prior_path = root / prior["path"]
        if sha256_file(prior_path) != prior["sha256"]:
            raise RuntimeError(f"prior amendment does not match {AMENDMENT_ID}")
    freeze_path = root / amendment["base_freeze"]["path"]
    if sha256_file(freeze_path) != amendment["base_freeze"]["sha256"]:
        raise RuntimeError(f"base FREEZE.json does not match ROS 1 {AMENDMENT_ID}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    amended_paths = {entry["path"] for entry in amendment.get("files", [])}
    freeze_checks = verify_paths(
        [entry for entry in freeze["files"] if entry["path"] not in amended_paths],
        root,
    )
    if not all(row["passed"] for row in freeze_checks):
        failed = [row for row in freeze_checks if not row["passed"]]
        raise RuntimeError(f"base frozen files no longer verify: {failed}")
    checks = verify_paths(amendment["files"], root)
    if not all(row["passed"] for row in checks):
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"ROS 1 {AMENDMENT_ID} verification failed: {failed}")
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
    if abs(controller_limit - MAX_CONTROLLER_ACCELERATION_MPS2) > 1e-9:
        raise RuntimeError(
            "verified VESC controller acceleration must equal the frozen "
            f"{MAX_CONTROLLER_ACCELERATION_MPS2:.1f} m/s^2 mechanical-lead setting"
        )
    course = site.get("course", {})
    localization = course.get("localization_system")
    if not localization or "REPLACE_WITH" in str(localization):
        raise RuntimeError("localization system must be resolved before preflight")
    ground_truth = course.get("evaluation_ground_truth_system")
    if not ground_truth or "REPLACE_WITH" in str(ground_truth):
        raise RuntimeError(
            "evaluation ground-truth availability must be declared before preflight"
        )
    localization_tf = site.get("topics", {}).get("localization_tf")
    if localization_tf:
        frames = site.get("frames", {})
        required_frames = (
            "odometry_frame_id",
            "localization_odom_frame_id",
            "base_frame_id",
        )
        unresolved = [
            key
            for key in required_frames
            if not frames.get(key) or "REPLACE_WITH" in str(frames.get(key))
        ]
        if unresolved:
            raise RuntimeError(
                f"Cartographer TF pose requires resolved frames: {unresolved}"
            )
    topics = site.get("topics", {})
    required_topic_keys = (
        "drive",
        "odometry",
        "deadman",
        "estop",
        "run_active",
        "safety_override",
    )
    for key in required_topic_keys:
        value = topics.get(key)
        if not value or "REPLACE_WITH" in str(value):
            raise RuntimeError(f"ROS 1 topic {key!r} must be resolved")
    required_topic_values = [str(topics[key]) for key in required_topic_keys]
    if len(set(required_topic_values)) != len(required_topic_values):
        raise RuntimeError("ROS 1 command, safety, and state topics must be distinct")
    bridge = site.get("safety_bridge", {})
    joy_topic = bridge.get("joy_topic")
    if not joy_topic or "REPLACE_WITH" in str(joy_topic):
        raise RuntimeError("safety bridge joystick topic must be resolved")
    try:
        deadman_index = int(bridge["deadman_button_index"])
        estop_index = int(bridge["estop_button_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("safety bridge button indices must be resolved integers") from exc
    if min(deadman_index, estop_index) < 0 or deadman_index == estop_index:
        raise RuntimeError("deadman and e-stop buttons must be distinct and nonnegative")
    if estop_index != 6:
        raise RuntimeError(
            "Frank's confirmed experiment software e-stop is joystick index 6"
        )
    joy_stale = float(bridge.get("joy_stale_seconds", 0.0))
    run_active_stale = float(bridge.get("run_active_stale_seconds", 0.0))
    if min(joy_stale, run_active_stale) <= 0.0 or max(
        joy_stale, run_active_stale
    ) > 0.25:
        raise RuntimeError(
            "joystick and runner heartbeat timeouts must be greater than zero "
            "and at most 0.25 seconds"
        )
    if float(bridge.get("publish_rate_hz", 0.0)) < 20.0:
        raise RuntimeError("safety bridge publish rate must be at least 20 Hz")
    clearance = float(bridge.get("deadman_clearance_seconds", 0.0))
    if abs(clearance - 1.0) > 1e-9:
        raise RuntimeError("autonomous mux-clearance interval must equal 1.0 second")


def ros1_bag_topics(site: dict) -> list[str]:
    topics = [
        site["topics"]["drive"],
        site["topics"]["odometry"],
        site["topics"]["deadman"],
        site["topics"]["estop"],
        site["topics"]["run_active"],
        site["topics"]["safety_override"],
    ]
    if site["topics"].get("localization_tf"):
        topics.append(site["topics"]["localization_tf"])
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
    archive = Path(run_dir) / "frozen_inputs" / AMENDMENT_ID.lower()
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
        "amendment": result.get("amendment_id") == AMENDMENT_ID,
    }
    if not all(checks.values()):
        raise RuntimeError(f"ROS 1 real-run preflight is not valid: {checks}")
    return result

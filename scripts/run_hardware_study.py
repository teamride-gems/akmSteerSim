#!/usr/bin/env python3
"""Execute exactly one frozen, blinded hardware-study run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.adapters import make_adapter
from hardware_study.bundles import load_bundle_from_manifest
from hardware_study.execution import execute_run
from hardware_study.integrity import sha256_file, verify_paths, write_json
from hardware_study.validation import validate_and_write


DEFAULT_CONFIG = "configs/hardware_study_v1.yaml"
DEFAULT_PREPARED = "reproducibility/hardware_validation/study_v1/prepared"
DEFAULT_FREEZE = "reproducibility/hardware_validation/study_v1/FREEZE.json"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_no_placeholders(value, location: str = "site") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_placeholders(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_placeholders(child, f"{location}[{index}]")
    elif isinstance(value, str) and "REPLACE_WITH" in value:
        raise RuntimeError(f"unresolved placeholder at {location}: {value}")


def verify_prepared(prepared_dir: Path) -> dict:
    manifest_path = prepared_dir / "PREPARED_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = verify_paths(manifest["files"], prepared_dir)
    if not all(check["passed"] for check in checks):
        failed = [check for check in checks if not check["passed"]]
        raise RuntimeError(f"prepared study verification failed: {failed}")
    return manifest


def verify_freeze(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks = verify_paths(freeze["files"], ROOT)
    if not all(check["passed"] for check in checks):
        failed = [check for check in checks if not check["passed"]]
        raise RuntimeError(f"study freeze verification failed: {failed}")
    return freeze


def verify_preflight(
    path: Path, config: dict, config_path: Path, site_path: Path
) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    age = time.time() - float(result["completed_unix_s"])
    maximum_age = float(result.get("maximum_age_hours", 12.0)) * 3600.0
    checks = {
        "passed": bool(result.get("passed")),
        "fresh": 0.0 <= age <= maximum_age,
        "config_hash": result.get("config_sha256")
        == sha256_file(config_path),
        "site_hash": result.get("site_sha256") == sha256_file(site_path),
        "study_id": result.get("study_id") == config["study_id"],
        "adapter": result.get("adapter") == "ros2",
    }
    if not all(checks.values()):
        raise RuntimeError(f"real-run preflight is not valid: {checks}")
    return result


def rosbag_topics(site: dict) -> list[str]:
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


def start_rosbag(run_dir: Path, site: dict):  # pragma: no cover - robot runtime
    bag_dir = run_dir / "rosbag"
    log_path = run_dir / "rosbag_process.log"
    command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        str(site["rosbag"].get("storage_id", "mcap")),
        "--output",
        str(bag_dir),
        *rosbag_topics(site),
    ]
    log_handle = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"rosbag exited before motion; inspect {log_path} (code {process.returncode})"
            )
        if bag_dir.exists() and any(item.is_file() for item in bag_dir.rglob("*")):
            return process, log_handle, {
                "relative_path": "rosbag",
                "topics": rosbag_topics(site),
                "command": command,
            }
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=3.0)
    log_handle.close()
    raise RuntimeError("rosbag did not create its output directory within 5 seconds")


def stop_rosbag(process, log_handle) -> None:  # pragma: no cover - robot runtime
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


def archive_inputs(
    run_dir: Path,
    config_path: Path,
    site_path: Path,
    prepared_manifest_path: Path,
    freeze_path: Path,
    schedule_row: dict,
    bundle: dict,
) -> None:
    archive = run_dir / "frozen_inputs"
    archive.mkdir(exist_ok=False)
    shutil.copy2(config_path, archive / "hardware_study_v1.yaml")
    shutil.copy2(site_path, archive / "hardware_site.yaml")
    shutil.copy2(prepared_manifest_path, archive / "PREPARED_MANIFEST.json")
    if freeze_path.is_file():
        shutil.copy2(freeze_path, archive / "FREEZE.json")
    write_json(archive / "schedule_row.json", schedule_row)
    write_json(archive / "command_bundle.json", bundle)


def retry_code_for_attempt(attempt: Path) -> str | None:
    launch_failure = attempt / "launch_failure.json"
    if launch_failure.is_file():
        payload = json.loads(launch_failure.read_text(encoding="utf-8"))
        if not payload.get("motion_started", False):
            return "logging_or_bag_start_failure_before_first_motion"
        return None
    manifest_path = attempt / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("motion_started"):
        return None
    reason = str(manifest.get("abort_reason") or "")
    if "start_pose_out_of_tolerance" in reason or "start_heading_out_of_tolerance" in reason:
        return "start_pose_out_of_tolerance_before_arm"
    if "adapter_not_ready" in reason or "telemetry_stale" in reason:
        return "missing_or_stale_telemetry_before_first_motion"
    return None


def next_expected_run(schedule: list[dict], output_root: Path, config: dict) -> str | None:
    allowed = config["run_validity"]["allowed_pre_motion_technical_reruns"]
    seen_ids = set()
    for row in schedule:
        seen_ids.add(row["run_id"])
        container = output_root / row["run_id"]
        attempts = sorted(container.glob("attempt_*")) if container.is_dir() else []
        if not attempts:
            later = [
                child.name
                for child in output_root.iterdir()
                if output_root.is_dir()
                and child.is_dir()
                and child.name not in seen_ids
            ] if output_root.is_dir() else []
            if later:
                raise RuntimeError(
                    f"out-of-order archives exist after missing {row['run_id']}: {later}"
                )
            return row["run_id"]
        outcomes = []
        for attempt in attempts:
            manifest_path = attempt / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("completed") or manifest.get("motion_started"):
                    outcomes.append(attempt)
                    continue
            code = retry_code_for_attempt(attempt)
            if code not in allowed:
                raise RuntimeError(f"attempt requires manual audit before proceeding: {attempt}")
        if len(outcomes) == 0:
            return row["run_id"]
        if len(outcomes) != 1 or outcomes[0] != attempts[-1]:
            raise RuntimeError(f"ambiguous attempts for {row['run_id']}")
        validation_path = outcomes[0] / "validation.json"
        if not validation_path.is_file():
            raise RuntimeError(f"outcome lacks validation: {outcomes[0]}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("eligible_outcome"):
            raise RuntimeError(f"outcome failed validation: {outcomes[0]}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run exactly one row of the frozen blinded hardware schedule."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter", choices=("mock", "ros2"), required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--site")
    parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED)
    parser.add_argument("--freeze", default=DEFAULT_FREEZE)
    parser.add_argument("--output-root", default="hardware_runs/study_v1")
    parser.add_argument("--preflight")
    parser.add_argument(
        "--allow-unfrozen-mock",
        action="store_true",
        help="Engineering-only escape hatch; never accepted by the real adapter.",
    )
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    prepared_dir = (ROOT / args.prepared_dir).resolve()
    freeze_path = (ROOT / args.freeze).resolve()
    output_root = (ROOT / args.output_root).resolve()
    if args.site:
        site_path = (ROOT / args.site).resolve()
    elif args.adapter == "mock":
        site_path = ROOT / "configs/hardware_site_mock.yaml"
    else:
        site_path = ROOT / "local_hardware_site.yaml"

    config = load_yaml(config_path)
    site = load_yaml(site_path)
    assert_no_placeholders(site)
    prepared = verify_prepared(prepared_dir)
    if prepared["study_id"] != config["study_id"]:
        raise RuntimeError("prepared manifest and config study identifiers disagree")
    if prepared["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("prepared manifest does not match the supplied config")
    if freeze_path.is_file():
        freeze = verify_freeze(freeze_path)
    elif args.adapter == "mock" and args.allow_unfrozen_mock:
        freeze = {"status": "UNFROZEN_MOCK_ONLY"}
    else:
        raise FileNotFoundError(f"required freeze record is missing: {freeze_path}")

    schedule = json.loads(
        (prepared_dir / "machine_schedule.json").read_text(encoding="utf-8")
    )
    matches = [row for row in schedule if row["run_id"] == args.run_id]
    if len(matches) != 1:
        raise RuntimeError(f"run id must occur exactly once in schedule: {args.run_id}")
    row = matches[0]
    if args.adapter == "ros2":
        if not args.preflight:
            raise RuntimeError("a recent passed ROS 2 preflight record is required")
        verify_preflight(
            (ROOT / args.preflight).resolve(), config, config_path, site_path
        )
    bundle, bundle_entry = load_bundle_from_manifest(prepared_dir, row["bundle_id"])

    expected_next = next_expected_run(schedule, output_root, config)
    if expected_next is None:
        raise RuntimeError("the complete 120-run schedule is already archived")
    if row["run_id"] != expected_next:
        raise RuntimeError(
            f"out-of-order run refused; the next authorized run is {expected_next}"
        )
    run_container = output_root / row["run_id"]
    existing_attempts = (
        sorted(run_container.glob("attempt_*")) if run_container.is_dir() else []
    )
    if existing_attempts:
        allowed = config["run_validity"]["allowed_pre_motion_technical_reruns"]
        retry_code = retry_code_for_attempt(existing_attempts[-1])
        if retry_code not in allowed:
            raise RuntimeError(
                "another attempt is forbidden after motion or an unapproved pre-motion failure"
            )
    attempt_number = len(existing_attempts) + 1
    run_dir = run_container / f"attempt_{attempt_number:03d}"
    run_dir.mkdir(parents=True)
    os.environ["ROS_DOMAIN_ID"] = str(site.get("ros_domain_id", 0))
    archive_inputs(
        run_dir,
        config_path,
        site_path,
        prepared_dir / "PREPARED_MANIFEST.json",
        freeze_path,
        row,
        bundle,
    )
    static_hashes = {
        "config_sha256": sha256_file(config_path),
        "site_sha256": sha256_file(site_path),
        "prepared_manifest_sha256": sha256_file(
            prepared_dir / "PREPARED_MANIFEST.json"
        ),
        "machine_schedule_sha256": sha256_file(
            prepared_dir / "machine_schedule.json"
        ),
        "bundle_sha256": bundle_entry["sha256"],
        "freeze_sha256": sha256_file(freeze_path) if freeze_path.is_file() else None,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    process = None
    log_handle = None
    bag_metadata = None
    adapter = None
    try:
        try:
            if args.adapter == "ros2":
                process, log_handle, bag_metadata = start_rosbag(run_dir, site)
            adapter = make_adapter(args.adapter, site)
            manifest = execute_run(
                adapter,
                bundle,
                row,
                config,
                site,
                run_dir,
                static_hashes,
                rosbag_metadata=bag_metadata,
            )
        except BaseException as exc:
            if not (run_dir / "run_manifest.json").is_file():
                write_json(
                    run_dir / "launch_failure.json",
                    {
                        "run_id": row["run_id"],
                        "motion_started": False,
                        "retry_code": "logging_or_bag_start_failure_before_first_motion",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "static_hashes": static_hashes,
                        "recorded_unix_s": time.time(),
                    },
                )
            raise
    finally:
        if adapter is not None:
            adapter.close()
        if process is not None:
            stop_rosbag(process, log_handle)
    result = validate_and_write(run_dir, config, bundle, row)
    summary = {
        "run_id": row["run_id"],
        "condition_code": row["condition_code"],
        "completed": manifest["completed"],
        "technical_valid": result["technical_valid"],
        "eligible_outcome": result["eligible_outcome"],
        "archive": str(run_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["eligible_outcome"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

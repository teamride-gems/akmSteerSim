#!/usr/bin/env python3
"""Capture Frank's ROS 1 interface and selected controller files before motion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.integrity import sha256_file, write_json
from hardware_study.ros1_runtime import (
    MAX_CONTROLLER_ACCELERATION_MPS2,
    validate_ros1_site,
    verify_ros1_amendment,
)


EXPECTED_TYPES = {
    "drive": "ackermann_msgs/AckermannDriveStamped",
    "odometry": "nav_msgs/Odometry",
    "localization_tf": "tf2_msgs/TFMessage",
}


def run_checked(command: list[str]) -> str:
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=15.0
    )
    return result.stdout.strip()


def find_named_values(value, key: str) -> list[float]:
    found = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if str(child_key) == key:
                found.append(float(child))
            found.extend(find_named_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_named_values(child, key))
    return found


def main() -> int:  # pragma: no cover - requires Frank's ROS 1 environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_hardware_site_ros1_draft.yaml")
    parser.add_argument("--vesc-config", required=True)
    parser.add_argument("--joy-config", required=True)
    parser.add_argument("--launch-file", action="append", default=[])
    parser.add_argument("--output", default="hardware_runs/configuration_capture")
    args = parser.parse_args()

    verify_ros1_amendment(ROOT)
    site_path = (ROOT / args.site).resolve()
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    validate_ros1_site(site)
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"configuration capture refuses overwrite: {output}")
    output.mkdir(parents=True)

    vesc_path = Path(args.vesc_config).expanduser().resolve()
    joy_path = Path(args.joy_config).expanduser().resolve()
    source_files = [vesc_path, joy_path, *[Path(p).expanduser().resolve() for p in args.launch_file]]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    vesc = yaml.safe_load(vesc_path.read_text(encoding="utf-8"))
    acceleration_values = find_named_values(vesc, "max_acceleration")
    if len(acceleration_values) != 1:
        raise RuntimeError(
            f"expected exactly one max_acceleration in VESC config, found {acceleration_values}"
        )
    if (
        acceleration_values[0] <= 0.0
        or acceleration_values[0] > MAX_CONTROLLER_ACCELERATION_MPS2
    ):
        raise RuntimeError(
            f"unsafe/unverified VESC max_acceleration: {acceleration_values[0]} m/s^2"
        )
    configured = float(site["vehicle_limits"]["controller_max_acceleration_mps2"])
    if abs(configured - acceleration_values[0]) > 1e-9:
        raise RuntimeError("site acceleration value does not match captured vesc.yaml")

    topic_rows = []
    expected_topics = [
        (key, site["topics"].get(key), expected_type)
        for key, expected_type in EXPECTED_TYPES.items()
    ]
    expected_topics.append(
        ("joy_source", site["safety_bridge"]["joy_topic"], "sensor_msgs/Joy")
    )
    for key, topic, expected_type in expected_topics:
        if not topic:
            continue
        observed_type = run_checked(["rostopic", "type", topic])
        if observed_type != expected_type:
            raise RuntimeError(
                f"{topic} has type {observed_type!r}; expected {expected_type!r}"
            )
        topic_rows.append(
            {
                "role": key,
                "topic": topic,
                "type": observed_type,
                "info": run_checked(["rostopic", "info", topic]),
            }
        )

    copied = []
    files_dir = output / "controller_files"
    files_dir.mkdir()
    for index, source in enumerate(source_files):
        destination = files_dir / f"{index:02d}_{source.name}"
        shutil.copy2(source, destination)
        copied.append(
            {
                "source_path": str(source),
                "archived_path": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
            }
        )
    shutil.copy2(site_path, output / "hardware_site_draft.yaml")
    manifest = {
        "schema_version": 1,
        "capture_id": "ROS1_CONFIGURATION_CAPTURE",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "ros_distro": run_checked(["rosversion", "-d"]),
        "ros_version": 1,
        "ros_nodes": run_checked(["rosnode", "list"]).splitlines(),
        "ros_topics": run_checked(["rostopic", "list"]).splitlines(),
        "required_topic_interfaces": topic_rows,
        "vesc_max_acceleration_mps2": acceleration_values[0],
        "files": copied,
        "site_sha256": sha256_file(site_path),
    }
    write_json(output / "configuration_manifest.json", manifest)
    print(json.dumps({"passed": True, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

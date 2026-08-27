#!/usr/bin/env python3
"""Fail-closed static and live preflight for the hardware study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.adapters import make_adapter
from hardware_study.execution import health_reasons, start_pose_errors
from hardware_study.integrity import sha256_file, verify_paths, write_json
from hardware_study.safety import check_runtime_safety


def no_placeholders(value) -> bool:
    if isinstance(value, dict):
        return all(no_placeholders(child) for child in value.values())
    if isinstance(value, list):
        return all(no_placeholders(child) for child in value)
    return not (isinstance(value, str) and "REPLACE_WITH" in value)


def maximum_predicted_radius(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(float(row["maximum_radius_m"]) for row in csv.DictReader(handle))


def unique_required_topics(site: dict) -> bool:
    values = [
        site["topics"]["drive"],
        site["topics"]["odometry"],
        site["topics"]["deadman"],
        site["topics"]["estop"],
    ]
    return all(isinstance(value, str) and value.startswith("/") for value in values) and (
        len(values) == len(set(values))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=("mock", "ros2"), required=True)
    parser.add_argument("--config", default="configs/hardware_study_v1.yaml")
    parser.add_argument("--site")
    parser.add_argument(
        "--prepared-dir",
        default="reproducibility/hardware_validation/study_v1/prepared",
    )
    parser.add_argument(
        "--freeze", default="reproducibility/hardware_validation/study_v1/FREEZE.json"
    )
    parser.add_argument("--output")
    parser.add_argument("--wheels-on-stands-verified", action="store_true")
    parser.add_argument("--physical-estop-tested", action="store_true")
    parser.add_argument("--course-cleared", action="store_true")
    parser.add_argument("--localization-checked", action="store_true")
    parser.add_argument("--zero-command-test", default="")
    parser.add_argument("--maximum-age-hours", type=float, default=12.0)
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    prepared_dir = (ROOT / args.prepared_dir).resolve()
    freeze_path = (ROOT / args.freeze).resolve()
    if args.site:
        site_path = (ROOT / args.site).resolve()
    elif args.adapter == "mock":
        site_path = ROOT / "configs/hardware_site_mock.yaml"
    else:
        site_path = ROOT / "local_hardware_site.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    site = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    prepared = json.loads(
        (prepared_dir / "PREPARED_MANIFEST.json").read_text(encoding="utf-8")
    )
    prepared_checks = verify_paths(prepared["files"], prepared_dir)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_checks = verify_paths(freeze["files"], ROOT)
    predicted_radius = maximum_predicted_radius(prepared_dir / "safety_envelope.csv")
    clear_radius = float(site["course"]["surveyed_clear_radius_m"])
    static_checks = {
        "site_has_no_placeholders": no_placeholders(site),
        "required_topics_are_distinct_absolute_names": unique_required_topics(site),
        "prepared_files_match": all(row["passed"] for row in prepared_checks),
        "freeze_files_match": all(row["passed"] for row in freeze_checks),
        "prepared_config_matches": prepared["config_sha256"]
        == sha256_file(config_path),
        "clear_radius_has_one_meter_margin": clear_radius >= predicted_radius + 1.0,
        "runtime_geofence_has_one_meter_stopping_margin": float(
            config["safety"]["maximum_geofence_radius_m"]
        )
        + 1.0
        <= clear_radius,
        "deadman_and_estop_buttons_are_distinct": int(
            site["safety_bridge"]["deadman_button_index"]
        )
        != int(site["safety_bridge"]["estop_button_index"]),
    }
    human_checks = {
        "wheels_on_stands_verified": args.wheels_on_stands_verified,
        "physical_estop_tested": args.physical_estop_tested,
        "course_cleared": args.course_cleared,
        "localization_checked": args.localization_checked,
        "zero_command_test_acknowledged": args.zero_command_test
        == "ZERO COMMAND TEST ON STANDS",
    }
    if args.adapter == "mock":
        human_checks = {key: True for key in human_checks}
    if not all(static_checks.values()):
        raise RuntimeError(f"static preflight failed: {static_checks}")
    if not all(human_checks.values()):
        raise RuntimeError(f"required signed checklist items missing: {human_checks}")

    os.environ["ROS_DOMAIN_ID"] = str(site.get("ros_domain_id", 0))
    adapter = make_adapter(args.adapter, site)
    live_rows = []
    live_checks = {}
    try:
        live_checks["adapter_ready"] = adapter.wait_for_ready(timeout_seconds=10.0)
        if not live_checks["adapter_ready"]:
            raise RuntimeError("adapter did not receive odometry and safety topics")
        initial = adapter.latest_telemetry()
        deadline = adapter.now() + (0.5 if args.adapter == "mock" else 2.0)
        if args.adapter == "mock":
            adapter.publish(0.0, 0.0, 0.5)
        else:
            while adapter.now() < deadline:
                adapter.publish(0.0, 0.0, 0.05)
                adapter.wait_until(adapter.now() + 0.05)
        live_rows.extend(adapter.drain_telemetry())
        final = adapter.latest_telemetry()
        now = adapter.now()
        errors = start_pose_errors(final, site["expected_start_pose"])
        live_checks.update(
            {
                "fresh_health": not health_reasons(adapter, now, config),
                "runtime_safety": not check_runtime_safety(final, initial, config),
                "start_position": errors["position_m"]
                <= float(config["run_validity"]["start_position_tolerance_m"]),
                "start_heading": errors["heading_rad"]
                <= math.radians(
                    float(config["run_validity"]["start_heading_tolerance_deg"])
                ),
                "stationary": abs(float(final["speed_mps"])) <= 0.03,
            }
        )
    finally:
        adapter.close()
    times = sorted(
        {
            float(row["received_monotonic_s"])
            for row in live_rows
            if row.get("received_monotonic_s") is not None
        }
    )
    gaps = np.diff(times) if len(times) > 1 else np.asarray([])
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
    live_checks["telemetry_rate"] = rate >= float(
        config["run_validity"]["minimum_telemetry_rate_hz"]
    )
    live_checks["telemetry_gap"] = bool(gaps.size) and float(np.max(gaps)) <= float(
        config["run_validity"]["telemetry_gap_max_seconds"]
    )
    passed = all(static_checks.values()) and all(human_checks.values()) and all(
        live_checks.values()
    )
    completed = time.time()
    if args.output:
        output_path = (ROOT / args.output).resolve()
    else:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(completed))
        output_path = ROOT / "hardware_runs/preflight" / f"preflight_{stamp}.json"
    result = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "adapter": args.adapter,
        "passed": passed,
        "completed_unix_s": completed,
        "maximum_age_hours": float(args.maximum_age_hours),
        "config_sha256": sha256_file(config_path),
        "site_sha256": sha256_file(site_path),
        "prepared_manifest_sha256": sha256_file(
            prepared_dir / "PREPARED_MANIFEST.json"
        ),
        "freeze_sha256": sha256_file(freeze_path),
        "static_checks": static_checks,
        "human_checks": human_checks,
        "live_checks": live_checks,
        "diagnostics": {
            "predicted_radius_m": predicted_radius,
            "surveyed_clear_radius_m": clear_radius,
            "telemetry_samples": len(times),
            "telemetry_rate_hz": rate,
            "maximum_telemetry_gap_s": float(np.max(gaps)) if gaps.size else None,
        },
    }
    write_json(output_path, result)
    print(json.dumps({"passed": passed, "output": str(output_path)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

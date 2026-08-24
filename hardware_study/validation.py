"""Post-run integrity, timing, telemetry, and eligibility validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .integrity import sha256_file, verify_hash_chain, write_json


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def validate_run(run_dir: Path, config: dict, bundle: dict, schedule_row: dict) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    log_path = run_dir / "records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chain = verify_hash_chain(log_path)
    records = chain.pop("records")
    commands = [row for row in records if row.get("record_type") == "command"]
    telemetry_records = [row for row in records if row.get("record_type") == "telemetry"]
    telemetry = [row["telemetry"] for row in telemetry_records]
    stop_commands = [
        row for row in records if row.get("record_type") == "safe_stop_command"
    ]
    expected_packets = len(bundle["conditions"][schedule_row["condition"]])
    expected_rows = bundle["conditions"][schedule_row["condition"]]
    completed = bool(manifest["completed"])
    post_motion_abort = bool(manifest["motion_started"] and not completed)
    lateness = np.asarray([float(row["lateness_s"]) for row in commands], dtype=float)
    telemetry_times = np.asarray(
        sorted(
            {
                float(row["received_monotonic_s"])
                for row in telemetry
                if row.get("received_monotonic_s") is not None
            }
        ),
        dtype=float,
    )
    gaps = np.diff(telemetry_times) if telemetry_times.size > 1 else np.array([])
    duration = float(telemetry_times[-1] - telemetry_times[0]) if telemetry_times.size > 1 else 0.0
    telemetry_rate = float((telemetry_times.size - 1) / duration) if duration > 0.0 else 0.0
    main_telemetry = [
        row["telemetry"]
        for row in telemetry_records
        if row.get("command_phase") == "main"
    ]
    finite_main = bool(main_telemetry) and all(
        math.isfinite(float(row[key]))
        for row in main_telemetry
        for key in ("x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rad_s")
    )
    validity = config["run_validity"]
    safety = config["safety"]
    rosbag = manifest.get("rosbag") or {}
    bag_path = run_dir / rosbag.get("relative_path", "__missing_rosbag__")
    bag_required = bool(validity["require_rosbag_for_real_adapter"]) and manifest[
        "adapter"
    ] != "mock"
    archive = run_dir / "frozen_inputs"
    static = manifest.get("static_hashes") or {}
    required_static = {
        "config_sha256",
        "site_sha256",
        "prepared_manifest_sha256",
        "machine_schedule_sha256",
        "bundle_sha256",
        "freeze_sha256",
        "runner_sha256",
    }
    archive_checks = {
        "config": (archive / "hardware_study_v1.yaml").is_file()
        and sha256_file(archive / "hardware_study_v1.yaml")
        == static.get("config_sha256"),
        "site": (archive / "hardware_site.yaml").is_file()
        and sha256_file(archive / "hardware_site.yaml") == static.get("site_sha256"),
        "prepared_manifest": (archive / "PREPARED_MANIFEST.json").is_file()
        and sha256_file(archive / "PREPARED_MANIFEST.json")
        == static.get("prepared_manifest_sha256"),
        "freeze": (archive / "FREEZE.json").is_file()
        and sha256_file(archive / "FREEZE.json") == static.get("freeze_sha256"),
        "bundle": (archive / "command_bundle.json").is_file()
        and sha256_file(archive / "command_bundle.json")
        == static.get("bundle_sha256"),
        "schedule_row": (archive / "schedule_row.json").is_file()
        and json.loads((archive / "schedule_row.json").read_text(encoding="utf-8"))
        == schedule_row,
    }
    sent_target_match = all(
        int(command["packet_index"]) < len(expected_rows)
        and command["phase"]
        == expected_rows[int(command["packet_index"])]["phase"]
        and int(command["phase_packet_index"])
        == int(expected_rows[int(command["packet_index"])]["phase_packet_index"])
        and float(command["study_target_steering_rad"])
        == float(expected_rows[int(command["packet_index"])]["target_steering_rad"])
        and float(command["study_target_speed_mps"])
        == float(expected_rows[int(command["packet_index"])]["target_speed_mps"])
        for command in commands
    )
    sent_steering = [float(row["sent_steering_rad"]) for row in commands]
    sent_speed = [float(row["sent_speed_mps"]) for row in commands]
    dt = float(bundle["packet_dt_seconds"])
    steering_steps = np.diff(np.asarray([0.0, *sent_steering]))
    speed_steps = np.diff(np.asarray([0.0, *sent_speed]))
    main_command_times = [
        float(row["planned_send_monotonic_s"])
        for row in commands
        if row["phase"] == "main"
    ]
    main_grid_covered = bool(
        bool(main_command_times and telemetry_times.size)
        and telemetry_times[0]
        <= min(main_command_times) + float(validity["telemetry_gap_max_seconds"])
        and telemetry_times[-1]
        >= max(main_command_times) - float(validity["telemetry_gap_max_seconds"])
    )
    checks = {
        "manifest_log_hash": manifest["records_sha256"] == sha256_file(log_path),
        "hash_chain": bool(chain["passed"])
        and manifest["terminal_record_sha256"] == chain["terminal_record_sha256"],
        "run_id": manifest["run_id"] == schedule_row["run_id"],
        "bundle_id": manifest["bundle_id"] == bundle["bundle_id"],
        "condition": manifest["condition"] == schedule_row["condition"],
        "static_hash_fields": required_static.issubset(static)
        and static.get("freeze_sha256") is not None,
        "archived_frozen_inputs": all(archive_checks.values()),
        "safe_stop_count": len(stop_commands) == int(safety["safe_stop_packets"]),
        "packet_count_or_post_motion_abort": (
            completed and len(commands) == expected_packets
        )
        or post_motion_abort,
        "packet_indices": [int(row["packet_index"]) for row in commands]
        == list(range(len(commands))),
        "packet_targets_and_phases": sent_target_match,
        "command_lateness_p95": bool(lateness.size)
        and float(np.percentile(lateness, 95))
        <= float(validity["command_lateness_p95_max_seconds"]),
        "command_lateness_max": bool(lateness.size)
        and float(np.max(lateness)) <= float(validity["command_lateness_max_seconds"]),
        "telemetry_rate": telemetry_rate >= float(validity["minimum_telemetry_rate_hz"]),
        "telemetry_gap": bool(gaps.size)
        and float(np.max(gaps)) <= float(validity["telemetry_gap_max_seconds"]),
        "main_localization": finite_main if completed else post_motion_abort,
        "main_time_grid_covered": main_grid_covered if completed else post_motion_abort,
        "steering_bound": all(
            abs(float(row["sent_steering_rad"]))
            <= float(safety["maximum_sent_abs_steering_rad"]) + 1e-12
            for row in commands
        ),
        "speed_bound": all(
            0.0 <= float(row["sent_speed_mps"])
            <= float(safety["maximum_command_speed_mps"]) + 1e-12
            for row in commands
        ),
        "observed_speed_bound": all(
            abs(float(row["speed_mps"]))
            <= float(safety["maximum_observed_speed_mps"]) + 1e-12
            for row in telemetry
            if row.get("speed_mps") is not None
        ),
        "steering_slew_bound": bool(commands)
        and bool(
            np.all(
                np.abs(steering_steps)
                <= float(safety["common_sent_steering_slew_rad_s"]) * dt + 1e-12
            )
        ),
        "acceleration_bound": bool(commands)
        and bool(
            np.all(
                np.abs(speed_steps)
                <= float(safety["common_sent_acceleration_mps2"]) * dt + 1e-12
            )
        ),
        "start_position": float(manifest["start_pose_errors"]["position_m"])
        <= float(validity["start_position_tolerance_m"]),
        "start_heading": float(manifest["start_pose_errors"]["heading_rad"])
        <= math.radians(float(validity["start_heading_tolerance_deg"])),
        "rosbag": (not bag_required) or (_tree_size(bag_path) > 0),
    }
    technical_valid = all(checks.values())
    eligible_outcome = bool(technical_valid and (completed or post_motion_abort))
    return {
        "run_id": manifest["run_id"],
        "block_id": manifest["block_id"],
        "condition": manifest["condition"],
        "condition_code": manifest["condition_code"],
        "adapter": manifest["adapter"],
        "completed": completed,
        "motion_started": bool(manifest["motion_started"]),
        "post_motion_abort": post_motion_abort,
        "abort_reason": manifest["abort_reason"],
        "technical_valid": technical_valid,
        "eligible_outcome": eligible_outcome,
        "checks": checks,
        "diagnostics": {
            "command_count": len(commands),
            "expected_command_count": expected_packets,
            "safe_stop_count": len(stop_commands),
            "telemetry_records": len(telemetry),
            "main_telemetry_records": len(main_telemetry),
            "telemetry_rate_hz": telemetry_rate,
            "maximum_telemetry_gap_s": float(np.max(gaps)) if gaps.size else None,
            "command_lateness_p95_s": float(np.percentile(lateness, 95))
            if lateness.size
            else None,
            "command_lateness_max_s": float(np.max(lateness)) if lateness.size else None,
            "hash_chain_errors": chain["errors"],
            "rosbag_size_bytes": _tree_size(bag_path),
            "archive_checks": archive_checks,
        },
    }


def validate_and_write(
    run_dir: Path, config: dict, bundle: dict, schedule_row: dict
) -> dict:
    result = validate_run(run_dir, config, bundle, schedule_row)
    write_json(Path(run_dir) / "validation.json", result)
    return result

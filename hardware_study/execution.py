"""One-run fail-closed executor shared by CLI and tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import traceback

from .integrity import HashChainWriter, sha256_file, write_json
from .safety import SafetyLimiter, check_runtime_safety


def angle_difference(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def start_pose_errors(actual: dict, expected: dict) -> dict:
    return {
        "position_m": math.hypot(
            float(actual["x_m"]) - float(expected["x_m"]),
            float(actual["y_m"]) - float(expected["y_m"]),
        ),
        "heading_rad": abs(
            angle_difference(float(actual["yaw_rad"]), float(expected["yaw_rad"]))
        ),
    }


def health_reasons(adapter, now: float, config: dict) -> list[str]:
    telemetry = adapter.latest_telemetry()
    safety = config["safety"]
    reasons = []
    if now - float(telemetry["received_monotonic_s"]) > float(
        safety["telemetry_stale_seconds"]
    ):
        reasons.append("telemetry_stale")
    deadman_time = telemetry.get("deadman_received_monotonic_s")
    if deadman_time is None or now - float(deadman_time) > float(
        safety["deadman_stale_seconds"]
    ):
        reasons.append("deadman_stale")
    estop_time = telemetry.get("estop_received_monotonic_s")
    if estop_time is None or now - float(estop_time) > float(
        safety["deadman_stale_seconds"]
    ):
        reasons.append("estop_status_stale")
    return reasons


def write_telemetry(writer: HashChainWriter, telemetry_rows: list[dict], context: dict) -> None:
    for telemetry in telemetry_rows:
        writer.write({"record_type": "telemetry", **context, "telemetry": telemetry})


def safe_stop(adapter, writer, config, context) -> int:
    safety = config["safety"]
    count = int(safety["safe_stop_packets"])
    dt = float(config["command_bundle"]["packet_dt_seconds"])
    sent = 0
    for index in range(count):
        adapter.publish(
            float(safety["safe_stop_steering_rad"]),
            float(safety["safe_stop_speed_mps"]),
            dt,
        )
        writer.write(
            {
                "record_type": "safe_stop_command",
                **context,
                "safe_stop_index": index,
                "sent_monotonic_s": adapter.now(),
                "steering_rad": float(safety["safe_stop_steering_rad"]),
                "speed_mps": float(safety["safe_stop_speed_mps"]),
            }
        )
        write_telemetry(writer, adapter.drain_telemetry(), context)
        sent += 1
        if adapter.is_real:
            adapter.wait_until(adapter.now() + dt)
    return sent


def execute_run(
    adapter,
    bundle: dict,
    schedule_row: dict,
    config: dict,
    site: dict,
    run_dir: Path,
    static_hashes: dict,
    rosbag_metadata: dict | None = None,
) -> dict:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        # A real run may pre-create only the bag and its process log.
        allowed = {"frozen_inputs", "rosbag", "rosbag_process.log"}
        disallowed = [item for item in run_dir.iterdir() if item.name not in allowed]
        if disallowed:
            raise FileExistsError(f"run directory is not empty: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    condition = schedule_row["condition"]
    packets = bundle["conditions"][condition]
    dt = float(bundle["packet_dt_seconds"])
    limiter = SafetyLimiter(
        maximum_speed_mps=float(config["safety"]["maximum_command_speed_mps"]),
        maximum_abs_steering_rad=float(config["safety"]["maximum_sent_abs_steering_rad"]),
        maximum_steering_slew_rad_s=float(
            config["safety"]["common_sent_steering_slew_rad_s"]
        ),
        maximum_acceleration_mps2=float(config["safety"]["common_sent_acceleration_mps2"]),
    )
    context = {
        "run_id": schedule_row["run_id"],
        "block_id": schedule_row["block_id"],
        "condition_code": schedule_row["condition_code"],
    }
    manifest = {
        "schema_version": 1,
        **context,
        "condition": condition,
        "adapter": adapter.name,
        "bundle_id": bundle["bundle_id"],
        "static_hashes": static_hashes,
        "rosbag": rosbag_metadata,
        "started_monotonic_s": None,
        "finished_monotonic_s": None,
        "motion_started": False,
        "completed": False,
        "abort_reason": None,
        "exception": None,
        "command_packets_sent": 0,
        "safe_stop_packets_sent": 0,
        "start_pose_errors": None,
    }
    log_path = run_dir / "records.jsonl"
    with HashChainWriter(log_path) as writer:
        try:
            if not adapter.wait_for_ready(timeout_seconds=5.0):
                raise RuntimeError("adapter_not_ready")
            initial = adapter.latest_telemetry()
            manifest["started_monotonic_s"] = adapter.now()
            errors = start_pose_errors(initial, site["expected_start_pose"])
            manifest["start_pose_errors"] = errors
            validity = config["run_validity"]
            if errors["position_m"] > float(validity["start_position_tolerance_m"]):
                raise RuntimeError("start_pose_out_of_tolerance_before_arm")
            if errors["heading_rad"] > math.radians(
                float(validity["start_heading_tolerance_deg"])
            ):
                raise RuntimeError("start_heading_out_of_tolerance_before_arm")
            initial_health = health_reasons(adapter, adapter.now(), config)
            initial_safety = check_runtime_safety(initial, initial, config)
            if initial_health or initial_safety:
                raise RuntimeError(
                    "pre_motion_safety_failed:"
                    + ",".join(sorted(set(initial_health + initial_safety)))
                )
            writer.write(
                {
                    "record_type": "run_start",
                    **context,
                    "adapter": adapter.name,
                    "condition": condition,
                    "bundle_id": bundle["bundle_id"],
                    "initial_telemetry": initial,
                    "static_hashes": static_hashes,
                }
            )
            write_telemetry(writer, adapter.drain_telemetry(), context)
            planned_start = adapter.now()
            for packet in packets:
                planned_send = planned_start + int(packet["packet_index"]) * dt
                adapter.wait_until(planned_send)
                now = adapter.now()
                telemetry = adapter.latest_telemetry()
                reasons = health_reasons(adapter, now, config)
                reasons.extend(check_runtime_safety(telemetry, initial, config))
                if reasons:
                    raise RuntimeError("runtime_safety_failed:" + ",".join(sorted(set(reasons))))
                sent = limiter.apply(
                    packet["target_steering_rad"], packet["target_speed_mps"], dt
                )
                if sent["speed_mps"] > 0.0:
                    manifest["motion_started"] = True
                adapter.publish(sent["steering_rad"], sent["speed_mps"], dt)
                writer.write(
                    {
                        "record_type": "command",
                        **context,
                        "condition": condition,
                        "packet_index": int(packet["packet_index"]),
                        "phase": packet["phase"],
                        "phase_packet_index": int(packet["phase_packet_index"]),
                        "planned_send_monotonic_s": planned_send,
                        "sent_monotonic_s": now,
                        "lateness_s": max(0.0, now - planned_send),
                        "study_target_steering_rad": float(packet["target_steering_rad"]),
                        "study_target_speed_mps": float(packet["target_speed_mps"]),
                        "sent_steering_rad": sent["steering_rad"],
                        "sent_speed_mps": sent["speed_mps"],
                        "limiter": {
                            key: sent[key]
                            for key in (
                                "target_steering_clipped",
                                "target_speed_clipped",
                                "steering_slew_limited",
                                "acceleration_limited",
                            )
                        },
                    }
                )
                manifest["command_packets_sent"] += 1
                write_telemetry(
                    writer,
                    adapter.drain_telemetry(),
                    {
                        **context,
                        "command_packet_index": int(packet["packet_index"]),
                        "command_phase": packet["phase"],
                        "command_phase_packet_index": int(packet["phase_packet_index"]),
                    },
                )
            manifest["completed"] = True
        except BaseException as exc:
            manifest["abort_reason"] = str(exc)
            manifest["exception"] = traceback.format_exc()
        finally:
            try:
                manifest["safe_stop_packets_sent"] = safe_stop(
                    adapter, writer, config, context
                )
            except Exception as stop_exc:
                manifest["abort_reason"] = (
                    (manifest["abort_reason"] + "; ") if manifest["abort_reason"] else ""
                ) + f"safe_stop_failed:{type(stop_exc).__name__}:{stop_exc}"
                manifest["completed"] = False
            manifest["finished_monotonic_s"] = adapter.now()
            writer.write(
                {
                    "record_type": "run_end",
                    **context,
                    "completed": manifest["completed"],
                    "abort_reason": manifest["abort_reason"],
                    "motion_started": manifest["motion_started"],
                }
            )
            manifest["terminal_record_sha256"] = writer.previous_hash
            manifest["record_count"] = writer.count
    manifest["records_sha256"] = sha256_file(log_path)
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest

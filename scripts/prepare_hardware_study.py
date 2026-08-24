#!/usr/bin/env python3
"""Generate the immutable command bundles and blinded hardware schedule."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.bundles import export_bundles
from hardware_study.design import balance_summary, make_schedule, select_sources
from hardware_study.integrity import sha256_file, verify_paths, write_json
from hardware_study.safety import SafetyLimiter
from utils.kinematic_bicycle_stack import KinematicPlantConfig, simulate_packet_commands
from utils.provenance import collect_provenance


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def limited_commands(packets: list[dict], config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    safety = config["safety"]
    limiter = SafetyLimiter(
        maximum_speed_mps=float(safety["maximum_command_speed_mps"]),
        maximum_abs_steering_rad=float(safety["maximum_sent_abs_steering_rad"]),
        maximum_steering_slew_rad_s=float(safety["common_sent_steering_slew_rad_s"]),
        maximum_acceleration_mps2=float(safety["common_sent_acceleration_mps2"]),
    )
    steering, speed = [], []
    counts = {
        "target_steering_clipped": 0,
        "target_speed_clipped": 0,
        "steering_slew_limited": 0,
        "acceleration_limited": 0,
    }
    dt = float(config["command_bundle"]["packet_dt_seconds"])
    for packet in packets:
        sent = limiter.apply(
            packet["target_steering_rad"], packet["target_speed_mps"], dt
        )
        steering.append(sent["steering_rad"])
        speed.append(sent["speed_mps"])
        for key in counts:
            counts[key] += int(sent[key])
    return np.asarray(steering), np.asarray(speed), counts


def safety_envelope(bundle_entries, output: Path, config: dict) -> list[dict]:
    rows = []
    plant = KinematicPlantConfig(
        wheelbase_m=0.33,
        steering_time_constant_s=0.12,
        speed_time_constant_s=0.20,
        steering_rate_limit_rad_s=3.2,
        acceleration_limit_m_s2=8.0,
        steering_bound_rad=float(config["safety"]["maximum_sent_abs_steering_rad"]),
        min_speed_m_s=0.0,
        max_speed_m_s=float(config["safety"]["maximum_command_speed_mps"]),
    )
    for entry in bundle_entries:
        bundle_path = output / entry["relative_path"]
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        for condition, packets in bundle["conditions"].items():
            steering, speed, limiter_counts = limited_commands(packets, config)
            trajectory = simulate_packet_commands(
                steering,
                speed,
                ticks_per_packet=5,
                dt_seconds=0.01,
                cfg=plant,
                initial_steering_rad=0.0,
            )
            states = trajectory["states"]
            positions = trajectory["positions"]
            rows.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "checkpoint": bundle["checkpoint"],
                    "spawn": bundle["spawn"],
                    "speed_cap_mps": bundle["speed_cap_mps"],
                    "condition": condition,
                    "min_x_m": float(np.min(positions[:, 0])),
                    "max_x_m": float(np.max(positions[:, 0])),
                    "min_y_m": float(np.min(positions[:, 1])),
                    "max_y_m": float(np.max(positions[:, 1])),
                    "maximum_radius_m": float(np.max(np.linalg.norm(positions, axis=1))),
                    "maximum_abs_heading_rad": float(np.max(np.abs(states[:, 2]))),
                    "terminal_x_m": float(positions[-1, 0]),
                    "terminal_y_m": float(positions[-1, 1]),
                    **limiter_counts,
                }
            )
    write_csv(output / "safety_envelope.csv", rows)
    maximum_radius = max(row["maximum_radius_m"] for row in rows)
    maximum_abs_y = max(max(abs(row["min_y_m"]), abs(row["max_y_m"])) for row in rows)
    maximum_abs_heading = max(row["maximum_abs_heading_rad"] for row in rows)
    (output / "SAFETY_ENVELOPE.md").write_text(
        "\n".join(
            [
                "# Conservative pre-hardware path envelope",
                "",
                "This is an engineering screen using the separate slow-actuator kinematic plant after the common hardware safety limiter. It is not physical evidence and must not replace the surveyed geofence or safety supervisor.",
                "",
                f"- Maximum predicted radius from the run start: `{maximum_radius:.3f} m`",
                f"- Maximum predicted absolute lateral displacement: `{maximum_abs_y:.3f} m`",
                f"- Maximum predicted absolute heading: `{maximum_abs_heading:.3f} rad`",
                f"- Frozen runtime geofence radius: `{config['safety']['maximum_geofence_radius_m']} m`",
                "- Required clear course: predicted envelope plus at least 1.0 m physical margin on every side.",
                "",
                "The first physical execution must still occur on stands and then use only the synthetic engineering pilot at walking speed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hardware_study_v1.yaml")
    parser.add_argument(
        "--protocol", default="reproducibility/hardware_validation/STUDY_PROTOCOL_V1.md"
    )
    parser.add_argument(
        "--output-dir", default="reproducibility/hardware_validation/study_v1/prepared"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    output = (ROOT / args.output_dir).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_entries = list(config["sealed_sources"].values())
    source_checks = verify_paths(source_entries, ROOT)
    if not all(check["passed"] for check in source_checks):
        raise RuntimeError(f"sealed source verification failed: {source_checks}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    sources = json.loads(
        (ROOT / config["sealed_sources"]["source_sequences"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    vehicle = yaml.safe_load(
        (ROOT / config["sealed_sources"]["vehicle_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    selected = select_sources(sources, config)
    selected_summary = [
        {
            "checkpoint": row["checkpoint"],
            "spawn": int(row["spawn"]),
            "suppression_fraction": float(row["suppression_fraction"]),
            "eligible_rank": int(row["eligible_rank"]),
            "eligible_count_checkpoint": int(row["eligible_count_checkpoint"]),
            **row["hardware_safety"],
        }
        for row in selected
    ]
    write_json(output / "selected_sources.json", selected_summary)
    bundle_entries, _ = export_bundles(selected, vehicle, config, output)
    schedule, condition_key = make_schedule(selected, config)
    write_json(output / "machine_schedule.json", schedule)
    write_json(
        output / "condition_key.json",
        {
            "warning": "Keep this file from the operator until the main study is complete.",
            "code_to_condition": condition_key,
        },
    )
    schedule_fields = list(schedule[0].keys())
    write_csv(output / "machine_schedule.csv", schedule, schedule_fields)
    operator_fields = [
        "run_index",
        "run_id",
        "block_id",
        "run_position",
        "checkpoint",
        "spawn",
        "speed_cap_mps",
        "bundle_id",
        "condition_code",
    ]
    write_csv(output / "operator_schedule.csv", schedule, operator_fields)
    balance = balance_summary(schedule)
    write_json(output / "randomization_balance.json", balance)
    envelope = safety_envelope(bundle_entries, output, config)
    provenance = collect_provenance(ROOT)
    write_json(output / "preparation_provenance.json", provenance)
    prepared_files = [
        output / "selected_sources.json",
        output / "bundle_manifest.json",
        output / "machine_schedule.json",
        output / "machine_schedule.csv",
        output / "operator_schedule.csv",
        output / "condition_key.json",
        output / "randomization_balance.json",
        output / "safety_envelope.csv",
        output / "SAFETY_ENVELOPE.md",
        output / "preparation_provenance.json",
    ] + [output / entry["relative_path"] for entry in bundle_entries]
    manifest = {
        "study_id": config["study_id"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "protocol_path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(protocol_path),
        "source_checks": source_checks,
        "selected_sources": len(selected),
        "bundle_count": len(bundle_entries),
        "schedule_runs": len(schedule),
        "schedule_blocks": len({row["block_id"] for row in schedule}),
        "randomization_balance": balance,
        "safety_envelope_rows": len(envelope),
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in prepared_files
        ],
    }
    write_json(output / "PREPARED_MANIFEST.json", manifest)
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                "# Hardware study preparation",
                "",
                "**Status: PREPARED — NO PHYSICAL OUTCOME DATA**",
                "",
                f"- Selected sources: `{len(selected)}`",
                f"- Immutable command bundles: `{len(bundle_entries)}`",
                f"- Randomized blocks: `{manifest['schedule_blocks']}`",
                f"- Scheduled main runs: `{len(schedule)}`",
                f"- Position-count range: `{balance['position_count_range']}`",
                f"- Directed carryover-count range: `{balance['transition_count_range']}`",
                f"- Maximum conservative predicted radius: `{max(row['maximum_radius_m'] for row in envelope):.3f} m`",
                "",
                "All bundle matching checks pass. The two clean conditions are byte-identical within each bundle; gate and timing placebo preserve exact accepted target and increment sequences; all five conditions have exact speed and packet counts.",
                "",
                "Physical execution remains blocked until a site file is captured and frozen, ROS topic and safety-bridge preflights pass on stands, and both the operator and safety supervisor sign the checklist.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

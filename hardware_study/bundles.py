"""Build immutable physical command bundles from sealed simulator streams."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .design import condition_sequences
from .integrity import sha256_file, write_json


def physical_speed(clean_actions: np.ndarray, vehicle: dict) -> np.ndarray:
    normalized = np.clip(clean_actions[:, 1], -1.0, 1.0)
    return float(vehicle["v_min"]) + 0.5 * (normalized + 1.0) * (
        float(vehicle["v_max"]) - float(vehicle["v_min"])
    )


def _packets(
    normalized_steering: np.ndarray,
    main_speed: np.ndarray,
    speed_cap: float,
    delta_max: float,
    config: dict,
) -> list[dict]:
    bundle_cfg = config["command_bundle"]
    preamble_count = int(bundle_cfg["preamble_packets"])
    postamble_count = int(bundle_cfg["postamble_packets"])
    preamble_speed = np.linspace(0.0, float(speed_cap), preamble_count)
    postamble_speed = np.linspace(float(main_speed[-1]), 0.0, postamble_count)
    steering = np.concatenate(
        [
            np.zeros(preamble_count),
            np.asarray(normalized_steering, dtype=float) * float(delta_max),
            np.zeros(postamble_count),
        ]
    )
    speed = np.concatenate([preamble_speed, main_speed, postamble_speed])
    phases = (
        ["preamble"] * preamble_count
        + ["main"] * normalized_steering.size
        + ["postamble"] * postamble_count
    )
    phase_indices = []
    counters = {"preamble": 0, "main": 0, "postamble": 0}
    for phase in phases:
        phase_indices.append(counters[phase])
        counters[phase] += 1
    return [
        {
            "packet_index": index,
            "phase": phase,
            "phase_packet_index": phase_index,
            "target_steering_rad": float(steering[index]),
            "target_speed_mps": float(speed[index]),
        }
        for index, (phase, phase_index) in enumerate(zip(phases, phase_indices))
    ]


def build_bundle(source: dict, speed_cap: float, vehicle: dict, config: dict) -> dict:
    sequences = condition_sequences(source)
    clean_actions = np.asarray(source["clean_actions"], dtype=float)
    source_speed = physical_speed(clean_actions, vehicle)
    main_speed = np.minimum(source_speed, float(speed_cap))
    expected_main = int(config["command_bundle"]["main_packets"])
    if clean_actions.shape != (expected_main, 2):
        raise ValueError(f"source action shape {clean_actions.shape} is not {(expected_main, 2)}")
    delta_max = float(vehicle["delta_max"])
    conditions = {
        condition: _packets(values, main_speed, speed_cap, delta_max, config)
        for condition, values in sequences.items()
    }
    gate_events = np.asarray(source["innovation_events"], dtype=bool)
    gate = sequences["innovation_gate"]
    timing = sequences["timing_placebo"]
    timing_event_indices = np.flatnonzero(np.r_[True, np.diff(timing) != 0.0])
    gate_event_indices = np.flatnonzero(gate_events)
    accepted_gate = gate[gate_event_indices]
    accepted_timing = timing[timing_event_indices]
    exact_targets = bool(np.array_equal(accepted_gate, accepted_timing))
    exact_increments = bool(
        np.array_equal(np.diff(accepted_gate), np.diff(accepted_timing))
    )
    condition_lengths = {key: len(value) for key, value in conditions.items()}
    all_targets = np.concatenate(
        [
            np.asarray([row["target_steering_rad"] for row in packets])
            for packets in conditions.values()
        ]
    )
    maximum_abs = float(np.max(np.abs(all_targets)))
    speed_token = str(float(speed_cap)).replace(".", "p")
    bundle_id = f"{source['checkpoint']}_spawn{int(source['spawn'])}_speed{speed_token}"
    return {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "checkpoint": source["checkpoint"],
        "spawn": int(source["spawn"]),
        "source_reset_seed": int(source["reset_seed"]),
        "source_jitter_seed": int(source["jitter_seed"]),
        "source_suppression_fraction": float(source["suppression_fraction"]),
        "speed_cap_mps": float(speed_cap),
        "packet_dt_seconds": float(config["command_bundle"]["packet_dt_seconds"]),
        "conditions": conditions,
        "matching_checks": {
            "clean_duplicates_exact": conditions["clean_a"] == conditions["clean_b"],
            "gate_timing_update_count_exact": int(accepted_gate.size)
            == int(accepted_timing.size),
            "gate_timing_accepted_targets_exact": exact_targets,
            "gate_timing_accepted_increments_exact": exact_increments,
            "condition_packet_counts_exact": len(set(condition_lengths.values())) == 1,
            "speed_commands_exact_across_conditions": all(
                [row["target_speed_mps"] for row in packets]
                == [row["target_speed_mps"] for row in conditions["clean_a"]]
                for packets in conditions.values()
            ),
        },
        "diagnostics": {
            "condition_packet_counts": condition_lengths,
            "gate_update_count": int(accepted_gate.size),
            "timing_update_count": int(accepted_timing.size),
            "maximum_abs_target_steering_rad": maximum_abs,
            "maximum_main_speed_mps": float(np.max(main_speed)),
            "minimum_main_speed_mps": float(np.min(main_speed)),
        },
    }


def export_bundles(
    selected: list[dict],
    vehicle: dict,
    config: dict,
    output_dir: Path,
) -> tuple[list[dict], dict]:
    bundle_dir = Path(output_dir) / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for speed in config["command_bundle"]["speed_caps_mps"]:
        for source in selected:
            bundle = build_bundle(source, float(speed), vehicle, config)
            if not all(bundle["matching_checks"].values()):
                raise RuntimeError(
                    f"bundle matching failed for {bundle['bundle_id']}: {bundle['matching_checks']}"
                )
            maximum = float(config["safety"]["maximum_study_target_abs_steering_rad"])
            if bundle["diagnostics"]["maximum_abs_target_steering_rad"] > maximum + 1e-12:
                raise RuntimeError(f"bundle exceeds frozen steering target bound: {bundle['bundle_id']}")
            path = bundle_dir / f"{bundle['bundle_id']}.json"
            write_json(path, bundle)
            manifest_rows.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "checkpoint": bundle["checkpoint"],
                    "spawn": bundle["spawn"],
                    "speed_cap_mps": bundle["speed_cap_mps"],
                    "suppression_fraction": bundle["source_suppression_fraction"],
                    "maximum_abs_target_steering_rad": bundle["diagnostics"][
                        "maximum_abs_target_steering_rad"
                    ],
                }
            )
    manifest = {
        "schema_version": 1,
        "bundle_count": len(manifest_rows),
        "bundles": manifest_rows,
    }
    write_json(Path(output_dir) / "bundle_manifest.json", manifest)
    return manifest_rows, manifest


def load_bundle_from_manifest(prepared_dir: Path, bundle_id: str) -> tuple[dict, dict]:
    prepared_dir = Path(prepared_dir)
    manifest = json.loads((prepared_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["bundles"] if row["bundle_id"] == bundle_id]
    if len(matches) != 1:
        raise KeyError(f"bundle {bundle_id!r} not uniquely present in manifest")
    entry = matches[0]
    path = prepared_dir / entry["relative_path"]
    if sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"bundle hash mismatch: {bundle_id}")
    return json.loads(path.read_text(encoding="utf-8")), entry

#!/usr/bin/env python3
"""Build an opaque-code ROS 1 operator view without semantic condition names."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.integrity import sha256_file


DEFAULT_SOURCE = "reproducibility/hardware_validation/study_v1/prepared"
DEFAULT_OUTPUT = "reproducibility/hardware_validation/study_v1/operator_prepared"


def write_json(path: Path, payload) -> None:
    """Write deterministic LF-only JSON on both Linux and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    )


def code_to_condition(schedule: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in schedule:
        code = str(row["condition_code"])
        condition = str(row["condition"])
        previous = mapping.setdefault(code, condition)
        if previous != condition:
            raise RuntimeError(f"condition code {code!r} is not globally stable")
    if len(mapping) != 5 or len(set(mapping.values())) != 5:
        raise RuntimeError("expected a one-to-one five-condition code mapping")
    return mapping


def write_opaque_safety_envelope(
    source_path: Path, destination_path: Path, mapping: dict[str, str]
) -> None:
    """Preserve the frozen envelope while replacing semantic labels with codes."""
    semantic_to_code = {semantic: code for code, semantic in mapping.items()}
    with source_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "condition" not in reader.fieldnames:
            raise RuntimeError("safety envelope must contain a condition column")
        rows = list(reader)
    for row in rows:
        semantic = str(row["condition"])
        if semantic not in semantic_to_code:
            raise RuntimeError(
                f"unrecognized semantic condition in safety envelope: {semantic!r}"
            )
        row["condition"] = semantic_to_code[semantic]
    with destination_path.open("x", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_operator_package(source: Path, output: Path) -> dict:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"operator package refuses overwrite: {output}")
    output.mkdir(parents=True)
    (output / "bundles").mkdir()

    source_manifest_path = source / "PREPARED_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    schedule = json.loads(
        (source / "machine_schedule.json").read_text(encoding="utf-8")
    )
    mapping = code_to_condition(schedule)

    opaque_schedule = []
    for row in schedule:
        opaque = {
            key: value
            for key, value in row.items()
            if key not in {"condition", "template_index"}
        }
        opaque["condition"] = str(row["condition_code"])
        opaque_schedule.append(opaque)
    write_json(output / "machine_schedule.json", opaque_schedule)

    source_bundle_manifest = json.loads(
        (source / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    operator_bundle_rows = []
    for entry in source_bundle_manifest["bundles"]:
        source_bundle_path = source / entry["relative_path"]
        if sha256_file(source_bundle_path) != entry["sha256"]:
            raise RuntimeError(f"source bundle hash mismatch: {entry['bundle_id']}")
        source_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))
        opaque_bundle = {
            "schema_version": 1,
            "bundle_id": source_bundle["bundle_id"],
            "checkpoint": source_bundle["checkpoint"],
            "spawn": source_bundle["spawn"],
            "speed_cap_mps": source_bundle["speed_cap_mps"],
            "packet_dt_seconds": source_bundle["packet_dt_seconds"],
            "conditions": {
                code: source_bundle["conditions"][semantic]
                for code, semantic in sorted(mapping.items())
            },
        }
        relative_path = Path("bundles") / f"{entry['bundle_id']}.json"
        destination = output / relative_path
        write_json(destination, opaque_bundle)
        operator_bundle_rows.append(
            {
                "bundle_id": entry["bundle_id"],
                "relative_path": relative_path.as_posix(),
                "sha256": sha256_file(destination),
                "checkpoint": entry["checkpoint"],
                "spawn": entry["spawn"],
                "speed_cap_mps": entry["speed_cap_mps"],
                "maximum_abs_target_steering_rad": entry[
                    "maximum_abs_target_steering_rad"
                ],
            }
        )
    operator_bundle_manifest = {
        "schema_version": 1,
        "bundle_count": len(operator_bundle_rows),
        "bundles": operator_bundle_rows,
    }
    write_json(output / "bundle_manifest.json", operator_bundle_manifest)

    operator_schedule_text = (source / "operator_schedule.csv").read_text(
        encoding="utf-8"
    )
    (output / "operator_schedule.csv").write_bytes(
        operator_schedule_text.encode("utf-8")
    )
    write_opaque_safety_envelope(
        source / "safety_envelope.csv", output / "safety_envelope.csv", mapping
    )
    readme = (
        "# Frank opaque operator inputs\n\n"
        "Use only `operator_schedule.csv` and the ROS 1 runner. Conditions are "
        "represented only by opaque codes in this directory. Do not inspect or "
        "compare packet bundles during collection. Semantic unblinding is held "
        "by the study lead until all outcomes are locked.\n"
    )
    (output / "README.md").write_bytes(readme.encode("utf-8"))

    sealed = [
        entry
        for entry in source_manifest["files"]
        if Path(entry["path"]).as_posix() == "condition_key.json"
    ]
    if len(sealed) != 1:
        raise RuntimeError("source manifest must declare exactly one condition key")
    visible_paths = [
        Path("machine_schedule.json"),
        Path("operator_schedule.csv"),
        Path("safety_envelope.csv"),
        Path("bundle_manifest.json"),
        Path("README.md"),
        *[Path(row["relative_path"]) for row in operator_bundle_rows],
    ]
    files = [
        {"path": path.as_posix(), "sha256": sha256_file(output / path)}
        for path in visible_paths
    ]
    files.append(dict(sealed[0]))
    manifest = {
        "schema_version": 1,
        "package_role": "ROS1_BLINDED_OPERATOR_INPUTS",
        "study_id": source_manifest["study_id"],
        "config_sha256": source_manifest["config_sha256"],
        "source_prepared_manifest_sha256": sha256_file(source_manifest_path),
        "bundle_count": len(operator_bundle_rows),
        "schedule_runs": len(opaque_schedule),
        "semantic_conditions_present": False,
        "files": files,
    }
    write_json(output / "PREPARED_MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_operator_package(ROOT / args.source, ROOT / args.output)
    print(
        json.dumps(
            {
                "output": str((ROOT / args.output).resolve()),
                "bundles": manifest["bundle_count"],
                "runs": manifest["schedule_runs"],
                "semantic_conditions_present": manifest[
                    "semantic_conditions_present"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

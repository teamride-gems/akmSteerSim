#!/usr/bin/env python3
"""Create the immutable code/data freeze record before physical outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.integrity import sha256_file, verify_paths, write_json


FROZEN_PATHS = [
    "configs/hardware_study_v1.yaml",
    "configs/hardware_site_template.yaml",
    "configs/hardware_site_mock.yaml",
    "configs/vehicle.yaml",
    "reproducibility/hardware_validation/STUDY_PROTOCOL_V1.md",
    "reproducibility/hardware_validation/study_v1/prepared/PREPARED_MANIFEST.json",
    "reproducibility/innovation_gate_second_stack/run_v1/source_sequences.json",
    "reproducibility/innovation_gate_second_stack/placebo_repair_v1_corrected/result.json",
    "hardware_study/__init__.py",
    "hardware_study/adapters.py",
    "hardware_study/analysis.py",
    "hardware_study/bundles.py",
    "hardware_study/design.py",
    "hardware_study/execution.py",
    "hardware_study/integrity.py",
    "hardware_study/pilot.py",
    "hardware_study/safety.py",
    "hardware_study/validation.py",
    "scripts/analyze_hardware_study.py",
    "scripts/capture_hardware_site.py",
    "scripts/hardware_preflight.py",
    "scripts/prepare_hardware_study.py",
    "scripts/ros2_hardware_safety_bridge.py",
    "scripts/run_hardware_engineering_pilot.py",
    "scripts/run_hardware_study.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="reproducibility/hardware_validation/study_v1/FREEZE.json"
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"freeze already exists and will not be overwritten: {output}"
        )
    prepared_path = ROOT / "reproducibility/hardware_validation/study_v1/prepared/PREPARED_MANIFEST.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    prepared_root = prepared_path.parent
    prepared_checks = verify_paths(prepared["files"], prepared_root)
    if not all(check["passed"] for check in prepared_checks):
        raise RuntimeError("prepared files changed; refusing to freeze")
    missing = [path for path in FROZEN_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen inputs are missing: {missing}")
    record = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_PHYSICAL_OUTCOMES",
        "study_id": prepared["study_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_outcomes_observed_at_freeze": False,
        "amendment_rule": "Never overwrite. Any necessary post-freeze change requires a numbered amendment that preserves this record and is disclosed before inspecting further outcomes.",
        "files": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in FROZEN_PATHS
        ],
    }
    write_json(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

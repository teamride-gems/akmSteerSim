#!/usr/bin/env python3
"""Apply the frozen analysis and correctly classify Frank ROS 1 outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.analysis import analyze_study, write_analysis
from hardware_study.ros1_runtime import verify_ros1_amendment


def classify_ros1_evidence(result: dict) -> dict:
    """Correct the frozen ROS 2-only evidence label for the amended adapter."""
    adapter_names = set(result.get("adapter_names", []))
    if adapter_names == {"ros1_ackermann_noetic"}:
        result["evidence_class"] = "PHYSICAL_HARDWARE"
    elif "ros1_ackermann_noetic" in adapter_names:
        result["evidence_class"] = "INVALID_MIXED_ADAPTERS"
        result.setdefault("invalid_reasons", []).append("mixed_adapter_evidence")
        result["verdict"] = "INVALID"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="hardware_runs/study_v1")
    parser.add_argument(
        "--prepared-dir",
        default="reproducibility/hardware_validation/study_v1/prepared",
        help="study-lead semantic prepared set; use only after outcome lock",
    )
    parser.add_argument("--config", default="configs/hardware_study_v1.yaml")
    parser.add_argument("--output", default="hardware_runs/study_v1_analysis")
    args = parser.parse_args()

    verify_ros1_amendment(ROOT)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    result = classify_ros1_evidence(
        analyze_study(ROOT / args.runs, ROOT / args.prepared_dir, config)
    )
    write_analysis(ROOT / args.output, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "evidence_class": result["evidence_class"],
                "output": str((ROOT / args.output).resolve()),
            },
            indent=2,
        )
    )
    return 0 if result["verdict"] != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

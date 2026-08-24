#!/usr/bin/env python3
"""Apply the frozen physical-study analysis without interactive choices."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="hardware_runs/study_v1")
    parser.add_argument(
        "--prepared-dir",
        default="reproducibility/hardware_validation/study_v1/prepared",
    )
    parser.add_argument("--config", default="configs/hardware_study_v1.yaml")
    parser.add_argument("--output", default="hardware_runs/study_v1_analysis")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    result = analyze_study(
        ROOT / args.runs,
        ROOT / args.prepared_dir,
        config,
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

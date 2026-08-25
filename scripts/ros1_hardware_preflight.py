#!/usr/bin/env python3
"""Run the frozen live preflight through the current ROS 1 amendment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.integrity import sha256_file, write_json
from hardware_study.ros1_adapter import Ros1AckermannAdapter
from hardware_study.ros1_runtime import (
    AMENDMENT_ID,
    AMENDMENT_PATH,
    validate_ros1_site,
    verify_operator_paths,
    verify_ros1_amendment,
)
from scripts import hardware_preflight as frozen_preflight
from scripts.run_hardware_study_ros1 import translate_arguments


def argument_value(arguments: list[str], name: str) -> str:
    if name not in arguments:
        raise ValueError(f"ROS 1 preflight requires an explicit {name} path")
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"missing value for {name}")
    return arguments[index + 1]


def make_ros1_adapter(name: str, site: dict):
    if name != "ros2":
        raise ValueError("ROS 1 preflight cannot construct a mock adapter")
    validate_ros1_site(site)
    return Ros1AckermannAdapter(site)


def main() -> int:
    amendment_path = ROOT / AMENDMENT_PATH
    verify_ros1_amendment(ROOT, amendment_path)
    original_arguments = list(sys.argv[1:])
    output_path = (ROOT / argument_value(original_arguments, "--output")).resolve()
    frozen_preflight.make_adapter = make_ros1_adapter
    frozen_preflight.verify_paths = verify_operator_paths
    sys.argv = [str(Path(__file__)), *translate_arguments(original_arguments)]
    result_code = frozen_preflight.main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result.update(
        {
            "adapter": "ros1",
            "adapter_name": "ros1_ackermann_noetic",
            "amendment_id": AMENDMENT_ID,
            "amendment_sha256": sha256_file(amendment_path),
            "preflight_wrapper_sha256": sha256_file(Path(__file__)),
        }
    )
    write_json(output_path, result)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())

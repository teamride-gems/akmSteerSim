#!/usr/bin/env python3
"""Capture the stationary site pose using the native ROS 1 adapter."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.ros1_adapter import Ros1AckermannAdapter
from hardware_study.ros1_runtime import validate_ros1_site, verify_ros1_amendment
from scripts import capture_hardware_site as frozen_capture


class ValidatedRos1Adapter(Ros1AckermannAdapter):
    def __init__(self, site: dict):
        validate_ros1_site(site)
        super().__init__(site, authorize_motion=False)


def argument_value(arguments: list[str], name: str, default: str) -> str:
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError(f"missing value for {name}")
    return arguments[index + 1]


def restore_template_course_fields(template_path: Path, output_path: Path) -> None:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    captured = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    captured["course"] = {
        **dict(template.get("course", {})),
        **dict(captured.get("course", {})),
    }
    output_path.write_text(
        yaml.safe_dump(captured, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    verify_ros1_amendment(ROOT)
    arguments = list(sys.argv[1:])
    if "--template" not in arguments:
        arguments.extend(["--template", "local_hardware_site_ros1_draft.yaml"])
    if "--output" not in arguments:
        arguments.extend(["--output", "local_hardware_site_ros1.yaml"])
    template_path = (ROOT / argument_value(
        arguments, "--template", "local_hardware_site_ros1_draft.yaml"
    )).resolve()
    output_path = (ROOT / argument_value(
        arguments, "--output", "local_hardware_site_ros1.yaml"
    )).resolve()
    frozen_capture.Ros2AckermannAdapter = ValidatedRos1Adapter
    sys.argv = [str(Path(__file__)), *arguments]
    result = frozen_capture.main()
    restore_template_course_fields(template_path, output_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

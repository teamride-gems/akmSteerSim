#!/usr/bin/env python3
"""Capture the stationary site pose using the native ROS 1 adapter."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.ros1_adapter import Ros1AckermannAdapter
from hardware_study.ros1_runtime import validate_ros1_site, verify_ros1_amendment
from scripts import capture_hardware_site as frozen_capture


class ValidatedRos1Adapter(Ros1AckermannAdapter):
    def __init__(self, site: dict):
        validate_ros1_site(site)
        super().__init__(site)


def main() -> int:
    verify_ros1_amendment(ROOT)
    arguments = list(sys.argv[1:])
    if "--template" not in arguments:
        arguments.extend(["--template", "local_hardware_site_ros1_draft.yaml"])
    if "--output" not in arguments:
        arguments.extend(["--output", "local_hardware_site_ros1.yaml"])
    frozen_capture.Ros2AckermannAdapter = ValidatedRos1Adapter
    sys.argv = [str(Path(__file__)), *arguments]
    return frozen_capture.main()


if __name__ == "__main__":
    raise SystemExit(main())


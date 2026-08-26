#!/usr/bin/env python3
"""Run an engineering-only qualification pilot under the current ROS 1 amendment."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.ros1_adapter import Ros1AckermannAdapter
from hardware_study.ros1_runtime import (
    archive_amendment_inputs,
    start_ros1_bag,
    stop_ros1_bag,
    validate_ros1_site,
    verify_operator_prepared,
    verify_ros1_amendment,
    verify_ros1_preflight,
)
from scripts import run_hardware_engineering_pilot as frozen_pilot
from scripts.run_hardware_study_ros1 import translate_arguments


def make_ros1_adapter(name: str, site: dict):
    if name != "ros2":
        raise ValueError("amended pilot cannot construct a mock adapter")
    validate_ros1_site(site)
    return Ros1AckermannAdapter(site, authorize_motion=True)


def main() -> int:
    verify_ros1_amendment(ROOT)
    original_execute = frozen_pilot.execute_run
    original_sha256 = frozen_pilot.sha256_file

    def execute_with_amendment(*args, **kwargs):
        run_dir = Path(args[5] if len(args) > 5 else kwargs["run_dir"])
        archive_amendment_inputs(ROOT, run_dir)
        return original_execute(*args, **kwargs)

    def amended_runner_hash(path):
        if Path(path).resolve() == Path(frozen_pilot.__file__).resolve():
            return original_sha256(Path(__file__))
        return original_sha256(path)

    frozen_pilot.make_adapter = make_ros1_adapter
    frozen_pilot.start_rosbag = start_ros1_bag
    frozen_pilot.stop_rosbag = stop_ros1_bag
    frozen_pilot.verify_preflight = verify_ros1_preflight
    frozen_pilot.verify_prepared = verify_operator_prepared
    frozen_pilot.execute_run = execute_with_amendment
    frozen_pilot.sha256_file = amended_runner_hash
    sys.argv = [str(Path(__file__)), *translate_arguments(sys.argv[1:])]
    return frozen_pilot.main()


if __name__ == "__main__":
    raise SystemExit(main())

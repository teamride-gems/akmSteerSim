#!/usr/bin/env python3
"""Run one frozen study row through amendment 001's native ROS 1 adapter."""

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
from scripts import run_hardware_study as frozen_runner


def translate_arguments(arguments: list[str]) -> list[str]:
    translated = list(arguments)
    if "--adapter" in translated:
        index = translated.index("--adapter")
        if index + 1 >= len(translated) or translated[index + 1] != "ros1":
            raise ValueError("the ROS 1 runner accepts only '--adapter ros1'")
        translated[index + 1] = "ros2"
    else:
        translated.extend(["--adapter", "ros2"])
    if "--site" not in translated:
        translated.extend(["--site", "local_hardware_site_ros1.yaml"])
    return translated


def make_ros1_adapter(name: str, site: dict):
    if name != "ros2":
        raise ValueError("amended real runner cannot construct a mock adapter")
    validate_ros1_site(site)
    return Ros1AckermannAdapter(site)


def main() -> int:
    amendment = verify_ros1_amendment(ROOT)
    original_archive = frozen_runner.archive_inputs
    original_sha256 = frozen_runner.sha256_file

    def archive_with_amendment(*args, **kwargs):
        original_archive(*args, **kwargs)
        run_dir = Path(args[0] if args else kwargs["run_dir"])
        archive_amendment_inputs(ROOT, run_dir)

    def amended_runner_hash(path):
        if Path(path).resolve() == Path(frozen_runner.__file__).resolve():
            return original_sha256(Path(__file__))
        return original_sha256(path)

    frozen_runner.make_adapter = make_ros1_adapter
    frozen_runner.start_rosbag = start_ros1_bag
    frozen_runner.stop_rosbag = stop_ros1_bag
    frozen_runner.verify_preflight = verify_ros1_preflight
    frozen_runner.verify_prepared = verify_operator_prepared
    frozen_runner.archive_inputs = archive_with_amendment
    frozen_runner.sha256_file = amended_runner_hash
    frozen_runner.DEFAULT_FREEZE = amendment["base_freeze"]["path"]
    sys.argv = [str(Path(__file__)), *translate_arguments(sys.argv[1:])]
    return frozen_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())


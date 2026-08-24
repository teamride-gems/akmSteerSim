#!/usr/bin/env python3
"""Capture a stationary surveyed start pose into a local hardware site file."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.adapters import Ros2AckermannAdapter


def circular_mean(values: list[float]) -> float:
    return math.atan2(
        sum(math.sin(value) for value in values),
        sum(math.cos(value) for value in values),
    )


def main() -> int:  # pragma: no cover - requires robot runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="configs/hardware_site_template.yaml")
    parser.add_argument("--output", default="local_hardware_site.yaml")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--course-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--safety-supervisor", required=True)
    parser.add_argument("--clear-radius-m", required=True, type=float)
    parser.add_argument("--localization-system", required=True)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--ros-domain-id", type=int)
    args = parser.parse_args()
    if args.clear_radius_m <= 0.0 or args.sample_seconds < 1.0:
        raise ValueError("clear radius must be positive and sampling must last at least 1 s")
    template_path = (ROOT / args.template).resolve()
    output_path = (ROOT / args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"site capture refuses to overwrite: {output_path}")
    site = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if args.ros_domain_id is not None:
        site["ros_domain_id"] = args.ros_domain_id
    os.environ["ROS_DOMAIN_ID"] = str(site["ros_domain_id"])
    adapter = Ros2AckermannAdapter(site)
    samples = []
    try:
        if not adapter.wait_for_ready(timeout_seconds=10.0):
            raise RuntimeError(
                "odometry/deadman/e-stop topics were not all observed within 10 seconds"
            )
        deadline = time.monotonic() + args.sample_seconds
        while time.monotonic() < deadline:
            telemetry = adapter.latest_telemetry()
            if abs(float(telemetry["speed_mps"])) > 0.03:
                raise RuntimeError("robot moved during start-pose capture")
            samples.append(telemetry)
            time.sleep(0.02)
    finally:
        adapter.close()
    if len(samples) < 40:
        raise RuntimeError("too few localization samples for site capture")
    x = statistics.median(float(row["x_m"]) for row in samples)
    y = statistics.median(float(row["y_m"]) for row in samples)
    yaw = circular_mean([float(row["yaw_rad"]) for row in samples])
    maximum_deviation = max(
        math.hypot(float(row["x_m"]) - x, float(row["y_m"]) - y)
        for row in samples
    )
    if maximum_deviation > 0.02:
        raise RuntimeError(
            f"localization was unstable during capture ({maximum_deviation:.3f} m)"
        )
    site.update(
        {
            "site_id": args.site_id,
            "robot_id": args.robot_id,
            "course_id": args.course_id,
            "operator": args.operator,
            "safety_supervisor": args.safety_supervisor,
        }
    )
    site["expected_start_pose"] = {"x_m": x, "y_m": y, "yaw_rad": yaw}
    site["course"] = {
        "surveyed_clear_radius_m": float(args.clear_radius_m),
        "localization_system": args.localization_system,
    }
    output_path.write_text(
        yaml.safe_dump(site, sort_keys=False), encoding="utf-8"
    )
    print(f"Captured {len(samples)} stationary samples into {output_path}")
    print(f"Start pose: x={x:.4f} m, y={y:.4f} m, yaw={yaw:.5f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

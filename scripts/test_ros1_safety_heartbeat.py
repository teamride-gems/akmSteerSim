#!/usr/bin/env python3
"""Publish a short, stands-only authorization heartbeat for bridge testing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_study.ros1_runtime import validate_ros1_site, verify_ros1_amendment


CONFIRMATION = "FRANK ON STANDS - TEST HEARTBEAT"


def validate_duration(value: float) -> float:
    duration = float(value)
    if not math.isfinite(duration) or duration < 2.0 or duration > 5.0:
        raise ValueError("heartbeat test duration must be between 2 and 5 seconds")
    return duration


def main() -> int:  # pragma: no cover - requires Frank's ROS 1 environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_hardware_site_ros1_draft.yaml")
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--stands-confirm", required=True)
    args = parser.parse_args()
    duration = validate_duration(args.duration_seconds)
    if args.stands_confirm != CONFIRMATION:
        raise RuntimeError(f"stands confirmation must be exactly: {CONFIRMATION}")

    verify_ros1_amendment(ROOT)
    site = yaml.safe_load((ROOT / args.site).read_text(encoding="utf-8"))
    validate_ros1_site(site)
    try:
        import rospy
        from std_msgs.msg import Bool
    except ImportError as exc:
        raise RuntimeError("heartbeat test requires rospy and std_msgs") from exc

    rospy.init_node("ride_safety_heartbeat_test", anonymous=False)
    publisher = rospy.Publisher(site["topics"]["run_active"], Bool, queue_size=10)
    rate = rospy.Rate(20.0)
    deadline = time.monotonic() + duration
    try:
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            publisher.publish(Bool(data=True))
            rate.sleep()
    finally:
        for _ in range(10):
            publisher.publish(Bool(data=False))
            rate.sleep()
        publisher.unregister()
    print("Heartbeat test completed and autonomous authorization was withdrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

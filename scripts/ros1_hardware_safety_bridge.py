#!/usr/bin/env python3
"""Fail-closed deadman and latched software e-stop bridge for ROS 1 Noetic."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


class SafetyState:
    """Thread-safe safety state kept independent of ROS for deterministic tests."""

    def __init__(
        self,
        deadman_index: int,
        estop_index: int,
        stale_seconds: float,
        deadman_clearance_seconds: float = 0.0,
    ):
        if deadman_index == estop_index or min(deadman_index, estop_index) < 0:
            raise ValueError("deadman and e-stop must be distinct nonnegative buttons")
        self.deadman_index = int(deadman_index)
        self.estop_index = int(estop_index)
        self.stale_seconds = float(stale_seconds)
        self.deadman_clearance_seconds = float(deadman_clearance_seconds)
        self.last_joy_time = None
        self.deadman_requested_since = None
        self.buttons = []
        self.estop_latched = True
        self.lock = threading.Lock()

    def _button(self, index: int) -> bool:
        return index < len(self.buttons) and bool(self.buttons[index])

    def _deadman_held_alone(self) -> bool:
        return self._button(self.deadman_index) and not any(
            bool(value)
            for index, value in enumerate(self.buttons)
            if index != self.deadman_index
        )

    def update(self, buttons: list[int], now: float | None = None) -> None:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            self.last_joy_time = current
            self.buttons = list(buttons)
            if self._button(self.estop_index):
                self.estop_latched = True
            if self._deadman_held_alone() and not self.estop_latched:
                if self.deadman_requested_since is None:
                    self.deadman_requested_since = current
            else:
                self.deadman_requested_since = None

    def snapshot(self, now: float | None = None) -> tuple[bool, bool]:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            fresh = self.last_joy_time is not None and (
                current - self.last_joy_time <= self.stale_seconds
            )
            clearance_elapsed = (
                self.deadman_requested_since is not None
                and current - self.deadman_requested_since
                >= self.deadman_clearance_seconds
            )
            deadman = (
                fresh
                and self._deadman_held_alone()
                and not self.estop_latched
                and clearance_elapsed
            )
            return bool(deadman), bool(self.estop_latched)

    def stop_override_required(self, now: float | None = None) -> bool:
        """Return true unless fresh raw deadman input authorizes mux release."""
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            fresh = self.last_joy_time is not None and (
                current - self.last_joy_time <= self.stale_seconds
            )
            return not (
                fresh
                and self._deadman_held_alone()
                and not self.estop_latched
            )

    def reset(self, now: float | None = None) -> tuple[bool, str]:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            fresh = self.last_joy_time is not None and (
                current - self.last_joy_time <= self.stale_seconds
            )
            safe = fresh and not any(bool(value) for value in self.buttons)
            if safe:
                self.estop_latched = False
                self.deadman_requested_since = None
                return True, "software e-stop reset; deadman remains released"
            return False, "reset denied: require fresh joystick with all buttons released"


def main() -> int:  # pragma: no cover - requires ROS 1 robot runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_hardware_site_ros1.yaml")
    args = parser.parse_args()
    site = yaml.safe_load((ROOT / args.site).read_text(encoding="utf-8"))
    bridge = site["safety_bridge"]
    state = SafetyState(
        int(bridge["deadman_button_index"]),
        int(bridge["estop_button_index"]),
        float(bridge["joy_stale_seconds"]),
        float(bridge.get("deadman_clearance_seconds", 1.0)),
    )
    try:
        import rospy
        from ackermann_msgs.msg import AckermannDriveStamped
        from sensor_msgs.msg import Joy
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger, TriggerResponse
    except ImportError as exc:
        raise RuntimeError(
            "ROS 1 safety bridge requires rospy, ackermann_msgs, sensor_msgs, "
            "std_msgs, and std_srvs"
        ) from exc

    rospy.init_node("ride_hardware_safety_bridge", anonymous=False)
    deadman_pub = rospy.Publisher(site["topics"]["deadman"], Bool, queue_size=10)
    estop_pub = rospy.Publisher(site["topics"]["estop"], Bool, queue_size=10)
    stop_pub = rospy.Publisher(
        site["topics"]["safety_override"],
        AckermannDriveStamped,
        queue_size=10,
    )

    def on_joy(message):
        state.update(list(message.buttons))

    def on_reset(request):
        del request
        success, message = state.reset()
        return TriggerResponse(success=success, message=message)

    rospy.Subscriber(bridge["joy_topic"], Joy, on_joy, queue_size=20)
    rospy.Service("/hardware_study/reset_software_estop", Trigger, on_reset)
    rate = rospy.Rate(float(bridge["publish_rate_hz"]))
    while not rospy.is_shutdown():
        deadman, estop = state.snapshot()
        deadman_pub.publish(Bool(data=deadman))
        estop_pub.publish(Bool(data=estop))
        if state.stop_override_required():
            stop = AckermannDriveStamped()
            stop.header.stamp = rospy.Time.now()
            stop.header.frame_id = site["frames"]["command_frame_id"]
            stop.drive.steering_angle = 0.0
            stop.drive.speed = 0.0
            stop_pub.publish(stop)
        rate.sleep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

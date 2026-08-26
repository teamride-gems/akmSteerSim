#!/usr/bin/env python3
"""Fail-closed runner authorization and latched e-stop bridge for ROS 1."""

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
        estop_index: int,
        joy_stale_seconds: float,
        run_active_stale_seconds: float,
        authorization_clearance_seconds: float = 0.0,
    ):
        if estop_index < 0:
            raise ValueError("e-stop must be a nonnegative button index")
        self.estop_index = int(estop_index)
        self.joy_stale_seconds = float(joy_stale_seconds)
        self.run_active_stale_seconds = float(run_active_stale_seconds)
        self.authorization_clearance_seconds = float(
            authorization_clearance_seconds
        )
        self.last_joy_time = None
        self.last_run_active_time = None
        self.run_active = False
        self.run_active_source_count = 0
        self.authorization_requested_since = None
        self.buttons = []
        self.estop_latched = True
        self.lock = threading.Lock()

    def _button(self, index: int) -> bool:
        return index < len(self.buttons) and bool(self.buttons[index])

    def _fresh(self, last_time, timeout: float, current: float) -> bool:
        return last_time is not None and current - float(last_time) <= float(timeout)

    def _raw_authorized(self, current: float) -> bool:
        return (
            self._fresh(self.last_joy_time, self.joy_stale_seconds, current)
            and self._fresh(
                self.last_run_active_time,
                self.run_active_stale_seconds,
                current,
            )
            and self.run_active
            and self.run_active_source_count == 1
            and not self.estop_latched
        )

    def _refresh_authorization(self, current: float) -> bool:
        raw = self._raw_authorized(current)
        if raw and self.authorization_requested_since is None:
            self.authorization_requested_since = current
        elif not raw:
            self.authorization_requested_since = None
        return raw

    def update_joy(self, buttons: list[int], now: float | None = None) -> None:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            self.last_joy_time = current
            self.buttons = list(buttons)
            if self._button(self.estop_index):
                self.estop_latched = True
            self._refresh_authorization(current)

    def update_run_active(self, active: bool, now: float | None = None) -> None:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            self.last_run_active_time = current
            self.run_active = bool(active)
            self._refresh_authorization(current)

    def update_run_active_source_count(
        self, source_count: int, now: float | None = None
    ) -> None:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            self.run_active_source_count = int(source_count)
            self._refresh_authorization(current)

    def snapshot(self, now: float | None = None) -> tuple[bool, bool]:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            raw = self._refresh_authorization(current)
            clearance_elapsed = (
                self.authorization_requested_since is not None
                and current - self.authorization_requested_since
                >= self.authorization_clearance_seconds
            )
            authorized = raw and clearance_elapsed
            return bool(authorized), bool(self.estop_latched)

    def stop_override_required(self, now: float | None = None) -> bool:
        """Return true unless fresh joystick and runner heartbeats allow release."""
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            return not self._refresh_authorization(current)

    def reset(self, now: float | None = None) -> tuple[bool, str]:
        with self.lock:
            current = time.monotonic() if now is None else float(now)
            fresh = self._fresh(
                self.last_joy_time, self.joy_stale_seconds, current
            )
            safe = fresh and not any(bool(value) for value in self.buttons)
            if safe:
                self.estop_latched = False
                self.authorization_requested_since = None
                self._refresh_authorization(current)
                return (
                    True,
                    "software e-stop reset; motion still requires a fresh runner heartbeat",
                )
            return False, "reset denied: require fresh joystick with all buttons released"


def main() -> int:  # pragma: no cover - requires ROS 1 robot runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_hardware_site_ros1.yaml")
    args = parser.parse_args()
    site = yaml.safe_load((ROOT / args.site).read_text(encoding="utf-8"))
    bridge = site["safety_bridge"]
    state = SafetyState(
        int(bridge["estop_button_index"]),
        float(bridge["joy_stale_seconds"]),
        float(bridge["run_active_stale_seconds"]),
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
    authorization_pub = rospy.Publisher(
        site["topics"]["deadman"], Bool, queue_size=10
    )
    estop_pub = rospy.Publisher(site["topics"]["estop"], Bool, queue_size=10)
    stop_pub = rospy.Publisher(
        site["topics"]["safety_override"],
        AckermannDriveStamped,
        queue_size=10,
    )

    def on_joy(message):
        state.update_joy(list(message.buttons))

    def on_run_active(message):
        state.update_run_active(bool(message.data))

    def on_reset(request):
        del request
        success, message = state.reset()
        return TriggerResponse(success=success, message=message)

    rospy.Subscriber(bridge["joy_topic"], Joy, on_joy, queue_size=20)
    run_active_subscriber = rospy.Subscriber(
        site["topics"]["run_active"], Bool, on_run_active, queue_size=20
    )
    rospy.Service("/hardware_study/reset_software_estop", Trigger, on_reset)
    rate = rospy.Rate(float(bridge["publish_rate_hz"]))
    while not rospy.is_shutdown():
        state.update_run_active_source_count(
            run_active_subscriber.get_num_connections()
        )
        authorized, estop = state.snapshot()
        authorization_pub.publish(Bool(data=authorized))
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

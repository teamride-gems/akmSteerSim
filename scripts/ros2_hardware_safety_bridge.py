#!/usr/bin/env python3
"""Publish fail-closed deadman and latched e-stop status from a ROS 2 joystick."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:  # pragma: no cover - requires ROS 2 robot runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="local_hardware_site.yaml")
    args = parser.parse_args()
    site = yaml.safe_load((ROOT / args.site).read_text(encoding="utf-8"))
    bridge = site["safety_bridge"]
    deadman_index = int(bridge["deadman_button_index"])
    estop_index = int(bridge["estop_button_index"])
    if deadman_index == estop_index or min(deadman_index, estop_index) < 0:
        raise RuntimeError("deadman and e-stop must use distinct nonnegative buttons")

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Joy
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger
    except ImportError as exc:
        raise RuntimeError(
            "This bridge requires ROS 2 packages rclpy, sensor_msgs, std_msgs, and std_srvs"
        ) from exc

    class SafetyBridge(Node):
        def __init__(self):
            super().__init__("ride_hardware_safety_bridge")
            self.last_joy_time = None
            self.buttons = []
            # Fail closed at startup. Reset is possible only with a fresh joystick,
            # the e-stop button released, and the deadman released.
            self.estop_latched = True
            self.deadman_pub = self.create_publisher(
                Bool, site["topics"]["deadman"], 10
            )
            self.estop_pub = self.create_publisher(Bool, site["topics"]["estop"], 10)
            self.create_subscription(Joy, bridge["joy_topic"], self.on_joy, 20)
            self.create_service(
                Trigger, "/hardware_study/reset_software_estop", self.on_reset
            )
            self.create_timer(1.0 / float(bridge["publish_rate_hz"]), self.publish)

        def button(self, index: int) -> bool:
            return index < len(self.buttons) and bool(self.buttons[index])

        def fresh(self) -> bool:
            return self.last_joy_time is not None and (
                time.monotonic() - self.last_joy_time
                <= float(bridge["joy_stale_seconds"])
            )

        def on_joy(self, message):
            self.last_joy_time = time.monotonic()
            self.buttons = list(message.buttons)
            if self.button(estop_index):
                self.estop_latched = True

        def on_reset(self, request, response):
            del request
            safe_to_reset = (
                self.fresh()
                and not self.button(estop_index)
                and not self.button(deadman_index)
            )
            if safe_to_reset:
                self.estop_latched = False
                response.success = True
                response.message = "software e-stop reset; deadman remains released"
            else:
                response.success = False
                response.message = (
                    "reset denied: require fresh joystick with e-stop and deadman released"
                )
            return response

        def publish(self):
            fresh = self.fresh()
            deadman = (
                fresh and self.button(deadman_index) and not self.estop_latched
            )
            deadman_message = Bool()
            deadman_message.data = bool(deadman)
            estop_message = Bool()
            estop_message.data = bool(self.estop_latched)
            self.deadman_pub.publish(deadman_message)
            self.estop_pub.publish(estop_message)

    rclpy.init(args=None)
    node = SafetyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

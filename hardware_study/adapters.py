"""Robot adapters for fast local mocks and optional ROS 2 Ackermann hardware."""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Optional

import numpy as np


class AdapterError(RuntimeError):
    pass


class MockAdapter:
    """Deterministic no-sleep adapter used by tests and rehearsal."""

    name = "mock"
    is_real = False

    def __init__(self, wheelbase_m: float = 0.33, telemetry_rate_hz: float = 100.0):
        self.wheelbase_m = float(wheelbase_m)
        self.telemetry_rate_hz = float(telemetry_rate_hz)
        self.virtual_time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.steering = 0.0
        self.deadman = True
        self.estop = False
        self._queue = deque()
        self._latest = self._telemetry()

    def _telemetry(self) -> dict:
        yaw_rate = (
            self.speed * math.tan(self.steering) / self.wheelbase_m
            if self.wheelbase_m > 0.0
            else 0.0
        )
        return {
            "received_monotonic_s": float(self.virtual_time),
            "source_stamp_s": float(self.virtual_time),
            "x_m": float(self.x),
            "y_m": float(self.y),
            "yaw_rad": float(self.yaw),
            "speed_mps": float(self.speed),
            "yaw_rate_rad_s": float(yaw_rate),
            "steering_rad": float(self.steering),
            "steering_source": "mock_state",
            "deadman": bool(self.deadman),
            "deadman_received_monotonic_s": float(self.virtual_time),
            "estop": bool(self.estop),
            "estop_received_monotonic_s": float(self.virtual_time),
            "battery_voltage_v": 24.0,
        }

    def now(self) -> float:
        return float(self.virtual_time)

    def wait_until(self, target_monotonic_s: float) -> None:
        if target_monotonic_s > self.virtual_time:
            self.virtual_time = float(target_monotonic_s)

    def wait_for_ready(self, timeout_seconds: float) -> bool:
        return True

    def publish(self, steering_rad: float, speed_mps: float, duration_s: float) -> None:
        substeps = max(1, int(round(float(duration_s) * self.telemetry_rate_hz)))
        dt = float(duration_s) / substeps
        target_steering = float(steering_rad)
        target_speed = float(speed_mps)
        for _ in range(substeps):
            self.steering += np.clip(target_steering - self.steering, -3.2 * dt, 3.2 * dt)
            self.speed += np.clip(target_speed - self.speed, -3.0 * dt, 3.0 * dt)
            yaw_rate = self.speed * math.tan(self.steering) / self.wheelbase_m
            self.x += self.speed * math.cos(self.yaw) * dt
            self.y += self.speed * math.sin(self.yaw) * dt
            self.yaw += yaw_rate * dt
            self.virtual_time += dt
            self._latest = self._telemetry()
            self._queue.append(dict(self._latest))

    def latest_telemetry(self) -> dict:
        return dict(self._latest)

    def drain_telemetry(self) -> list[dict]:
        output = list(self._queue)
        self._queue.clear()
        return output

    def close(self) -> None:
        pass


class Ros2AckermannAdapter:
    """Minimal ROS 2 adapter using standard Ackermann, odometry, and safety messages."""

    name = "ros2_ackermann"
    is_real = True

    def __init__(self, site: dict):
        try:
            import rclpy
            from ackermann_msgs.msg import AckermannDriveStamped
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Bool
        except ImportError as exc:  # pragma: no cover - requires robot runtime
            raise AdapterError(
                "ROS 2 adapter requires rclpy, ackermann_msgs, nav_msgs, and std_msgs"
            ) from exc
        self._rclpy = rclpy
        self._AckermannDriveStamped = AckermannDriveStamped
        self._lock = threading.Lock()
        self._queue = deque()
        self._latest: Optional[dict] = None
        self._deadman = None
        self._deadman_time = None
        self._estop = None
        self._estop_time = None
        self._joint_steering = None
        self._battery_voltage = None
        self.site = site
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("ride_hardware_study_runner")
        self.publisher = self.node.create_publisher(
            AckermannDriveStamped, site["topics"]["drive"], 10
        )
        self.node.create_subscription(
            Odometry, site["topics"]["odometry"], self._odom_callback, 50
        )
        self.node.create_subscription(
            Bool, site["topics"]["deadman"], self._deadman_callback, 20
        )
        self.node.create_subscription(
            Bool, site["topics"]["estop"], self._estop_callback, 20
        )
        joint_topic = site["topics"].get("joint_states")
        if joint_topic:
            try:
                from sensor_msgs.msg import JointState
            except ImportError as exc:
                raise AdapterError("joint-state feedback requires sensor_msgs") from exc
            self.node.create_subscription(JointState, joint_topic, self._joint_callback, 50)
        battery_topic = site["topics"].get("battery_state")
        if battery_topic:
            try:
                from sensor_msgs.msg import BatteryState
            except ImportError as exc:
                raise AdapterError("battery feedback requires sensor_msgs") from exc
            self.node.create_subscription(
                BatteryState, battery_topic, self._battery_callback, 10
            )
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def now(self) -> float:
        return time.monotonic()

    def wait_until(self, target_monotonic_s: float) -> None:
        remaining = target_monotonic_s - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def _spin(self):  # pragma: no cover - requires robot runtime
        while self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.01)

    @staticmethod
    def _yaw(quaternion) -> float:
        siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
        return math.atan2(siny, cosy)

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def _odom_callback(self, message):  # pragma: no cover - requires robot runtime
        now = time.monotonic()
        pose = message.pose.pose
        twist = message.twist.twist
        speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        yaw_rate = float(twist.angular.z)
        feedback = self.site["steering_feedback"]
        if self._joint_steering is not None:
            steering = float(self._joint_steering)
            source = "joint_state"
        elif speed >= float(feedback["minimum_speed_for_kinematic_estimate_mps"]):
            steering = math.atan(float(feedback["wheelbase_m"]) * yaw_rate / speed)
            source = "kinematic_from_odometry"
        else:
            steering = None
            source = "unavailable_below_speed_threshold"
        with self._lock:
            telemetry = {
                "received_monotonic_s": now,
                "source_stamp_s": self._stamp_seconds(message.header.stamp),
                "x_m": float(pose.position.x),
                "y_m": float(pose.position.y),
                "yaw_rad": self._yaw(pose.orientation),
                "speed_mps": speed,
                "yaw_rate_rad_s": yaw_rate,
                "steering_rad": steering,
                "steering_source": source,
                "deadman": bool(self._deadman) if self._deadman is not None else False,
                "deadman_received_monotonic_s": self._deadman_time,
                "estop": bool(self._estop) if self._estop is not None else True,
                "estop_received_monotonic_s": self._estop_time,
                "battery_voltage_v": self._battery_voltage,
            }
            self._latest = telemetry
            self._queue.append(dict(telemetry))

    def _deadman_callback(self, message):  # pragma: no cover
        with self._lock:
            self._deadman = bool(message.data)
            self._deadman_time = time.monotonic()

    def _estop_callback(self, message):  # pragma: no cover
        with self._lock:
            self._estop = bool(message.data)
            self._estop_time = time.monotonic()

    def _joint_callback(self, message):  # pragma: no cover
        name = self.site["steering_feedback"]["steering_joint_name"]
        if name not in message.name:
            return
        index = list(message.name).index(name)
        with self._lock:
            self._joint_steering = float(message.position[index])

    def _battery_callback(self, message):  # pragma: no cover
        with self._lock:
            self._battery_voltage = float(message.voltage)

    def wait_for_ready(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            with self._lock:
                ready = (
                    self._latest is not None
                    and self._deadman is not None
                    and self._estop is not None
                )
            if ready:
                return True
            time.sleep(0.02)
        return False

    def publish(self, steering_rad: float, speed_mps: float, duration_s: float) -> None:
        message = self._AckermannDriveStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.site["frames"]["command_frame_id"]
        message.drive.steering_angle = float(steering_rad)
        message.drive.speed = float(speed_mps)
        self.publisher.publish(message)

    def latest_telemetry(self) -> dict:
        with self._lock:
            if self._latest is None:
                raise AdapterError("no odometry received")
            return dict(self._latest)

    def drain_telemetry(self) -> list[dict]:
        with self._lock:
            output = list(self._queue)
            self._queue.clear()
        return output

    def close(self) -> None:  # pragma: no cover
        try:
            self.node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()
            self._thread.join(timeout=2.0)


def make_adapter(name: str, site: dict):
    if name == "mock":
        return MockAdapter(
            wheelbase_m=float(site["steering_feedback"]["wheelbase_m"])
        )
    if name == "ros2":
        return Ros2AckermannAdapter(site)
    raise ValueError(f"unknown adapter: {name}")

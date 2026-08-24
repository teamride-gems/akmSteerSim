"""Native ROS 1 Noetic adapter for the frozen hardware-study executor.

ROS imports are intentionally delayed until construction so the repository remains
testable on machines without ROS.  Tests may inject a small runtime object that
provides the same message classes and rospy surface.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Optional

from .adapters import AdapterError


def load_ros1_runtime():  # pragma: no cover - requires the robot runtime
    try:
        import rospy
        from ackermann_msgs.msg import AckermannDriveStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import BatteryState, JointState
        from std_msgs.msg import Bool
    except ImportError as exc:
        raise AdapterError(
            "ROS 1 adapter requires rospy, ackermann_msgs, nav_msgs, sensor_msgs, "
            "and std_msgs in a sourced ROS Noetic environment"
        ) from exc

    class Runtime:
        pass

    runtime = Runtime()
    runtime.rospy = rospy
    runtime.AckermannDriveStamped = AckermannDriveStamped
    runtime.Odometry = Odometry
    runtime.Bool = Bool
    runtime.JointState = JointState
    runtime.BatteryState = BatteryState
    return runtime


class Ros1AckermannAdapter:
    """ROS 1 Ackermann adapter with fail-closed safety telemetry."""

    name = "ros1_ackermann_noetic"
    is_real = True

    def __init__(self, site: dict, runtime=None):
        self.runtime = runtime or load_ros1_runtime()
        self.rospy = self.runtime.rospy
        self.site = site
        self._lock = threading.Lock()
        self._queue = deque()
        self._latest: Optional[dict] = None
        self._deadman = None
        self._deadman_time = None
        self._estop = None
        self._estop_time = None
        self._joint_steering = None
        self._battery_voltage = None
        self._subscribers = []

        if not self.rospy.core.is_initialized():
            self.rospy.init_node(
                "ride_hardware_study_runner", anonymous=False, disable_signals=True
            )
        self.publisher = self.rospy.Publisher(
            site["topics"]["drive"],
            self.runtime.AckermannDriveStamped,
            queue_size=10,
        )
        self._subscribers.append(
            self.rospy.Subscriber(
                site["topics"]["odometry"],
                self.runtime.Odometry,
                self._odom_callback,
                queue_size=50,
            )
        )
        self._subscribers.append(
            self.rospy.Subscriber(
                site["topics"]["deadman"],
                self.runtime.Bool,
                self._deadman_callback,
                queue_size=20,
            )
        )
        self._subscribers.append(
            self.rospy.Subscriber(
                site["topics"]["estop"],
                self.runtime.Bool,
                self._estop_callback,
                queue_size=20,
            )
        )
        joint_topic = site["topics"].get("joint_states")
        if joint_topic:
            self._subscribers.append(
                self.rospy.Subscriber(
                    joint_topic,
                    self.runtime.JointState,
                    self._joint_callback,
                    queue_size=50,
                )
            )
        battery_topic = site["topics"].get("battery_state")
        if battery_topic:
            self._subscribers.append(
                self.rospy.Subscriber(
                    battery_topic,
                    self.runtime.BatteryState,
                    self._battery_callback,
                    queue_size=10,
                )
            )

    def now(self) -> float:
        return time.monotonic()

    def wait_until(self, target_monotonic_s: float) -> None:
        remaining = float(target_monotonic_s) - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    @staticmethod
    def _yaw(quaternion) -> float:
        siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
        return math.atan2(siny, cosy)

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.to_sec())

    def _odom_callback(self, message):  # pragma: no cover - exercised with fakes
        now = time.monotonic()
        pose = message.pose.pose
        twist = message.twist.twist
        speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        yaw_rate = float(twist.angular.z)
        feedback = self.site["steering_feedback"]
        with self._lock:
            if self._joint_steering is not None:
                steering = float(self._joint_steering)
                source = "joint_state"
            elif speed >= float(feedback["minimum_speed_for_kinematic_estimate_mps"]):
                steering = math.atan(float(feedback["wheelbase_m"]) * yaw_rate / speed)
                source = "kinematic_from_odometry"
            else:
                steering = None
                source = "unavailable_below_speed_threshold"
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

    def _deadman_callback(self, message):
        with self._lock:
            self._deadman = bool(message.data)
            self._deadman_time = time.monotonic()

    def _estop_callback(self, message):
        with self._lock:
            self._estop = bool(message.data)
            self._estop_time = time.monotonic()

    def _joint_callback(self, message):
        name = self.site["steering_feedback"]["steering_joint_name"]
        if name not in message.name:
            return
        index = list(message.name).index(name)
        with self._lock:
            self._joint_steering = float(message.position[index])

    def _battery_callback(self, message):
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
        del duration_s
        message = self.runtime.AckermannDriveStamped()
        message.header.stamp = self.rospy.Time.now()
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

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.unregister()
        self.publisher.unregister()


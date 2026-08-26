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
        from tf2_msgs.msg import TFMessage
    except ImportError as exc:
        raise AdapterError(
            "ROS 1 adapter requires rospy, ackermann_msgs, nav_msgs, sensor_msgs, "
            "std_msgs, and tf2_msgs in a sourced ROS Noetic environment"
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
    runtime.TFMessage = TFMessage
    return runtime


class Ros1AckermannAdapter:
    """ROS 1 Ackermann adapter with fail-closed safety telemetry."""

    name = "ros1_ackermann_noetic"
    is_real = True

    def __init__(self, site: dict, runtime=None, authorize_motion: bool = False):
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
        self._odom_state = None
        self._tf_transforms = {}
        self._subscribers = []
        self._authorize_motion = bool(authorize_motion)
        self.run_active_publisher = None

        if not self.rospy.core.is_initialized():
            self.rospy.init_node(
                "ride_hardware_study_runner", anonymous=False, disable_signals=True
            )
        self.publisher = self.rospy.Publisher(
            site["topics"]["drive"],
            self.runtime.AckermannDriveStamped,
            queue_size=10,
        )
        if self._authorize_motion:
            self.run_active_publisher = self.rospy.Publisher(
                site["topics"]["run_active"], self.runtime.Bool, queue_size=10
            )
            self._publish_run_active(True)
        self._subscribers.append(
            self.rospy.Subscriber(
                site["topics"]["odometry"],
                self.runtime.Odometry,
                self._odom_callback,
                queue_size=50,
            )
        )
        localization_tf_topic = site["topics"].get("localization_tf")
        if localization_tf_topic:
            self._subscribers.append(
                self.rospy.Subscriber(
                    localization_tf_topic,
                    self.runtime.TFMessage,
                    self._tf_callback,
                    queue_size=100,
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
        while True:
            self._publish_run_active(True)
            remaining = float(target_monotonic_s) - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(remaining, 0.05))

    def _publish_run_active(self, active: bool) -> None:
        if self.run_active_publisher is None:
            return
        message = self.runtime.Bool()
        message.data = bool(active)
        self.run_active_publisher.publish(message)

    @staticmethod
    def _yaw(quaternion) -> float:
        siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy = 1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2)
        return math.atan2(siny, cosy)

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.to_sec())

    @staticmethod
    def _normalized_frame(frame: str) -> str:
        return str(frame).lstrip("/")

    def _steering(self, speed: float, yaw_rate: float) -> tuple[Optional[float], str]:
        feedback = self.site["steering_feedback"]
        if self._joint_steering is not None:
            return float(self._joint_steering), "joint_state"
        if speed >= float(feedback["minimum_speed_for_kinematic_estimate_mps"]):
            value = math.atan(float(feedback["wheelbase_m"]) * yaw_rate / speed)
            return value, "kinematic_from_odometry"
        return None, "unavailable_below_speed_threshold"

    def _append_telemetry(
        self,
        *,
        source_stamp_s: float,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        speed_mps: float,
        yaw_rate_rad_s: float,
        pose_source: str,
    ) -> None:
        now = time.monotonic()
        steering, steering_source = self._steering(speed_mps, yaw_rate_rad_s)
        telemetry = {
            "received_monotonic_s": now,
            "source_stamp_s": float(source_stamp_s),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "yaw_rad": float(yaw_rad),
            "speed_mps": float(speed_mps),
            "yaw_rate_rad_s": float(yaw_rate_rad_s),
            "steering_rad": steering,
            "steering_source": steering_source,
            "pose_source": pose_source,
            "deadman": bool(self._deadman) if self._deadman is not None else False,
            "deadman_received_monotonic_s": self._deadman_time,
            "estop": bool(self._estop) if self._estop is not None else True,
            "estop_received_monotonic_s": self._estop_time,
            "battery_voltage_v": self._battery_voltage,
        }
        self._latest = telemetry
        self._queue.append(dict(telemetry))

    def _odom_callback(self, message):  # pragma: no cover - exercised with fakes
        now = time.monotonic()
        pose = message.pose.pose
        twist = message.twist.twist
        speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        yaw_rate = float(twist.angular.z)
        with self._lock:
            self._odom_state = {
                "received_monotonic_s": now,
                "source_stamp_s": self._stamp_seconds(message.header.stamp),
                "speed_mps": speed,
                "yaw_rate_rad_s": yaw_rate,
            }
            if not self.site["topics"].get("localization_tf"):
                self._append_telemetry(
                    source_stamp_s=self._odom_state["source_stamp_s"],
                    x_m=pose.position.x,
                    y_m=pose.position.y,
                    yaw_rad=self._yaw(pose.orientation),
                    speed_mps=speed,
                    yaw_rate_rad_s=yaw_rate,
                    pose_source="odometry",
                )

    def _tf_callback(self, message):  # pragma: no cover - exercised with fakes
        now = time.monotonic()
        frames = self.site["frames"]
        world = self._normalized_frame(frames["odometry_frame_id"])
        localization_odom = self._normalized_frame(frames["localization_odom_frame_id"])
        base = self._normalized_frame(frames.get("base_frame_id", "base_link"))
        with self._lock:
            for transform in message.transforms:
                parent = self._normalized_frame(transform.header.frame_id)
                child = self._normalized_frame(transform.child_frame_id)
                self._tf_transforms[(parent, child)] = (now, transform)
            world_to_odom = self._tf_transforms.get((world, localization_odom))
            odom_to_base = self._tf_transforms.get((localization_odom, base))
            if not world_to_odom or not odom_to_base or self._odom_state is None:
                return
            stale_seconds = float(frames.get("localization_stale_seconds", 0.10))
            if (
                now - world_to_odom[0] > stale_seconds
                or now - odom_to_base[0] > stale_seconds
                or now - self._odom_state["received_monotonic_s"] > stale_seconds
            ):
                return
            first = world_to_odom[1]
            second = odom_to_base[1]
            first_yaw = self._yaw(first.transform.rotation)
            second_yaw = self._yaw(second.transform.rotation)
            first_translation = first.transform.translation
            second_translation = second.transform.translation
            x_m = (
                float(first_translation.x)
                + math.cos(first_yaw) * float(second_translation.x)
                - math.sin(first_yaw) * float(second_translation.y)
            )
            y_m = (
                float(first_translation.y)
                + math.sin(first_yaw) * float(second_translation.x)
                + math.cos(first_yaw) * float(second_translation.y)
            )
            source_stamp_s = max(
                self._stamp_seconds(first.header.stamp),
                self._stamp_seconds(second.header.stamp),
            )
            self._append_telemetry(
                source_stamp_s=source_stamp_s,
                x_m=x_m,
                y_m=y_m,
                yaw_rad=math.atan2(
                    math.sin(first_yaw + second_yaw),
                    math.cos(first_yaw + second_yaw),
                ),
                speed_mps=self._odom_state["speed_mps"],
                yaw_rate_rad_s=self._odom_state["yaw_rate_rad_s"],
                pose_source="composed_tf",
            )

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
            self._publish_run_active(True)
            with self._lock:
                safety_observed = self._deadman is not None and self._estop is not None
                safety_authorized = (
                    self._deadman is True and self._estop is False
                    if self._authorize_motion
                    else safety_observed
                )
                ready = self._latest is not None and safety_authorized
            if ready:
                return True
            time.sleep(0.02)
        return False

    def publish(self, steering_rad: float, speed_mps: float, duration_s: float) -> None:
        del duration_s
        self._publish_run_active(True)
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
        if self.run_active_publisher is not None:
            for _ in range(3):
                self._publish_run_active(False)
                time.sleep(0.01)
        for subscriber in self._subscribers:
            subscriber.unregister()
        self.publisher.unregister()
        if self.run_active_publisher is not None:
            self.run_active_publisher.unregister()

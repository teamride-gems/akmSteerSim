"""Common final command limiter and runtime safety predicates."""

from __future__ import annotations

from dataclasses import dataclass
import math


class SafetyViolation(RuntimeError):
    pass


@dataclass
class SafetyLimiter:
    maximum_speed_mps: float
    maximum_abs_steering_rad: float
    maximum_steering_slew_rad_s: float
    maximum_acceleration_mps2: float
    last_steering_rad: float = 0.0
    last_speed_mps: float = 0.0

    def reset(self) -> None:
        self.last_steering_rad = 0.0
        self.last_speed_mps = 0.0

    def apply(self, target_steering_rad: float, target_speed_mps: float, dt: float) -> dict:
        values = (target_steering_rad, target_speed_mps, dt)
        if not all(math.isfinite(float(value)) for value in values):
            raise SafetyViolation("nonfinite command or timestep")
        if dt <= 0.0:
            raise SafetyViolation("command timestep must be positive")
        bounded_target_steering = max(
            -self.maximum_abs_steering_rad,
            min(self.maximum_abs_steering_rad, float(target_steering_rad)),
        )
        bounded_target_speed = max(
            0.0, min(self.maximum_speed_mps, float(target_speed_mps))
        )
        maximum_steering_change = self.maximum_steering_slew_rad_s * float(dt)
        maximum_speed_change = self.maximum_acceleration_mps2 * float(dt)
        steering = max(
            self.last_steering_rad - maximum_steering_change,
            min(
                self.last_steering_rad + maximum_steering_change,
                bounded_target_steering,
            ),
        )
        speed = max(
            self.last_speed_mps - maximum_speed_change,
            min(self.last_speed_mps + maximum_speed_change, bounded_target_speed),
        )
        limited = {
            "steering_rad": float(steering),
            "speed_mps": float(speed),
            "target_steering_clipped": bounded_target_steering
            != float(target_steering_rad),
            "target_speed_clipped": bounded_target_speed != float(target_speed_mps),
            "steering_slew_limited": steering != bounded_target_steering,
            "acceleration_limited": speed != bounded_target_speed,
        }
        self.last_steering_rad = float(steering)
        self.last_speed_mps = float(speed)
        return limited


def check_runtime_safety(telemetry: dict, start_pose: dict, config: dict) -> list[str]:
    safety = config["safety"]
    reasons = []
    required = ("x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rad_s")
    for key in required:
        value = telemetry.get(key)
        if value is None or not math.isfinite(float(value)):
            reasons.append(f"invalid_telemetry_{key}")
    if reasons:
        return reasons
    radius = math.hypot(
        float(telemetry["x_m"]) - float(start_pose["x_m"]),
        float(telemetry["y_m"]) - float(start_pose["y_m"]),
    )
    if radius > float(safety["maximum_geofence_radius_m"]):
        reasons.append("geofence_exceeded")
    if abs(float(telemetry["speed_mps"])) > float(
        safety["maximum_observed_speed_mps"]
    ):
        reasons.append("observed_speed_exceeded")
    if abs(float(telemetry["yaw_rate_rad_s"])) > float(
        safety["maximum_abs_yaw_rate_rad_s"]
    ):
        reasons.append("yaw_rate_exceeded")
    if bool(telemetry.get("estop", False)):
        reasons.append("estop_asserted")
    if not bool(telemetry.get("deadman", False)):
        reasons.append("deadman_released")
    return reasons

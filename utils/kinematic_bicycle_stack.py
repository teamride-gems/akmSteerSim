"""Standalone kinematic-bicycle plant used for cross-stack transport tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KinematicPlantConfig:
    wheelbase_m: float
    steering_time_constant_s: float
    speed_time_constant_s: float
    steering_rate_limit_rad_s: float
    acceleration_limit_m_s2: float
    steering_bound_rad: float
    min_speed_m_s: float = 0.0
    max_speed_m_s: float = 6.0

    def validate(self) -> None:
        positive = (
            self.wheelbase_m,
            self.steering_time_constant_s,
            self.speed_time_constant_s,
            self.steering_rate_limit_rad_s,
            self.acceleration_limit_m_s2,
            self.steering_bound_rad,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("plant time constants, limits, bound, and wheelbase must be positive")
        if self.min_speed_m_s > self.max_speed_m_s:
            raise ValueError("minimum speed exceeds maximum speed")


def derivative(state: Sequence[float], command: Sequence[float], cfg: KinematicPlantConfig):
    cfg.validate()
    x, y, heading, speed, steering = np.asarray(state, dtype=float)
    steering_target, speed_target = np.asarray(command, dtype=float)
    steering_target = float(np.clip(steering_target, -cfg.steering_bound_rad, cfg.steering_bound_rad))
    speed_target = float(np.clip(speed_target, cfg.min_speed_m_s, cfg.max_speed_m_s))
    steering_rate = float(
        np.clip(
            (steering_target - steering) / cfg.steering_time_constant_s,
            -cfg.steering_rate_limit_rad_s,
            cfg.steering_rate_limit_rad_s,
        )
    )
    acceleration = float(
        np.clip(
            (speed_target - speed) / cfg.speed_time_constant_s,
            -cfg.acceleration_limit_m_s2,
            cfg.acceleration_limit_m_s2,
        )
    )
    return np.array(
        [
            speed * np.cos(heading),
            speed * np.sin(heading),
            speed * np.tan(steering) / cfg.wheelbase_m,
            acceleration,
            steering_rate,
        ],
        dtype=float,
    )


def rk4_step(
    state: Sequence[float],
    command: Sequence[float],
    dt_seconds: float,
    cfg: KinematicPlantConfig,
) -> np.ndarray:
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive")
    state = np.asarray(state, dtype=float)
    k1 = derivative(state, command, cfg)
    k2 = derivative(state + 0.5 * dt_seconds * k1, command, cfg)
    k3 = derivative(state + 0.5 * dt_seconds * k2, command, cfg)
    k4 = derivative(state + dt_seconds * k3, command, cfg)
    result = state + (dt_seconds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[3] = np.clip(result[3], cfg.min_speed_m_s, cfg.max_speed_m_s)
    result[4] = np.clip(result[4], -cfg.steering_bound_rad, cfg.steering_bound_rad)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite independent-plant state")
    return result


def simulate_packet_commands(
    steering_commands_rad: Sequence[float],
    speed_commands_m_s: Sequence[float],
    ticks_per_packet: int,
    dt_seconds: float,
    cfg: KinematicPlantConfig,
    initial_steering_rad: float = 0.0,
) -> dict:
    steering = np.asarray(steering_commands_rad, dtype=float).reshape(-1)
    speed = np.asarray(speed_commands_m_s, dtype=float).reshape(-1)
    if steering.size == 0 or steering.shape != speed.shape:
        raise ValueError("steering and speed packet arrays must be nonempty and aligned")
    if ticks_per_packet <= 0:
        raise ValueError("ticks_per_packet must be positive")
    state = np.array([0.0, 0.0, 0.0, float(speed[0]), float(initial_steering_rad)])
    states = []
    for packet_index in range(steering.size):
        command = np.array([steering[packet_index], speed[packet_index]], dtype=float)
        for _ in range(int(ticks_per_packet)):
            state = rk4_step(state, command, dt_seconds, cfg)
            states.append(state.copy())
    array = np.asarray(states, dtype=float)
    return {
        "states": array,
        "positions": array[:, :2],
        "terminal_position": array[-1, :2],
        "steering_tv_rad": float(np.sum(np.abs(np.diff(array[:, 4])))),
        "maximum_abs_steering_rad": float(np.max(np.abs(array[:, 4]))),
    }

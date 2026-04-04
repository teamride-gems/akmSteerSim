import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# Shared robot command interface
# ============================================================

ROBOT_COMMAND_NAMES = ("steering_angle", "speed")
ROBOT_COMMAND_UNITS = ("rad", "m/s")


# ============================================================
# Basic helpers
# ============================================================


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)



def _require_dim(x: Any, expected_dim: int, name: str = "vector") -> np.ndarray:
    arr = _to_numpy(x)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"{name} must have dimension {expected_dim}, got {arr.shape[0]}.")
    return arr



def _cfg(config: Dict[str, Any], key: str, default: float) -> float:
    return float(config.get(key, default))



def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))



def scale_from_unit_interval(x: float, low: float, high: float) -> float:
    return low + x * (high - low)



def scale_from_signed_unit(x: float, low: float, high: float) -> float:
    return low + 0.5 * (x + 1.0) * (high - low)



def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)



def get_steering_bounds(config: Dict[str, Any]) -> Tuple[float, float]:
    return (
        _cfg(config, "min_steering_angle", -0.4189),
        _cfg(config, "max_steering_angle", 0.4189),
    )



def get_speed_bounds(config: Dict[str, Any]) -> Tuple[float, float]:
    return (
        _cfg(config, "min_speed", 0.0),
        _cfg(config, "max_speed", 5.0),
    )



def get_wheelbase(config: Dict[str, Any]) -> float:
    return _cfg(config, "wheelbase", 0.33)



def get_curvature_bounds(config: Dict[str, Any]) -> Tuple[float, float]:
    min_steering, max_steering = get_steering_bounds(config)
    wheelbase = max(get_wheelbase(config), 1e-8)
    return (
        math.tan(min_steering) / wheelbase,
        math.tan(max_steering) / wheelbase,
    )


# ============================================================
# Per-dimension policy output mapping
# ============================================================
#
# IMPORTANT: SB3's SAC uses SquashedDiagGaussianDistribution, which
# applies tanh internally to squash Gaussian samples to [-1, 1].
# The raw_action arriving at env.step() is ALREADY in [-1, 1].
#
# The "linear" mode performs a simple affine rescaling from [-1, 1]
# to [low, high] with no additional nonlinearity. This is the correct
# default for SB3 — it gives the agent equal access to the full
# physical range of each dimension.
#
# The "tanh" and "sigmoid" modes apply ADDITIONAL nonlinearities on
# top of SB3's tanh, creating double-squashing that compresses the
# effective action range. These are retained for compatibility but
# should NOT be used with SB3's default SAC.
# ============================================================



def map_dimension_from_spec(raw_value: float, dim_spec: Dict[str, Any]) -> float:
    mode = dim_spec.get("mode", "linear")

    if mode == "identity":
        return float(raw_value)

    if mode == "linear":
        low = float(dim_spec["low"])
        high = float(dim_spec["high"])
        return scale_from_signed_unit(clip(float(raw_value), -1.0, 1.0), low, high)

    if mode == "tanh":
        low = float(dim_spec["low"])
        high = float(dim_spec["high"])
        return scale_from_signed_unit(math.tanh(float(raw_value)), low, high)

    if mode == "sigmoid":
        low = float(dim_spec["low"])
        high = float(dim_spec["high"])
        return scale_from_unit_interval(sigmoid(float(raw_value)), low, high)

    if mode == "clip":
        low = float(dim_spec["low"])
        high = float(dim_spec["high"])
        return clip(float(raw_value), low, high)

    raise ValueError(f"Unknown per-dimension mapping mode: {mode}")



def apply_policy_output_spec(
    raw_action: Any,
    policy_output_spec: Sequence[Dict[str, Any]],
    name: str = "raw_action",
) -> np.ndarray:
    a = _require_dim(raw_action, len(policy_output_spec), name=name)
    mapped = np.empty_like(a, dtype=float)
    for i, dim_spec in enumerate(policy_output_spec):
        mapped[i] = map_dimension_from_spec(float(a[i]), dim_spec)
    return mapped


# ============================================================
# Shared robot constraints
# ============================================================


# --- preserve extra keys (e.g. pre_constraint_*) through constraints ---

def apply_ackermann_command_constraints(
    command: Dict[str, float],
    config: Dict[str, Any],
    prev_command: Optional[Dict[str, float]] = None,
    dt: Optional[float] = None,
) -> Dict[str, float]:
    steering = float(command["steering_angle"])
    speed = float(command["speed"])

    min_steering, max_steering = get_steering_bounds(config)
    min_speed, max_speed = get_speed_bounds(config)

    steering = clip(steering, min_steering, max_steering)
    speed = clip(speed, min_speed, max_speed)

    if prev_command is not None and dt is not None and dt > 0.0:
        prev_steering = float(prev_command["steering_angle"])
        prev_speed = float(prev_command["speed"])

        max_steering_rate = config.get("max_steering_rate")
        if max_steering_rate is not None:
            max_delta_steering = float(max_steering_rate) * dt
            steering = clip(
                steering,
                prev_steering - max_delta_steering,
                prev_steering + max_delta_steering,
            )
            steering = clip(steering, min_steering, max_steering)

        max_acceleration = config.get("max_acceleration")
        if max_acceleration is not None:
            max_delta_speed = float(max_acceleration) * dt
            speed = clip(
                speed,
                prev_speed - max_delta_speed,
                prev_speed + max_delta_speed,
            )
            speed = clip(speed, min_speed, max_speed)

    result = dict(command)  # preserve any extra keys (pre_constraint_*)
    result["steering_angle"] = steering
    result["speed"] = speed
    return result


# ============================================================
# Ackermann geometry helpers
# ============================================================



def steering_from_curvature(curvature: float, wheelbase: float) -> float:
    return math.atan(wheelbase * curvature)



def curvature_from_steering(steering_angle: float, wheelbase: float) -> float:
    return math.tan(steering_angle) / max(wheelbase, 1e-8)


# ============================================================
# Action-space-specific interpretation functions
# ============================================================



def interpret_steer_speed(action: Any, config: Dict[str, Any]) -> Dict[str, float]:
    a = _require_dim(action, 2, name="action")
    return {
        "steering_angle": float(a[0]),
        "speed": float(a[1]),
    }



def interpret_curvature_speed(action: Any, config: Dict[str, Any]) -> Dict[str, float]:
    a = _require_dim(action, 2, name="action")
    return {
        "curvature": float(a[0]),
        "speed": float(a[1]),
    }



def interpret_lookahead_point(action: Any, config: Dict[str, Any]) -> Dict[str, float]:
    a = _require_dim(action, 3, name="action")
    return {
        "lookahead_x": float(a[0]),
        "lookahead_y": float(a[1]),
        "speed": float(a[2]),
    }



def interpret_bezier(action: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    action = [p1_x, p1_y, p2_x, p2_y, speed]

    Endpoint is fixed on the forward axis to keep the primitive local and
    comparable across rollouts. The receding-horizon follower determines the
    immediate steering command from the primitive.
    """
    a = _require_dim(action, 5, name="action")
    end_x = _cfg(config, "bezier_end_x", 4.0)

    return {
        "p0": np.array([0.0, 0.0], dtype=float),
        "p1": np.array([float(a[0]), float(a[1])], dtype=float),
        "p2": np.array([float(a[2]), float(a[3])], dtype=float),
        "p3": np.array([end_x, 0.0], dtype=float),
        "speed": float(a[4]),
    }


# ============================================================
# Representation constraints
# ============================================================



def enforce_lookahead_validity(
    representation: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, float]:
    min_speed, max_speed = get_speed_bounds(config)
    return {
        "lookahead_x": clip(
            float(representation["lookahead_x"]),
            _cfg(config, "lookahead_min_x", 0.5),
            _cfg(config, "lookahead_max_x", 5.0),
        ),
        "lookahead_y": clip(
            float(representation["lookahead_y"]),
            -_cfg(config, "lookahead_max_abs_y", 2.0),
            _cfg(config, "lookahead_max_abs_y", 2.0),
        ),
        "speed": clip(float(representation["speed"]), min_speed, max_speed),
    }



def enforce_bezier_validity(
    representation: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    p0 = np.array(representation["p0"], dtype=float, copy=True)
    p1 = np.array(representation["p1"], dtype=float, copy=True)
    p2 = np.array(representation["p2"], dtype=float, copy=True)
    p3 = np.array(representation["p3"], dtype=float, copy=True)

    min_x = _cfg(config, "bezier_min_x", 0.5)
    max_x = _cfg(config, "bezier_max_x", 5.0)
    max_abs_y = _cfg(config, "bezier_max_abs_y", 2.0)
    min_dx = _cfg(config, "bezier_min_dx", 0.2)

    p1[0] = clip(p1[0], min_x, max_x)
    p2[0] = clip(p2[0], min_x, max_x)

    max_feasible_p1_x = max_x - min_dx
    p1[0] = clip(p1[0], min_x, max_feasible_p1_x)
    p2[0] = clip(max(p2[0], p1[0] + min_dx), p1[0] + min_dx, max_x)

    p1[1] = clip(p1[1], -max_abs_y, max_abs_y)
    p2[1] = clip(p2[1], -max_abs_y, max_abs_y)

    min_speed, max_speed = get_speed_bounds(config)
    speed = clip(float(representation["speed"]), min_speed, max_speed)

    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "speed": speed,
    }


# ============================================================
# Bezier helpers
# ============================================================



def bezier_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        (u ** 3) * p0
        + 3.0 * (u ** 2) * t * p1
        + 3.0 * u * (t ** 2) * p2
        + (t ** 3) * p3
    )



def bezier_first_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        3.0 * (u ** 2) * (p1 - p0)
        + 6.0 * u * t * (p2 - p1)
        + 3.0 * (t ** 2) * (p3 - p2)
    )



def bezier_second_derivative(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
    u = 1.0 - t
    return (
        6.0 * u * (p2 - 2.0 * p1 + p0)
        + 6.0 * t * (p3 - 2.0 * p2 + p1)
    )



def bezier_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> float:
    d1 = bezier_first_derivative(p0, p1, p2, p3, t)
    d2 = bezier_second_derivative(p0, p1, p2, p3, t)

    x1, y1 = d1[0], d1[1]
    x2, y2 = d2[0], d2[1]

    denom = (x1 * x1 + y1 * y1) ** 1.5
    if denom <= 1e-8:
        return 0.0

    return (x1 * y2 - y1 * x2) / denom



def sample_bezier_curve(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    num_samples: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, 1.0, int(num_samples))
    points = np.array([bezier_point(p0, p1, p2, p3, t) for t in ts], dtype=float)
    return ts, points



def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)

    diffs = points[1:] - points[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.zeros(points.shape[0], dtype=float)
    s[1:] = np.cumsum(seg_lengths)
    return s



def find_nearest_point_index(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    dists = np.linalg.norm(points, axis=1)
    return int(np.argmin(dists))



def find_target_point_along_curve(points: np.ndarray, lookahead_distance: float) -> np.ndarray:
    if len(points) == 0:
        return np.array([1.0, 0.0], dtype=float)

    s = cumulative_arc_length(points)
    i0 = find_nearest_point_index(points)
    s0 = s[i0]
    s_target = s0 + lookahead_distance

    i_target = int(np.searchsorted(s, s_target, side="left"))
    i_target = min(max(i_target, i0), len(points) - 1)
    return points[i_target]



def max_abs_curvature_on_prefix(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t_end: float = 0.5,
    num: int = 25,
) -> float:
    t_end = clip(float(t_end), 0.0, 1.0)
    ts = np.linspace(0.0, t_end, int(num))
    vals = [abs(bezier_curvature(p0, p1, p2, p3, float(t))) for t in ts]
    return max(vals) if vals else 0.0


# ============================================================
# Conversion to shared robot command space
# ============================================================



def command_from_steer_speed(representation: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "steering_angle": float(representation["steering_angle"]),
        "speed": float(representation["speed"]),
    }



def command_from_curvature_speed(representation: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    curvature = float(representation["curvature"])
    speed = float(representation["speed"])
    wheelbase = get_wheelbase(config)

    return {
        "steering_angle": steering_from_curvature(curvature, wheelbase),
        "speed": speed,
    }



def command_from_lookahead_point(representation: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
    x = float(representation["lookahead_x"])
    y = float(representation["lookahead_y"])
    speed = float(representation["speed"])
    wheelbase = get_wheelbase(config)

    denom = x * x + y * y
    curvature = 0.0 if denom <= 1e-8 else (2.0 * y / denom)

    min_curvature, max_curvature = get_curvature_bounds(config)
    curvature = clip(curvature, min_curvature, max_curvature)

    return {
        "steering_angle": steering_from_curvature(curvature, wheelbase),
        "speed": speed,
    }



def command_from_bezier(representation: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    p0 = np.array(representation["p0"], dtype=float)
    p1 = np.array(representation["p1"], dtype=float)
    p2 = np.array(representation["p2"], dtype=float)
    p3 = np.array(representation["p3"], dtype=float)
    requested_speed = float(representation["speed"])

    wheelbase = get_wheelbase(config)
    min_speed, max_speed = get_speed_bounds(config)
    min_curvature, max_curvature = get_curvature_bounds(config)

    num_samples = int(config.get("bezier_num_samples", 60))
    lookahead_distance = _cfg(config, "bezier_lookahead_distance", 1.0)

    _, points = sample_bezier_curve(p0, p1, p2, p3, num_samples=num_samples)
    target = find_target_point_along_curve(points, lookahead_distance)

    x = float(target[0])
    y = float(target[1])

    denom = x * x + y * y
    curvature = 0.0 if denom <= 1e-8 else (2.0 * y / denom)
    curvature = clip(curvature, min_curvature, max_curvature)
    steering_angle = steering_from_curvature(curvature, wheelbase)

    speed = clip(requested_speed, min_speed, max_speed)

    return {
        "steering_angle": steering_angle,
        "speed": speed,
    }


# ============================================================
# Metadata helpers
# ============================================================



def default_metadata(
    category: Optional[str] = None,
    debug_plot: bool = False,
    path_like: bool = False,
    paper_role: str = "main",
    inductive_bias_level: int = 0,
) -> Dict[str, Any]:
    return {
        "category": category,
        "debug_plot": bool(debug_plot),
        "path_like": bool(path_like),
        "paper_role": paper_role,
        "inductive_bias_level": inductive_bias_level,
    }


# ============================================================
# Registry schema
# ============================================================


@dataclass(frozen=True)
class ActionSpaceSpec:
    policy_dim: int
    policy_dim_names: List[str]
    units: List[str]
    policy_output_spec: List[Dict[str, Any]]
    interpret: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
    to_command: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    representation_constraints: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None


# ============================================================
# Action space definitions
# ============================================================
# All dimensions use "linear" mode: affine rescaling from [-1, 1]
# to [low, high] with no additional nonlinearity. This is the correct
# pairing with SB3's SAC, which already applies tanh squashing
# internally. Using "tanh" or "sigmoid" here would double-squash,
# compressing the effective action range and creating unequal
# coverage across action spaces.
# ============================================================


def _make_action_spaces() -> Dict[str, ActionSpaceSpec]:
    return {
        "steer_speed": ActionSpaceSpec(
            policy_dim=2,
            policy_dim_names=["steering_angle", "speed"],
            units=["rad", "m/s"],
            policy_output_spec=[
                {"mode": "linear", "low": -0.4189, "high": 0.4189},
                {"mode": "linear", "low": 0.0, "high": 5.0},
            ],
            interpret=interpret_steer_speed,
            to_command=command_from_steer_speed,
            description="Direct steering angle and speed control.",
            metadata=default_metadata(
                category="direct",
                paper_role="main",
                inductive_bias_level=0,
            ),
        ),
        "curvature_speed": ActionSpaceSpec(
            policy_dim=2,
            policy_dim_names=["curvature", "speed"],
            units=["1/m", "m/s"],
            policy_output_spec=[
                {"mode": "linear", "low": -2.0, "high": 2.0},
                {"mode": "linear", "low": 0.0, "high": 5.0},
            ],
            interpret=interpret_curvature_speed,
            to_command=command_from_curvature_speed,
            description="Curvature and speed control aligned with Ackermann geometry.",
            metadata=default_metadata(
                category="kinematic",
                paper_role="main",
                inductive_bias_level=1,
            ),
        ),
        "lookahead_point": ActionSpaceSpec(
            policy_dim=3,
            policy_dim_names=["lookahead_x", "lookahead_y", "speed"],
            units=["m", "m", "m/s"],
            policy_output_spec=[
                {"mode": "linear", "low": 0.5, "high": 5.0},
                {"mode": "linear", "low": -2.0, "high": 2.0},
                {"mode": "linear", "low": 0.0, "high": 5.0},
            ],
            interpret=interpret_lookahead_point,
            representation_constraints=enforce_lookahead_validity,
            to_command=command_from_lookahead_point,
            description="Local target point in the robot frame plus speed.",
            metadata=default_metadata(
                category="geometric",
                debug_plot=True,
                paper_role="main",
                inductive_bias_level=2,
            ),
        ),
        "bezier": ActionSpaceSpec(
            policy_dim=5,
            policy_dim_names=["p1_x", "p1_y", "p2_x", "p2_y", "speed"],
            units=["m", "m", "m", "m", "m/s"],
            policy_output_spec=[
                {"mode": "linear", "low": 0.5, "high": 5.0},
                {"mode": "linear", "low": -2.0, "high": 2.0},
                {"mode": "linear", "low": 0.5, "high": 5.0},
                {"mode": "linear", "low": -2.0, "high": 2.0},
                {"mode": "linear", "low": 0.0, "high": 5.0},
            ],
            interpret=interpret_bezier,
            representation_constraints=enforce_bezier_validity,
            to_command=command_from_bezier,
            description=(
                "Short-horizon cubic Bezier path primitive plus speed, executed via "
                "receding-horizon pure-pursuit-style path following."
            ),
            metadata=default_metadata(
                category="path",
                debug_plot=True,
                path_like=True,
                paper_role="main",
                inductive_bias_level=3,
            ),
        ),
    }


ACTION_SPACES: Dict[str, ActionSpaceSpec] = _make_action_spaces()


# ============================================================
# Registry validation and config-aware refresh
# ============================================================



def validate_action_space_spec(name: str, spec: ActionSpaceSpec) -> None:
    if spec.policy_dim <= 0:
        raise ValueError(f"{name}: policy_dim must be positive.")
    if len(spec.policy_dim_names) != spec.policy_dim:
        raise ValueError(f"{name}: policy_dim_names length must equal policy_dim.")
    if len(spec.units) != spec.policy_dim:
        raise ValueError(f"{name}: units length must equal policy_dim.")
    if len(spec.policy_output_spec) != spec.policy_dim:
        raise ValueError(f"{name}: policy_output_spec length must equal policy_dim.")
    if not callable(spec.interpret):
        raise ValueError(f"{name}: interpret must be callable.")
    if not callable(spec.to_command):
        raise ValueError(f"{name}: to_command must be callable.")
    if spec.representation_constraints is not None and not callable(spec.representation_constraints):
        raise ValueError(f"{name}: representation_constraints must be callable when provided.")



def validate_all_action_spaces() -> None:
    for name, spec in ACTION_SPACES.items():
        validate_action_space_spec(name, spec)



def refresh_action_space_bounds(config: Dict[str, Any]) -> None:
    min_curvature, max_curvature = get_curvature_bounds(config)
    min_speed, max_speed = get_speed_bounds(config)
    min_steering, max_steering = get_steering_bounds(config)

    ACTION_SPACES["steer_speed"] = ActionSpaceSpec(
        **{
            **ACTION_SPACES["steer_speed"].__dict__,
            "policy_output_spec": [
                {"mode": "linear", "low": min_steering, "high": max_steering},
                {"mode": "linear", "low": min_speed, "high": max_speed},
            ],
        }
    )

    ACTION_SPACES["curvature_speed"] = ActionSpaceSpec(
        **{
            **ACTION_SPACES["curvature_speed"].__dict__,
            "policy_output_spec": [
                {"mode": "linear", "low": min_curvature, "high": max_curvature},
                {"mode": "linear", "low": min_speed, "high": max_speed},
            ],
        }
    )

    ACTION_SPACES["lookahead_point"] = ActionSpaceSpec(
        **{
            **ACTION_SPACES["lookahead_point"].__dict__,
            "policy_output_spec": [
                {
                    "mode": "linear",
                    "low": _cfg(config, "lookahead_min_x", 0.5),
                    "high": _cfg(config, "lookahead_max_x", 5.0),
                },
                {
                    "mode": "linear",
                    "low": -_cfg(config, "lookahead_max_abs_y", 2.0),
                    "high": _cfg(config, "lookahead_max_abs_y", 2.0),
                },
                {"mode": "linear", "low": min_speed, "high": max_speed},
            ],
        }
    )

    ACTION_SPACES["bezier"] = ActionSpaceSpec(
        **{
            **ACTION_SPACES["bezier"].__dict__,
            "policy_output_spec": [
                {
                    "mode": "linear",
                    "low": _cfg(config, "bezier_min_x", 0.5),
                    "high": _cfg(config, "bezier_max_x", 5.0),
                },
                {
                    "mode": "linear",
                    "low": -_cfg(config, "bezier_max_abs_y", 2.0),
                    "high": _cfg(config, "bezier_max_abs_y", 2.0),
                },
                {
                    "mode": "linear",
                    "low": _cfg(config, "bezier_min_x", 0.5),
                    "high": _cfg(config, "bezier_max_x", 5.0),
                },
                {
                    "mode": "linear",
                    "low": -_cfg(config, "bezier_max_abs_y", 2.0),
                    "high": _cfg(config, "bezier_max_abs_y", 2.0),
                },
                {"mode": "linear", "low": min_speed, "high": max_speed},
            ],
        }
    )

    validate_all_action_spaces()


validate_all_action_spaces()


# ============================================================
# Registry query helpers
# ============================================================



def get_action_space_names() -> List[str]:
    return list(ACTION_SPACES.keys())



def get_action_space_spec(action_space_name: str) -> ActionSpaceSpec:
    try:
        return ACTION_SPACES[action_space_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown action space '{action_space_name}'. Available: {get_action_space_names()}"
        ) from exc



def get_policy_dim(action_space_name: str) -> int:
    return int(get_action_space_spec(action_space_name).policy_dim)



def get_policy_dim_names(action_space_name: str) -> List[str]:
    return list(get_action_space_spec(action_space_name).policy_dim_names)



def get_action_space_units(action_space_name: str) -> List[str]:
    return list(get_action_space_spec(action_space_name).units)



def get_policy_output_spec(action_space_name: str) -> List[Dict[str, Any]]:
    return list(get_action_space_spec(action_space_name).policy_output_spec)



def get_interpret_function(action_space_name: str):
    return get_action_space_spec(action_space_name).interpret



def get_to_command_function(action_space_name: str):
    return get_action_space_spec(action_space_name).to_command



def get_representation_constraints(action_space_name: str):
    return get_action_space_spec(action_space_name).representation_constraints



def get_action_space_description(action_space_name: str) -> str:
    return get_action_space_spec(action_space_name).description



def get_action_space_metadata(action_space_name: str) -> Dict[str, Any]:
    return dict(get_action_space_spec(action_space_name).metadata)


# ============================================================
# Debug / inspection utilities
# ============================================================



def trace_action_pipeline(
    action_space_name: str,
    raw_action: Any,
    config: Dict[str, Any],
    prev_command: Optional[Dict[str, float]] = None,
    dt: Optional[float] = None,
    apply_final_constraints: bool = True,
) -> Dict[str, Any]:
    mapped_action = raw_action_to_mapped_action(action_space_name, raw_action)
    representation = action_to_representation(action_space_name, mapped_action, config)
    unconstrained_command = representation_to_command(action_space_name, representation, config)

    final_command = dict(unconstrained_command)
    if apply_final_constraints:
        final_command = apply_ackermann_command_constraints(
            final_command,
            config,
            prev_command=prev_command,
            dt=dt,
        )

    return {
        "action_space_name": action_space_name,
        "raw_action": _to_numpy(raw_action),
        "mapped_action": np.array(mapped_action, dtype=float),
        "representation": representation,
        "command_before_final_constraints": unconstrained_command,
        "command_after_final_constraints": final_command,
    }



def sample_command_statistics(
    action_space_name: str,
    config: Dict[str, Any],
    num_samples: int = 2048,
    raw_action_std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    if rng is None:
        rng = np.random.default_rng(0)

    dim = get_policy_dim(action_space_name)
    raw_actions = rng.normal(loc=0.0, scale=raw_action_std, size=(int(num_samples), dim))

    steer_vals: List[float] = []
    speed_vals: List[float] = []
    steer_sat_count = 0
    speed_sat_count = 0

    min_steering, max_steering = get_steering_bounds(config)
    min_speed, max_speed = get_speed_bounds(config)

    for raw_action in raw_actions:
        cmd = raw_action_to_command(action_space_name, raw_action, config)
        steering = float(cmd["steering_angle"])
        speed = float(cmd["speed"])

        steer_vals.append(steering)
        speed_vals.append(speed)

        if abs(steering - min_steering) < 1e-9 or abs(steering - max_steering) < 1e-9:
            steer_sat_count += 1
        if abs(speed - min_speed) < 1e-9 or abs(speed - max_speed) < 1e-9:
            speed_sat_count += 1

    steer_arr = np.asarray(steer_vals, dtype=float)
    speed_arr = np.asarray(speed_vals, dtype=float)

    return {
        "steering_min": float(np.min(steer_arr)),
        "steering_max": float(np.max(steer_arr)),
        "steering_mean": float(np.mean(steer_arr)),
        "steering_std": float(np.std(steer_arr)),
        "speed_min": float(np.min(speed_arr)),
        "speed_max": float(np.max(speed_arr)),
        "speed_mean": float(np.mean(speed_arr)),
        "speed_std": float(np.std(speed_arr)),
        "steering_saturation_fraction": float(steer_sat_count / num_samples),
        "speed_saturation_fraction": float(speed_sat_count / num_samples),
    }


# ============================================================
# Canonical end-to-end pipeline
# ============================================================



def raw_action_to_mapped_action(
    action_space_name: str,
    raw_action: Any,
) -> np.ndarray:
    spec = get_action_space_spec(action_space_name)
    return apply_policy_output_spec(raw_action, spec.policy_output_spec, name="raw_action")



def action_to_representation(
    action_space_name: str,
    action: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    spec = get_action_space_spec(action_space_name)
    representation = spec.interpret(action, config)

    if spec.representation_constraints is not None:
        representation = spec.representation_constraints(representation, config)

    return representation



def representation_to_command(
    action_space_name: str,
    representation: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, float]:
    spec = get_action_space_spec(action_space_name)
    return spec.to_command(representation, config)


# --- save pre-constraint values before applying constraints ---

def raw_action_to_command(
    action_space_name: str,
    raw_action: Any,
    config: Dict[str, Any],
    prev_command: Optional[Dict[str, float]] = None,
    dt: Optional[float] = None,
    apply_final_constraints: bool = True,
) -> Dict[str, float]:
    mapped_action = raw_action_to_mapped_action(action_space_name, raw_action)

    representation = action_to_representation(
        action_space_name,
        mapped_action,
        config,
    )

    command = representation_to_command(
        action_space_name,
        representation,
        config,
    )

    # save pre-constraint values for analysis
    command["pre_constraint_steering"] = command["steering_angle"]
    command["pre_constraint_speed"] = command["speed"]

    if apply_final_constraints:
        command = apply_ackermann_command_constraints(
            command,
            config,
            prev_command=prev_command,
            dt=dt,
        )

    return command



def action_to_command(
    action_space_name: str,
    action: Any,
    config: Dict[str, Any],
    prev_command: Optional[Dict[str, float]] = None,
    dt: Optional[float] = None,
    apply_final_constraints: bool = True,
) -> Dict[str, float]:
    representation = action_to_representation(
        action_space_name,
        action,
        config,
    )

    command = representation_to_command(
        action_space_name,
        representation,
        config,
    )

    # save pre-constraint values for analysis
    command["pre_constraint_steering"] = command["steering_angle"]
    command["pre_constraint_speed"] = command["speed"]

    if apply_final_constraints:
        command = apply_ackermann_command_constraints(
            command,
            config,
            prev_command=prev_command,
            dt=dt,
        )

    return command
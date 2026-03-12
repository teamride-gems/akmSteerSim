import math
import numpy as np


# Shared robot command interface

ROBOT_COMMAND_NAMES = ["steering_angle", "speed"]
ROBOT_COMMAND_UNITS = ["rad", "m/s"]


# Basic helpers

def _to_numpy(x):
    """Convert an input action-like object to a flat numpy array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr



def _require_dim(x, expected_dim, name="vector"):
    """Raise a clear error if a vector does not have the expected size."""
    arr = _to_numpy(x)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"{name} must have dimension {expected_dim}, got {arr.shape[0]}."
        )
    return arr



def scale_from_unit_interval(x, low, high):
    """Scale x from [0, 1] to [low, high]."""
    return low + x * (high - low)



def scale_from_signed_unit(x, low, high):
    """Scale x from [-1, 1] to [low, high]."""
    return low + 0.5 * (x + 1.0) * (high - low)



def clip(value, low, high):
    """Clip value to range."""
    return max(low, min(high, value))


# Shared robot constraints

def apply_ackermann_command_constraints(command, config, prev_command=None, dt=None):
    """
    Apply shared robot-level constraints to the final command.

    Expected config fields (defaults provided if omitted):
        min_steering_angle
        max_steering_angle
        min_speed
        max_speed
        max_steering_rate      (optional)
        max_acceleration       (optional)

    command format:
        {
            "steering_angle": float,
            "speed": float,
        }

    prev_command format is the same. dt is the control timestep in seconds.
    """

    # Extract steering and speed from the command
    steering = float(command["steering_angle"])
    speed = float(command["speed"])

    # Retrieve absolute limits from configuration
    min_steering = float(config.get("min_steering_angle", -0.4189))
    max_steering = float(config.get("max_steering_angle", 0.4189))
    min_speed = float(config.get("min_speed", 0.0))
    max_speed = float(config.get("max_speed", 5.0))

    # Apply absolute bounds
    steering = clip(steering, min_steering, max_steering)
    speed = clip(speed, min_speed, max_speed)

    # Apply rate constraints (optional)
    if prev_command is not None and dt is not None and dt > 0.0:
        # Steering rate constraint
        if "max_steering_rate" in config:
            # Maximum allowed steering change during this timestep
            max_delta_steering = float(config["max_steering_rate"]) * dt

            prev_steering = float(prev_command["steering_angle"])

            # Clamp steering relative to the previous steering value
            steering = clip(
                steering,
                prev_steering - max_delta_steering,
                prev_steering + max_delta_steering,
            )

            # Reapply absolute limits to ensure bounds are still respected
            steering = clip(steering, min_steering, max_steering)

        # Acceleration constraint
        if "max_acceleration" in config:
            # Maximum allowed speed change during this timestep
            max_delta_speed = float(config["max_acceleration"]) * dt

            prev_speed = float(prev_command["speed"])

            # Clamp speed relative to the previous speed value
            speed = clip(
                speed,
                prev_speed - max_delta_speed,
                prev_speed + max_delta_speed,
            )

            # Reapply absolute limits
            speed = clip(speed, min_speed, max_speed)

    # Return constrained command
    return {
        "steering_angle": steering,
        "speed": speed,
    }


# ============================================================
# Action-space-specific interpretation functions
# ============================================================


def interpret_steer_speed(bounded_action, config):
    """
    Directly interpret the bounded action as steering angle and speed.

    Expects bounded_action in [-1, 1]^2 when using tanh.
    """
    a = _require_dim(bounded_action, 2, name="bounded_action")

    steering = scale_from_signed_unit(
        a[0],
        float(config.get("min_steering_angle", -0.4189)),
        float(config.get("max_steering_angle", 0.4189)),
    )
    speed = scale_from_signed_unit(
        a[1],
        float(config.get("min_speed", 0.0)),
        float(config.get("max_speed", 5.0)),
    )

    return {
        "steering_angle": steering,
        "speed": speed,
    }



def interpret_curvature_speed(bounded_action, config):
    """
    Interpret bounded action as curvature and speed.

    curvature has units 1/m.
    """
    a = _require_dim(bounded_action, 2, name="bounded_action")

    max_abs_curvature = float(config.get("max_abs_curvature", 1.0))
    curvature = max_abs_curvature * a[0]
    speed = scale_from_signed_unit(
        a[1],
        float(config.get("min_speed", 0.0)),
        float(config.get("max_speed", 5.0)),
    )

    return {
        "curvature": curvature,
        "speed": speed,
    }



def interpret_lookahead_point(bounded_action, config):
    """
    Interpret bounded action as a lookahead point in the robot frame.

    x is forward distance and y is lateral offset.
    """
    a = _require_dim(bounded_action, 2, name="bounded_action")

    min_x = float(config.get("lookahead_min_x", 0.5))
    max_x = float(config.get("lookahead_max_x", 5.0))
    max_abs_y = float(config.get("lookahead_max_abs_y", 2.0))
    target_speed = scale_from_signed_unit(
        float(config.get("lookahead_default_speed_normalized", 0.0)),
        float(config.get("min_speed", 0.0)),
        float(config.get("max_speed", 5.0)),
    )

    x = scale_from_signed_unit(a[0], min_x, max_x)
    y = max_abs_y * a[1]

    return {
        "lookahead_x": x,
        "lookahead_y": y,
        "speed": target_speed,
    }



def interpret_bezier(bounded_action, config):
    """
    Interpret bounded action as control points for a local cubic Bezier curve.

    Assumes the curve starts at the robot origin p0 = (0, 0), and ends at
    p3 = (end_x, 0). The action specifies interior control points p1 and p2.

    bounded_action = [p1_x, p1_y, p2_x, p2_y]
    """
    a = _require_dim(bounded_action, 4, name="bounded_action")

    min_x = float(config.get("bezier_min_x", 0.5))
    max_x = float(config.get("bezier_max_x", 5.0))
    max_abs_y = float(config.get("bezier_max_abs_y", 2.0))
    end_x = float(config.get("bezier_end_x", 4.0))
    speed = scale_from_signed_unit(
        float(config.get("bezier_default_speed_normalized", 0.0)),
        float(config.get("min_speed", 0.0)),
        float(config.get("max_speed", 5.0)),
    )

    p1_x = scale_from_signed_unit(a[0], min_x, max_x)
    p1_y = max_abs_y * a[1]
    p2_x = scale_from_signed_unit(a[2], min_x, max_x)
    p2_y = max_abs_y * a[3]

    return {
        "p0": np.array([0.0, 0.0], dtype=float),
        "p1": np.array([p1_x, p1_y], dtype=float),
        "p2": np.array([p2_x, p2_y], dtype=float),
        "p3": np.array([end_x, 0.0], dtype=float),
        "speed": speed,
    }


# Representation constraints

def enforce_lookahead_validity(representation, config):
    """
    Enforce basic validity constraints for a lookahead-point representation.

    Mainly ensures that x remains forward and not too close to zero.
    """
    x = float(representation["lookahead_x"])
    y = float(representation["lookahead_y"])
    speed = float(representation["speed"])

    min_x = float(config.get("lookahead_min_x", 0.5))
    max_x = float(config.get("lookahead_max_x", 5.0))
    max_abs_y = float(config.get("lookahead_max_abs_y", 2.0))

    x = clip(x, min_x, max_x)
    y = clip(y, -max_abs_y, max_abs_y)

    return {
        "lookahead_x": x,
        "lookahead_y": y,
        "speed": speed,
    }



def enforce_bezier_validity(representation, config):
    """
    Enforce simple structural validity for a cubic Bezier segment.

    Current rules:
      - p1_x and p2_x are clamped to a forward interval
      - p2_x is forced to be at least p1_x + min_dx
      - y coordinates are clamped

    This avoids backward / badly ordered control points while still allowing
    substantial geometric freedom.
    """
    p0 = np.array(representation["p0"], dtype=float)
    p1 = np.array(representation["p1"], dtype=float)
    p2 = np.array(representation["p2"], dtype=float)
    p3 = np.array(representation["p3"], dtype=float)
    speed = float(representation["speed"])

    min_x = float(config.get("bezier_min_x", 0.5))
    max_x = float(config.get("bezier_max_x", 5.0))
    max_abs_y = float(config.get("bezier_max_abs_y", 2.0))
    min_dx = float(config.get("bezier_min_dx", 0.2))

    p1[0] = clip(p1[0], min_x, max_x)
    p2[0] = clip(p2[0], min_x, max_x)

    if p2[0] < p1[0] + min_dx:
        p2[0] = p1[0] + min_dx

    p2[0] = clip(p2[0], min_x, max_x)
    if p2[0] < p1[0]:
        p1[0] = clip(p2[0] - min_dx, min_x, max_x)

    p1[1] = clip(p1[1], -max_abs_y, max_abs_y)
    p2[1] = clip(p2[1], -max_abs_y, max_abs_y)

    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "speed": speed,
    }


# Conversion to shared robot command space

def command_from_steer_speed(representation, config):
    """Direct pass-through into the shared robot command space."""
    return {
        "steering_angle": float(representation["steering_angle"]),
        "speed": float(representation["speed"]),
    }



def command_from_curvature_speed(representation, config):
    """
    Convert curvature and speed into steering angle and speed.

    Uses the Ackermann bicycle relation:
        delta = arctan(L * curvature)
    """
    curvature = float(representation["curvature"])
    speed = float(representation["speed"])
    wheelbase = float(config.get("wheelbase", 0.33))

    steering_angle = math.atan(wheelbase * curvature)

    return {
        "steering_angle": steering_angle,
        "speed": speed,
    }



def command_from_lookahead_point(representation, config):
    """
    Convert a lookahead point in the robot frame into steering and speed.

    Uses a pure-pursuit-like relation:
        curvature = 2y / (x^2 + y^2)
        delta = arctan(L * curvature)
    """
    x = float(representation["lookahead_x"])
    y = float(representation["lookahead_y"])
    speed = float(representation["speed"])
    wheelbase = float(config.get("wheelbase", 0.33))

    denom = x * x + y * y
    if denom <= 1e-8:
        curvature = 0.0
    else:
        curvature = 2.0 * y / denom

    steering_angle = math.atan(wheelbase * curvature)

    return {
        "steering_angle": steering_angle,
        "speed": speed,
    }



def bezier_point(p0, p1, p2, p3, t):
    """Evaluate a cubic Bezier curve at parameter t in [0, 1]."""
    u = 1.0 - t
    return (
        (u ** 3) * p0
        + 3.0 * (u ** 2) * t * p1
        + 3.0 * u * (t ** 2) * p2
        + (t ** 3) * p3
    )



def bezier_first_derivative(p0, p1, p2, p3, t):
    """Evaluate the first derivative of a cubic Bezier curve."""
    u = 1.0 - t
    return (
        3.0 * (u ** 2) * (p1 - p0)
        + 6.0 * u * t * (p2 - p1)
        + 3.0 * (t ** 2) * (p3 - p2)
    )



def bezier_second_derivative(p0, p1, p2, p3, t):
    """Evaluate the second derivative of a cubic Bezier curve."""
    u = 1.0 - t
    return (
        6.0 * u * (p2 - 2.0 * p1 + p0)
        + 6.0 * t * (p3 - 2.0 * p2 + p1)
    )



def bezier_curvature(p0, p1, p2, p3, t):
    """
    Compute planar curvature of a cubic Bezier curve at t.

    curvature = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    """
    d1 = bezier_first_derivative(p0, p1, p2, p3, t)
    d2 = bezier_second_derivative(p0, p1, p2, p3, t)

    x1, y1 = d1[0], d1[1]
    x2, y2 = d2[0], d2[1]

    denom = (x1 * x1 + y1 * y1) ** 1.5
    if denom <= 1e-8:
        return 0.0

    return (x1 * y2 - y1 * x2) / denom



def command_from_bezier(representation, config):
    """
    Convert a local Bezier segment into steering and speed.

    The intended usage is receding-horizon: the robot tracks only the initial
    portion of the curve and then replans. Here we approximate this by using
    the curvature near the beginning of the curve at parameter t_track.
    """
    p0 = np.array(representation["p0"], dtype=float)
    p1 = np.array(representation["p1"], dtype=float)
    p2 = np.array(representation["p2"], dtype=float)
    p3 = np.array(representation["p3"], dtype=float)
    speed = float(representation["speed"])

    wheelbase = float(config.get("wheelbase", 0.33))
    t_track = float(config.get("bezier_track_t", 0.05))
    t_track = clip(t_track, 0.0, 1.0)

    curvature = bezier_curvature(p0, p1, p2, p3, t_track)
    steering_angle = math.atan(wheelbase * curvature)

    return {
        "steering_angle": steering_angle,
        "speed": speed,
    }


# Metadata helpers


def default_metadata(category=None, debug_plot=False, path_like=False):
    return {
        "category": category,
        "debug_plot": bool(debug_plot),
        "path_like": bool(path_like),
    }


# Action space registry

ACTION_SPACES = {
    "steer_speed": {
        "policy_dim": 2,
        "policy_dim_names": ["steering_angle", "speed"],
        "units": ["rad", "m/s"],
        "bounds_mode": "tanh",
        "interpret": interpret_steer_speed,
        "to_command": command_from_steer_speed,
        "description": "Direct steering angle and speed control.",
        "metadata": default_metadata(category="direct", debug_plot=False, path_like=False),
    },
    "curvature_speed": {
        "policy_dim": 2,
        "policy_dim_names": ["curvature", "speed"],
        "units": ["1/m", "m/s"],
        "bounds_mode": "tanh",
        "interpret": interpret_curvature_speed,
        "to_command": command_from_curvature_speed,
        "description": "Curvature and speed control converted to steering angle and speed.",
        "metadata": default_metadata(category="kinematic", debug_plot=False, path_like=False),
    },
    "lookahead_point": {
        "policy_dim": 2,
        "policy_dim_names": ["lookahead_x", "lookahead_y"],
        "units": ["m", "m"],
        "bounds_mode": "tanh",
        "interpret": interpret_lookahead_point,
        "representation_constraints": enforce_lookahead_validity,
        "to_command": command_from_lookahead_point,
        "description": "Lookahead point in the robot frame converted to steering angle and speed.",
        "metadata": default_metadata(category="geometric", debug_plot=True, path_like=False),
    },
    "bezier": {
        "policy_dim": 4,
        "policy_dim_names": ["p1_x", "p1_y", "p2_x", "p2_y"],
        "units": ["m", "m", "m", "m"],
        "bounds_mode": "tanh",
        "interpret": interpret_bezier,
        "representation_constraints": enforce_bezier_validity,
        "to_command": command_from_bezier,
        "description": "Local cubic Bezier segment converted into steering angle and speed using its initial tracked portion.",
        "metadata": default_metadata(category="geometric", debug_plot=True, path_like=True),
    },
}


# Registry query helpers


def get_action_space_names():
    return list(ACTION_SPACES.keys())

def get_action_space_spec(action_space_name):
    """
    Return the full specification dictionary for an action space.

    This is the lowest-level accessor used internally by all
    other helper functions.
    """
    if action_space_name not in ACTION_SPACES:
        raise KeyError(
            f"Unknown action space '{action_space_name}'. Available: {get_action_space_names()}"
        )

    return ACTION_SPACES[action_space_name]


def get_policy_dim(action_space_name):
    """Return the dimensionality of the policy output."""
    return int(get_action_space_spec(action_space_name)["policy_dim"])



def get_policy_dim_names(action_space_name):
    """Return human-readable names for the policy output dimensions."""
    return list(get_action_space_spec(action_space_name)["policy_dim_names"])



def get_action_space_units(action_space_name):
    """Return the physical or geometric units for each action dimension."""
    return list(get_action_space_spec(action_space_name)["units"])



def get_bounds_mode(action_space_name):
    """Return the bounding mode used for raw policy outputs."""
    return get_action_space_spec(action_space_name).get("bounds_mode", "tanh")



def get_interpret_function(action_space_name):
    """Return the interpretation function for the action space."""
    return get_action_space_spec(action_space_name)["interpret"]



def get_to_command_function(action_space_name):
    """Return the function that converts representations to robot commands."""
    return get_action_space_spec(action_space_name)["to_command"]



def get_representation_constraints(action_space_name):
    """
    Return the representation constraint function if it exists.

    Some action spaces require additional structural constraints
    (for example Bézier ordering or lookahead positivity).
    """
    return get_action_space_spec(action_space_name).get("representation_constraints", None)



def get_action_space_description(action_space_name):
    """Return the textual description of the action space."""
    return get_action_space_spec(action_space_name).get("description", "")



def get_action_space_metadata(action_space_name):
    """Return optional metadata associated with the action space."""
    return dict(get_action_space_spec(action_space_name).get("metadata", {}))

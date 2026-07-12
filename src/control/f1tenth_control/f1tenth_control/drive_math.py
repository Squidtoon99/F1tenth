"""Pure action -> Ackermann mapping (no ROS imports, easily unit-tested).

Mirrors ``f1tenth_env/env.py`` ``_apply_actions``: throttle scales speed by
``max_speed`` and steering scales the center steering angle by ``max_steer``.
Negative throttle is a brake; ``brake_behavior`` controls whether that means a hard
stop (speed 0) or reverse (negative speed).

Force mode maps throttle to a normalized longitudinal command in [-1, 1] carried
on ``AckermannDrive.acceleration`` (see ADR 0005); negative values mean brake.
"""

from __future__ import annotations


def map_action_to_drive(
    throttle: float,
    steering: float,
    max_speed: float,
    max_steer: float,
    clip_actions: float = 1.0,
    brake_behavior: str = "stop",
) -> tuple[float, float]:
    """Return ``(speed_mps, steering_angle_rad)`` for an AckermannDrive command."""
    throttle = max(-clip_actions, min(clip_actions, float(throttle)))
    steering = max(-clip_actions, min(clip_actions, float(steering)))

    if brake_behavior == "reverse":
        speed = throttle * max_speed
    else:  # "stop": negative throttle commands a stop, not reverse
        speed = max(throttle, 0.0) * max_speed

    steering_angle = steering * max_steer
    return speed, steering_angle


def map_action_to_force(
    throttle: float,
    steering: float,
    max_steer: float,
    clip_actions: float = 1.0,
) -> tuple[float, float]:
    """Return ``(longitudinal_cmd, steering_angle_rad)``.

    ``longitudinal_cmd`` is normalized in ``[-clip_actions, clip_actions]``:
    positive = drive force, negative = brake force, zero = coast.
    """
    throttle = max(-clip_actions, min(clip_actions, float(throttle)))
    steering = max(-clip_actions, min(clip_actions, float(steering)))
    return throttle, steering * max_steer


def force_to_motor_currents(
    longitudinal_cmd: float,
    i_drive_max_a: float,
    i_brake_max_a: float,
) -> tuple[float, float]:
    """Map normalized longitudinal command to ``(i_drive, i_brake)`` amps.

    Mutual exclusion: at most one of drive/brake is nonzero.
    Non-finite input maps to coast (0, 0).
    """
    if not math_isfinite(longitudinal_cmd):
        return 0.0, 0.0
    cmd = float(longitudinal_cmd)
    if cmd > 0.0:
        return min(cmd, 1.0) * float(i_drive_max_a), 0.0
    if cmd < 0.0:
        return 0.0, min(-cmd, 1.0) * float(i_brake_max_a)
    return 0.0, 0.0


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def lag_alpha(control_dt: float, t_delta: float) -> float:
    """First-order lag blend factor matching ``F1tenthEnv`` steer dynamics.

    ``steer_lag_alpha = control_dt / (t_delta + control_dt)`` with training
    defaults (10 Hz, ``t_delta=0.1``) this is 0.5 per control step.
    """
    t_delta = max(float(t_delta), 1e-9)
    control_dt = max(float(control_dt), 1e-9)
    return control_dt / (t_delta + control_dt)


def step_first_order_lag(state: float, target: float, alpha: float) -> float:
    """Advance a scalar first-order lag toward ``target``."""
    alpha = max(0.0, min(1.0, float(alpha)))
    return state + alpha * (target - state)

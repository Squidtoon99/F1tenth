"""Map normalized policy actions to calibrated physical actuator fields."""

from __future__ import annotations

import math

from f1tenth_policy import integrate_steering_delta

__all__ = [
    "integrate_steering_delta",
    "normalized_action_to_physical",
]


def _map_action_to_force(
    throttle: float,
    steering: float,
    max_steer: float,
    clip_actions: float = 1.0,
) -> tuple[float, float]:
    throttle = max(-clip_actions, min(clip_actions, float(throttle)))
    steering = max(-clip_actions, min(clip_actions, float(steering)))
    return throttle, steering * max_steer


def _force_to_motor_currents(
    longitudinal_cmd: float,
    i_drive_max_a: float,
    i_brake_max_a: float,
) -> tuple[float, float]:
    if not math.isfinite(longitudinal_cmd):
        return 0.0, 0.0
    cmd = float(longitudinal_cmd)
    if cmd > 0.0:
        return min(cmd, 1.0) * float(i_drive_max_a), 0.0
    if cmd < 0.0:
        return 0.0, min(-cmd, 1.0) * float(i_brake_max_a)
    return 0.0, 0.0


def normalized_action_to_physical(
    longitudinal: float,
    steering: float,
    *,
    i_drive_max_a: float,
    i_brake_max_a: float,
    max_steer: float,
    steering_angle_to_servo_gain: float,
    steering_angle_to_servo_offset: float,
    clip_actions: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Return ``(drive_a, brake_a, servo, long_norm, steer_norm)``."""
    if not math.isfinite(longitudinal) or not math.isfinite(steering):
        return 0.0, 0.0, steering_angle_to_servo_offset, 0.0, 0.0
    long_norm, steer_rad = _map_action_to_force(
        longitudinal, steering, max_steer, clip_actions=clip_actions
    )
    drive_a, brake_a = _force_to_motor_currents(
        long_norm, i_drive_max_a, i_brake_max_a
    )
    servo = steering_angle_to_servo_gain * steer_rad + steering_angle_to_servo_offset
    steer_norm = steer_rad / max_steer if max_steer > 0.0 else 0.0
    if not all(
        math.isfinite(v)
        for v in (drive_a, brake_a, servo, long_norm, steer_norm)
    ):
        return 0.0, 0.0, steering_angle_to_servo_offset, 0.0, 0.0
    return drive_a, brake_a, servo, long_norm, steer_norm

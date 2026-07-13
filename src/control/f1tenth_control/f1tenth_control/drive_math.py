"""Pure action -> Ackermann / VESC current mapping (no ROS imports).

Longitudinal action is always force/brake effort (ADR 0006):

* ``throttle > 0`` — drive effort → motor current
* ``throttle = 0`` — coast
* ``throttle < 0`` — brake effort → brake current

The mux carrier is ``AckermannDrive.acceleration``; ``vesc_actuator`` maps that
to ``/commands/motor/current`` and ``/commands/motor/brake``.
"""

from __future__ import annotations


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

    ``steer_lag_alpha = control_dt / (t_delta + control_dt)``.
    """
    t_delta = max(float(t_delta), 1e-9)
    control_dt = max(float(control_dt), 1e-9)
    return control_dt / (t_delta + control_dt)


def step_first_order_lag(state: float, target: float, alpha: float) -> float:
    """Advance a scalar first-order lag toward ``target``."""
    alpha = max(0.0, min(1.0, float(alpha)))
    return state + alpha * (target - state)

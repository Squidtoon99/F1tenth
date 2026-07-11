"""Throttle/brake -> per-wheel axle torque (Nm), wheel order [LR, RR, LF, RF].

Two throttle interpretations, selected by ``params.throttle_mode``:

- ``"speed"`` (VESC-style, default): throttle is a normalized *speed command*
  ``v_cmd = throttle * max_speed`` -- the exact mapping the deployed car uses in
  ``drive_math.py`` -- tracked by an acceleration-limited proportional speed loop,
  with the drive force capped by a brushless-motor torque limit (``Kt * I_max``)
  and the traction cap. Being closed-loop and self-regulating, this is the stable,
  reliable default and matches the deployed action semantics.
- ``"force"`` (alternative): throttle in [-1, 1] maps to an open-loop drive force
  capped by ``power_max`` and a friction traction cap, split AWD by
  ``k_drive_front``; negative throttle brakes.

Returned torque is the net axle torque per wheel (drive minus brake / rolling
resistance) fed into the wheel-spin ODE alongside the tyre longitudinal reaction.
"""

from __future__ import annotations

import torch


def _split_drive(params, f_drive: torch.Tensor) -> torch.Tensor:
    r = params.wheel_radius
    kf = params.k_drive_front
    f_front = kf * f_drive
    f_rear = (1.0 - kf) * f_drive
    tau_rear = f_rear * 0.5 * r
    tau_front = f_front * 0.5 * r
    sign = params.drive_torque_sign
    return sign * torch.stack([tau_rear, tau_rear, tau_front, tau_front], dim=-1)


def _traction_cap(params, mu: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    return mu * mass * params.gravity


def _force_mode_drive(
    params, throttle: torch.Tensor, v_mag: torch.Tensor,
    mu: torch.Tensor, mass: torch.Tensor,
) -> torch.Tensor:
    f = throttle * params.f_drive_max
    f = torch.minimum(f, params.power_max / v_mag.clamp_min(params.v_eps))
    return torch.minimum(f, _traction_cap(params, mu, mass))


def _speed_mode_drive(
    params, throttle: torch.Tensor, v_long: torch.Tensor,
    mu: torch.Tensor, mass: torch.Tensor,
) -> torch.Tensor:
    v_cmd = throttle * params.max_speed
    err = v_cmd - v_long
    a_des = (params.vesc_kp * err / mass).clamp(
        -params.vesc_accel_limit, params.vesc_accel_limit
    )
    f = mass * a_des
    motor_cap = params.motor_kt * params.motor_i_max / max(params.wheel_radius, 1e-6)
    f = f.clamp(-motor_cap, motor_cap)
    trac = _traction_cap(params, mu, mass)
    return f.clamp(-trac, trac)


def wheel_axle_torques(
    params,
    throttle_cmd: torch.Tensor,
    v_long: torch.Tensor,
    omega: torch.Tensor,
    mu: torch.Tensor,
    mass: torch.Tensor,
    drive_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Net per-wheel axle torque (N,4) for the given throttle command.

    ``drive_scale`` (N,) is an optional per-env multiplier on the drive force
    (domain randomization of motor/gearing strength); brakes are left unscaled.
    """
    r = params.wheel_radius
    v_mag = v_long.abs()
    ds = None if drive_scale is None else drive_scale.unsqueeze(1)

    if params.throttle_mode == "speed":
        f_drive = _speed_mode_drive(params, throttle_cmd, v_long, mu, mass)
        if drive_scale is not None:
            f_drive = f_drive * drive_scale
        tau = _split_drive(params, f_drive)
        # Speed mode already produces signed force (drive + regen/brake), so no
        # separate friction-brake stage; add light coast rolling resistance only.
        if params.c_roll > 0.0:
            roll = _coast_roll(params, throttle_cmd.abs() < 1e-3, omega)
            tau = tau + roll
        return tau

    throttle = throttle_cmd.clamp_min(0.0)
    brake = (-throttle_cmd).clamp_min(0.0)
    f_drive = _force_mode_drive(params, throttle, v_mag, mu, mass)
    tau_drive = _split_drive(params, f_drive)
    if ds is not None:
        tau_drive = tau_drive * ds

    brake_sign = torch.sign(omega)
    brake_sign = torch.where(brake_sign == 0, torch.ones_like(brake_sign), brake_sign)
    f_brake = brake * params.f_brake_max
    tau_brake = -brake_sign * (f_brake * 0.25 * r).unsqueeze(1)

    drive_mask = (throttle > 1e-3).unsqueeze(1)
    brake_mask = (brake > 1e-3).unsqueeze(1)
    coast_mask = ~(drive_mask | brake_mask)

    tau = torch.zeros_like(tau_drive)
    tau = torch.where(drive_mask, tau_drive, tau)
    tau = torch.where(brake_mask, tau_brake, tau)
    if params.c_roll > 0.0:
        tau = torch.where(coast_mask, _coast_roll(params, coast_mask[:, 0], omega), tau)
    return tau


def _coast_roll(params, coast: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    roll_sign = torch.sign(omega)
    roll_sign = torch.where(roll_sign == 0, torch.ones_like(roll_sign), roll_sign)
    return -roll_sign * params.c_roll

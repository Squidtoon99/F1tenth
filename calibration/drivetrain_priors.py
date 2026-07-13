"""Drivetrain priors and seed force-per-amp for Velineon 3500 + VESC 6."""

from __future__ import annotations

# Traxxas Velineon 3500 (sensorless 4-pole BLDC): Kt ≈ 60/(2π·Kv).
VELINEON_3500_KV = 3500.0
VELINEON_KT_NM_PER_A = 60.0 / (2.0 * 3.141592653589793 * VELINEON_3500_KV)  # ≈ 0.00273

# Stock Slash 4X4 VXL priors — verify on the race car before trusting.
STOCK_PINION = 13
STOCK_SPUR = 54
STOCK_DIFF_PINION = 13
STOCK_DIFF_SPUR = 37
STOCK_GEAR_RATIO = (STOCK_SPUR / STOCK_PINION) * (STOCK_DIFF_SPUR / STOCK_DIFF_PINION)
# ≈ 11.82:1

# TRAMPA VESC 6 MKV controller limits (not operating setpoints).
VESC6_CONT_A = 80.0
VESC6_PEAK_A = 120.0

# Motor published continuous/burst (upper-bound references only).
MOTOR_CONT_A_REF = 65.0
MOTOR_BURST_A_REF = 100.0


def seed_force_per_amp(
    *,
    kt_nm_per_a: float = VELINEON_KT_NM_PER_A,
    gear_ratio: float = STOCK_GEAR_RATIO,
    efficiency: float = 0.85,
    wheel_radius_m: float = 0.05,
) -> float:
    """Wheel longitudinal force per ampere of motor phase current.

    ``F = I_phase × Kt × gear_ratio × efficiency / wheel_radius``.
    """
    if wheel_radius_m <= 0.0:
        raise ValueError("wheel_radius_m must be > 0")
    return float(kt_nm_per_a * gear_ratio * efficiency / wheel_radius_m)


def seed_force_envelope(
    *,
    i_drive_max_a: float,
    i_brake_max_a: float,
    force_per_amp: float | None = None,
) -> dict:
    """Map current limits to simulator ``f_drive_max`` / ``f_brake_max`` seeds."""
    fpa = seed_force_per_amp() if force_per_amp is None else float(force_per_amp)
    return {
        "force_per_amp": fpa,
        "f_drive_max": fpa * float(i_drive_max_a),
        "f_brake_max": fpa * float(i_brake_max_a),
        "kt_nm_per_a": VELINEON_KT_NM_PER_A,
        "gear_ratio": STOCK_GEAR_RATIO,
        "note": "seed only; replace with fit_current_to_force / fit_brake_current",
    }

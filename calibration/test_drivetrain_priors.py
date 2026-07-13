"""Unit tests for Velineon / gearing seed force helpers."""

from drivetrain_priors import (
    STOCK_GEAR_RATIO,
    VELINEON_KT_NM_PER_A,
    seed_force_envelope,
    seed_force_per_amp,
)


def test_kt_near_expected():
    assert abs(VELINEON_KT_NM_PER_A - 0.002728) < 1e-5


def test_stock_gear_ratio():
    assert abs(STOCK_GEAR_RATIO - 11.822485) < 1e-3


def test_seed_force_per_amp_positive():
    fpa = seed_force_per_amp()
    assert fpa > 0.4
    assert fpa < 0.8


def test_seed_envelope_scales_with_current():
    env = seed_force_envelope(i_drive_max_a=10.0, i_brake_max_a=20.0)
    assert abs(env["f_drive_max"] - 10.0 * env["force_per_amp"]) < 1e-9
    assert abs(env["f_brake_max"] - 20.0 * env["force_per_amp"]) < 1e-9

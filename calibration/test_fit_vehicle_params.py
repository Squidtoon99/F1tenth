"""Synthetic-data unit tests for calibration fits (no ROS, no bags required)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from bag_io import linear_fit, resample  # noqa: E402
from fit_imu import G, fit_imu_static  # noqa: E402
from fit_longitudinal import fit_speed_to_erpm, fit_tire_mu_lower_bound  # noqa: E402
from fit_steering import fit_steer_lag  # noqa: E402


def test_linear_fit_exact():
    x = np.linspace(-2.0, 2.0, 50)
    y = 4300.0 * x + 10.0
    slope, intercept, r2 = linear_fit(x, y)
    assert math.isclose(slope, 4300.0, rel_tol=1e-9)
    assert math.isclose(intercept, 10.0, abs_tol=1e-8)
    assert r2 > 0.999


def test_resample_identity():
    t = np.linspace(0.0, 1.0, 11)
    series = np.column_stack([t, t * 2.0])
    out = resample(series, t)
    assert np.allclose(out[:, 1], series[:, 1])


def test_imu_static_si_units():
    n = 200
    t = np.linspace(0.0, 4.0, n)
    imu = np.column_stack(
        [
            t,
            np.full(n, 0.01),
            np.full(n, -0.02),
            np.full(n, G),
            np.zeros(n),
            np.zeros(n),
            np.full(n, 0.001),
        ]
    )
    fit = fit_imu_static(imu)
    assert fit["status"] in ("identified", "weakly_identified")
    assert fit["accel_unit"] == "m_s2"
    assert math.isclose(fit["accel_scale"], 1.0)
    assert fit["gravity_error_ms2"] < 0.5


def test_imu_static_g_and_deg():
    n = 200
    t = np.linspace(0.0, 4.0, n)
    imu = np.column_stack(
        [
            t,
            np.full(n, -0.02),
            np.full(n, 0.07),
            np.full(n, 1.0),  # 1 g
            np.zeros(n),
            np.zeros(n),
            np.full(n, -0.235),  # deg/s-ish bias
        ]
    )
    # Inflate gyro noise to trigger deg/s detection.
    imu[:, 6] += np.random.default_rng(0).normal(0.0, 0.8, n)
    fit = fit_imu_static(imu)
    assert fit["accel_unit"] == "g"
    assert math.isclose(fit["accel_scale"], G, rel_tol=1e-6)
    assert fit["gyro_unit"] == "deg_s"


def test_speed_to_erpm_recovery():
    t = np.linspace(0.0, 10.0, 500)
    speed = np.where(t < 5.0, 2.0, 4.0)
    erpm = 4300.0 * speed + 5.0
    ack = np.column_stack([t, speed, np.zeros_like(t), np.zeros_like(t)])
    motor = np.column_stack([t, erpm])
    fit = fit_speed_to_erpm(ack, motor)
    assert fit["status"] == "identified"
    assert abs(fit["speed_to_erpm_gain"] - 4300.0) < 1.0
    assert abs(fit["speed_to_erpm_offset"] - 5.0) < 1.0


def test_tire_mu_fit_rejects_short_imu_spike():
    t = np.arange(0.0, 10.0, 0.02)
    n = t.size
    static = np.column_stack(
        [
            t,
            np.full(n, -0.02),
            np.full(n, 0.07),
            np.ones(n),
            np.zeros((n, 3)),
        ]
    )
    imu = static.copy()
    imu[:, 1] += 0.6
    imu[20::40, 1] += 4.0
    odom = np.column_stack([t, np.full(n, 2.0), np.zeros(n)])

    fit = fit_tire_mu_lower_bound(imu, odom, static)

    assert fit["status"] == "weakly_identified"
    assert 0.55 < fit["tire_friction_lower_bound"] < 0.7
    assert fit["a_horiz_raw_p99"] > fit["a_horiz_p99"]


def test_steer_lag_recovery():
    rng = np.random.default_rng(0)
    control_dt = 0.1
    t_delta_true = 0.1
    alpha = control_dt / (t_delta_true + control_dt)
    t = np.arange(0.0, 20.0, 0.02)
    # Square-ish steer command.
    steer = np.where((t % 4.0) < 2.0, 0.3, -0.3)
    filt = np.zeros_like(steer)
    for i in range(1, len(steer)):
        filt[i] = filt[i - 1] + alpha * (steer[i] - filt[i - 1])
    # Yaw rate roughly proportional to filtered steer.
    yaw = 3.0 * filt + rng.normal(0.0, 0.01, size=steer.shape)
    ack = np.column_stack([t, np.ones_like(t), steer, np.zeros_like(t)])
    odom = np.column_stack([t, np.full_like(t, 2.0), yaw])
    fit = fit_steer_lag(ack, odom, control_dt=control_dt)
    assert fit["status"] == "weakly_identified"
    assert abs(fit["t_delta"] - t_delta_true) < 0.08

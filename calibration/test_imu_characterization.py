"""Synthetic tests for IMU bias/noise characterization (no rosbags on disk)."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from bag_io import G, build_typestore  # noqa: E402
from characterize_imu import main as characterize_main  # noqa: E402
from imu_characterization import (  # noqa: E402
    MissingChannelsError,
    calibration_from_static_fit,
    characterize_imu_bag,
    compare_actor_streams,
    convert_driver_imu,
    correlate_with_core,
    load_imu_calibration,
    per_axis_stats,
    replay_policy_imu,
    write_characterization_report,
)
from fit_imu import fit_imu_static  # noqa: E402


def _static_imu(
    *,
    n: int = 400,
    duration_s: float = 20.0,
    ax: float = -0.02,
    ay: float = 0.07,
    az: float = 1.0,
    gz: float = -0.235,
    noise: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, duration_s, n)
    imu = np.column_stack(
        [
            t,
            np.full(n, ax) + rng.normal(0.0, noise, n),
            np.full(n, ay) + rng.normal(0.0, noise, n),
            np.full(n, az) + rng.normal(0.0, noise, n),
            rng.normal(0.0, noise, n),
            rng.normal(0.0, noise, n),
            np.full(n, gz) + rng.normal(0.0, noise, n),
        ]
    )
    return imu


def _empty_series() -> dict[str, np.ndarray]:
    z7 = np.zeros((0, 7), dtype=float)
    z3 = np.zeros((0, 3), dtype=float)
    z17 = np.zeros((0, 17), dtype=float)
    return {
        "/odom": z3,
        "/sensors/imu/raw": z7,
        "/sensor_policy/imu_raw_record": z7,
        "/sensor_policy/imu_actor_record": z7,
        "/pf/pose/odom": np.zeros((0, 4), dtype=float),
        "/ackermann_cmd": np.zeros((0, 4), dtype=float),
        "/teleop": np.zeros((0, 4), dtype=float),
        "/drive": np.zeros((0, 4), dtype=float),
        "/commands/motor/speed": np.zeros((0, 2), dtype=float),
        "/commands/motor/current": np.zeros((0, 2), dtype=float),
        "/commands/motor/brake": np.zeros((0, 2), dtype=float),
        "/commands/servo/position": np.zeros((0, 2), dtype=float),
        "/sensors/servo_position_command": np.zeros((0, 2), dtype=float),
        "/sensors/core": z17,
    }


def test_per_axis_stats_reports_drift_and_outliers():
    t = np.linspace(0.0, 10.0, 500)
    values = 0.05 * t + np.random.default_rng(1).normal(0.0, 0.02, t.size)
    gyro = np.zeros((t.size, 3))
    imu = np.column_stack([t, values, values * 0.5, np.full(t.size, G), gyro])
    stats = per_axis_stats(imu)
    assert stats["status"] == "ok"
    assert stats["sample_count"] == t.size
    assert stats["rate_hz"] > 40.0
    ax = stats["channels"]["ax"]
    assert math.isclose(ax["mean"], float(np.mean(values)), rel_tol=0.05)
    assert ax["drift_slope_per_s"] > 0.0
    assert ax["outlier_count"] >= 0


def test_missing_imu_raises_clear_error():
    series = _empty_series()
    with pytest.raises(MissingChannelsError) as exc:
        characterize_imu_bag(series)
    assert "/sensors/imu/raw" in exc.value.missing


def test_require_core_fails_when_absent():
    series = _empty_series()
    series["/sensors/imu/raw"] = _static_imu(n=200)
    with pytest.raises(MissingChannelsError) as exc:
        characterize_imu_bag(series, require_core=True)
    assert "/sensors/core" in exc.value.missing


def test_characterize_static_bag_proposes_params():
    raw = _static_imu()
    series = _empty_series()
    series["/sensors/imu/raw"] = raw
    result = characterize_imu_bag(series, condition="static_on")
    assert result["condition"] == "static_on"
    assert result["static_fit"]["status"] in ("identified", "weakly_identified")
    assert result["converted_si"]["status"] == "ok"
    assert result["replayed_actor"]["status"] == "ok"
    assert result["telemetry_correlation"]["status"] == "absent"
    proposed = result["proposed_sensor_policy_params"]
    assert "imu_ax_bias" in proposed
    assert "imu_accel_to_ms2" in proposed
    # Do not null gravity in converted SI / deploy params.
    assert proposed["imu_az_bias"] == 0.0
    az_mean = result["converted_si"]["channels"]["az"]["mean"]
    assert abs(az_mean - G) < 0.5


def test_replay_policy_imu_freezes_actor_channels():
    raw = _static_imu(n=800, duration_s=40.0)
    fit = fit_imu_static(raw)
    cal = calibration_from_static_fit(fit)
    replayed = replay_policy_imu(raw, cal, control_hz=20.0)
    assert replayed.shape[0] > 100
    az = replayed[:, 3]
    gx = replayed[:, 4]
    gy = replayed[:, 5]
    assert np.allclose(az, G, atol=1e-4)
    assert np.allclose(gx, 0.0, atol=1e-6)
    assert np.allclose(gy, 0.0, atol=1e-6)


def test_recorded_vs_replay_parity_on_synthetic():
    raw = _static_imu(n=600, duration_s=30.0)
    fit = fit_imu_static(raw)
    cal = calibration_from_static_fit(fit)
    replayed = replay_policy_imu(raw, cal)
    series = _empty_series()
    series["/sensors/imu/raw"] = raw
    series["/sensor_policy/imu_actor_record"] = replayed
    result = characterize_imu_bag(series, cal=cal)
    cmp = result["recorded_vs_replay"]
    assert cmp is not None
    assert cmp["max_abs_diff"] < 1e-4


def test_correlate_with_core_detects_current_coupling():
    raw = _static_imu(n=400, duration_s=20.0, noise=0.005)
    fit = fit_imu_static(raw)
    cal = calibration_from_static_fit(fit)
    converted = convert_driver_imu(raw, cal).copy()
    t = converted[:, 0]
    current = 2.0 + 3.0 * np.sin(2.0 * np.pi * 0.5 * t)
    rng = np.random.default_rng(2)
    # Rolling std of ax tracks |current| when vibration noise scales with load.
    converted[:, 1] += rng.normal(0.0, 1.0, t.size) * (
        0.02 + 0.2 * np.abs(current)
    )
    core = np.column_stack(
        [
            t,
            np.full(t.size, 35.0),
            np.full(t.size, 40.0),
            current,
            np.zeros((t.size, 13)),
        ]
    )
    corr = correlate_with_core(converted, core)
    assert corr["status"] == "ok"
    assert abs(corr["channels"]["ax"]["std_vs_abs_current_motor"]) > 0.2


def test_load_imu_calibration_from_sensor_policy_yaml():
    payload = {
        "sensor_policy": {
            "ros__parameters": {
                "imu_accel_to_ms2": 9.81,
                "imu_gyro_to_rads": 0.01745,
                "imu_ax_sign": -1.0,
                "imu_ax_bias": 0.123,
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(payload, fh)
        path = Path(fh.name)
    try:
        cal = load_imu_calibration(path)
        assert cal is not None
        assert math.isclose(cal.accel_to_ms2, 9.81)
        assert cal.ax_sign == -1.0
        assert math.isclose(cal.ax_bias, 0.123)
    finally:
        path.unlink(missing_ok=True)


def test_load_imu_calibration_from_nested_vehicle_obs_block():
    payload = {
        "vehicle_obs": {
            "ros__parameters": {
                "imu_ax_bias": -0.02114,
                "imu_ay_bias": 0.07107,
            }
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(payload, fh)
        path = Path(fh.name)
    try:
        cal = load_imu_calibration(path)
        assert cal is not None
        assert math.isclose(cal.ax_bias, -0.02114)
        assert math.isclose(cal.ay_bias, 0.07107)
    finally:
        path.unlink(missing_ok=True)


def test_write_characterization_report_roundtrip():
    raw = _static_imu(n=120)
    series = _empty_series()
    series["/sensors/imu/raw"] = raw
    result = characterize_imu_bag(series, condition="static_off")
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "imu.md"
        write_characterization_report(result, report)
        text = report.read_text(encoding="utf-8")
        assert "IMU characterization" in text
        assert "static_off" in text
        assert "Proposed sensor_policy IMU params" in text


def test_compare_actor_streams_resampled():
    t = np.linspace(0.0, 1.0, 21)
    recorded = np.column_stack(
        [t, np.sin(t), np.cos(t), np.full(t.size, G), np.zeros((t.size, 3))]
    )
    replayed = recorded.copy()
    replayed[:, 1] += 0.001
    cmp = compare_actor_streams(recorded, replayed)
    assert cmp is not None
    assert cmp["max_abs_diff"] >= 0.001
    assert cmp["mean_abs_diff"] > 0.0


def test_characterize_cli_missing_bag_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        characterize_main(["--bag", "/tmp/does-not-exist-imu-bag"])
    assert exc.value.code != 0


def test_build_typestore_registers_custom_msgs_without_conflict():
    ts = build_typestore()
    assert "vesc_msgs/msg/VescStateStamped" in ts.fielddefs
    assert "std_msgs/msg/Float32MultiArray" in ts.fielddefs


def test_convert_driver_imu_applies_sign_and_bias():
    raw = _static_imu(n=50, ax=-0.02, ay=0.07, az=1.0, gz=-0.235, noise=0.0)
    fit = fit_imu_static(raw)
    cal = calibration_from_static_fit(fit)
    converted = convert_driver_imu(raw, cal)
    assert converted.shape == raw.shape
    assert abs(float(np.mean(converted[:, 1]))) < 0.05
    assert abs(float(np.mean(converted[:, 2]))) < 0.05
    assert abs(float(np.mean(converted[:, 3])) - G) < 0.2

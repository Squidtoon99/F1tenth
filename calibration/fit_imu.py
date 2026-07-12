"""Static IMU unit / bias / sign estimation."""

from __future__ import annotations

import numpy as np

from bag_io import G, detect_accel_scale


def fit_imu_static(imu: np.ndarray) -> dict:
    """Fit IMU conversion from a stationary bag.

    Columns: t, ax, ay, az, gx, gy, gz.

    Returns sign conventions that map body accel toward (0, 0, +g) in m/s^2
    and gyro toward rad/s when the raw units are either SI or (g, deg/s).
    """
    if imu.shape[0] < 10:
        return {
            "status": "NOT_IDENTIFIABLE",
            "reason": "insufficient_imu_samples",
        }

    mean = imu[:, 1:].mean(axis=0)
    std = imu[:, 1:].std(axis=0)
    ax, ay, az = mean[0], mean[1], mean[2]
    gx, gy, gz = mean[3], mean[4], mean[5]

    accel_scale, accel_unit = detect_accel_scale(az)

    # Gyro: VESC raw IMU often reports deg/s. Prefer deg/s when static |bias| or
    # noise is large relative to typical rad/s bias (~0.01), or when accel was in g
    # (same sensor family).
    gyro_std = float(np.std(imu[:, 6]))
    if gyro_std > 0.5 or abs(gz) > 0.15 or accel_unit == "g":
        gyro_scale = np.pi / 180.0
        gyro_unit = "deg_s"
    else:
        gyro_scale = 1.0
        gyro_unit = "rad_s"

    ax_si = ax * accel_scale
    ay_si = ay * accel_scale
    az_si = az * accel_scale

    # Choose signs so gravity lands on +z and small horizontal biases stay small.
    imu_ax_sign = -1.0 if abs(-ax_si) < abs(ax_si) and abs(ax_si) > 0.2 else 1.0
    imu_ay_sign = -1.0 if abs(-ay_si) < abs(ay_si) and abs(ay_si) > 0.2 else 1.0
    imu_az_sign = 1.0 if az_si >= 0.0 else -1.0
    imu_yaw_rate_sign = 1.0

    corrected = np.array(
        [
            imu_ax_sign * ax_si,
            imu_ay_sign * ay_si,
            imu_az_sign * az_si,
        ]
    )
    gravity_err = float(np.linalg.norm(corrected - np.array([0.0, 0.0, G])))

    return {
        "status": "identified" if gravity_err < 1.5 else "weakly_identified",
        "accel_unit": accel_unit,
        "gyro_unit": gyro_unit,
        "accel_scale": float(accel_scale),
        "gyro_scale": float(gyro_scale),
        "imu_ax_sign": float(imu_ax_sign),
        "imu_ay_sign": float(imu_ay_sign),
        "imu_az_sign": float(imu_az_sign),
        "imu_yaw_rate_sign": float(imu_yaw_rate_sign),
        "bias_accel_raw": {
            "ax": float(ax),
            "ay": float(ay),
            "az": float(az),
        },
        "bias_gyro_raw": {
            "gx": float(gx),
            "gy": float(gy),
            "gz": float(gz),
        },
        "std_accel_raw": {
            "ax": float(std[0]),
            "ay": float(std[1]),
            "az": float(std[2]),
        },
        "std_gyro_raw": {
            "gx": float(std[3]),
            "gy": float(std[4]),
            "gz": float(std[5]),
        },
        "gravity_error_ms2": gravity_err,
    }

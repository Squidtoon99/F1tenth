from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from f1tenth_rl_agent.sensor_interfaces import (
    GRAVITY_MS2,
    IMU_DIM,
    IMU_START,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    LIDAR_START,
    NUM_OBS,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_HISTORY,
    STEER_T,
    STEER_T1,
    STEER_T2,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    VESC_CURRENT,
    VESC_SPEED,
)

LIDAR_FOV_DEG = 270.0
LIDAR_NUM_BEAMS = LIDAR_DIM


@dataclass(frozen=True)
class ImuCalibration:
    accel_to_ms2: float = 9.80665
    gyro_to_rads: float = math.pi / 180.0
    ax_sign: float = 1.0
    ay_sign: float = 1.0
    az_sign: float = 1.0
    gx_sign: float = 1.0
    gy_sign: float = 1.0
    gz_sign: float = 1.0
    ax_bias: float = 0.0
    ay_bias: float = 0.0
    az_bias: float = 0.0
    gx_bias: float = 0.0
    gy_bias: float = 0.0
    gz_bias: float = 0.0


@dataclass(frozen=True)
class RawImuSample:
    stamp_s: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


def beam_angles_rad() -> np.ndarray:
    return (
        LIDAR_ANGLE_MIN
        + np.arange(LIDAR_NUM_BEAMS, dtype=np.float64) * LIDAR_ANGLE_INCREMENT
    )


def pack_lidar_from_scan(
    scan_angle_min: float,
    scan_angle_increment: float,
    scan_ranges: np.ndarray | list[float],
    *,
    scan_range_min: float = 0.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        out = np.full(LIDAR_NUM_BEAMS, LIDAR_RANGE_MAX, dtype=np.float32)
    else:
        out.fill(LIDAR_RANGE_MAX)
    ranges = np.asarray(scan_ranges, dtype=np.float64)
    if ranges.size == 0:
        return out

    src_angles = scan_angle_min + np.arange(ranges.size, dtype=np.float64) * (
        scan_angle_increment
    )
    targets = beam_angles_rad()
    right = np.searchsorted(src_angles, targets, side="left")
    right = np.clip(right, 0, ranges.size - 1)
    left = np.maximum(right - 1, 0)
    pick = np.where(
        np.abs(src_angles[left] - targets) <= np.abs(src_angles[right] - targets),
        left,
        right,
    )
    values = ranges[pick]
    valid = np.isfinite(values) & (values >= float(scan_range_min))
    values = np.where(valid, values, LIDAR_RANGE_MAX)
    values = np.clip(values, LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)
    out[:] = values.astype(np.float32)
    return out


def convert_raw_imu(
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    cal: ImuCalibration,
) -> tuple[float, float, float, float, float, float]:
    ax_si = cal.ax_sign * cal.accel_to_ms2 * (float(ax) - cal.ax_bias)
    ay_si = cal.ay_sign * cal.accel_to_ms2 * (float(ay) - cal.ay_bias)
    az_si = cal.az_sign * cal.accel_to_ms2 * (float(az) - cal.az_bias)
    gx_si = cal.gx_sign * cal.gyro_to_rads * (float(gx) - cal.gx_bias)
    gy_si = cal.gy_sign * cal.gyro_to_rads * (float(gy) - cal.gy_bias)
    gz_si = cal.gz_sign * cal.gyro_to_rads * (float(gz) - cal.gz_bias)
    return ax_si, ay_si, az_si, gx_si, gy_si, gz_si


def average_dynamic_imu(samples: list[RawImuSample]) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    ax = float(np.mean([s.ax for s in samples]))
    ay = float(np.mean([s.ay for s in samples]))
    gz = float(np.mean([s.gz for s in samples]))
    return ax, ay, gz


def actor_imu_from_interval(
    samples: list[RawImuSample],
    cal: ImuCalibration,
    *,
    freeze_const_channels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    converted = [
        convert_raw_imu(s.ax, s.ay, s.az, s.gx, s.gy, s.gz, cal) for s in samples
    ]
    if not converted:
        raw_mean = np.zeros(IMU_DIM, dtype=np.float32)
        actor = np.array(
            [0.0, 0.0, GRAVITY_MS2, 0.0, 0.0, 0.0], dtype=np.float32
        )
        return actor, raw_mean

    raw_mean = np.mean(converted, axis=0).astype(np.float32)
    ax, ay, gz = average_dynamic_imu(
        [
            RawImuSample(
                stamp_s=s.stamp_s,
                ax=c[0],
                ay=c[1],
                az=c[2],
                gx=c[3],
                gy=c[4],
                gz=c[5],
            )
            for s, c in zip(samples, converted)
        ]
    )
    if freeze_const_channels:
        actor = np.array([ax, ay, GRAVITY_MS2, 0.0, 0.0, gz], dtype=np.float32)
    else:
        actor = raw_mean.copy()
    return actor, raw_mean


def pack_actor_observation(
    lidar: np.ndarray,
    imu: np.ndarray,
    speed_mps: float,
    vesc_current_a: float,
    throttle_current: float,
    throttle_predecessor: float,
    executed_steer_history: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    if out is None:
        obs = np.zeros(NUM_OBS, dtype=np.float32)
    else:
        obs = out
        obs.fill(0.0)
    obs[LIDAR_START:IMU_START] = np.asarray(lidar, dtype=np.float32).reshape(-1)[
        :LIDAR_DIM
    ]
    obs[IMU_START : IMU_START + IMU_DIM] = np.asarray(imu, dtype=np.float32).reshape(
        -1
    )[:IMU_DIM]
    obs[VESC_SPEED] = float(speed_mps)
    obs[VESC_CURRENT] = float(vesc_current_a)
    hist = np.asarray(executed_steer_history, dtype=np.float32).reshape(-1)
    if hist.size < STEER_HISTORY:
        raise ValueError(
            f"executed_steer_history length={hist.size}; expected {STEER_HISTORY}"
        )
    steer_t = float(hist[0])
    steer_t1 = float(hist[1])
    steer_t2 = float(hist[2])
    steer_t3 = float(hist[3])
    obs[THROTTLE_CURRENT] = float(throttle_current)
    obs[THROTTLE_PRED] = float(throttle_predecessor)
    obs[STEER_T] = steer_t
    obs[STEER_T1] = steer_t1
    obs[STEER_T2] = steer_t2
    obs[STEER_DELTA0] = steer_t - steer_t1
    obs[STEER_DELTA1] = steer_t1 - steer_t2
    obs[STEER_DELTA2] = steer_t2 - steer_t3
    return obs


def observation_is_finite(obs: np.ndarray) -> bool:
    return bool(np.isfinite(obs).all())

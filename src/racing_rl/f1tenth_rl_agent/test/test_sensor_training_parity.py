"""Training vs ROS packing parity on identical synthetic inputs (no mocks)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from f1tenth_policy.current import (
    TRAINING_I_BRAKE_MAX_A,
    TRAINING_I_DRIVE_MAX_A,
    applied_current_fraction,
)
from f1tenth_policy.layout import (
    CONTROL_HZ,
    HISTORY_STRIDE,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    MAX_STEER_RAD,
    OBS_PREPROCESSING_VERSION,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_HISTORY,
    STEER_T,
    STEER_T1,
    STEER_T2,
    STEERING_DELTA_MAX_RAD,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    TWIST_VX_SIGN_BODY_FORWARD,
    VESC_CURRENT,
    VESC_SPEED,
)
from f1tenth_rl_agent.sensor_action import (
    integrate_steering_delta,
    normalized_action_to_physical,
)
from f1tenth_rl_agent.sensor_preprocessing import (
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    body_accel_to_driver,
    driver_accel_to_body,
    pack_actor_observation,
    pack_lidar_from_scan,
)


def test_control_rate_and_preprocessing_version_match_training():
    assert CONTROL_HZ == pytest.approx(20.0)
    assert OBS_PREPROCESSING_VERSION == 5
    assert HISTORY_STRIDE == 2
    assert STEER_HISTORY == 7
    assert STEERING_DELTA_MAX_RAD == pytest.approx(math.pi / 120.0)
    assert MAX_STEER_RAD == pytest.approx(0.33)
    assert TWIST_VX_SIGN_BODY_FORWARD == pytest.approx(1.0)


def test_native_lidar_order_and_clip_match_training_layout():
    rng = np.random.default_rng(0)
    ranges = rng.uniform(0.2, 12.0, size=LIDAR_DIM).astype(np.float64)
    ranges[3] = np.inf
    ranges[7] = np.nan
    ranges[11] = 0.01
    packed = pack_lidar_from_scan(
        LIDAR_ANGLE_MIN, LIDAR_ANGLE_INCREMENT, ranges, scan_range_min=0.05
    )
    expected = np.clip(ranges, LIDAR_RANGE_MIN, LIDAR_RANGE_MAX)
    expected[3] = LIDAR_RANGE_MAX
    expected[7] = LIDAR_RANGE_MAX
    expected[11] = LIDAR_RANGE_MAX
    np.testing.assert_allclose(packed, expected.astype(np.float32), atol=1e-6)


def test_driver_imu_rotation_recovers_training_body_accel():
    body_ax, body_ay, body_gz = 1.5, -0.4, 0.25
    driver_ax, driver_ay = body_accel_to_driver(body_ax, body_ay)
    assert driver_accel_to_body(driver_ax, driver_ay) == pytest.approx(
        (body_ax, body_ay)
    )
    actor_imu, _ = actor_imu_from_interval(
        [RawImuSample(0.0, driver_ax, driver_ay, 9.81, 0.0, 0.0, body_gz)],
        ImuCalibration(accel_to_ms2=1.0, gyro_to_rads=1.0),
    )
    assert actor_imu[0] == pytest.approx(body_ax)
    assert actor_imu[1] == pytest.approx(body_ay)
    assert actor_imu[2] == pytest.approx(9.81)
    assert actor_imu[3] == pytest.approx(0.0)
    assert actor_imu[4] == pytest.approx(0.0)
    assert actor_imu[5] == pytest.approx(body_gz)


def test_vesc_speed_current_and_command_history_match_training_pack():
    lidar = np.full(LIDAR_DIM, 4.0, dtype=np.float32)
    imu = np.array([0.1, -0.2, 9.81, 0.0, 0.0, 0.3], dtype=np.float32)
    odom_vx = 2.4
    speed = TWIST_VX_SIGN_BODY_FORWARD * odom_vx
    signed_a = 24.0
    vesc_current = applied_current_fraction(
        signed_a, TRAINING_I_DRIVE_MAX_A, TRAINING_I_BRAKE_MAX_A
    )
    steer_hist = np.array(
        [0.12, 0.105, 0.09, 0.06, 0.03, 0.015, 0.00], dtype=np.float32
    )
    assert steer_hist.size == STEER_HISTORY
    obs = pack_actor_observation(
        lidar, imu, speed, vesc_current, 0.30, 0.10, steer_hist
    )
    assert obs[VESC_SPEED] == pytest.approx(2.4)
    assert obs[VESC_CURRENT] == pytest.approx(0.30)
    assert obs[THROTTLE_CURRENT] == pytest.approx(0.30)
    assert obs[THROTTLE_PRED] == pytest.approx(0.10)
    assert obs[STEER_T] == pytest.approx(0.12)
    assert obs[STEER_T1] == pytest.approx(0.09)
    assert obs[STEER_T2] == pytest.approx(0.03)
    assert obs[STEER_DELTA0] == pytest.approx(0.03)
    assert obs[STEER_DELTA1] == pytest.approx(0.06)
    assert obs[STEER_DELTA2] == pytest.approx(0.03)


def test_action_scaling_80_40_gate_and_steer_x_0_33():
    drive, brake, servo, long_norm, steer_norm = normalized_action_to_physical(
        1.0,
        1.0,
        i_drive_max_a=TRAINING_I_DRIVE_MAX_A,
        i_brake_max_a=TRAINING_I_BRAKE_MAX_A,
        max_steer=MAX_STEER_RAD,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive == pytest.approx(80.0)
    assert brake == pytest.approx(0.0)
    assert long_norm == pytest.approx(1.0)
    assert steer_norm == pytest.approx(1.0)
    assert servo == pytest.approx(-1.2135 * 0.33 + 0.4495)

    drive_b, brake_b, _, long_b, _ = normalized_action_to_physical(
        -1.0,
        0.0,
        i_drive_max_a=TRAINING_I_DRIVE_MAX_A,
        i_brake_max_a=TRAINING_I_BRAKE_MAX_A,
        max_steer=MAX_STEER_RAD,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive_b == pytest.approx(0.0)
    assert brake_b == pytest.approx(40.0)
    assert long_b == pytest.approx(-1.0)

    steer = integrate_steering_delta(
        0.0, 1.0, delta_max_rad=STEERING_DELTA_MAX_RAD, max_steer=MAX_STEER_RAD
    )
    assert steer == pytest.approx(STEERING_DELTA_MAX_RAD)

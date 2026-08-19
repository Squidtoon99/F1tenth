"""Pure-Python tests for 1,097-D sensor preprocessing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.sensor_preprocessing import (
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    beam_angles_rad,
    convert_raw_imu,
    pack_actor_observation,
    pack_lidar_from_scan,
)


def test_beam_layout_matches_training_fov():
    angles = beam_angles_rad()
    assert angles.shape == (si.LIDAR_DIM,)
    assert abs(float(angles[0]) - math.radians(-135.0)) < 1e-6
    assert abs(float(angles[-1]) - math.radians(135.0)) < 1e-6
    assert abs(float(angles[1] - angles[0]) - math.radians(0.25)) < 1e-6


def test_lidar_packing_clamps_and_maps_no_return():
    scan_min = math.radians(-135.0)
    inc = math.radians(0.25)
    ranges = np.full(1081, 1.5, dtype=np.float32)
    ranges[100] = float("inf")
    ranges[200] = float("nan")
    ranges[300] = 0.01
    packed = pack_lidar_from_scan(scan_min, inc, ranges, scan_range_min=0.05)
    assert packed.shape == (si.LIDAR_DIM,)
    assert packed[100] == pytest.approx(si.LIDAR_RANGE_MAX)
    assert packed[200] == pytest.approx(si.LIDAR_RANGE_MAX)
    assert packed[300] == pytest.approx(si.LIDAR_RANGE_MAX)
    assert np.all(packed >= si.LIDAR_RANGE_MIN)
    assert np.all(packed <= si.LIDAR_RANGE_MAX)


def test_imu_conversion_sign_bias_and_frozen_channels():
    cal = ImuCalibration(
        accel_to_ms2=9.81,
        gyro_to_rads=math.pi / 180.0,
        ax_sign=-1.0,
        ay_sign=1.0,
        ax_bias=0.1,
        ay_bias=0.0,
    )
    ax, ay, az, gx, gy, gz = convert_raw_imu(0.2, 0.0, 1.0, 0.0, 0.0, 10.0, cal)
    assert ax == pytest.approx(-1.0 * 9.81 * (0.2 - 0.1))
    assert ay == pytest.approx(0.0)
    assert gz == pytest.approx(10.0 * math.pi / 180.0)

    samples = [
        RawImuSample(0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 5.0),
        RawImuSample(0.01, 0.3, 0.0, 1.0, 0.0, 0.0, 15.0),
    ]
    actor_imu, raw_mean = actor_imu_from_interval(samples, cal, freeze_const_channels=True)
    assert actor_imu[2] == pytest.approx(si.GRAVITY_MS2)
    assert actor_imu[3] == pytest.approx(0.0)
    assert actor_imu[4] == pytest.approx(0.0)
    assert raw_mean.shape == (6,)


def test_actor_observation_layout_offsets():
    lidar = np.linspace(0.1, 5.0, si.LIDAR_DIM, dtype=np.float32)
    imu = np.array([1, 2, 9.81, 0, 0, 0.3], dtype=np.float32)
    steer_hist = np.array([-0.2, 0.1, 0.0, 0.05], dtype=np.float32)
    obs = pack_actor_observation(
        lidar, imu, 3.0, 0.4, 0.5, 0.1, steer_hist
    )
    assert obs.shape == (si.NUM_OBS,)
    np.testing.assert_allclose(obs[:si.LIDAR_DIM], lidar)
    np.testing.assert_allclose(obs[si.IMU_START : si.IMU_START + 6], imu)
    assert obs[si.VESC_SPEED] == pytest.approx(3.0)
    assert obs[si.VESC_CURRENT] == pytest.approx(0.4)
    assert obs[si.THROTTLE_CURRENT] == pytest.approx(0.5)
    assert obs[si.THROTTLE_PRED] == pytest.approx(0.1)
    assert obs[si.STEER_T] == pytest.approx(-0.2)
    assert obs[si.STEER_T1] == pytest.approx(0.1)
    assert obs[si.STEER_T2] == pytest.approx(0.0)
    assert obs[si.STEER_DELTA0] == pytest.approx(-0.3)
    assert obs[si.STEER_DELTA1] == pytest.approx(0.1)
    assert obs[si.STEER_DELTA2] == pytest.approx(-0.05)


def test_actor_layout_offsets_match_training_fixture():
    import json
    import os

    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "actor_layout_v2.json"
    )
    with open(fixture_path, encoding="utf-8") as f:
        layout = json.load(f)

    assert si.ACTOR_LAYOUT_VERSION == layout["actor_layout_version"]
    assert si.NUM_OBS == layout["num_obs"]
    assert si.PROPRIO_DIM == layout["proprio_dim"]
    assert si.LIDAR_DIM == layout["lidar_dim"]
    assert si.IMU_DIM == layout["imu_dim"]
    assert si.IMU_START == layout["imu_start"]
    assert si.VESC_SPEED == layout["vesc_speed"]
    assert si.VESC_CURRENT == layout["vesc_current"]
    assert si.THROTTLE_CURRENT == layout["throttle_current"]
    assert si.THROTTLE_PRED == layout["throttle_pred"]
    assert si.STEER_T == layout["steer_t"]
    assert si.STEER_T1 == layout["steer_t1"]
    assert si.STEER_T2 == layout["steer_t2"]
    assert si.STEER_DELTA0 == layout["steer_delta0"]
    assert si.STEER_DELTA1 == layout["steer_delta1"]
    assert si.STEER_DELTA2 == layout["steer_delta2"]
    assert si.OBS_PREPROCESSING_VERSION == layout["observation_preprocessing_version"]
    assert layout["vesc_current_semantics"].startswith("signed_applied_current_a")


def test_deploy_shaped_scan_replay_matches_training_layout():
    """Synthetic LaserScan/IMU/odom → packed obs matches frozen 1,097-D layout.

    Offset parity with ``training/f1tenth_env/sensors.py`` is covered by the
    training actor-schema tests; this gate stays Warp-free for the ROS image.
    """
    assert si.NUM_OBS == 1097
    assert si.PROPRIO_DIM == 16
    assert si.LIDAR_DIM == 1081
    assert si.IMU_START == 1081
    assert si.VESC_SPEED == 1087
    assert si.VESC_CURRENT == 1088
    assert si.THROTTLE_CURRENT == 1089
    assert si.STEER_T == 1091
    assert si.STEER_DELTA2 == 1096
    assert si.OBS_PREPROCESSING_VERSION == 3
    assert si.ACTOR_LAYOUT_VERSION == 2

    rng = np.random.default_rng(0)
    scan_min = math.radians(-135.0)
    inc = math.radians(0.25)
    ranges = rng.uniform(0.2, 8.0, size=1081).astype(np.float32)
    ranges[10] = np.inf
    ranges[20] = np.nan
    lidar = pack_lidar_from_scan(scan_min, inc, ranges, scan_range_min=0.05)

    cal = ImuCalibration(
        accel_to_ms2=9.80665,
        gyro_to_rads=math.pi / 180.0,
        ax_bias=-0.02114,
        ay_bias=0.07107,
    )
    samples = [
        RawImuSample(0.00, 0.05, -0.02, 1.01, 0.1, -0.2, 3.0),
        RawImuSample(0.02, 0.07, 0.01, 0.99, -0.1, 0.1, 5.0),
        RawImuSample(0.04, 0.06, 0.00, 1.00, 0.0, 0.0, 4.0),
    ]
    actor_imu, _ = actor_imu_from_interval(samples, cal, freeze_const_channels=True)
    assert actor_imu[2] == pytest.approx(si.GRAVITY_MS2)
    assert actor_imu[3] == pytest.approx(0.0)
    assert actor_imu[4] == pytest.approx(0.0)

    twist_vx_sign = -1.0
    odom_vx = -1.25
    speed = twist_vx_sign * odom_vx
    vesc_current_fraction = si.applied_current_fraction(1.75, 10.0, 10.0)
    steer_hist = np.array([-0.1, 0.15, 0.0, 0.0], dtype=np.float32)
    obs = pack_actor_observation(
        lidar, actor_imu, speed, vesc_current_fraction, 0.2, -0.05, steer_hist
    )

    assert obs.shape == (si.NUM_OBS,)
    assert np.isfinite(obs).all()
    assert obs[si.VESC_SPEED] == pytest.approx(1.25)
    assert obs[si.VESC_CURRENT] == pytest.approx(vesc_current_fraction)
    assert obs[si.VESC_CURRENT] == pytest.approx(0.175)
    assert obs[si.THROTTLE_CURRENT] == pytest.approx(0.2)
    assert obs[si.THROTTLE_PRED] == pytest.approx(-0.05)
    assert obs[si.STEER_T] == pytest.approx(-0.1)
    assert obs[si.STEER_T1] == pytest.approx(0.15)
    assert obs[si.STEER_DELTA0] == pytest.approx(-0.25)
    np.testing.assert_allclose(
        obs[si.IMU_START + 2 : si.IMU_START + 5],
        [si.GRAVITY_MS2, 0.0, 0.0],
        atol=1e-6,
    )


def test_real_checkpoint_deploy_shaped_preprocess_then_infer():
    """End-to-end: deploy-shaped obs → obs_norm → deterministic action bounds."""
    import os

    from f1tenth_rl_agent.policy_model import load_sensor_actor, load_sensor_obs_norm

    ckpt = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        ),
        "training",
        "outputs",
        "runs",
        "7210e365",
        "checkpoints",
        "policy_271360000.pt",
    )
    if not os.path.isfile(ckpt):
        pytest.skip(f"ignored real checkpoint not present: {ckpt}")

    rng = np.random.default_rng(1)
    scan_min = math.radians(-135.0)
    inc = math.radians(0.25)
    ranges = rng.uniform(0.5, 12.0, size=1081).astype(np.float32)
    lidar = pack_lidar_from_scan(scan_min, inc, ranges)
    cal = ImuCalibration(ax_bias=-0.02114, ay_bias=0.07107)
    samples = [
        RawImuSample(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        RawImuSample(0.03, 0.02, -0.01, 1.0, 0.0, 0.0, 1.5),
    ]
    imu, _ = actor_imu_from_interval(samples, cal, freeze_const_channels=True)
    obs = pack_actor_observation(
        lidar,
        imu,
        speed_mps=0.0,
        vesc_current_a=0.0,
        throttle_current=0.0,
        throttle_predecessor=0.0,
        executed_steer_history=np.zeros(si.STEER_HISTORY, dtype=np.float32),
    )

    import torch

    device = torch.device("cpu")
    # Old checkpoints use the pre-break layout and must be rejected outright.
    reject = r"policy_format_version|actor_layout_version|actor_obs_dim"
    with pytest.raises(ValueError, match=reject):
        load_sensor_actor(ckpt, "actor", device)
    with pytest.raises(ValueError, match=reject):
        load_sensor_obs_norm(ckpt, device, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
    assert obs.shape == (si.NUM_OBS,)

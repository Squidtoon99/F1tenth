"""Real-Warp LiDAR/IMU geometry, DR, determinism, and with_sensors interface tests."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import warp as wp

from config import DEFAULT_CONFIG
from conftest import make_circle_track, make_straight_track
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.kernel import OpponentParams, ResetParams, SensorParams
from f1tenth_env.sensors import imu_stage_kernel, lidar_solo_kernel, lidar_stage_kernel
from f1tenth_env.utils import build_corridor_distance_field
from f1tenth_env.warp_env import (
    _CorridorDistanceStorage,
    _EnvironmentStorage,
    _VehicleStorage,
)
from f1tenth_sim.params import VehicleParams

_NUM_BEAMS = 1081
_FWD = 540
_LEFT = 900
_RIGHT = 180
_CAR_LENGTH = 0.568
_CAR_WIDTH = 0.296
_EDT_RES = 0.025
_UST10_ACCURACY_M = 0.040


def _configure_cpu():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _sensor_params(**overrides) -> SensorParams:
    sensor = DEFAULT_CONFIG["sensor"]
    fov = math.radians(float(sensor["fov_deg"]))
    num_beams = int(sensor["num_beams"])
    params = SensorParams()
    params.num_beams = num_beams
    params.angle_min = float(-0.5 * fov)
    params.angle_increment = float(fov / float(num_beams - 1))
    params.range_min = float(sensor["range_min_m"])
    params.range_max = float(sensor["range_max_m"])
    params.reliable_range = float(sensor["reliable_range_m"])
    params.lidar_offset_x = float(sensor["lidar_offset_x"])
    params.lidar_offset_y = float(sensor["lidar_offset_y"])
    params.lidar_offset_yaw = float(sensor["lidar_offset_yaw"])
    params.max_march_steps = max(1, int(sensor.get("max_march_steps", 512)))
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def _reset_params(seed: int = 0) -> ResetParams:
    params = ResetParams()
    params.seed = int(seed)
    return params


def _opponent_params() -> OpponentParams:
    params = OpponentParams()
    params.car_length = float(_CAR_LENGTH)
    params.car_width = float(_CAR_WIDTH)
    return params


def _beam_world_angle(yaw: float, beam_id: int, sensor: SensorParams) -> float:
    return (
        float(yaw)
        + float(sensor.lidar_offset_yaw)
        + float(sensor.angle_min)
        + float(beam_id) * float(sensor.angle_increment)
    )


def analytic_parallel_corridor_range(
    origin_x: float,
    origin_y: float,
    angle: float,
    half_width: float,
    range_max: float,
) -> float:
    """Exact ray hit distance for infinite walls at y = +/- half_width."""
    direction_y = math.sin(angle)
    if abs(direction_y) < 1.0e-12:
        return float(range_max)
    hits = []
    for wall_y in (half_width, -half_width):
        travel = (wall_y - origin_y) / direction_y
        if travel >= 0.0:
            hits.append(travel)
    if not hits:
        return float(range_max)
    return float(min(min(hits), range_max))


def _launch_lidar(
    *,
    centerline,
    width_left,
    width_right,
    ego_pose,
    ego_yaw,
    opponent_pose=None,
    opponent_yaw=0.0,
    sensor=None,
    seed=0,
    env_overrides=None,
    resolution=_EDT_RES,
):
    sensor = sensor or _sensor_params()
    host = build_corridor_distance_field(
        centerline, width_left, width_right, resolution=resolution
    )
    corridor = _CorridorDistanceStorage(host, "cpu")
    vehicle = VehicleParams.from_config(DEFAULT_CONFIG["env"])
    ego = _VehicleStorage(1, torch.device("cpu"), vehicle)
    env = _EnvironmentStorage(1, torch.device("cpu"))
    ego.tensor["x"][0] = float(ego_pose[0])
    ego.tensor["y"][0] = float(ego_pose[1])
    ego.tensor["yaw"][0] = float(ego_yaw)
    if env_overrides:
        for key, value in env_overrides.items():
            env.tensor[key][0] = value
    ranges = torch.zeros(1, int(sensor.num_beams), dtype=torch.float32)
    reset = _reset_params(seed)
    opponent = _opponent_params()
    if opponent_pose is None:
        wp.launch(
            lidar_solo_kernel,
            dim=(1, int(sensor.num_beams)),
            inputs=[
                ego.buffers,
                env.buffers,
                corridor.data,
                sensor,
                opponent,
                reset,
                wp.from_torch(ranges),
            ],
            device="cpu",
        )
    else:
        opp = _VehicleStorage(1, torch.device("cpu"), vehicle)
        opp.tensor["x"][0] = float(opponent_pose[0])
        opp.tensor["y"][0] = float(opponent_pose[1])
        opp.tensor["yaw"][0] = float(opponent_yaw)
        wp.launch(
            lidar_stage_kernel,
            dim=(1, int(sensor.num_beams)),
            inputs=[
                ego.buffers,
                opp.buffers,
                env.buffers,
                corridor.data,
                sensor,
                opponent,
                reset,
                0,
                wp.from_torch(ranges),
            ],
            device="cpu",
        )
    wp.synchronize()
    return ranges, host, sensor


def _make_env(num_envs=4, *, opponent=False, enable_dr=False, sensor_dr=None):
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dr = dict(cfg["env"]["domain_randomization"])
    dr["enabled"] = bool(enable_dr)
    if enable_dr and sensor_dr:
        dr.update(sensor_dr)
    if not enable_dr:
        # Keep sensor DR at no-op defaults even when vehicle DR is off.
        for key in list(dr):
            if key.startswith("lidar_") or key.startswith("imu_"):
                if key.endswith("_range"):
                    dr[key] = [0.0, 0.0]
    cfg["env"]["domain_randomization"] = dr
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    if opponent:
        cfg["env"]["opponent_strategy"] = "scripted"
    else:
        cfg["env"]["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def test_sensor_angle_layout_is_ust10lx(warp_runtime):
    del warp_runtime
    sensor = _sensor_params()
    assert int(sensor.num_beams) == _NUM_BEAMS
    assert abs(float(sensor.angle_min) - math.radians(-135.0)) < 1e-6
    assert abs(float(sensor.angle_increment) - math.radians(0.25)) < 1e-6
    span = float(sensor.angle_min) + float(sensor.angle_increment) * (
        _NUM_BEAMS - 1
    )
    assert abs(span - math.radians(135.0)) < 1e-6
    assert abs(_beam_world_angle(0.0, _FWD, sensor)) < 1e-6
    assert abs(_beam_world_angle(0.0, _LEFT, sensor) - math.pi / 2.0) < 1e-6
    assert abs(_beam_world_angle(0.0, _RIGHT, sensor) + math.pi / 2.0) < 1e-6


def test_circle_lateral_beams_match_half_width(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    radius = 20.0
    centerline, width_left, width_right = make_circle_track(
        radius=radius, n=360, w_left=half, w_right=half
    )
    ranges, host, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
    )
    assert ranges.shape == (1, _NUM_BEAMS)
    assert abs(float(ranges[0, _LEFT]) - half) <= host.resolution
    assert abs(float(ranges[0, _RIGHT]) - half) <= host.resolution
    assert abs(float(ranges[0, _LEFT]) - half) <= _UST10_ACCURACY_M
    assert abs(float(ranges[0, _RIGHT]) - half) <= _UST10_ACCURACY_M


def test_straight_forward_beam_matches_analytic_wall(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    centerline, width_left, width_right = make_straight_track(
        length=80.0, n=320, w_left=half, w_right=half
    )
    ego_x, ego_y, ego_yaw = 40.0, 0.0, math.pi / 2.0
    ranges, host, sensor = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(ego_x, ego_y),
        ego_yaw=ego_yaw,
    )
    angle = _beam_world_angle(ego_yaw, _FWD, sensor)
    origin_x = ego_x + math.cos(ego_yaw) * float(sensor.lidar_offset_x)
    origin_y = ego_y + math.sin(ego_yaw) * float(sensor.lidar_offset_x)
    expected = analytic_parallel_corridor_range(
        origin_x, origin_y, angle, half, float(sensor.range_max)
    )
    assert abs(float(ranges[0, _FWD]) - expected) <= host.resolution
    assert abs(float(ranges[0, _FWD]) - (half - 0.27)) <= _UST10_ACCURACY_M


def test_edt_sphere_trace_matches_analytic_within_one_cell(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 2.0
    centerline, width_left, width_right = make_straight_track(
        length=60.0, n=240, w_left=half, w_right=half
    )
    ego_x, ego_y, ego_yaw = 30.0, 0.25, 0.0
    ranges, host, sensor = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(ego_x, ego_y),
        ego_yaw=ego_yaw,
    )
    errors = []
    origin_x = ego_x + math.cos(ego_yaw) * float(sensor.lidar_offset_x)
    origin_y = ego_y + math.sin(ego_yaw) * float(sensor.lidar_offset_x)
    for beam_id in (_LEFT, _RIGHT, _FWD + 180, _FWD - 180):
        angle = _beam_world_angle(ego_yaw, beam_id, sensor)
        expected = analytic_parallel_corridor_range(
            origin_x, origin_y, angle, half, float(sensor.range_max)
        )
        if expected >= float(sensor.range_max) - 1e-6:
            continue
        errors.append(abs(float(ranges[0, beam_id]) - expected))
    assert errors
    assert max(errors) <= host.resolution


def test_lidar_range_min_max_sentinels(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    radius = 20.0
    centerline, width_left, width_right = make_circle_track(
        radius=radius, n=360, w_left=half, w_right=half
    )
    sensor = _sensor_params()
    # Near the outer wall, the outward beam clamps up to range_min.
    near, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius + half - 0.02 - float(sensor.lidar_offset_x), 0.0),
        ego_yaw=0.0,
        sensor=sensor,
    )
    assert float(near[0, _FWD]) == pytest.approx(float(sensor.range_min), abs=1e-5)

    # Cap range_max below the analytic wall distance so the sentinel is hit.
    capped = _sensor_params(range_max=0.5)
    far, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
        sensor=capped,
    )
    assert torch.all(far[0] <= float(capped.range_max) + 1e-5)
    assert torch.any(far[0] >= float(capped.range_max) - 1e-4)


def test_opponent_obb_shortens_forward_beam(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    radius = 20.0
    gap = 3.0
    centerline, width_left, width_right = make_circle_track(
        radius=radius, n=360, w_left=half, w_right=half
    )
    solo, _, sensor = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
    )
    duel, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
        opponent_pose=(radius, gap),
        opponent_yaw=math.pi / 2.0,
    )
    expected = gap - float(sensor.lidar_offset_x) - 0.5 * _CAR_LENGTH
    assert float(duel[0, _FWD]) == pytest.approx(expected, abs=1e-3)
    assert float(duel[0, _FWD]) < float(solo[0, _FWD])
    assert float(solo[0, _FWD]) < float(sensor.range_max)


def test_clean_imu_matches_vehicle_state(warp_runtime):
    del warp_runtime
    _configure_cpu()
    vehicle = VehicleParams.from_config(DEFAULT_CONFIG["env"])
    ego = _VehicleStorage(1, torch.device("cpu"), vehicle)
    env = _EnvironmentStorage(1, torch.device("cpu"))
    ego.tensor["ax"][0] = 1.25
    ego.tensor["ay"][0] = -0.5
    ego.tensor["yaw_rate"][0] = 0.4
    imu = torch.zeros(1, 6, dtype=torch.float32)
    sim = vehicle.to_warp(sim_dt=0.005, control_dt=0.05)
    wp.launch(
        imu_stage_kernel,
        dim=1,
        inputs=[
            ego.buffers,
            env.buffers,
            _reset_params(0),
            sim,
            0,
            wp.from_torch(imu),
        ],
        device="cpu",
    )
    wp.synchronize()
    assert float(imu[0, 0]) == pytest.approx(1.25, abs=1e-5)
    assert float(imu[0, 1]) == pytest.approx(-0.5, abs=1e-5)
    assert float(imu[0, 2]) == pytest.approx(float(sim.gravity), abs=1e-5)
    assert float(imu[0, 3]) == pytest.approx(0.0, abs=1e-5)
    assert float(imu[0, 4]) == pytest.approx(0.0, abs=1e-5)
    assert float(imu[0, 5]) == pytest.approx(0.4, abs=1e-5)


def test_imu_dr_stays_within_bias_noise_bounds(warp_runtime):
    del warp_runtime
    _configure_cpu()
    vehicle = VehicleParams.from_config(DEFAULT_CONFIG["env"])
    ego = _VehicleStorage(1, torch.device("cpu"), vehicle)
    env = _EnvironmentStorage(1, torch.device("cpu"))
    ax, ay, yaw_rate = 0.8, 0.2, -0.15
    ego.tensor["ax"][0] = ax
    ego.tensor["ay"][0] = ay
    ego.tensor["yaw_rate"][0] = yaw_rate
    accel_bias = 0.05
    gyro_bias = 0.02
    accel_std = 0.03
    gyro_std = 0.01
    env.tensor["imu_accel_bias_x"][0] = accel_bias
    env.tensor["imu_accel_bias_y"][0] = accel_bias
    env.tensor["imu_accel_bias_z"][0] = accel_bias
    env.tensor["imu_gyro_bias_x"][0] = gyro_bias
    env.tensor["imu_gyro_bias_y"][0] = gyro_bias
    env.tensor["imu_gyro_bias_z"][0] = gyro_bias
    env.tensor["imu_accel_noise_std"][0] = accel_std
    env.tensor["imu_gyro_noise_std"][0] = gyro_std
    imu = torch.zeros(1, 6, dtype=torch.float32)
    sim = vehicle.to_warp(sim_dt=0.005, control_dt=0.05)
    wp.launch(
        imu_stage_kernel,
        dim=1,
        inputs=[
            ego.buffers,
            env.buffers,
            _reset_params(7),
            sim,
            0,
            wp.from_torch(imu),
        ],
        device="cpu",
    )
    wp.synchronize()
    # Gaussian samples are unbounded in theory; bound by a generous multiple of std
    # so the DR path is exercised without accepting pathological outliers.
    accel_tol = abs(accel_bias) + 6.0 * accel_std
    gyro_tol = abs(gyro_bias) + 6.0 * gyro_std
    assert abs(float(imu[0, 0]) - ax) <= accel_tol
    assert abs(float(imu[0, 1]) - ay) <= accel_tol
    assert float(imu[0, 2]) == pytest.approx(float(sim.gravity), abs=1e-5)
    assert float(imu[0, 3]) == pytest.approx(0.0, abs=1e-5)
    assert float(imu[0, 4]) == pytest.approx(0.0, abs=1e-5)
    assert abs(float(imu[0, 5]) - yaw_rate) <= gyro_tol


def test_lidar_dropout_fraction_and_far_range(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    radius = 50.0
    centerline, width_left, width_right = make_circle_track(
        radius=radius, n=480, w_left=half, w_right=half
    )
    clean, _, sensor = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
        seed=0,
    )
    far_hits = clean[0] > float(sensor.reliable_range)
    assert int(far_hits.sum()) > 0

    dropped, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
        seed=0,
        env_overrides={"lidar_far_dropout_prob": 1.0},
    )
    assert torch.all(dropped[0][far_hits] >= float(sensor.range_max) - 1e-4)

    noisy, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(radius, 0.0),
        ego_yaw=math.pi / 2.0,
        seed=11,
        env_overrides={"lidar_dropout_prob": 0.5},
    )
    frac = float((noisy[0] >= float(sensor.range_max) - 1e-4).float().mean())
    assert 0.35 <= frac <= 0.65


def test_dr_off_lidar_is_deterministic_and_clean(warp_runtime):
    del warp_runtime
    _configure_cpu()
    half = 1.5
    centerline, width_left, width_right = make_circle_track(
        radius=20.0, n=360, w_left=half, w_right=half
    )
    first, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(20.0, 0.0),
        ego_yaw=math.pi / 2.0,
        seed=3,
    )
    second, _, _ = _launch_lidar(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        ego_pose=(20.0, 0.0),
        ego_yaw=math.pi / 2.0,
        seed=3,
    )
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert abs(float(first[0, _LEFT]) - half) <= _UST10_ACCURACY_M


def test_dual_env_sensor_scans_are_bit_identical(warp_runtime):
    del warp_runtime
    first = _make_env(num_envs=4, opponent=False)
    second = _make_env(num_envs=4, opponent=False)
    try:
        obs_a, _ = first.reset(seed=17, with_sensors=True)
        obs_b, _ = second.reset(seed=17, with_sensors=True)
        assert torch.equal(obs_a["frenet"], obs_b["frenet"])
        assert torch.equal(obs_a["lidar"], obs_b["lidar"])
        assert torch.equal(obs_a["imu"], obs_b["imu"])
        generator = torch.Generator().manual_seed(0)
        for _ in range(8):
            actions = torch.rand(4, 2, generator=generator) * 0.5
            out_a, _, _, _ = first.step(
                actions, n_steps=first.control_interval, with_sensors=True
            )
            out_b, _, _, _ = second.step(
                actions, n_steps=second.control_interval, with_sensors=True
            )
            assert torch.equal(out_a["frenet"], out_b["frenet"])
            assert torch.equal(out_a["lidar"], out_b["lidar"])
            assert torch.equal(out_a["imu"], out_b["imu"])
    finally:
        first.close()
        second.close()


def test_with_sensors_dict_shapes_and_launch_count(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=4)
    try:
        obs, _ = env.reset(seed=1, with_sensors=True)
        assert set(obs) == {"frenet", "actor", "lidar", "imu"}
        assert obs["frenet"].shape == (4, 392)
        assert obs["actor"].shape == (4, 1097)
        assert obs["lidar"].shape == (4, _NUM_BEAMS)
        assert obs["imu"].shape == (4, 6)
        assert torch.isfinite(obs["lidar"]).all()
        assert torch.isfinite(obs["imu"]).all()

        out, reward, done, extras = env.step(
            torch.zeros(4, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert set(out) == {"frenet", "actor", "lidar", "imu"}
        assert out["frenet"].shape == (4, 392)
        assert out["actor"].shape == (4, 1097)
        assert out["lidar"].shape == (4, _NUM_BEAMS)
        assert out["imu"].shape == (4, 6)
        assert env.step_launch_count == 4
        assert torch.isfinite(reward).all()
        assert done.shape == (4,)
        assert "dr/lidar_range_noise_std" in extras["metrics"]
        assert "dr/imu_accel_bias_x" in extras["metrics"]
        assert "dr/vesc_speed_bias" in extras["metrics"]
    finally:
        env.close()


def test_with_sensors_false_matches_flat_obs_and_skips_kernels(warp_runtime):
    del warp_runtime
    off = _make_env(num_envs=4)
    on = _make_env(num_envs=4)
    try:
        flat, _ = off.reset(seed=9, with_sensors=False)
        bundled, _ = on.reset(seed=9, with_sensors=True)
        assert isinstance(flat, torch.Tensor)
        assert flat.shape == (4, 392)
        assert torch.equal(flat, bundled["frenet"])

        actions = torch.tensor([[0.3, 0.1]]).repeat(4, 1)
        flat_step, reward_off, done_off, _ = off.step(
            actions, n_steps=off.control_interval, with_sensors=False
        )
        bundled_step, reward_on, done_on, _ = on.step(
            actions, n_steps=on.control_interval, with_sensors=True
        )
        assert off.step_launch_count == 3
        assert on.step_launch_count == 4
        assert torch.equal(flat_step, bundled_step["frenet"])
        assert torch.equal(reward_off, reward_on)
        assert torch.equal(done_off, done_on)
        assert torch.equal(
            off.read_state()["base_pos"], on.read_state()["base_pos"]
        )
    finally:
        off.close()
        on.close()


def test_env_imu_tracks_read_state_when_clean(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2)
    try:
        env.reset(seed=2, with_sensors=True)
        action = torch.tensor([[0.6, 0.2]]).repeat(2, 1)
        for _ in range(5):
            out, _, _, _ = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
        state = env.read_state()
        imu = out["imu"]
        assert torch.allclose(imu[:, 0], state["base_lin_acc"][:, 0], atol=1e-5)
        assert torch.allclose(imu[:, 1], state["base_lin_acc"][:, 1], atol=1e-5)
        assert torch.allclose(imu[:, 5], state["base_ang_vel"][:, 2], atol=1e-5)
        assert torch.allclose(
            imu[:, 2],
            torch.full((2,), float(env._sim_params.gravity)),
            atol=1e-5,
        )
        assert torch.allclose(imu[:, 3:5], torch.zeros(2, 2), atol=1e-5)
    finally:
        env.close()


def test_has_opponent_env_forward_beam_sees_car(warp_runtime):
    del warp_runtime
    solo = _make_env(num_envs=1, opponent=False)
    duel = _make_env(num_envs=1, opponent=True)
    try:
        solo.reset(seed=1)
        state = solo.read_state()
        pose = state["base_pos"][0, :2].clone()
        quat = state["base_quat"][0]
        yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
        gap = 2.5
        heading = torch.tensor([math.cos(yaw), math.sin(yaw)])
        opp_pose = pose + gap * heading
        solo_obs, _ = solo.reset_to(
            pose, yaw, 0.0, seed=1, with_sensors=True
        )
        duel_obs, _ = duel.reset_to(
            pose,
            yaw,
            0.0,
            opponent_pose=opp_pose,
            opponent_yaw=yaw,
            opponent_speed=0.0,
            seed=1,
            with_sensors=True,
        )
        forward = float(solo_obs["lidar"][0, _FWD])
        duel_forward = float(duel_obs["lidar"][0, _FWD])
        expected = (
            gap
            - float(duel._sensor_params.lidar_offset_x)
            - 0.5 * float(duel._opponent_params.car_length)
        )
        assert duel_forward == pytest.approx(expected, abs=0.08)
        assert duel_forward < forward
    finally:
        solo.close()
        duel.close()

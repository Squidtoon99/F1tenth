"""Austin spawn: Warp 1,097-D actor obs vs gym occupancy / ROS packing."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "training"
for path in (
    _ROOT,
    _TRAINING,
    _ROOT / "src/racing_rl/f1tenth_rl_agent",
    _ROOT / "libs/f1tenth_policy",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.map_lidar import raycast_distance_field  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_env.kernel import OpponentParams, ResetParams  # noqa: E402
from f1tenth_env.sensors import lidar_solo_kernel  # noqa: E402
from f1tenth_env.utils import (  # noqa: E402
    CorridorDistanceData,
    load_occupancy_distance_field,
)
from f1tenth_env.warp_env import (  # noqa: E402
    _CorridorDistanceStorage,
    _EnvironmentStorage,
    _VehicleStorage,
)
from f1tenth_policy.layout import (  # noqa: E402
    GRAVITY_MS2,
    IMU_START,
    LIDAR_DIM,
    STEER_DELTA0,
    STEER_T,
    THROTTLE_CURRENT,
    VESC_CURRENT,
    VESC_SPEED,
)
from f1tenth_rl_agent.gym_certify import (  # noqa: E402
    GYM_LIDAR_TF_XYYAW,
    WARP_LIDAR_TF_XYYAW,
    WarpLongitudinalPlant,
    lidar_frame_pose,
    spawn_pose_from_centerline,
)
from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    beam_angles_rad,
    body_accel_to_driver,
    pack_actor_observation,
    pack_lidar_from_scan,
)
from f1tenth_rl_agent.spawn_utils import load_map_assets  # noqa: E402
from f1tenth_sim.params import VehicleParams  # noqa: E402
from standalone_trainer import build_env_cfg  # noqa: E402

_AUSTIN_YAML = _TRAINING / "assets" / "Austin.yaml"
_AUSTIN_CL = _TRAINING / "assets" / "Austin_centerline.csv"
_AUSTIN_SHA = "16501f0cf2333305c305efb6cc3110c7a31e2843263067ddc52edfe826633318"
_FWD = 540
_RIGHT = 180
_LEFT = 900


def _configure_cpu():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _spawn():
    return spawn_pose_from_centerline(_AUSTIN_CL)


def _warp_field():
    return load_occupancy_distance_field(
        _AUSTIN_YAML, expected_sha256=_AUSTIN_SHA
    )


def _gym_occupancy_field() -> CorridorDistanceData:
    """Trinary field: unknown cells stamped -1, matching Warp and the gym patch."""
    return load_occupancy_distance_field(
        _AUSTIN_YAML, expected_sha256=_AUSTIN_SHA
    )


def _unknown_as_free_field() -> CorridorDistanceData:
    assets = load_map_assets(_AUSTIN_YAML)
    occupied = assets.occupancy_map <= 128.0
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~occupied).astype(np.float32)
    distance *= float(assets.resolution)
    return CorridorDistanceData(
        distance=np.ascontiguousarray(distance.reshape(-1)),
        width=int(assets.occupancy_map.shape[1]),
        height=int(assets.occupancy_map.shape[0]),
        origin=(float(assets.origin[0]), float(assets.origin[1])),
        resolution=float(assets.resolution),
    )


def _raycast(field, pose_xyyaw, lidar_tf):
    lx, ly, lyaw = lidar_frame_pose(*pose_xyyaw, lidar_tf)
    return raycast_distance_field(
        field,
        (lx, ly, lyaw),
        beam_angles_rad(),
        0.06,
        30.0,
        lidar_offset=(0.0, 0.0, 0.0),
    ).astype(np.float32)


def _warp_kernel_scan(field, pose_xyyaw, lidar_tf):
    _configure_cpu()
    sensor = F1tenthEnv(
        num_envs=1,
        env_cfg=_austin_env_cfg(),
        obs_cfg=DEFAULT_CONFIG["obs"],
        reward_cfg=DEFAULT_CONFIG["reward"],
    )
    try:
        params = sensor._sensor_params
        params.lidar_offset_x = float(lidar_tf[0])
        params.lidar_offset_y = float(lidar_tf[1])
        params.lidar_offset_yaw = float(lidar_tf[2])
        corridor = _CorridorDistanceStorage(field, "cpu")
        vehicle = VehicleParams.from_config(DEFAULT_CONFIG["env"])
        ego = _VehicleStorage(1, torch.device("cpu"), vehicle)
        env = _EnvironmentStorage(1, torch.device("cpu"))
        ego.tensor["x"][0] = float(pose_xyyaw[0])
        ego.tensor["y"][0] = float(pose_xyyaw[1])
        ego.tensor["yaw"][0] = float(pose_xyyaw[2])
        ranges = torch.zeros(1, LIDAR_DIM, dtype=torch.float32)
        reset = ResetParams()
        opponent = OpponentParams()
        wp.launch(
            lidar_solo_kernel,
            dim=(1, LIDAR_DIM),
            inputs=[
                ego.buffers,
                env.buffers,
                corridor.data,
                params,
                opponent,
                reset,
                wp.from_torch(ranges),
            ],
            device="cpu",
        )
        wp.synchronize()
        return ranges[0].numpy()
    finally:
        sensor.close()


def _austin_env_cfg():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = "Austin"
    cfg["env"]["opponent_strategy"] = None
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_stationary_probability"] = 1.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["sensor"]["lidar_map_yaml"] = str(_AUSTIN_YAML)
    cfg["sensor"]["lidar_map_sha256"] = _AUSTIN_SHA
    return build_env_cfg(cfg)


def _make_austin_env():
    _configure_cpu()
    return F1tenthEnv(
        num_envs=1,
        env_cfg=_austin_env_cfg(),
        obs_cfg=DEFAULT_CONFIG["obs"],
        reward_cfg=DEFAULT_CONFIG["reward"],
    )


def test_austin_spawn_heading_is_centerline_tangent():
    x, y, yaw = _spawn()
    assert (x, y) == (0.0, 0.0)
    assert abs(yaw + 0.6524) < 1e-3


def test_gym_occupancy_matches_warp_trinary_at_spawn(warp_runtime):
    del warp_runtime
    pose = _spawn()
    warp_scan = _raycast(_warp_field(), pose, WARP_LIDAR_TF_XYYAW)
    gym_scan = _raycast(_gym_occupancy_field(), pose, GYM_LIDAR_TF_XYYAW)
    mae = float(np.mean(np.abs(warp_scan - gym_scan)))
    assert mae < 0.05
    assert abs(float(warp_scan[_LEFT]) - float(gym_scan[_LEFT])) < 0.05
    unknown_free = _raycast(_unknown_as_free_field(), pose, GYM_LIDAR_TF_XYYAW)
    assert float(np.mean(np.abs(warp_scan - unknown_free))) > 1.0


def test_warp_kernel_matches_occupancy_raycast_at_spawn(warp_runtime):
    del warp_runtime
    pose = _spawn()
    field = _warp_field()
    python_scan = _raycast(field, pose, WARP_LIDAR_TF_XYYAW)
    kernel_scan = _warp_kernel_scan(field, pose, WARP_LIDAR_TF_XYYAW)
    assert abs(float(python_scan[_FWD]) - float(kernel_scan[_FWD])) < 0.25
    assert abs(float(python_scan[_LEFT]) - float(kernel_scan[_LEFT])) < 0.25
    hit = (python_scan < 29.0) & (kernel_scan < 29.0)
    assert float(np.mean(np.abs(python_scan[hit] - kernel_scan[hit]))) < 0.35


def test_scan_from_vehicle_origin_misses_lidar_mount(warp_runtime):
    del warp_runtime
    pose = _spawn()
    field = _warp_field()
    mounted = _raycast(field, pose, WARP_LIDAR_TF_XYYAW)
    unmounted = _raycast(field, pose, (0.0, 0.0, 0.0))
    assert float(np.max(np.abs(mounted - unmounted))) >= 0.15


def test_actor_obs_at_austin_spawn_matches_ros_packing(warp_runtime):
    del warp_runtime
    pose = _spawn()
    env = _make_austin_env()
    try:
        obs, _ = env.reset_to(
            [pose[0], pose[1]], pose[2], 0.0, seed=0, with_sensors=True
        )
        actor = obs["actor"][0].detach().cpu().numpy()
        lidar = _warp_kernel_scan(_warp_field(), pose, WARP_LIDAR_TF_XYYAW)
        packed_lidar = pack_lidar_from_scan(
            float(beam_angles_rad()[0]),
            float(beam_angles_rad()[1] - beam_angles_rad()[0]),
            lidar,
            scan_range_min=0.06,
        )
        lidar_mae = float(np.mean(np.abs(actor[:LIDAR_DIM] - packed_lidar)))
        assert lidar_mae < 0.05
        imu = actor[IMU_START : IMU_START + 6]
        assert imu[0] == pytest.approx(0.0, abs=0.05)
        assert imu[1] == pytest.approx(0.0, abs=0.05)
        assert imu[2] == pytest.approx(GRAVITY_MS2, abs=0.05)
        assert imu[3] == pytest.approx(0.0, abs=1e-5)
        assert imu[4] == pytest.approx(0.0, abs=1e-5)
        assert imu[5] == pytest.approx(0.0, abs=0.05)
        assert actor[VESC_SPEED] == pytest.approx(0.0, abs=0.05)
        assert actor[VESC_CURRENT] == pytest.approx(0.0, abs=0.05)
        assert actor[THROTTLE_CURRENT] == pytest.approx(0.0, abs=0.05)
        assert actor[STEER_T] == pytest.approx(0.0, abs=1e-5)
        assert actor[STEER_DELTA0] == pytest.approx(0.0, abs=1e-5)

        ax_driver, ay_driver = body_accel_to_driver(0.0, 0.0)
        ros_imu, _ = actor_imu_from_interval(
            [RawImuSample(0.0, ax_driver, ay_driver, GRAVITY_MS2, 0.0, 0.0, 0.0)],
            ImuCalibration(accel_to_ms2=1.0, gyro_to_rads=1.0),
        )
        np.testing.assert_allclose(ros_imu, imu, atol=0.05)
        hist = np.zeros(7, dtype=np.float32)
        packed = pack_actor_observation(
            packed_lidar, ros_imu, 0.0, 0.0, 0.0, 0.0, hist
        )
        proprio_err = np.abs(packed[LIDAR_DIM:] - actor[LIDAR_DIM:])
        assert float(np.max(proprio_err)) < 0.05
    finally:
        env.close()


def test_one_second_full_throttle_matches_warp_plant(warp_runtime):
    del warp_runtime
    pose = _spawn()
    env = _make_austin_env()
    plant = WarpLongitudinalPlant()
    try:
        env.reset_to([pose[0], pose[1]], pose[2], 0.0, seed=0, with_sensors=True)
        vx_plant = 0.0
        vx_env = 0.0
        action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        for _ in range(20):
            out, _, _, _ = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
            vx_env = float(out["actor"][0, VESC_SPEED])
            vx_plant = plant.step(vx_plant, 80.0, 0.0)
        assert 4.0 < vx_plant < 8.0
        assert abs(vx_env - vx_plant) < 1.0
        steer = float(out["actor"][0, STEER_T])
        assert abs(steer) < 0.02
    finally:
        env.close()

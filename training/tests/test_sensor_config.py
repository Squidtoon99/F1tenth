"""Focused tests for sensor config, SensorParams, and corridor EDT bake/upload."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from config import DEFAULT_CONFIG
from conftest import build_track_state, make_circle_track
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.kernel import CorridorDistanceField, SensorParams
from analysis.map_lidar import raycast_distance_field
from f1tenth_env.utils import build_corridor_distance_field
from f1tenth_env.utils import load_occupancy_distance_field
from f1tenth_env.warp_env import _CorridorDistanceStorage
from standalone_trainer import build_config, build_env_cfg, parse_args


def test_default_sensor_block_matches_ust10lx():
    sensor = DEFAULT_CONFIG["sensor"]
    assert sensor["num_beams"] == 1081
    assert sensor["fov_deg"] == 270.0
    assert sensor["range_min_m"] == 0.06
    assert sensor["range_max_m"] == 30.0
    assert sensor["reliable_range_m"] == 10.0
    assert sensor["beam_decimation"] == 1
    assert sensor["max_march_steps"] == 512
    assert sensor["lidar_offset_x"] == 0.27
    assert sensor["lidar_offset_y"] == 0.0
    assert sensor["lidar_offset_yaw"] == 0.0
    assert sensor["lidar_dropout_bin_m"] == 0.0


def test_root_sensor_patch_reaches_warp_env(warp_runtime):
    """build_config sensor.* patches must reach SensorParams via build_env_cfg."""
    del warp_runtime
    patch = {
        "sensor": {
            "max_march_steps": 128,
        }
    }
    args, explicit = parse_args([])
    merged = {"env": {"episode_length": 60}, "sensor": patch["sensor"]}
    cfg = build_config(args, patch=merged, explicit=explicit)
    assert "sensor" not in cfg["env"]

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=build_env_cfg(cfg),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        # Passing cfg["env"] alone would ignore the patch and keep defaults.
        assert env.sensor_cfg["num_beams"] == 1081
        assert env.sensor_cfg["beam_decimation"] == 1
        assert env.sensor_cfg["max_march_steps"] == 128
        assert env.num_lidar_beams == 1081
        assert int(env._sensor_params.max_march_steps) == 128
        assert "vesc_current_scale_a" not in env.sensor_cfg
    finally:
        env.close()


def test_lidar_map_patch_selects_occupancy_distance_field(warp_runtime, tmp_path):
    del warp_runtime
    pgm = tmp_path / "map.pgm"
    pgm.write_bytes(b"P5\n3 2\n255\n" + bytes([0, 254, 254, 254, 254, 0]))
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        "resolution: 0.5\n"
        "origin: [7.0, 8.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
    )
    args, explicit = parse_args([])
    cfg = build_config(
        args,
        patch={
            "env": {"episode_length": 60},
            "sensor": {
                "lidar_map_yaml": str(map_yaml),
                "lidar_map_variant_count": 3,
                "lidar_wall_offset_max_m": 0.05,
                "lidar_map_variant_seed": 23,
            },
        },
        explicit=explicit,
    )
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )

    env = F1tenthEnv(
        num_envs=1,
        env_cfg=build_env_cfg(cfg),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        assert env._corridor_host.origin == (7.0, 8.0)
        assert (env._corridor_host.width, env._corridor_host.height) == (3, 2)
        assert env._corridor_host.variant_count == 3
    finally:
        env.close()


def test_root_sensor_decimation_patch_rejected_by_env(warp_runtime):
    del warp_runtime
    patch = {
        "env": {"episode_length": 60},
        "sensor": {"beam_decimation": 2},
    }
    args, explicit = parse_args([])
    cfg = build_config(args, patch=patch, explicit=explicit)
    assert cfg["sensor"]["beam_decimation"] == 2
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    with pytest.raises(ValueError, match="beam_decimation=1"):
        F1tenthEnv(
            num_envs=1,
            env_cfg=build_env_cfg(cfg),
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        )


def test_default_env_cfg_keeps_ust10lx_sensor_geometry(warp_runtime):
    del warp_runtime
    args, explicit = parse_args([])
    cfg = build_config(
        args, patch={"env": {"episode_length": 60}}, explicit=explicit
    )
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=build_env_cfg(cfg),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        assert env.num_lidar_beams == 1081
        assert int(env._sensor_params.max_march_steps) == 512
        assert float(env._sensor_params.lidar_offset_x) == pytest.approx(0.27)
    finally:
        env.close()


def test_default_current_scale_is_80a_drive_20a_brake_envelope():
    env = DEFAULT_CONFIG["env"]
    assert env["i_drive_max_a"] == pytest.approx(80.0)
    assert env["i_brake_max_a"] == pytest.approx(20.0)
    assert env["i_slew_a_per_s"] == pytest.approx(200.0)
    assert env["f_drive_max"] == pytest.approx(23.0)
    assert env["power_max"] == pytest.approx(320.0)
    assert env["warp_sim"]["longitudinal_slew_rate_per_s"] == pytest.approx(2.5)
    assert env["warp_sim"]["longitudinal_slew_rate_per_s"] == pytest.approx(
        env["i_slew_a_per_s"] / env["i_drive_max_a"]
    )


def test_default_sensor_dr_matches_measured_evidence():
    dr = DEFAULT_CONFIG["env"]["domain_randomization"]
    assert dr["lidar_extrinsic_xy_range"] == [-0.01, 0.01]
    assert dr["lidar_extrinsic_yaw_range"] == [-0.008726646, 0.008726646]
    assert dr["tire_friction_range"] == [0.5, 1.132]
    assert dr["imu_accel_bias_range"] == [-0.15, 0.15]
    assert dr["imu_gyro_bias_range"] == [
        -0.006607313505033768,
        0.006607313505033768,
    ]
    assert dr["imu_accel_noise_std_range"] == [
        0.007283811484662825,
        0.04370286890797695,
    ]
    assert dr["imu_gyro_noise_std_range"] == [
        0.0006188990447305095,
        0.0037133942683830567,
    ]
    assert dr["imu_axis_misalign_range"] == [0.0, 0.02]
    assert dr["vesc_current_bias_range"] == [-0.005, 0.005]
    assert dr["vesc_current_noise_std_range"] == [0.0, 0.005]
    noop_keys = [
        "lidar_range_noise_std_range",
        "lidar_far_dropout_prob_range",
        "lidar_dropout_prob_range",
        "lidar_angle_bias_range",
        "vesc_speed_bias_range",
        "vesc_speed_noise_std_range",
    ]
    for key in noop_keys:
        assert key in dr
        lo, hi = dr[key]
        assert lo == 0.0 and hi == 0.0, key


def test_sensor_params_struct_matches_config_geometry():
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
    params.lidar_dropout_bin_m = float(sensor.get("lidar_dropout_bin_m", 0.0))
    params.max_march_steps = max(1, int(sensor.get("max_march_steps", 512)))

    assert int(params.num_beams) == 1081
    assert abs(float(params.angle_min) - math.radians(-135.0)) < 1e-6
    assert abs(float(params.angle_increment) - math.radians(0.25)) < 1e-6
    assert int(params.max_march_steps) == 512
    span = float(params.angle_min) + float(params.angle_increment) * (
        num_beams - 1
    )
    assert abs(span - math.radians(135.0)) < 1e-6


def test_corridor_distance_field_centerline_near_half_width(real_modules):
    half_width = 1.5
    centerline, width_left, width_right = make_circle_track(
        radius=20.0, n=360, w_left=half_width, w_right=half_width
    )
    host = build_corridor_distance_field(
        centerline, width_left, width_right, resolution=0.05
    )
    assert host.width > 0 and host.height > 0
    assert host.distance.shape == (host.height * host.width,)
    assert host.resolution == 0.05

    # Sample a few centerline vertices; distance should be ~ half-width.
    grid = host.distance.reshape(host.height, host.width)
    ox, oy = host.origin
    res = host.resolution
    errors = []
    for point in centerline[::45]:
        ix = int(np.floor((point[0] - ox) / res))
        iy = int(np.floor((point[1] - oy) / res))
        assert 0 <= ix < host.width and 0 <= iy < host.height
        errors.append(abs(float(grid[iy, ix]) - half_width))
    assert max(errors) < 2.0 * res

    # A wall vertex should land near a zero-distance cell.
    left, right = real_modules.utils.compute_track_boundaries(
        centerline, width_left, width_right
    )
    wall_dists = []
    for point in np.concatenate([left[::90], right[::90]], axis=0):
        ix = int(np.floor((point[0] - ox) / res))
        iy = int(np.floor((point[1] - oy) / res))
        if 0 <= ix < host.width and 0 <= iy < host.height:
            wall_dists.append(float(grid[iy, ix]))
    assert wall_dists
    assert min(wall_dists) <= res


def test_load_occupancy_distance_field_uses_ros_map_coordinates(tmp_path):
    pgm = tmp_path / "map.pgm"
    pgm.write_bytes(b"P5\n3 2\n255\n" + bytes([0, 205, 254, 254, 0, 205]))
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        "resolution: 0.5\n"
        "origin: [1.0, 2.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
    )

    field = load_occupancy_distance_field(map_yaml)
    grid = field.distance.reshape(field.height, field.width)

    assert (field.width, field.height) == (3, 2)
    assert field.origin == (1.0, 2.0)
    assert field.resolution == pytest.approx(0.5)
    # PGM rows are top-to-bottom; the distance field is bottom-to-top in map space.
    assert grid[0, 1] == 0.0
    assert grid[1, 0] == 0.0
    # Unknown trinary pixels (205) terminate rays at max range; they are not walls.
    assert grid[0, 2] < 0.0


def test_occupancy_unknown_gap_returns_max_range(tmp_path):
    image = np.full((21, 41), 254, dtype=np.uint8)
    image[:, 20] = 0
    image[10, 20] = 254
    image[:, 21:] = 205
    pgm_image = np.flipud(image)
    pgm = tmp_path / "map.pgm"
    pgm.write_bytes(
        f"P5\n{pgm_image.shape[1]} {pgm_image.shape[0]}\n255\n".encode()
        + pgm_image.tobytes()
    )
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        "resolution: 0.05\n"
        "origin: [0.0, 0.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
    )
    field = load_occupancy_distance_field(map_yaml)
    scan = raycast_distance_field(
        field, (0.5, 0.525, 0.0), np.array([0.0]), 0.02, 30.0
    )
    assert float(scan[0]) == 30.0


def test_load_occupancy_distance_field_rejects_wrong_image_hash(tmp_path):
    pgm = tmp_path / "map.pgm"
    pgm.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 254, 254, 0]))
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        "resolution: 0.05\n"
        "origin: [0.0, 0.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_occupancy_distance_field(map_yaml, expected_sha256="0" * 64)


def test_occupancy_distance_variants_are_seeded_bounded_and_keep_base(tmp_path):
    image = np.full((31, 31), 254, dtype=np.uint8)
    image[:, 15] = 0
    pgm = tmp_path / "map.pgm"
    pgm.write_bytes(b"P5\n31 31\n255\n" + image.tobytes())
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\n"
        "mode: trinary\n"
        "resolution: 0.05\n"
        "origin: [0.0, 0.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n"
    )

    base = load_occupancy_distance_field(map_yaml)
    first = load_occupancy_distance_field(
        map_yaml,
        variant_count=4,
        max_wall_offset_m=0.10,
        variant_seed=17,
    )
    second = load_occupancy_distance_field(
        map_yaml,
        variant_count=4,
        max_wall_offset_m=0.10,
        variant_seed=17,
    )
    changed_seed = load_occupancy_distance_field(
        map_yaml,
        variant_count=4,
        max_wall_offset_m=0.10,
        variant_seed=18,
    )

    assert first.variant_count == 4
    fields = first.distance.reshape(4, first.height, first.width)
    np.testing.assert_array_equal(fields[0].reshape(-1), base.distance)
    np.testing.assert_array_equal(first.distance, second.distance)
    assert not np.array_equal(first.distance, changed_seed.distance)
    assert any(not np.array_equal(fields[0], variant) for variant in fields[1:])
    assert np.max(np.abs(fields[1:] - fields[0])) <= 0.15


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_corridor_distance_field_uploads_to_warp(real_modules, device):
    del real_modules
    centerline, width_left, width_right = make_circle_track(
        radius=12.0, n=180, w_left=1.0, w_right=1.0
    )
    host = build_corridor_distance_field(
        centerline, width_left, width_right, resolution=0.05
    )
    storage = _CorridorDistanceStorage(host, device)
    field: CorridorDistanceField = storage.data
    assert int(field.width) == host.width
    assert int(field.height) == host.height
    assert abs(float(field.resolution) - host.resolution) < 1e-9
    assert abs(float(field.origin[0]) - host.origin[0]) < 1e-6
    assert abs(float(field.origin[1]) - host.origin[1]) < 1e-6
    uploaded = field.distance.numpy().reshape(host.height, host.width)
    np.testing.assert_allclose(
        uploaded, host.distance.reshape(host.height, host.width), rtol=0.0, atol=0.0
    )


def test_build_track_state_feeds_corridor_bake(real_modules):
    centerline, width_left, width_right = make_circle_track(radius=15.0, n=240)
    state = build_track_state(
        real_modules.utils, centerline, width_left, width_right
    )
    host = build_corridor_distance_field(
        state["centerline"],
        state["w_tr_left"],
        state["w_tr_right"],
        resolution=0.05,
    )
    assert host.distance.min() == 0.0
    assert host.distance.max() > 0.5


def test_straight_corridor_distance_field_centerline_is_half_width(real_modules):
    del real_modules
    from conftest import make_straight_track

    half_width = 1.5
    centerline, width_left, width_right = make_straight_track(
        length=80.0, n=320, w_left=half_width, w_right=half_width
    )
    host = build_corridor_distance_field(
        centerline, width_left, width_right, resolution=0.025
    )
    grid = host.distance.reshape(host.height, host.width)
    ox, oy = host.origin
    res = host.resolution
    ix = int(np.floor((40.0 - ox) / res))
    iy = int(np.floor((0.0 - oy) / res))
    assert 0 <= ix < host.width and 0 <= iy < host.height
    assert abs(float(grid[iy, ix]) - half_width) <= res

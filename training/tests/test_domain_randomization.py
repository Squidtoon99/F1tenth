"""Warp tests for config-gated domain randomization."""

from __future__ import annotations

import copy
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_sim import VehicleParams  # noqa: E402

_SENSOR_DR_METRICS = (
    "dr/lidar_extrinsic_x",
    "dr/lidar_extrinsic_y",
    "dr/lidar_extrinsic_yaw",
    "dr/lidar_angle_bias",
    "dr/lidar_range_noise_std",
    "dr/lidar_dropout_prob",
    "dr/lidar_far_dropout_prob",
    "dr/imu_accel_bias_x",
    "dr/imu_accel_bias_y",
    "dr/imu_accel_bias_z",
    "dr/imu_gyro_bias_x",
    "dr/imu_gyro_bias_y",
    "dr/imu_gyro_bias_z",
    "dr/imu_accel_noise_std",
    "dr/imu_gyro_noise_std",
    "dr/imu_axis_misalign",
    "dr/vesc_speed_bias",
    "dr/vesc_current_bias",
    "dr/vesc_speed_noise_std",
    "dr/vesc_current_noise_std",
)


def _make_env(
    *,
    enable_dr: bool,
    num_envs: int = 4,
    sensor_dr: dict | None = None,
) -> F1tenthEnv:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dr = dict(cfg["env"]["domain_randomization"])
    dr["enabled"] = enable_dr
    if enable_dr:
        dr["tire_friction_range"] = [0.5, 0.8]
        dr["vehicle_mass_range"] = [3.0, 4.0]
        dr["action_latency_steps_range"] = [0, 2]
        dr["obs_noise_std_range"] = [0.01, 0.05]
        if sensor_dr:
            dr.update(sensor_dr)
    cfg["env"]["domain_randomization"] = dr
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _collect_dr_samples(env: F1tenthEnv, resets: int) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "tire_friction": [],
        "vehicle_mass": [],
        "action_latency_steps": [],
        "obs_noise_std": [],
    }
    for _ in range(resets):
        env.reset()
        metrics = env.extras.get("metrics", {})
        samples["tire_friction"].extend(
            metrics["dr/tire_friction"].detach().cpu().tolist()
        )
        samples["vehicle_mass"].extend(
            metrics["dr/vehicle_mass"].detach().cpu().tolist()
        )
        samples["action_latency_steps"].extend(
            metrics["dr/action_latency_steps"].detach().cpu().tolist()
        )
        samples["obs_noise_std"].extend(
            metrics["dr/obs_noise_std"].detach().cpu().tolist()
        )
    return samples


def _collect_sensor_dr_samples(
    env: F1tenthEnv, resets: int
) -> dict[str, list[float]]:
    samples = {key: [] for key in _SENSOR_DR_METRICS}
    for _ in range(resets):
        env.reset()
        metrics = env.extras.get("metrics", {})
        for key in _SENSOR_DR_METRICS:
            samples[key].extend(metrics[key].detach().cpu().tolist())
    return samples


def test_dr_disabled_matches_baseline(warp_runtime):
    num_envs = 4
    base_tf = float(DEFAULT_CONFIG["env"]["tire_friction"])
    base_mass = VehicleParams.from_config(DEFAULT_CONFIG["env"]).mass
    env = _make_env(enable_dr=False, num_envs=num_envs)
    try:
        env.reset()
        metrics = env.extras["metrics"]
        assert torch.allclose(
            metrics["dr/tire_friction"],
            torch.full((num_envs,), base_tf),
        )
        assert torch.allclose(
            metrics["dr/vehicle_mass"],
            torch.full((num_envs,), base_mass),
        )
        assert torch.all(
            metrics["dr/action_latency_steps"] == 1.0
        ), "baseline keeps simulate_action_latency=1 step"
        assert torch.allclose(metrics["dr/obs_noise_std"], torch.zeros(num_envs))
        for key in _SENSOR_DR_METRICS:
            assert torch.allclose(
                metrics[key], torch.zeros(num_envs)
            ), f"{key} should be zero when DR is disabled"
        loads = env.read_wheel_state()["tyre_load"]
        assert torch.allclose(loads, torch.ones_like(loads))
    finally:
        env.close()


def test_dr_enabled_samples_vary_and_respect_bounds(warp_runtime):
    num_envs = 4
    env = _make_env(enable_dr=True, num_envs=num_envs)
    try:
        samples = _collect_dr_samples(env, resets=6)
        for key, vals in samples.items():
            assert len(set(round(v, 4) for v in vals)) > 1, f"{key} did not vary"
        assert all(0.5 <= v <= 0.8 for v in samples["tire_friction"])
        assert all(3.0 <= v <= 4.0 for v in samples["vehicle_mass"])
        assert all(0 <= v <= 2 for v in samples["action_latency_steps"])
        assert all(0.01 <= v <= 0.05 for v in samples["obs_noise_std"])

        control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
        obs, _ = env.reset()
        assert torch.equal(obs[:, 384:390], torch.zeros_like(obs[:, 384:390]))
        for _ in range(120):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.4
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
            assert torch.equal(obs[:, 384:390], torch.zeros_like(obs[:, 384:390]))
    finally:
        env.close()


def test_sensor_dr_metrics_vary_within_bounds(warp_runtime):
    num_envs = 4
    sensor_dr = {
        "lidar_range_noise_std_range": [0.01, 0.04],
        "lidar_far_dropout_prob_range": [0.1, 0.4],
        "lidar_dropout_prob_range": [0.05, 0.2],
        "lidar_angle_bias_range": [-0.02, 0.02],
        "lidar_extrinsic_xy_range": [-0.03, 0.03],
        "lidar_extrinsic_yaw_range": [-0.01, 0.01],
        "imu_accel_bias_range": [-0.05, 0.05],
        "imu_gyro_bias_range": [-0.02, 0.02],
        "imu_accel_noise_std_range": [0.01, 0.03],
        "imu_gyro_noise_std_range": [0.005, 0.02],
        "imu_axis_misalign_range": [-0.01, 0.01],
    }
    env = _make_env(enable_dr=True, num_envs=num_envs, sensor_dr=sensor_dr)
    try:
        samples = _collect_sensor_dr_samples(env, resets=8)
        bounds = {
            "dr/lidar_range_noise_std": (0.01, 0.04),
            "dr/lidar_far_dropout_prob": (0.1, 0.4),
            "dr/lidar_dropout_prob": (0.05, 0.2),
            "dr/lidar_angle_bias": (-0.02, 0.02),
            "dr/lidar_extrinsic_x": (-0.03, 0.03),
            "dr/lidar_extrinsic_y": (-0.03, 0.03),
            "dr/lidar_extrinsic_yaw": (-0.01, 0.01),
            "dr/imu_accel_bias_x": (-0.05, 0.05),
            "dr/imu_accel_bias_y": (-0.05, 0.05),
            "dr/imu_accel_bias_z": (-0.05, 0.05),
            "dr/imu_gyro_bias_x": (-0.02, 0.02),
            "dr/imu_gyro_bias_y": (-0.02, 0.02),
            "dr/imu_gyro_bias_z": (-0.02, 0.02),
            "dr/imu_accel_noise_std": (0.01, 0.03),
            "dr/imu_gyro_noise_std": (0.005, 0.02),
            "dr/imu_axis_misalign": (-0.01, 0.01),
        }
        for key, (lo, hi) in bounds.items():
            vals = samples[key]
            assert len(set(round(v, 5) for v in vals)) > 1, f"{key} did not vary"
            assert all(lo - 1e-6 <= v <= hi + 1e-6 for v in vals), key

        obs, _, _, extras = env.step(
            torch.zeros(num_envs, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert obs["lidar"].shape == (num_envs, 1081)
        assert obs["imu"].shape == (num_envs, 6)
        assert torch.isfinite(obs["lidar"]).all()
        assert torch.isfinite(obs["imu"]).all()
        assert extras["metrics"]["dr/lidar_dropout_prob"].shape == (num_envs,)
    finally:
        env.close()


def test_sensor_dr_off_yields_clean_deterministic_scans(warp_runtime):
    num_envs = 2
    env = _make_env(enable_dr=False, num_envs=num_envs)
    try:
        first, _ = env.reset(seed=4, with_sensors=True)
        second, _ = env.reset(seed=4, with_sensors=True)
        assert torch.equal(first["lidar"], second["lidar"])
        assert torch.equal(first["imu"], second["imu"])
        metrics = env.extras["metrics"]
        for key in _SENSOR_DR_METRICS:
            assert torch.allclose(metrics[key], torch.zeros(num_envs)), key
    finally:
        env.close()

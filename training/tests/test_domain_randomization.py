"""Warp tests for config-gated domain randomization."""

from __future__ import annotations

import copy
import json
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_sim.params import STEER_RATE_UNBOUNDED, VehicleParams  # noqa: E402

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
            metrics["dr/action_latency_steps"] == 0.0
        ), "baseline keeps simulate_action_latency=0 policy steps"
        assert torch.allclose(metrics["dr/obs_noise_std"], torch.zeros(num_envs))
        for key in _SENSOR_DR_METRICS:
            assert torch.allclose(
                metrics[key], torch.zeros(num_envs)
            ), f"{key} should be zero when DR is disabled"
        loads = env.read_wheel_state()["tyre_load"]
        assert torch.allclose(loads, torch.ones_like(loads))
        params = VehicleParams.from_config(DEFAULT_CONFIG["env"])
        c_f, c_r = params.axle_stiffness()
        assert torch.allclose(
            metrics["dr/yaw_inertia"],
            torch.full((num_envs,), params.izz),
        )
        assert torch.allclose(
            metrics["dr/cornering_stiffness_front"],
            torch.full((num_envs,), c_f),
            atol=1e-4,
        )
        assert torch.allclose(
            metrics["dr/cornering_stiffness_rear"],
            torch.full((num_envs,), c_r),
            atol=1e-4,
        )
        assert torch.allclose(
            metrics["dr/steer_time_constant"],
            torch.full((num_envs,), params.t_delta),
        )
        assert torch.allclose(
            metrics["dr/steer_rate_max"],
            torch.full((num_envs,), STEER_RATE_UNBOUNDED),
        )
        assert torch.allclose(
            metrics["dr/steer_gain"],
            torch.full((num_envs,), params.steer_gain),
        )
        assert metrics["dr/yaw_inertia"].device == env.device
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
        assert torch.equal(obs[:, 384:392], torch.zeros_like(obs[:, 384:392]))
        for _ in range(120):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.4
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
            assert torch.equal(obs[:, 384:392], torch.zeros_like(obs[:, 384:392]))
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


def test_default_dr_keeps_nominal_dynamics(warp_runtime):
    num_envs = 8
    env = _make_env(enable_dr=True, num_envs=num_envs)
    params = VehicleParams.from_config(DEFAULT_CONFIG["env"])
    try:
        env.reset()
        metrics = env.extras["metrics"]
        assert torch.allclose(
            metrics["dr/yaw_inertia"],
            torch.full((num_envs,), params.izz),
        )
        mass = metrics["dr/vehicle_mass"].detach()
        expected_c_f = []
        expected_c_r = []
        for value in mass.cpu().tolist():
            c_f, c_r = params.axle_stiffness(value)
            expected_c_f.append(c_f)
            expected_c_r.append(c_r)
        assert torch.allclose(
            metrics["dr/cornering_stiffness_front"].cpu(),
            torch.tensor(expected_c_f),
            atol=1e-4,
        )
        assert torch.allclose(
            metrics["dr/cornering_stiffness_rear"].cpu(),
            torch.tensor(expected_c_r),
            atol=1e-4,
        )
        assert torch.allclose(
            metrics["dr/steer_time_constant"],
            torch.full((num_envs,), params.t_delta),
        )
        assert torch.allclose(
            metrics["dr/steer_rate_max"],
            torch.full((num_envs,), STEER_RATE_UNBOUNDED),
        )
        assert torch.allclose(
            metrics["dr/drive_scale"],
            torch.ones(num_envs),
        )
        assert torch.allclose(
            metrics["dr/steer_gain"],
            torch.full((num_envs,), params.steer_gain),
        )
        assert metrics["dr/yaw_inertia"].device == env.device
        assert metrics["dr/cornering_stiffness_front"].device == env.device
    finally:
        env.close()


_DYNAMICS_DR = {
    "yaw_inertia_range": [0.045, 0.14],
    "cornering_stiffness_front_range": [90.0, 200.0],
    "cornering_stiffness_rear_range": [90.0, 200.0],
    "steer_time_constant_range": [0.03, 0.15],
    "steer_rate_max_range": [2.5, 5.0],
    "drive_scale_range": [0.8, 1.2],
    "vehicle_mass_range": [3.1, 3.79],
    "tire_friction_range": [0.65, 1.1],
}


def test_dynamics_dr_samples_vary_and_respect_bounds(warp_runtime):
    num_envs = 8
    env = _make_env(enable_dr=True, num_envs=num_envs, sensor_dr=_DYNAMICS_DR)
    keys = {
        "dr/yaw_inertia": (0.045, 0.14),
        "dr/cornering_stiffness_front": (90.0, 200.0),
        "dr/cornering_stiffness_rear": (90.0, 200.0),
        "dr/steer_time_constant": (0.03, 0.15),
        "dr/steer_rate_max": (2.5, 5.0),
        "dr/drive_scale": (0.8, 1.2),
        "dr/vehicle_mass": (3.1, 3.79),
        "dr/tire_friction": (0.65, 1.1),
    }
    try:
        samples = {key: [] for key in keys}
        for _ in range(8):
            env.reset()
            metrics = env.extras["metrics"]
            for key in keys:
                tensor = metrics[key]
                assert tensor.device == env.device, key
                samples[key].extend(tensor.detach().cpu().tolist())
        for key, (lo, hi) in keys.items():
            vals = samples[key]
            assert len(set(round(v, 4) for v in vals)) > 1, f"{key} did not vary"
            assert all(lo - 1e-5 <= v <= hi + 1e-5 for v in vals), key

        control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
        obs, _ = env.reset()
        for _ in range(40):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.4
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_austin_e2e_ppo_dr_config_enables_dynamics_ranges():
    path = os.path.join(_REPO_ROOT, "configs", "austin_e2e_ppo_dr.json")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    dr = cfg["env"]["domain_randomization"]
    assert dr["enabled"] is True
    assert dr["yaw_inertia_range"] == [0.045, 0.14]
    assert dr["tire_friction_range"] == [0.65, 1.1]
    assert dr["cornering_stiffness_front_range"] == [90.0, 200.0]
    assert dr["cornering_stiffness_rear_range"] == [90.0, 200.0]
    assert dr["vehicle_mass_range"] == [3.1, 3.79]
    assert dr["steer_time_constant_range"] == [0.03, 0.15]
    assert dr["steer_rate_max_range"] == [2.5, 5.0]
    assert dr["drive_scale_range"] == [0.8, 1.2]
    assert dr["action_latency_steps_range"] == [0, 0]
    assert "steer_gain_range" not in dr
    assert cfg["ppo"]["entropy_coef_end"] == 0.0
    assert cfg["ppo"]["entropy_coef_horizon_transitions"] == 32000000
    assert cfg["env"]["track"] == "Austin"


def test_austin_e2e_ppo_dr_config_is_a_valid_patch():
    from standalone_trainer import load_config_patch

    path = os.path.join(_REPO_ROOT, "configs", "austin_e2e_ppo_dr.json")
    patch, _meta = load_config_patch(path)
    assert patch["env"]["domain_randomization"]["yaw_inertia_range"] == [
        0.045,
        0.14,
    ]


def test_steer_gain_and_latency_dr_samples_vary_and_respect_bounds(warp_runtime):
    num_envs = 8
    extra = {
        **_DYNAMICS_DR,
        "steer_gain_range": [1.0, 1.10714],
        "action_latency_steps_range": [0, 3],
    }
    env = _make_env(enable_dr=True, num_envs=num_envs, sensor_dr=extra)
    keys = {
        "dr/steer_gain": (1.0, 1.10714),
        "dr/action_latency_steps": (0.0, 3.0),
        "dr/yaw_inertia": (0.045, 0.14),
        "dr/tire_friction": (0.65, 1.1),
    }
    try:
        samples = {key: [] for key in keys}
        for _ in range(8):
            env.reset()
            metrics = env.extras["metrics"]
            for key in keys:
                tensor = metrics[key]
                assert tensor.device == env.device, key
                samples[key].extend(tensor.detach().cpu().tolist())
        for key, (lo, hi) in keys.items():
            vals = samples[key]
            assert len(set(round(v, 4) for v in vals)) > 1, f"{key} did not vary"
            assert all(lo - 1e-5 <= v <= hi + 1e-5 for v in vals), key
        assert env.env_cfg["t_delta"] == 0.09
        from f1tenth_sim.dynamics import COMMAND_DELAY_SUBSTEPS

        assert COMMAND_DELAY_SUBSTEPS == 2

        control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
        obs, _ = env.reset()
        for _ in range(40):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.4
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_austin_e2e_ppo_baseline_v6_dr_config():
    from standalone_trainer import load_config_patch

    path = os.path.join(_REPO_ROOT, "configs", "austin_e2e_ppo_baseline_v6_dr.json")
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    dr = cfg["env"]["domain_randomization"]
    assert dr["enabled"] is True
    assert dr["yaw_inertia_range"] == [0.045, 0.14]
    assert dr["tire_friction_range"] == [0.65, 1.1]
    assert dr["cornering_stiffness_front_range"] == [90.0, 200.0]
    assert dr["cornering_stiffness_rear_range"] == [90.0, 200.0]
    assert dr["vehicle_mass_range"] == [3.1, 3.79]
    assert dr["steer_time_constant_range"] == [0.03, 0.15]
    assert dr["steer_rate_max_range"] == [2.5, 5.0]
    assert dr["drive_scale_range"] == [0.8, 1.2]
    assert dr["steer_gain_range"] == [1.0, 1.10714]
    assert dr["action_latency_steps_range"] == [0, 3]
    assert cfg["env"]["t_delta"] == 0.09
    assert cfg["env"]["steer_gain"] == 1.10714
    assert cfg["reward"]["reward_scales"]["tal"] == 0.2
    assert cfg["ppo"]["entropy_coef"] == 0.01
    assert cfg["ppo"]["entropy_coef_end"] == 0.0
    assert cfg["ppo"]["entropy_coef_horizon_transitions"] == 32000000
    assert "caps_coef" not in cfg.get("ppo", {})
    assert "reversal_coef_throttle" not in cfg.get("ppo", {})
    assert "log_std_max" not in cfg.get("ppo", {})
    assert "lr_schedule" not in cfg.get("ppo", {})
    assert cfg["env"]["track"] == "Austin"
    patch, _meta = load_config_patch(path)
    assert patch["env"]["domain_randomization"]["steer_gain_range"] == [
        1.0,
        1.10714,
    ]
    assert patch["env"]["domain_randomization"]["action_latency_steps_range"] == [
        0,
        3,
    ]

"""Real-Warp tests for full-parity sensor-policy self-play."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.sensors import (
    ACTOR_CMD_CURRENT_START,
    ACTOR_CMD_PRED_START,
    ACTOR_IMU_START,
    ACTOR_LIDAR_DIM,
    ACTOR_OBS_DIM,
    ACTOR_VESC_CURRENT,
    ACTOR_VESC_SPEED,
)
from qrsac import SquashedGaussianMLPActor
from standalone_trainer import ObsNormalizer, SelfPlayManager

_FWD = 540
_CAR_LENGTH = 0.568


def _configure_cpu():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _make_env(
    *,
    num_envs: int = 2,
    opponent_strategy="policy",
    enable_dr: bool = False,
    sensor_dr: dict | None = None,
) -> F1tenthEnv:
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dr = dict(cfg["env"]["domain_randomization"])
    dr["enabled"] = enable_dr
    if sensor_dr:
        dr.update(sensor_dr)
    cfg["env"]["domain_randomization"] = dr
    cfg["env"]["opponent_strategy"] = opponent_strategy
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
        "sensor": cfg["sensor"],
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _make_actor(obs_dim: int = ACTOR_OBS_DIM, seed: int = 0):
    torch.manual_seed(seed)
    return SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=2,
        # Match env default PolicyOpponent width from make_opponent.
        hidden_sizes=[512, 512, 512],
        activation=nn.ReLU,
        act_limit=1.0,
    )


def _seed_opponent(env: F1tenthEnv, seed: int = 0) -> None:
    actor = _make_actor(seed=seed)
    env.refresh_opponent_policy(
        actor.state_dict(), torch.zeros(ACTOR_OBS_DIM), torch.ones(ACTOR_OBS_DIM)
    )


def test_opponent_viewpoint_sees_ego_obb(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=1, opponent_strategy="policy")
    try:
        env.reset(seed=1, with_sensors=True)
        state = env.read_state()
        pose = state["base_pos"][0, :2].clone()
        quat = state["base_quat"][0]
        yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
        gap = 2.5
        heading = torch.tensor([math.cos(yaw), math.sin(yaw)])
        opp_pose = pose + gap * heading
        obs, _ = env.reset_to(
            pose,
            yaw,
            0.0,
            opponent_pose=opp_pose,
            opponent_yaw=yaw + math.pi,
            opponent_speed=0.0,
            seed=1,
            with_sensors=True,
        )
        expected = gap - 0.5 * _CAR_LENGTH
        ego_fwd = float(obs["actor"][0, _FWD])
        opp_fwd = float(obs["opponent_actor"][0, _FWD])
        assert ego_fwd == pytest.approx(expected, abs=0.08)
        assert opp_fwd == pytest.approx(expected, abs=0.08)
        assert ego_fwd < float(env._sensor_params.range_max) - 1.0
        assert opp_fwd < float(env._sensor_params.range_max) - 1.0
    finally:
        env.close()


def test_ego_and_opponent_sensor_noise_are_independent(warp_runtime):
    del warp_runtime
    env = _make_env(
        num_envs=1,
        opponent_strategy="policy",
        enable_dr=True,
        sensor_dr={
            "lidar_range_noise_std_range": [0.05, 0.05],
            "lidar_dropout_prob_range": [0.0, 0.0],
            "lidar_far_dropout_prob_range": [0.0, 0.0],
            "imu_accel_noise_std_range": [0.05, 0.05],
            "imu_gyro_noise_std_range": [0.02, 0.02],
            "vesc_speed_noise_std_range": [0.1, 0.1],
            "vesc_current_noise_std_range": [0.2, 0.2],
            "vesc_speed_bias_range": [0.0, 0.0],
            "vesc_current_bias_range": [0.0, 0.0],
            "imu_accel_bias_range": [0.0, 0.0],
            "imu_gyro_bias_range": [0.0, 0.0],
        },
    )
    try:
        env.reset(seed=3, with_sensors=True)
        state = env.read_state()
        pose = state["base_pos"][0, :2].clone()
        quat = state["base_quat"][0]
        yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
        # Identical pose/state so any shared RNG key would produce equal noise.
        obs, _ = env.reset_to(
            pose,
            yaw,
            1.0,
            opponent_pose=pose,
            opponent_yaw=yaw,
            opponent_speed=1.0,
            seed=3,
            with_sensors=True,
        )
        ego = obs["actor"][0]
        opp = obs["opponent_actor"][0]
        assert not torch.allclose(
            ego[ACTOR_IMU_START : ACTOR_IMU_START + 6],
            opp[ACTOR_IMU_START : ACTOR_IMU_START + 6],
            atol=1e-6,
        )
        assert ego[ACTOR_VESC_SPEED] != pytest.approx(
            float(opp[ACTOR_VESC_SPEED]), abs=1e-6
        )
        assert ego[ACTOR_VESC_CURRENT] != pytest.approx(
            float(opp[ACTOR_VESC_CURRENT]), abs=1e-6
        )
        # LiDAR noise also differs under the view-id key (same geometry).
        assert not torch.allclose(
            ego[:ACTOR_LIDAR_DIM], opp[:ACTOR_LIDAR_DIM], atol=1e-6
        )
    finally:
        env.close()


def test_sensor_selfplay_is_deterministic(warp_runtime):
    del warp_runtime
    first = _make_env(num_envs=2, opponent_strategy="policy")
    second = _make_env(num_envs=2, opponent_strategy="policy")
    try:
        obs_a, _ = first.reset(seed=11, with_sensors=True)
        obs_b, _ = second.reset(seed=11, with_sensors=True)
        assert torch.equal(obs_a["actor"], obs_b["actor"])
        assert torch.equal(obs_a["opponent_actor"], obs_b["opponent_actor"])
        _seed_opponent(first, seed=42)
        _seed_opponent(second, seed=42)
        actions = torch.tensor([[0.4, -0.2], [-0.1, 0.3]])
        for _ in range(3):
            out_a, _, _, _ = first.step(
                actions, n_steps=first.control_interval, with_sensors=True
            )
            out_b, _, _, _ = second.step(
                actions, n_steps=second.control_interval, with_sensors=True
            )
            assert torch.equal(out_a["actor"], out_b["actor"])
            assert torch.equal(out_a["opponent_actor"], out_b["opponent_actor"])
            assert torch.equal(out_a["frenet"], out_b["frenet"])
    finally:
        first.close()
        second.close()


def test_opponent_command_history_is_causal(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=1, opponent_strategy="policy")
    try:
        obs, _ = env.reset(seed=4, with_sensors=True)
        assert torch.equal(
            obs["opponent_actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_PRED_START + 2],
            torch.zeros(4),
        )
        # Seed a known frozen policy so opponent actions are non-zero and repeatable.
        actor = _make_actor(seed=9)
        mean = torch.zeros(ACTOR_OBS_DIM)
        var = torch.ones(ACTOR_OBS_DIM)
        env.refresh_opponent_policy(actor.state_dict(), mean, var)
        with torch.no_grad():
            a0 = env._policy_opponent.act_observation(obs["opponent_actor"])
        out1, _, _, _ = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval, with_sensors=True
        )
        assert torch.allclose(
            out1["opponent_actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_CURRENT_START + 2],
            a0[0],
            atol=1e-5,
        )
        assert torch.allclose(
            out1["opponent_actor"][0, ACTOR_CMD_PRED_START:ACTOR_CMD_PRED_START + 2],
            torch.zeros(2),
            atol=1e-5,
        )
        with torch.no_grad():
            a1 = env._policy_opponent.act_observation(out1["opponent_actor"])
        out2, _, _, _ = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval, with_sensors=True
        )
        assert torch.allclose(
            out2["opponent_actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_CURRENT_START + 2],
            a1[0],
            atol=1e-5,
        )
        assert torch.allclose(
            out2["opponent_actor"][0, ACTOR_CMD_PRED_START:ACTOR_CMD_PRED_START + 2],
            a0[0],
            atol=1e-5,
        )
    finally:
        env.close()


def test_privileged_snapshot_schema_is_rejected(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=1, opponent_strategy="policy")
    try:
        env.reset(seed=0, with_sensors=True)
        privileged = _make_actor(obs_dim=390, seed=1)
        with pytest.raises(ValueError, match="390|expected sensor actor dim 1093"):
            env.refresh_opponent_policy(
                privileged.state_dict(),
                torch.zeros(390),
                torch.ones(390),
            )
        with pytest.raises(ValueError, match="expected sensor actor dim 1093"):
            env.refresh_opponent_policy(
                _make_actor(seed=2).state_dict(),
                torch.zeros(390),
                torch.ones(390),
            )
    finally:
        env.close()


def test_first_batch_normalizer_before_snapshot_bootstrap(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2, opponent_strategy="policy")
    try:
        models = type("M", (), {})()
        models.actor = _make_actor(seed=5)
        normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
        assert float(normalizer.count) == pytest.approx(1e-8)

        obs, _ = env.reset(seed=6, with_sensors=True)
        normalizer.update(obs["actor"])
        assert float(normalizer.count) > 1.0

        mgr = SelfPlayManager(
            pool_size=2,
            snapshot_interval_transitions=10_000,
            refresh_interval_transitions=10_000,
        )
        mgr.seed_snapshot(
            SelfPlayManager.make_snapshot(models, normalizer, transitions=0)
        )
        assert len(mgr.pool) == 1
        assert not torch.equal(mgr.pool[0]["mean"], torch.zeros(ACTOR_OBS_DIM))
        mgr.bootstrap_opponent(env)
        assert env._policy_opponent.obs_mean is not None
        assert torch.allclose(env._policy_opponent.obs_mean.cpu(), mgr.pool[0]["mean"])
    finally:
        env.close()


def test_1v0_skips_opponent_actor_stream(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2, opponent_strategy=None)
    try:
        obs, _ = env.reset(seed=7, with_sensors=True)
        assert "opponent_actor" not in obs
        assert "actor" in obs
        out, _, _, _ = env.step(
            torch.zeros(2, 2), n_steps=env.control_interval, with_sensors=True
        )
        assert "opponent_actor" not in out
        assert env.step_launch_count == 4
    finally:
        env.close()


def test_policy_1v1_covers_all_rows_with_opponent_sensors(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=4, opponent_strategy="policy")
    try:
        obs, _ = env.reset(seed=8, with_sensors=True)
        assert obs["opponent_actor"].shape == (4, ACTOR_OBS_DIM)
        assert torch.isfinite(obs["opponent_actor"]).all()
        assert torch.count_nonzero(obs["opponent_actor"][:, :ACTOR_LIDAR_DIM]) > 0
        actor = _make_actor(seed=3)
        env.refresh_opponent_policy(
            actor.state_dict(), torch.zeros(ACTOR_OBS_DIM), torch.ones(ACTOR_OBS_DIM)
        )
        out, _, _, _ = env.step(
            torch.zeros(4, 2), n_steps=env.control_interval, with_sensors=True
        )
        assert out["opponent_actor"].shape == (4, ACTOR_OBS_DIM)
        assert env.step_launch_count == 5
        # Policy inference consumed the prior opponent-centric 1,093-D observation.
        assert env._policy_opponent.obs_dim == ACTOR_OBS_DIM
    finally:
        env.close()


def test_scripted_1v1_skips_unused_opponent_actor_render(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2, opponent_strategy="scripted")
    try:
        obs, _ = env.reset(seed=9, with_sensors=True)
        assert "opponent_actor" not in obs
        out, _, _, _ = env.step(
            torch.zeros(2, 2), n_steps=env.control_interval, with_sensors=True
        )
        assert "opponent_actor" not in out
        assert env.step_launch_count == 4
    finally:
        env.close()

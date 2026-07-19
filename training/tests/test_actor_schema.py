"""Focused real-Warp tests for the frozen 1,093-D causal actor observation."""

from __future__ import annotations

import copy

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.sensors import (
    ACTOR_CMD_CURRENT_START,
    ACTOR_CMD_PRED_START,
    ACTOR_IMU_DIM,
    ACTOR_IMU_START,
    ACTOR_LIDAR_DIM,
    ACTOR_OBS_DIM,
    ACTOR_VESC_CURRENT,
    ACTOR_VESC_SPEED,
    NATIVE_NUM_BEAMS,
    VESC_CURRENT_SCALE_A,
)


def _configure_cpu():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _make_env(
    *,
    num_envs: int = 4,
    opponent_strategy=None,
    sensor_dr: dict | None = None,
    enable_dr: bool = False,
) -> F1tenthEnv:
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dr = dict(cfg["env"]["domain_randomization"])
    dr["enabled"] = enable_dr
    if sensor_dr:
        dr.update(sensor_dr)
    cfg["env"]["domain_randomization"] = dr
    cfg["env"]["opponent_strategy"] = opponent_strategy
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


def test_actor_layout_constants_match_frozen_offsets():
    assert ACTOR_OBS_DIM == 1093
    assert NATIVE_NUM_BEAMS == 1081
    assert ACTOR_LIDAR_DIM == 1081
    assert ACTOR_IMU_START == 1081
    assert ACTOR_IMU_DIM == 6
    assert ACTOR_VESC_SPEED == 1087
    assert ACTOR_VESC_CURRENT == 1088
    assert ACTOR_CMD_CURRENT_START == 1089
    assert ACTOR_CMD_PRED_START == 1091
    assert VESC_CURRENT_SCALE_A == 10.0
    assert DEFAULT_CONFIG["obs"]["num_actor_obs"] == 1093
    assert DEFAULT_CONFIG["sensor"]["vesc_current_scale_a"] == 10.0


def test_native_beam_decimation_rejected(warp_runtime):
    del warp_runtime
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["sensor"]["beam_decimation"] = 2
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": 1},
        **cfg["env"],
        "sensor": cfg["sensor"],
    }
    with pytest.raises(ValueError, match="beam_decimation=1"):
        F1tenthEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        )


def test_native_beam_count_rejected(warp_runtime):
    del warp_runtime
    _configure_cpu()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["sensor"]["num_beams"] = 541
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": 1},
        **cfg["env"],
        "sensor": cfg["sensor"],
    }
    with pytest.raises(ValueError, match="num_beams=1081"):
        F1tenthEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        )


def test_actor_obs_layout_and_no_privileged_leak(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2)
    try:
        obs, extras = env.reset(seed=0, with_sensors=True)
        assert set(obs) >= {"frenet", "actor", "lidar", "imu"}
        assert obs["actor"].shape == (2, ACTOR_OBS_DIM)
        assert obs["frenet"].shape == (2, 390)
        assert obs["lidar"].data_ptr() == obs["actor"][:, :ACTOR_LIDAR_DIM].data_ptr()
        assert extras["observations"]["critic"] is obs["frenet"]
        assert extras["observations"]["actor"] is obs["actor"]
        # Actor must not embed the privileged opponent block / tyre truth region.
        assert obs["actor"].shape[-1] == ACTOR_OBS_DIM
        assert not torch.equal(
            obs["actor"][:, :390],
            obs["frenet"],
        )
        out, _, _, extras = env.step(
            torch.zeros(2, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert out["actor"].shape == (2, ACTOR_OBS_DIM)
        assert torch.isfinite(out["actor"]).all()
        assert env.step_launch_count == 4
    finally:
        env.close()


def test_command_causality_current_and_predecessor(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=1)
    try:
        obs, _ = env.reset(seed=1, with_sensors=True)
        assert torch.allclose(
            obs["actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_PRED_START + 2],
            torch.zeros(4),
        )

        a0 = torch.tensor([[0.4, -0.2]])
        out0, _, _, _ = env.step(
            a0, n_steps=env.control_interval, with_sensors=True
        )
        cur0 = out0["actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_CURRENT_START + 2]
        pred0 = out0["actor"][0, ACTOR_CMD_PRED_START:ACTOR_CMD_PRED_START + 2]
        assert torch.allclose(cur0, a0[0], atol=1e-5)
        assert torch.allclose(pred0, torch.zeros(2), atol=1e-5)

        a1 = torch.tensor([[-0.3, 0.5]])
        out1, _, _, _ = env.step(
            a1, n_steps=env.control_interval, with_sensors=True
        )
        cur1 = out1["actor"][0, ACTOR_CMD_CURRENT_START:ACTOR_CMD_CURRENT_START + 2]
        pred1 = out1["actor"][0, ACTOR_CMD_PRED_START:ACTOR_CMD_PRED_START + 2]
        assert torch.allclose(cur1, a1[0], atol=1e-5)
        assert torch.allclose(pred1, a0[0], atol=1e-5)
    finally:
        env.close()


def test_reset_clears_command_history(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2)
    try:
        env.reset(seed=2, with_sensors=True)
        action = torch.tensor([[0.7, 0.1], [0.2, -0.4]])
        for _ in range(3):
            env.step(action, n_steps=env.control_interval, with_sensors=True)

        obs, _ = env.reset(seed=3, with_sensors=True)
        cmds = obs["actor"][:, ACTOR_CMD_CURRENT_START:ACTOR_CMD_PRED_START + 2]
        assert torch.allclose(cmds, torch.zeros_like(cmds))
    finally:
        env.close()


def test_ping_pong_actor_buffers_stable_across_step(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=3)
    try:
        first, _ = env.reset(seed=5, with_sensors=True)
        first_actor = first["actor"]
        first_snapshot = first_actor.clone()
        first_ptr = first_actor.data_ptr()

        second, _, _, _ = env.step(
            torch.full((3, 2), 0.25),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert second["actor"].data_ptr() != first_ptr
        assert torch.equal(first_actor, first_snapshot)
        assert not torch.equal(first_actor, second["actor"])
    finally:
        env.close()


def test_vesc_proxies_match_wheel_speed_and_current_scale(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=1)
    try:
        env.reset(seed=6, with_sensors=True)
        # Drive a few steps so applied_effort and wheel omega are nonzero.
        action = torch.tensor([[0.8, 0.0]])
        out = None
        for _ in range(8):
            out, _, _, _ = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
        omega = env._ego.tensor["omega"][0]
        wheel_radius = float(env._sim_params.wheel_radius)
        expected_speed = float(omega.mean() * wheel_radius)
        expected_current = VESC_CURRENT_SCALE_A * float(
            env._ego.tensor["applied_effort"][0]
        )
        assert out["actor"][0, ACTOR_VESC_SPEED] == pytest.approx(
            expected_speed, abs=1e-4
        )
        assert out["actor"][0, ACTOR_VESC_CURRENT] == pytest.approx(
            expected_current, abs=1e-4
        )
    finally:
        env.close()


def test_actor_critic_share_post_transaction_state(warp_runtime):
    """Actor and privileged critic streams must reflect the same ego state."""
    del warp_runtime
    env = _make_env(num_envs=2)
    try:
        env.reset(seed=11, with_sensors=True)
        action = torch.tensor([[0.55, 0.15], [0.35, -0.1]])
        out = None
        for _ in range(6):
            out, _, _, _ = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
        state = env.read_state()
        wheels = env.read_wheel_state()
        # Critic Frenet block: linear/angular velocity slots match live state.
        assert torch.allclose(out["frenet"][:, :2], state["base_lin_vel"][:, :2])
        assert torch.allclose(out["frenet"][:, 2], state["base_ang_vel"][:, 2])
        # Actor VESC speed matches the same wheel state (no one-step lag).
        wheel_radius = float(env._sim_params.wheel_radius)
        expected_speed = wheels["dof_vel"].mean(dim=-1) * wheel_radius
        assert torch.allclose(
            out["actor"][:, ACTOR_VESC_SPEED], expected_speed, atol=1e-4
        )
        assert torch.allclose(
            out["actor"][:, ACTOR_CMD_CURRENT_START:ACTOR_CMD_CURRENT_START + 2],
            action,
            atol=1e-5,
        )
    finally:
        env.close()


def test_vesc_dr_bias_and_noise_bounds(warp_runtime):
    del warp_runtime
    env = _make_env(
        num_envs=1,
        enable_dr=True,
        sensor_dr={
            "vesc_speed_bias_range": [0.5, 0.5],
            "vesc_current_bias_range": [-1.0, -1.0],
            "vesc_speed_noise_std_range": [0.0, 0.0],
            "vesc_current_noise_std_range": [0.0, 0.0],
        },
    )
    try:
        env.reset(seed=7, with_sensors=True)
        action = torch.tensor([[0.6, 0.0]])
        out = None
        for _ in range(5):
            out, _, _, extras = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
        assert float(extras["metrics"]["dr/vesc_speed_bias"][0]) == pytest.approx(0.5)
        assert float(extras["metrics"]["dr/vesc_current_bias"][0]) == pytest.approx(
            -1.0
        )
        omega = env._ego.tensor["omega"][0]
        wheel_radius = float(env._sim_params.wheel_radius)
        base_speed = float(omega.mean() * wheel_radius)
        base_current = VESC_CURRENT_SCALE_A * float(
            env._ego.tensor["applied_effort"][0]
        )
        assert out["actor"][0, ACTOR_VESC_SPEED] == pytest.approx(
            base_speed + 0.5, abs=1e-4
        )
        assert out["actor"][0, ACTOR_VESC_CURRENT] == pytest.approx(
            base_current - 1.0, abs=1e-4
        )
    finally:
        env.close()


def test_opponent_actor_buffers_stable_for_policy_selfplay(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2, opponent_strategy="policy")
    try:
        obs, extras = env.reset(seed=8, with_sensors=True)
        assert "opponent_actor" in obs
        assert obs["opponent_actor"].shape == (2, ACTOR_OBS_DIM)
        assert extras["observations"]["opponent_actor"] is obs["opponent_actor"]
        assert torch.isfinite(obs["opponent_actor"]).all()
        assert torch.count_nonzero(obs["opponent_actor"][:, :ACTOR_LIDAR_DIM]) > 0
        first = obs["opponent_actor"]
        first_snapshot = first.clone()
        second, _, _, _ = env.step(
            torch.zeros(2, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert second["opponent_actor"].data_ptr() != first.data_ptr()
        assert torch.equal(first, first_snapshot)
        # Physics/transaction/observation + fused ego and opponent sensor actors.
        assert env.step_launch_count == 5
    finally:
        env.close()


def test_scripted_opponent_skips_opponent_actor_stream(warp_runtime):
    del warp_runtime
    env = _make_env(num_envs=2, opponent_strategy="scripted")
    try:
        obs, _ = env.reset(seed=9, with_sensors=True)
        assert "opponent_actor" not in obs
        assert "actor" in obs
        out, _, _, _ = env.step(
            torch.zeros(2, 2),
            n_steps=env.control_interval,
            with_sensors=True,
        )
        assert "opponent_actor" not in out
        # Ego-only fused sensor actor (+ physics/transaction/observation).
        assert env.step_launch_count == 4
    finally:
        env.close()

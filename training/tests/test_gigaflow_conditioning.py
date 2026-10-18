"""CPU tests for Gigaflow own-agent conditioning and Table A3 knobs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.gigaflow_cond import (
    C_ACCEL,
    C_THROTTLE,
    C_VEL,
    GIGAFLOW_ACTOR_OBS_DIM,
    GIGAFLOW_C_DIM,
    GIGAFLOW_CRITIC_OBS_DIM,
    format_gigaflow_spec,
    sample_agent_conditioning,
    sample_x,
    speed_cap_from_c,
)
from f1tenth_env.sensors import ACTOR_LIDAR_DIM
from f1tenth_policy.actor import apply_orthogonal_init, make_actor
from f1tenth_policy.layout import GRU_HIDDEN_DIM
from ppo import advantage_filter_running_max_mask
from standalone_trainer import (
    build_config,
    build_ppo_models,
    parse_args,
    unpack_sensor_observations,
)

_TRAINING_ROOT = Path(__file__).resolve().parents[1]


def _gigaflow_cfg(**env_extra):
    patch = json.loads(
        (_TRAINING_ROOT / "configs" / "gigaflow_v2.json").read_text()
    )
    args, explicit = parse_args([])
    args.num_envs = 8
    cfg = build_config(args, patch=patch, explicit=explicit)
    cfg["env"].update(env_extra)
    cfg["scenarios"] = []
    return cfg


def _make_env(cfg, num_envs=2, opponent=False):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    env_cfg = copy.deepcopy(cfg["env"])
    env_cfg["domain_randomization"] = copy.deepcopy(
        env_cfg["domain_randomization"]
    )
    env_cfg["domain_randomization"]["enabled"] = False
    env_cfg["gigaflow_force_track"] = "Oschersleben"
    env_cfg["gigaflow_force_variant"] = "width_0.8"
    if opponent:
        env_cfg["opponent_strategy"] = "policy"
        env_cfg["actor_type"] = cfg["model"]["actor_type"]
        env_cfg["actor_hidden_layers"] = list(
            cfg["model"]["actor_hidden_layers"]
        )
        env_cfg["lidar_pool_bins"] = int(cfg["model"]["lidar_pool_bins"])
        env_cfg["lidar_projection_dim"] = int(
            cfg["model"]["lidar_projection_dim"]
        )
        env_cfg["lidar_dim"] = ACTOR_LIDAR_DIM
    else:
        env_cfg["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def test_sample_x_splits_below_and_above_one():
    rng = np.random.default_rng(0)
    draws = np.array([sample_x(rng, 1.5) for _ in range(400)])
    assert np.all(draws >= 1.0 / 1.5 - 1e-6)
    assert np.all(draws <= 1.5 + 1e-6)
    assert (draws < 1.0).any() and (draws > 1.0).any()


def test_sample_agent_conditioning_shape_and_fixed_embodiment():
    rng = np.random.default_rng(1)
    c = sample_agent_conditioning(rng, 16, car_length=0.568, car_width=0.296)
    assert c.shape == (16, GIGAFLOW_C_DIM)
    assert np.allclose(c[:, 0], 0.568)
    assert np.allclose(c[:, 1], 0.296)
    assert np.all(np.isfinite(c))


def test_spec_prints_c_and_table_a3():
    text = format_gigaflow_spec()
    assert "C_dynamics" in text
    assert "C_reward" in text
    assert "Table A3" in text
    assert "X(1.25)" in text
    assert "running_max" in text


def test_running_max_advantage_filter_drops_small_rows():
    adv = torch.tensor([[0.0, 0.2], [0.0, 10.0]])
    keep, threshold, running = advantage_filter_running_max_mask(
        adv, enabled=True, eta=0.01, running_max=0.0
    )
    assert running == pytest.approx(10.0)
    assert threshold == pytest.approx(0.1)
    assert keep[1, 1]
    assert not keep[0, 0]


def test_orthogonal_init_sets_linear_weights():
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=GIGAFLOW_ACTOR_OBS_DIM,
        act_dim=2,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_dim=ACTOR_LIDAR_DIM,
        proprio_dim=GIGAFLOW_ACTOR_OBS_DIM - ACTOR_LIDAR_DIM,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    apply_orthogonal_init(actor)
    gram = actor.mu_layer.weight @ actor.mu_layer.weight.T
    assert torch.allclose(
        gram, torch.eye(gram.shape[0]) * 2.0, atol=1e-4, rtol=1e-3
    )


def test_two_agents_different_c_speed_caps_on_width_variant():
    cfg = _gigaflow_cfg()
    env = _make_env(cfg, num_envs=2, opponent=False)
    try:
        raw, extras = env.reset(with_sensors=True)
        actor, critic = unpack_sensor_observations(raw)
        assert actor.shape[-1] == GIGAFLOW_ACTOR_OBS_DIM
        assert critic.shape[-1] == GIGAFLOW_CRITIC_OBS_DIM
        assert extras["gigaflow_track"] == "Oschersleben"
        assert extras["gigaflow_variant"] == "width_0.8"
        env._c_ego[:, C_THROTTLE] = 1.0
        env._c_ego[:, C_ACCEL] = 1.0
        env._c_ego[:, C_VEL] = torch.tensor([1.0 / 1.5, 1.5])
        env._apply_gigaflow_plant(torch.ones(2, dtype=torch.bool))
        env._write_conditioning_columns()
        caps = speed_cap_from_c(env._c_ego[:, C_VEL].numpy(), 6.0)
        assert caps[0] == pytest.approx(4.0)
        assert caps[1] == pytest.approx(9.0)
        assert torch.allclose(env._ego.tensor["drive_scale"], torch.ones(2))
        assert not torch.equal(actor[0, -GIGAFLOW_C_DIM:], actor[1, -GIGAFLOW_C_DIM:])
        assert torch.equal(
            critic[:, GIGAFLOW_CRITIC_OBS_DIM - GIGAFLOW_C_DIM :],
            env._c_ego,
        )
        for _ in range(40):
            actions = torch.zeros(2, 2)
            actions[:, 0] = 1.0
            raw, reward, done, extras = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
            actor, critic = unpack_sensor_observations(raw)
            assert torch.isfinite(actor).all()
            assert torch.isfinite(critic).all()
            assert torch.isfinite(reward).all()
            assert extras["gigaflow_track"] == "Oschersleben"
            assert extras["gigaflow_variant"] == "width_0.8"
        vx = env._ego.tensor["vx"]
        assert torch.isfinite(vx).all()
        assert float(vx[0]) <= 4.35
        assert float(vx[1]) > float(vx[0]) + 0.4
        assert critic.shape[-1] == 402
        assert actor[:, :ACTOR_LIDAR_DIM].shape[-1] == ACTOR_LIDAR_DIM
    finally:
        env.close()


def test_one_optimizer_step_lidar_actor_privileged_critic():
    cfg = _gigaflow_cfg()
    cfg["ppo"]["rollout_steps"] = 4
    cfg["ppo"]["env_minibatches"] = 8
    cfg["ppo"]["epochs"] = 1
    cfg["model"]["actor_hidden_layers"] = [32, 32]
    cfg["model"]["critic_hidden_layers"] = [32, 32]
    cfg["model"]["lidar_pool_bins"] = 16
    device = torch.device("cpu")
    from f1tenth_policy.normalizer import ObsNormalizer

    actor_norm = ObsNormalizer(GIGAFLOW_ACTOR_OBS_DIM, device=device)
    critic_norm = ObsNormalizer(GIGAFLOW_CRITIC_OBS_DIM, device=device)
    models, trainer = build_ppo_models(
        cfg, device, actor_norm, critic_norm, num_envs=8
    )
    assert models.actor.obs_dim == GIGAFLOW_ACTOR_OBS_DIM
    assert models.actor.lidar_dim == ACTOR_LIDAR_DIM
    assert models.actor.proprio_dim == GIGAFLOW_ACTOR_OBS_DIM - ACTOR_LIDAR_DIM
    assert trainer.critic_obs_dim == GIGAFLOW_CRITIC_OBS_DIM
    env = _make_env(cfg, num_envs=8, opponent=False)
    try:
        raw, extras = env.reset(with_sensors=True)
        actor_obs, critic_obs = unpack_sensor_observations(raw)
        assert extras["gigaflow_variant"] == "width_0.8"
        trainer.initialize(actor_obs, critic_obs)
        metrics = None
        for _ in range(trainer.rollout_steps):
            action = trainer.act(actor_obs, critic_obs)
            raw, reward, done, extras = env.step(
                action, n_steps=env.control_interval, with_sensors=True
            )
            actor_obs, critic_obs = unpack_sensor_observations(raw)
            timed_out = extras["termination"]["time_out"]
            metrics = trainer.observe(
                actor_obs,
                critic_obs,
                reward,
                done,
                timed_out=timed_out,
                timeout_critic_obs=extras["observations"]["terminal_critic"],
            )
        assert metrics is not None
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert trainer.value_clip == 0.0
        assert trainer.advantage_filter_mode == "running_max"
    finally:
        env.close()

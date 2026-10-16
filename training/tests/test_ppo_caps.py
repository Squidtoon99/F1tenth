from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from f1tenth_policy.layout import executed_plant_action
from ppo import (
    PPOTrainer,
    action_reversal_penalty,
    caps_spatial_loss,
    caps_temporal_loss,
    grad_caps_temporal_loss,
)
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import validate_algorithm_config, validate_config_patch


CRITIC_DIM = 24
ACT_DIM = 2


def test_caps_temporal_loss_zero_on_constant_mean():
    mean = torch.ones(4, 8, 2)
    reset = torch.zeros(4, 8, dtype=torch.bool)
    reset[:, 0] = True
    assert float(caps_temporal_loss(mean, reset)) == 0.0


def test_caps_temporal_loss_positive_and_skips_resets():
    mean = torch.zeros(1, 3, 2)
    mean[0, 1, 0] = 1.0
    mean[0, 2, 0] = 1.0
    reset = torch.tensor([[True, False, True]])
    loss = float(caps_temporal_loss(mean, reset))
    assert loss == pytest.approx(1.0)
    reset_none = torch.zeros(1, 3, dtype=torch.bool)
    assert float(caps_temporal_loss(mean, reset_none)) == pytest.approx(0.5)


def test_default_caps_coef_is_off():
    ppo = DEFAULT_CONFIG["ppo"]
    assert ppo["caps_coef"] == 0.0
    assert ppo["caps_spatial_coef"] == 0.0
    assert ppo["grad_caps_coef"] == 0.0
    validate_config_patch({"ppo": {"caps_coef": 0.1}})
    validate_config_patch({"ppo": {"caps_spatial_coef": 0.1}})
    validate_config_patch({"ppo": {"grad_caps_coef": 0.1}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["caps_coef"] = 0.1
    validate_algorithm_config(cfg, num_envs=8)
    cfg["ppo"]["caps_coef"] = -0.1
    with pytest.raises(ValueError, match="caps_coef"):
        validate_algorithm_config(cfg, num_envs=8)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["grad_caps_coef"] = -0.1
    with pytest.raises(ValueError, match="grad_caps_coef"):
        validate_algorithm_config(cfg, num_envs=8)


def _tiny_trainer(**kwargs) -> PPOTrainer:
    torch.manual_seed(0)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    return PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=4,
        action_clip=1.0,
        advantage_filter_enabled=False,
        **kwargs,
    )


def _run_one_rollout(trainer: PPOTrainer):
    num_envs = 4
    torch.manual_seed(1)
    actor_obs = torch.randn(num_envs, ACTOR_OBS_DIM)
    critic_obs = torch.randn(num_envs, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    reward = torch.zeros(num_envs)
    done = torch.zeros(num_envs, dtype=torch.bool)
    metrics = None
    for step in range(trainer.rollout_steps):
        actor_obs = torch.randn(num_envs, ACTOR_OBS_DIM)
        critic_obs = torch.randn(num_envs, CRITIC_DIM)
        trainer.act(actor_obs, critic_obs)
        if step == trainer.rollout_steps - 1:
            done[0] = True
        metrics = trainer.observe(actor_obs, critic_obs, reward, done)
    return metrics


def test_caps_coef_zero_matches_baseline_update():
    off = _tiny_trainer(caps_coef=0.0)
    metrics = _run_one_rollout(off)
    assert metrics is not None
    assert metrics["caps_loss"] == 0.0
    assert metrics["caps_spatial_loss"] == 0.0
    assert metrics["grad_caps_loss"] == 0.0


def test_caps_coef_adds_positive_smoothness_loss():
    on = _tiny_trainer(caps_coef=0.1)
    metrics = _run_one_rollout(on)
    assert metrics is not None
    assert metrics["caps_loss"] > 0.0


def test_grad_caps_zero_on_constant_slew_and_positive_on_zigzag():
    reset = torch.zeros(1, 3, dtype=torch.bool)
    ramp = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
    assert float(grad_caps_temporal_loss(ramp, reset)) == pytest.approx(0.0)
    zigzag = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]])
    assert float(grad_caps_temporal_loss(zigzag, reset)) > 0.0
    assert float(caps_temporal_loss(ramp, reset)) > 0.0


def test_grad_caps_skips_resets():
    mean = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]])
    reset = torch.tensor([[False, True, False]])
    assert float(grad_caps_temporal_loss(mean, reset)) == pytest.approx(0.0)


def test_caps_spatial_loss_zero_on_identical_means():
    mean = torch.ones(2, 4, 2)
    assert float(caps_spatial_loss(mean, mean.clone())) == pytest.approx(0.0)
    other = mean.clone()
    other[..., 0] = 2.0
    assert float(caps_spatial_loss(mean, other)) == pytest.approx(1.0)


def test_grad_caps_coef_logs_positive_loss():
    metrics = _run_one_rollout(_tiny_trainer(grad_caps_coef=0.1))
    assert metrics is not None
    assert metrics["grad_caps_loss"] > 0.0


def test_caps_spatial_coef_logs_positive_loss():
    metrics = _run_one_rollout(_tiny_trainer(caps_spatial_coef=0.1))
    assert metrics is not None
    assert metrics["caps_spatial_loss"] > 0.0


def test_caps_and_reversal_use_plant_prefix():
    chunk = torch.zeros(2, 5, 8)
    chunk[..., 2] = torch.tensor([0.0, 1.0, -1.0, 1.0, -1.0])
    plant = executed_plant_action(chunk)
    assert plant.shape[-1] == 2
    reset = torch.zeros(2, 5, dtype=torch.bool)
    assert float(caps_temporal_loss(plant, reset)) == pytest.approx(0.0)
    assert float(caps_temporal_loss(chunk, reset)) > 0.0
    assert float(grad_caps_temporal_loss(plant, reset)) == pytest.approx(0.0)
    assert float(grad_caps_temporal_loss(chunk, reset)) > 0.0
    perturbed = plant.clone()
    perturbed[..., 0] = plant[..., 0] + 1.0
    assert float(caps_spatial_loss(plant, perturbed)) == pytest.approx(1.0)
    other = chunk.clone()
    other[..., 0] = 1.0
    other[..., 2] = 0.0
    prefix_spatial = float(
        caps_spatial_loss(executed_plant_action(chunk), executed_plant_action(other))
    )
    assert float(caps_spatial_loss(chunk, other)) > prefix_spatial
    plant_penalty, plant_rates = action_reversal_penalty(
        plant, reset, channel_weights=(1.0, 1.0)
    )
    chunk_penalty, chunk_rates = action_reversal_penalty(
        chunk, reset, channel_weights=(1.0, 1.0)
    )
    assert plant_rates.numel() == 2
    assert chunk_rates.numel() == 8
    assert float(plant_penalty) == pytest.approx(0.0)
    assert float(plant_rates[0]) == pytest.approx(0.0)
    assert float(chunk_rates[2]) > 0.0

    torch.manual_seed(0)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=8,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=4,
        action_clip=1.0,
        advantage_filter_enabled=False,
        caps_spatial_coef=0.1,
    )
    metrics = _run_one_rollout(trainer)
    assert metrics is not None
    assert metrics["caps_spatial_loss"] > 0.0
    assert 0.0 <= float(metrics["reversal_rate_throttle"]) <= 1.0
    assert 0.0 <= float(metrics["reversal_rate_steer"]) <= 1.0

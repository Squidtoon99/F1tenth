from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import PPOTrainer, caps_temporal_loss
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
    assert DEFAULT_CONFIG["ppo"]["caps_coef"] == 0.0
    validate_config_patch({"ppo": {"caps_coef": 0.1}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["caps_coef"] = 0.1
    validate_algorithm_config(cfg, num_envs=8)
    cfg["ppo"]["caps_coef"] = -0.1
    with pytest.raises(ValueError, match="caps_coef"):
        validate_algorithm_config(cfg, num_envs=8)


def _tiny_trainer(caps_coef: float) -> PPOTrainer:
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
        caps_coef=caps_coef,
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
    off = _tiny_trainer(0.0)
    metrics = _run_one_rollout(off)
    assert metrics is not None
    assert metrics["caps_loss"] == 0.0


def test_caps_coef_adds_positive_smoothness_loss():
    on = _tiny_trainer(0.1)
    metrics = _run_one_rollout(on)
    assert metrics is not None
    assert metrics["caps_loss"] > 0.0

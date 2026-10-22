"""Critic warmup freezes the actor; hinged KL penalizes drift from a ref policy."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import PPOTrainer, gaussian_kl
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import validate_algorithm_config, validate_config_patch

CRITIC_DIM = 24
ACT_DIM = 2
NUM_ENVS = 8


def _small_actor(seed=0):
    torch.manual_seed(seed)
    return make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )


def _make_trainer(actor=None, **kwargs):
    if actor is None:
        actor = _small_actor()
    defaults = dict(
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=NUM_ENVS,
        action_clip=1.0,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    defaults.update(kwargs)
    return PPOTrainer(actor, CRITIC_DIM, **defaults)


def _clone_params(module):
    return [p.detach().cpu().clone() for p in module.parameters()]


def _run_one_update(trainer, seed=0):
    torch.manual_seed(seed)
    actor_obs = torch.randn(NUM_ENVS, ACTOR_OBS_DIM)
    critic_obs = torch.randn(NUM_ENVS, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    reward = torch.randn(NUM_ENVS)
    done = torch.zeros(NUM_ENVS, dtype=torch.bool)
    metrics = None
    for _ in range(trainer.rollout_steps):
        actor_obs = torch.randn(NUM_ENVS, ACTOR_OBS_DIM)
        critic_obs = torch.randn(NUM_ENVS, CRITIC_DIM)
        trainer.act(actor_obs, critic_obs)
        metrics = trainer.observe(actor_obs, critic_obs, reward, done)
    return metrics


def test_gaussian_kl_identical_is_zero_and_unit_shift_is_half():
    mean = torch.zeros(4, 8, 2)
    log_std = torch.zeros_like(mean)
    assert float(gaussian_kl(mean, log_std, mean, log_std)) == pytest.approx(0.0, abs=1e-6)
    mean_p = torch.zeros(1, 2)
    mean_q = torch.tensor([[1.0, 0.0]])
    zeros = torch.zeros(1, 2)
    assert float(gaussian_kl(mean_p, zeros, mean_q, zeros)) == pytest.approx(0.5, abs=1e-5)


def test_default_config_warmup_and_kl_are_off():
    ppo = DEFAULT_CONFIG["ppo"]
    assert ppo["critic_warmup_updates"] == 0
    assert ppo["kl_coef"] == pytest.approx(0.0)
    assert ppo["kl_hinge"] == pytest.approx(0.0)
    assert ppo["freeze_actor_normalizer"] is False
    validate_config_patch({"ppo": {"critic_warmup_updates": 3, "kl_coef": 0.2}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["critic_warmup_updates"] = 2
    cfg["ppo"]["kl_coef"] = 0.1
    cfg["ppo"]["kl_hinge"] = 0.05
    validate_algorithm_config(cfg, num_envs=8)
    cfg["ppo"]["critic_warmup_updates"] = -1
    with pytest.raises(ValueError, match="critic_warmup_updates"):
        validate_algorithm_config(cfg, num_envs=8)


def test_critic_warmup_freezes_actor_and_updates_critic():
    actor = _small_actor(0)
    trainer = _make_trainer(actor, critic_warmup_updates=1)
    actor_before = _clone_params(trainer.actor)
    critic_before = _clone_params(trainer.value_critic)
    metrics = _run_one_update(trainer, seed=1)
    assert metrics["critic_warmup_active"] == pytest.approx(1.0)
    assert metrics["actor_updates"] == 0
    assert metrics["value_updates"] > 0
    for left, right in zip(actor_before, _clone_params(trainer.actor)):
        assert torch.equal(left, right)
    moved = False
    for left, right in zip(critic_before, _clone_params(trainer.value_critic)):
        if not torch.equal(left, right):
            moved = True
            break
    assert moved


def test_hinged_kl_keeps_live_actor_near_ref():
    live = _small_actor(0)
    ref = copy.deepcopy(live)
    with torch.no_grad():
        live.mu_layer.bias.add_(1.5)
    trainer = _make_trainer(
        live,
        kl_ref_actor=ref,
        kl_coef=5.0,
        kl_hinge=0.0,
        entropy_coef=0.0,
    )
    metrics = _run_one_update(trainer, seed=2)
    assert metrics["base_kl"] > 0.0
    assert metrics["kl_penalty"] > 0.0
    assert metrics["critic_warmup_active"] == pytest.approx(0.0)
    assert metrics["actor_updates"] > 0

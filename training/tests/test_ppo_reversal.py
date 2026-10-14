from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import PPOTrainer, action_reversal_penalty
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import validate_algorithm_config, validate_config_patch


CRITIC_DIM = 24
ACT_DIM = 2
WEIGHT = 0.5


def _weights(throttle=WEIGHT, steer=0.0):
    return torch.tensor([throttle, steer], dtype=torch.float32)


def _no_reset(steps):
    return torch.zeros(1, steps, dtype=torch.bool)


def test_reversal_ramp_is_zero():
    actions = torch.tensor(
        [[[0.0, 0.0], [0.25, 0.1], [0.5, 0.2], [0.75, 0.3]]],
        dtype=torch.float32,
    )
    penalty, rates = action_reversal_penalty(
        actions, _no_reset(4), channel_weights=_weights()
    )
    assert float(penalty) == pytest.approx(0.0)
    assert torch.all(rates == 0)


def test_reversal_hard_brake_is_zero():
    actions = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]],
        dtype=torch.float32,
    )
    penalty, rates = action_reversal_penalty(
        actions, _no_reset(5), channel_weights=_weights()
    )
    assert float(penalty) == pytest.approx(0.0)
    assert torch.all(rates == 0)


def test_reversal_chatter_costs_four_w():
    actions = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    penalty, rates = action_reversal_penalty(
        actions, _no_reset(3), channel_weights=_weights()
    )
    assert float(penalty) == pytest.approx(4.0 * WEIGHT)
    assert float(rates[0]) == pytest.approx(1.0)
    assert float(rates[1]) == pytest.approx(0.0)


def test_reversal_reset_masks_triplets():
    actions = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    mid_reset = torch.tensor([[False, True, False]])
    penalty, rates = action_reversal_penalty(
        actions, mid_reset, channel_weights=_weights()
    )
    assert float(penalty) == pytest.approx(0.0)
    assert torch.all(rates == 0)

    start_reset = torch.tensor([[True, False, False]])
    penalty_start, _ = action_reversal_penalty(
        actions, start_reset, channel_weights=_weights()
    )
    assert float(penalty_start) == pytest.approx(4.0 * WEIGHT)


def test_reversal_lag2_catches_period_four_square_wave():
    actions = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    reset = _no_reset(5)
    lag1, _ = action_reversal_penalty(
        actions, reset, channel_weights=_weights(), lag2=False
    )
    lag2, _ = action_reversal_penalty(
        actions, reset, channel_weights=_weights(), lag2=True
    )
    assert float(lag1) == pytest.approx(0.0)
    assert float(lag2) == pytest.approx(2.0 * WEIGHT)


def test_default_reversal_keys_are_off():
    ppo = DEFAULT_CONFIG["ppo"]
    assert ppo["reversal_coef_throttle"] == 0.0
    assert ppo["reversal_coef_steer"] == 0.0
    assert ppo["reversal_deadband"] == 0.0
    assert ppo["reversal_lag2"] is False
    validate_config_patch(
        {
            "ppo": {
                "reversal_coef_throttle": 0.1,
                "reversal_lag2": True,
            }
        }
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["reversal_coef_throttle"] = 0.1
    validate_algorithm_config(cfg, num_envs=8)
    cfg["ppo"]["reversal_coef_throttle"] = -0.1
    with pytest.raises(ValueError, match="reversal_coef_throttle"):
        validate_algorithm_config(cfg, num_envs=8)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["reversal_lag2"] = 1
    with pytest.raises(ValueError, match="reversal_lag2"):
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


def test_reversal_coefs_zero_leave_baseline_metrics_zero():
    metrics = _run_one_rollout(_tiny_trainer())
    assert metrics is not None
    assert metrics["reversal_loss"] == 0.0
    assert metrics["reversal_rate_throttle"] == 0.0
    assert metrics["reversal_rate_steer"] == 0.0


def test_reversal_coef_logs_penalty_and_rates():
    trainer = _tiny_trainer(reversal_coef_throttle=0.5)
    assert trainer._need_action_mean is True
    metrics = _run_one_rollout(trainer)
    assert metrics is not None
    assert math.isfinite(metrics["reversal_loss"])
    assert metrics["reversal_loss"] >= 0.0
    assert 0.0 <= metrics["reversal_rate_throttle"] <= 1.0
    assert 0.0 <= metrics["reversal_rate_steer"] <= 1.0

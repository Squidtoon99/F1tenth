from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import (
    PPOTrainer,
    ValueCritic,
    apply_timeout_bootstrap_rewards,
    compute_gae,
    pure_timeout_mask,
)
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM

CRITIC_DIM = 24
ACT_DIM = 2


def test_pure_timeout_mask_ignores_crash_flags():
    termination = {
        "time_out": torch.tensor([True, True, False]),
        "out_of_bounds": torch.tensor([False, True, False]),
        "not_moving": torch.tensor([False, False, False]),
        "invalid_state": torch.tensor([False, False, False]),
        "collision": torch.tensor([False, False, False]),
    }
    mask = pure_timeout_mask(termination)
    assert torch.equal(mask, torch.tensor([True, False, False]))


def test_timeout_bootstrap_adds_gamma_v_of_terminal_obs():
    torch.manual_seed(0)
    critic = ValueCritic(CRITIC_DIM, [16, 16])
    reward = torch.tensor([1.0, 1.0])
    timed_out = torch.tensor([True, False])
    timeout_obs = torch.ones(2, CRITIC_DIM)
    with torch.no_grad():
        values = critic(timeout_obs)
    gamma = 0.5
    augmented = apply_timeout_bootstrap_rewards(
        reward,
        timed_out,
        timeout_obs,
        value_critic=critic,
        critic_normalizer=None,
        gamma=gamma,
        device=torch.device("cpu"),
    )
    assert float(augmented[0]) == pytest.approx(1.0 + gamma * float(values[0]))
    assert float(augmented[1]) == pytest.approx(1.0)


def test_timeout_gae_bootstraps_instead_of_zeroing():
    torch.manual_seed(1)
    critic = ValueCritic(CRITIC_DIM, [16, 16])
    timeout_obs = torch.full((1, CRITIC_DIM), 2.0)
    with torch.no_grad():
        v_term = float(critic(timeout_obs)[0])
    gamma = 0.5
    reward = apply_timeout_bootstrap_rewards(
        torch.tensor([1.0]),
        torch.tensor([True]),
        timeout_obs,
        value_critic=critic,
        critic_normalizer=None,
        gamma=gamma,
        device=torch.device("cpu"),
    )
    rewards = reward.unsqueeze(0)
    values = torch.tensor([[3.0]])
    dones = torch.tensor([[True]])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        bootstrap_value=torch.tensor([99.0]),
        gamma=gamma,
        gae_lambda=0.95,
    )
    assert float(advantages[0, 0]) == pytest.approx(1.0 + gamma * v_term - 3.0)
    assert float(returns[0, 0]) == pytest.approx(1.0 + gamma * v_term)


def test_ppo_observe_bootstraps_pure_timeout_not_true_terminal():
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
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=2,
        num_epochs=1,
        env_minibatch_size=1,
        gamma=0.5,
        gae_lambda=0.95,
        action_clip=1.0,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    actor_obs = torch.zeros(1, ACTOR_OBS_DIM)
    critic_obs = torch.zeros(1, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    timeout_obs = torch.ones(1, CRITIC_DIM)
    with torch.no_grad():
        v_term = float(trainer.value_critic(timeout_obs)[0])

    trainer.act(actor_obs, critic_obs)
    trainer.observe(
        actor_obs,
        critic_obs,
        torch.tensor([1.0]),
        torch.tensor([True]),
        timed_out=torch.tensor([True]),
        timeout_critic_obs=timeout_obs,
    )
    trainer.act(actor_obs, critic_obs)
    metrics = trainer.observe(
        actor_obs,
        critic_obs,
        torch.zeros(1),
        torch.tensor([False]),
    )
    assert metrics is not None
    stored = float(trainer.last_rollout.rewards[0, 0])
    assert stored == pytest.approx(1.0 + trainer.gamma * v_term)
    assert stored != pytest.approx(1.0)

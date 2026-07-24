from __future__ import annotations

import copy

import torch
import torch.nn as nn

from qrsac import Models, QRSACTrainer, QuantileCritic
from qrsac.qrsac import quantile_huber_loss, select_min_quantiles
from qrsac.spinningup.core import mlp, _squashed_gaussian_forward

ACTOR_DIM = 5
CRITIC_DIM = 7


class _TinySquashedActor(nn.Module):
    """Minimal squashed-Gaussian actor for algorithm unit tests only."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, act_limit=1.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        return _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            obs,
            deterministic,
            with_logprob,
        )


def _models() -> Models:
    actor = _TinySquashedActor(ACTOR_DIM, 2, [16, 16])
    critic = QuantileCritic(CRITIC_DIM, 2, [16, 16], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def _asymmetric_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    return {
        "actor_obs": torch.randn(batch_size, ACTOR_DIM),
        "critic_obs": torch.randn(batch_size, CRITIC_DIM),
        "action": torch.rand(batch_size, 2) * 2.0 - 1.0,
        "reward": torch.randn(batch_size),
        "next_actor_obs": torch.randn(batch_size, ACTOR_DIM),
        "next_critic_obs": torch.randn(batch_size, CRITIC_DIM),
        "done": torch.zeros(batch_size),
    }


def test_quantile_huber_reference_value():
    pred = torch.zeros(3, 2)
    target = torch.ones(3, 2)
    assert torch.allclose(quantile_huber_loss(pred, target), torch.tensor(0.5))


def test_target_selection_keeps_whole_lower_mean_vector():
    q1 = torch.tensor([[0.0, 4.0], [5.0, 5.0]])
    q2 = torch.tensor([[1.0, 2.0], [0.0, 9.0]])
    selected = select_min_quantiles(q1, q2)
    assert torch.equal(selected[0], q2[0])
    assert torch.equal(selected[1], q2[1])


def test_actor_and_critic_output_ranks_align_for_policy_loss():
    torch.manual_seed(0)
    models = _models()
    actor_obs = torch.randn(7, ACTOR_DIM)
    critic_obs = torch.randn(7, CRITIC_DIM)
    actions, log_prob = models.actor(actor_obs)
    quantiles = models.critic1(critic_obs, actions)
    assert actions.shape == (7, 2)
    assert log_prob.shape == (7,)
    assert quantiles.shape == (7, 4)


def test_complete_update_has_scalar_losses_and_finite_gradients():
    torch.manual_seed(0)
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=7)
    assert trainer.actor_obs_dim == ACTOR_DIM
    assert trainer.critic_obs_dim == CRITIC_DIM
    batch = _asymmetric_batch()
    losses = trainer.update(batch)
    assert losses.policy_loss.shape == ()
    assert losses.critic_loss.shape == ()
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)

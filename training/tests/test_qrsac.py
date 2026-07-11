from __future__ import annotations

import copy

import torch
import torch.nn as nn

from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor
from qrsac.qrsac import quantile_huber_loss, select_min_quantiles


def _models() -> Models:
    actor = SquashedGaussianMLPActor(5, 2, [16, 16], nn.ReLU, 1.0)
    critic = QuantileCritic(5, 2, [16, 16], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


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
    obs = torch.randn(7, 5)
    actions, log_prob = models.actor(obs)
    quantiles = models.critic1(obs, actions)
    assert actions.shape == (7, 2)
    assert log_prob.shape == (7,)
    assert quantiles.shape == (7, 4)
    assert log_prob.unsqueeze(-1).shape == quantiles.mean(dim=-1, keepdim=True).shape


def test_complete_update_has_scalar_losses_and_finite_gradients():
    torch.manual_seed(0)
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=7)
    batch_size = 8
    batch = {
        "obs": torch.randn(batch_size, 5),
        "action": torch.rand(batch_size, 2) * 2.0 - 1.0,
        "reward": torch.randn(batch_size),
        "next_obs": torch.randn(batch_size, 5),
        "done": torch.zeros(batch_size),
    }
    target_before = {
        key: value.clone() for key, value in models.critic1_target.state_dict().items()
    }

    losses = trainer.update(batch)

    assert losses.policy_loss.shape == ()
    assert losses.critic_loss.shape == ()
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in models.actor.parameters()
    )
    assert any(
        not torch.equal(target_before[key], value)
        for key, value in models.critic1_target.state_dict().items()
    )

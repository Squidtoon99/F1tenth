from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor
from qrsac.qrsac import quantile_huber_loss, select_min_quantiles

ACTOR_DIM = 5
CRITIC_DIM = 7


def _models() -> Models:
    actor = SquashedGaussianMLPActor(ACTOR_DIM, 2, [16, 16], nn.ReLU, 1.0)
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
    assert log_prob.unsqueeze(-1).shape == quantiles.mean(dim=-1, keepdim=True).shape


def test_complete_update_has_scalar_losses_and_finite_gradients():
    torch.manual_seed(0)
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=7)
    assert trainer.actor_obs_dim == ACTOR_DIM
    assert trainer.critic_obs_dim == CRITIC_DIM
    batch = _asymmetric_batch()
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


def test_legacy_symmetric_batch_is_rejected():
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=7)
    with pytest.raises(KeyError, match="legacy symmetric"):
        trainer.update(
            {
                "obs": torch.randn(4, ACTOR_DIM),
                "action": torch.randn(4, 2),
                "reward": torch.randn(4),
                "next_obs": torch.randn(4, ACTOR_DIM),
                "done": torch.zeros(4),
            }
        )


def test_swapped_actor_critic_shapes_are_rejected():
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=7)
    batch = _asymmetric_batch(4)
    batch["actor_obs"] = torch.randn(4, CRITIC_DIM)
    batch["next_actor_obs"] = torch.randn(4, CRITIC_DIM)
    with pytest.raises(ValueError, match="must not be swapped"):
        trainer.update(batch)


def test_build_models_uses_independent_actor_critic_dims():
    from config import DEFAULT_CONFIG
    from standalone_trainer import build_models

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["num_quantiles"] = 4
    models, trainer = build_models(cfg, torch.device("cpu"), compile=False)
    assert trainer.actor_obs_dim == cfg["obs"]["num_actor_obs"] == 1093
    assert trainer.critic_obs_dim == cfg["obs"]["num_obs"] == 390
    assert models.actor.obs_dim == 1093
    assert models.critic1.backbone[0].weight.shape[1] == 390 + 2


def test_fused_compile_regions_leave_modules_eager():
    """Recommendation #1: compile two phases, not six per-module graphs."""
    models = _models()
    trainer = QRSACTrainer(
        models, torch.device("cpu"), n_step=3, compile=True, compile_mode="default"
    )
    assert trainer._policy_phase is not trainer._critic_phase
    # Raw modules stay uncompiled so tensors never cross per-module CUDA graphs.
    assert not isinstance(trainer.actor, torch._dynamo.eval_frame.OptimizedModule)
    assert not isinstance(trainer.critic1, torch._dynamo.eval_frame.OptimizedModule)
    assert not isinstance(
        trainer.critic1_target, torch._dynamo.eval_frame.OptimizedModule
    )
    assert not hasattr(trainer, "_guard")


def test_target_and_policy_loss_match_reference_routing():
    """Fused policy phase must preserve QR-SAC targets, routing, and losses."""
    torch.manual_seed(0)
    models = _models()
    trainer = QRSACTrainer(
        models, torch.device("cpu"), n_step=3, gamma=0.9, alpha=0.2, kappa=1.0
    )
    batch = _asymmetric_batch(5)

    actor_obs = batch["actor_obs"]
    critic_obs = batch["critic_obs"]
    action = batch["action"]
    reward = batch["reward"]
    next_actor_obs = batch["next_actor_obs"]
    next_critic_obs = batch["next_critic_obs"]
    done = batch["done"]
    discount = trainer.gamma**trainer.n_step

    # Align stochastic actor samples: targets consume the first draw, policy the second.
    torch.manual_seed(123)
    with torch.no_grad():
        actions_next, log_prob_next = models.actor(next_actor_obs)
        q1_next = models.critic1_target(next_critic_obs, actions_next)
        q2_next = models.critic2_target(next_critic_obs, actions_next)
        min_q = select_min_quantiles(q1_next, q2_next)
        ref_target = reward.unsqueeze(-1) + discount * (1.0 - done.unsqueeze(-1)) * (
            min_q - trainer.alpha * log_prob_next.unsqueeze(-1)
        )
        sampled_actions, log_prob = models.actor(actor_obs)
        q1 = models.critic1(critic_obs, sampled_actions)
        q2 = models.critic2(critic_obs, sampled_actions)
        q = torch.minimum(
            q1.mean(dim=-1, keepdim=True), q2.mean(dim=-1, keepdim=True)
        )
        ref_policy = (trainer.alpha * log_prob.unsqueeze(-1) - q).mean()

    for p in trainer.critic_params:
        p.requires_grad = False
    torch.manual_seed(123)
    policy_loss, target = trainer._policy_phase(
        trainer.actor,
        trainer.critic1,
        trainer.critic2,
        trainer.critic1_target,
        trainer.critic2_target,
        actor_obs,
        critic_obs,
        next_actor_obs,
        next_critic_obs,
        reward,
        done,
        discount,
        trainer.alpha,
    )
    assert torch.allclose(target, ref_target, rtol=1e-5, atol=1e-5)
    assert torch.allclose(policy_loss, ref_policy, rtol=1e-5, atol=1e-5)

    for p in trainer.critic_params:
        p.requires_grad = True
    critic_loss = trainer._critic_phase(
        trainer.critic1,
        trainer.critic2,
        critic_obs,
        action,
        target.detach(),
        trainer.kappa,
        trainer.quantile_fractions,
    )
    q1_obs = models.critic1(critic_obs, action)
    q2_obs = models.critic2(critic_obs, action)
    ref_critic = quantile_huber_loss(
        q1_obs, ref_target, kappa=trainer.kappa, taus=trainer.quantile_fractions
    ) + quantile_huber_loss(
        q2_obs, ref_target, kappa=trainer.kappa, taus=trainer.quantile_fractions
    )
    assert torch.allclose(critic_loss, ref_critic, rtol=1e-5, atol=1e-5)


def test_update_ordering_actor_before_critic_and_polyak():
    torch.manual_seed(1)
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=2, smooth_factor=0.1)
    batch = _asymmetric_batch(6)

    actor_before = {
        k: v.clone() for k, v in models.actor.state_dict().items()
    }
    critic_before = {
        k: v.clone() for k, v in models.critic1.state_dict().items()
    }
    target_before = {
        k: v.clone() for k, v in models.critic1_target.state_dict().items()
    }

    losses = trainer.update(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    assert any(
        not torch.equal(actor_before[k], v)
        for k, v in models.actor.state_dict().items()
    )
    assert any(
        not torch.equal(critic_before[k], v)
        for k, v in models.critic1.state_dict().items()
    )
    # Polyak must move targets toward the updated online critics.
    assert any(
        not torch.equal(target_before[k], v)
        for k, v in models.critic1_target.state_dict().items()
    )

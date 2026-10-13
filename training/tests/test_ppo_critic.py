"""Real-module tests for PPO critic knobs. Defaults must match today's loss."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import (
    PPOTrainer,
    Rollout,
    apply_timeout_bootstrap_rewards,
    explained_variance,
    value_prediction_loss,
)
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import (
    ActorModels,
    ObsNormalizer,
    build_ppo_models,
    load_resume_state,
    save_resume_state,
    validate_algorithm_config,
    validate_config_patch,
)

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


def _clone_rollout(rollout):
    kwargs = {}
    for field in dataclasses.fields(Rollout):
        value = getattr(rollout, field.name)
        if field.name == "optimized":
            kwargs[field.name] = False
        else:
            kwargs[field.name] = value.detach().clone()
    return Rollout(**kwargs)


def _snapshot(trainer):
    return {
        "actor": copy.deepcopy(trainer.actor.state_dict()),
        "critic": copy.deepcopy(trainer.value_critic.state_dict()),
        "actor_opt": copy.deepcopy(trainer.actor_optimizer.state_dict()),
        "value_opt": copy.deepcopy(trainer.value_optimizer.state_dict()),
    }


def _restore(trainer, states):
    trainer.actor.load_state_dict(states["actor"])
    trainer.value_critic.load_state_dict(states["critic"])
    trainer.actor_optimizer.load_state_dict(states["actor_opt"])
    trainer.value_optimizer.load_state_dict(states["value_opt"])
    trainer.value_critic.zero_grad(set_to_none=True)
    trainer.actor.zero_grad(set_to_none=True)


def _collect_frozen_batch(seed=0, num_envs=NUM_ENVS, reward=None, **trainer_kwargs):
    torch.manual_seed(seed)
    trainer = _make_trainer(env_minibatch_size=num_envs, **trainer_kwargs)
    states = _snapshot(trainer)
    actor_obs = torch.randn(num_envs, ACTOR_OBS_DIM)
    critic_obs = torch.randn(num_envs, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    if reward is None:
        reward = torch.randn(num_envs)
    done = torch.zeros(num_envs, dtype=torch.bool)
    for _ in range(trainer.rollout_steps):
        trainer.act(actor_obs, critic_obs)
        trainer.observe(actor_obs, critic_obs, reward, done)
    frozen = _clone_rollout(trainer.last_rollout)
    return trainer, frozen, states, critic_obs


def _legacy_clipped_mse(new_values, old_values, returns, value_clip=0.2):
    clipped = old_values + (new_values - old_values).clamp(-value_clip, value_clip)
    return 0.5 * torch.maximum(
        (new_values - returns).square(), (clipped - returns).square()
    )


def _tiny_cfg(**ppo_overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["model"]["actor_hidden_layers"] = [32, 32]
    cfg["model"]["critic_hidden_layers"] = [32, 32]
    cfg["model"]["lidar_pool_bins"] = 16
    cfg["model"]["lidar_aug_enabled"] = False
    cfg["ppo"].update(ppo_overrides)
    return cfg


def test_default_config_keeps_current_critic_knobs():
    ppo = DEFAULT_CONFIG["ppo"]
    assert ppo["value_clip"] == pytest.approx(0.2)
    assert ppo["value_normalization"] is False
    assert ppo["value_max_grad_norm"] is None
    assert ppo["value_huber_delta"] == pytest.approx(0.0)


def test_default_value_loss_matches_legacy_clip_mse_bit_for_bit():
    torch.manual_seed(0)
    new_values = torch.randn(4, 8, requires_grad=True)
    new_clone = new_values.detach().clone().requires_grad_()
    old_values = torch.randn(4, 8)
    returns = torch.randn(4, 8)
    legacy = _legacy_clipped_mse(new_values, old_values, returns)
    got, _ = value_prediction_loss(
        new_clone, old_values, returns, value_clip=0.2, huber_delta=0.0
    )
    assert torch.equal(got, legacy)
    legacy.mean().backward()
    got.mean().backward()
    assert torch.equal(new_values.grad, new_clone.grad)


def test_default_huber_and_norm_off_match_legacy_square():
    torch.manual_seed(3)
    new_values = torch.randn(5, requires_grad=True)
    old_values = torch.randn(5)
    returns = torch.randn(5)
    mse = 0.5 * (new_values - returns).square()
    got, active = value_prediction_loss(
        new_values, old_values, returns, value_clip=0.0, huber_delta=0.0
    )
    assert torch.equal(got, mse)
    assert not bool(active.any())


def test_default_trainer_update_matches_legacy_loss_and_critic_step():
    trainer, frozen, states, _ = _collect_frozen_batch(seed=1, num_envs=1)
    _restore(trainer, states)
    new_values = trainer.value_critic(frozen.critic_obs)
    legacy = _legacy_clipped_mse(
        new_values, frozen.old_values, frozen.returns
    ).mean()
    trainer.value_optimizer.zero_grad(set_to_none=True)
    legacy.backward()
    nn.utils.clip_grad_norm_(
        trainer.value_critic.parameters(), trainer.max_grad_norm
    )
    trainer.value_optimizer.step()
    after_legacy = copy.deepcopy(trainer.value_critic.state_dict())
    loss_legacy = float(legacy.detach())

    _restore(trainer, states)
    metrics = trainer.update(_clone_rollout(frozen))
    assert metrics["value_loss"] == pytest.approx(loss_legacy, abs=0.0, rel=0.0)
    for name, tensor in after_legacy.items():
        assert torch.equal(tensor, trainer.value_critic.state_dict()[name]), name


def test_omitted_knobs_match_explicit_defaults_on_fixed_batch():
    torch.manual_seed(2)
    actor_a = _small_actor(seed=2)
    actor_b = copy.deepcopy(actor_a)
    left = _make_trainer(actor=actor_a)
    right = _make_trainer(
        actor=actor_b,
        value_clip=0.2,
        value_normalization=False,
        value_max_grad_norm=None,
        value_huber_delta=0.0,
    )
    right.value_critic.load_state_dict(left.value_critic.state_dict())
    right.value_optimizer.load_state_dict(left.value_optimizer.state_dict())
    assert left.value_clip == pytest.approx(0.2)
    assert left.value_max_grad_norm == pytest.approx(left.max_grad_norm)
    assert left.value_normalizer is None
    assert left.value_huber_delta == pytest.approx(0.0)

    def _one_update(trainer, seed=4):
        torch.manual_seed(seed)
        actor_obs = torch.randn(NUM_ENVS, ACTOR_OBS_DIM)
        critic_obs = torch.randn(NUM_ENVS, CRITIC_DIM)
        trainer.initialize(actor_obs, critic_obs)
        reward = torch.randn(NUM_ENVS)
        done = torch.zeros(NUM_ENVS, dtype=torch.bool)
        metrics = None
        for _ in range(trainer.rollout_steps):
            trainer.act(actor_obs, critic_obs)
            metrics = trainer.observe(actor_obs, critic_obs, reward, done)
        return metrics

    torch.manual_seed(4)
    m_left = _one_update(left, seed=4)
    torch.manual_seed(4)
    m_right = _one_update(right, seed=4)
    assert m_left["value_loss"] == pytest.approx(m_right["value_loss"], abs=0.0)
    for name, tensor in left.value_critic.state_dict().items():
        assert torch.equal(tensor, right.value_critic.state_dict()[name]), name


def test_explained_variance_on_known_inputs():
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(explained_variance(returns, returns)) == pytest.approx(1.0)
    constant = torch.full_like(returns, float(returns.mean()))
    assert float(explained_variance(returns, constant)) == pytest.approx(0.0)
    shifted = returns + 1.0
    assert float(explained_variance(returns, shifted)) == pytest.approx(1.0)
    reversed_v = torch.tensor([4.0, 3.0, 2.0, 1.0])
    ev_wrong = float(explained_variance(returns, reversed_v))
    assert ev_wrong < 1.0


def test_trainer_explained_variance_is_one_when_values_match_returns():
    trainer, frozen, states, _ = _collect_frozen_batch(seed=5)
    _restore(trainer, states)
    frozen.old_values.copy_(frozen.returns)
    metrics = trainer.update(frozen)
    assert metrics["explained_variance"] == pytest.approx(1.0)
    assert metrics["explained_variance"] <= 1.0
    assert metrics["explained_variance_post"] <= 1.0


def test_trainer_logs_explained_variance_on_a_real_rollout():
    trainer, frozen, states, _ = _collect_frozen_batch(seed=6)
    _restore(trainer, states)
    metrics = trainer.update(frozen)
    assert metrics["explained_variance"] <= 1.0
    assert metrics["explained_variance_post"] <= 1.0
    assert "return_mean" in metrics
    assert "return_std" in metrics
    assert "value_mean" in metrics
    assert "value_clip_active_fraction" in metrics


def test_value_clip_zero_loss_equals_plain_mse():
    torch.manual_seed(7)
    new_values = torch.randn(4, 8)
    old_values = torch.randn(4, 8)
    returns = torch.randn(4, 8)
    got, active = value_prediction_loss(
        new_values, old_values, returns, value_clip=0.0, huber_delta=0.0
    )
    mse = 0.5 * (new_values - returns).square()
    assert torch.equal(got, mse)
    assert not bool(active.any())

    trainer, frozen, states, _ = _collect_frozen_batch(
        seed=7, num_envs=1, value_clip=0
    )
    _restore(trainer, states)
    new_values = trainer.value_critic(frozen.critic_obs)
    expected = 0.5 * (new_values - frozen.returns).square().mean()
    metrics = trainer.update(_clone_rollout(frozen))
    assert metrics["value_loss"] == pytest.approx(float(expected.detach()), abs=0.0)
    assert metrics["value_clip_fraction"] == pytest.approx(0.0)
    assert metrics["value_clip_active_fraction"] == pytest.approx(0.0)


def test_value_clip_none_matches_zero():
    torch.manual_seed(8)
    actor = _small_actor(seed=8)
    t0 = _make_trainer(actor=copy.deepcopy(actor), value_clip=0)
    t_none = _make_trainer(actor=copy.deepcopy(actor), value_clip=None)
    assert t0.value_clip == pytest.approx(0.0)
    assert t_none.value_clip == pytest.approx(0.0)


def test_disabling_value_clip_removes_zeroed_row_effect():
    new_clip = torch.tensor([1.0, 0.05], requires_grad=True)
    new_off = torch.tensor([1.0, 0.05], requires_grad=True)
    old = torch.zeros(2)
    returns = torch.tensor([10.0, 10.0])
    per_on, active_on = value_prediction_loss(
        new_clip, old, returns, value_clip=0.2, huber_delta=0.0
    )
    per_off, active_off = value_prediction_loss(
        new_off, old, returns, value_clip=0.0, huber_delta=0.0
    )
    assert bool(active_on[0])
    assert not bool(active_off.any())
    per_on.sum().backward()
    per_off.sum().backward()
    assert float(new_clip.grad[0]) == pytest.approx(0.0)
    assert abs(float(new_off.grad[0])) > 0.0
    assert abs(float(new_off.grad[1])) > 0.0

    trainer, frozen, states, _ = _collect_frozen_batch(
        seed=9, value_clip=0.2, value_lr=0.05, num_epochs=4
    )
    _restore(trainer, states)
    metrics_on = trainer.update(_clone_rollout(frozen))
    trainer_off, frozen_off, states_off, _ = _collect_frozen_batch(
        seed=9, value_clip=0, value_lr=0.05, num_epochs=4
    )
    _restore(trainer_off, states_off)
    metrics_off = trainer_off.update(_clone_rollout(frozen_off))
    assert metrics_off["value_clip_active_fraction"] == pytest.approx(0.0)
    assert metrics_on["value_clip_active_fraction"] >= 0.0


def test_popart_rescale_preserves_unnormalized_outputs():
    trainer = _make_trainer(value_normalization=True)
    obs = torch.randn(6, CRITIC_DIM)
    with torch.no_grad():
        before = trainer._value(obs).clone()
    mu_old, sigma_old = trainer._value_mu_sigma()
    trainer.value_normalizer.update(
        torch.linspace(-20.0, 40.0, 80).unsqueeze(-1)
    )
    mu_new, sigma_new = trainer._value_mu_sigma()
    assert not torch.equal(mu_old, mu_new) or not torch.equal(sigma_old, sigma_new)
    trainer._popart_rescale(mu_old, sigma_old, mu_new, sigma_new)
    with torch.no_grad():
        after = trainer._value(obs)
    assert torch.allclose(before, after, atol=1e-6, rtol=0.0)


def test_value_normalization_old_values_and_bootstrap_are_raw_units():
    trainer = _make_trainer(value_normalization=True)
    with torch.no_grad():
        last = trainer.value_critic.net[-1]
        last.weight.zero_()
        last.bias.fill_(0.5)
    trainer.value_normalizer.mean.fill_(3.0)
    trainer.value_normalizer.var.fill_(4.0)
    mu, sigma = trainer._value_mu_sigma()
    expected = 3.0 + float(sigma) * 0.5
    obs = torch.zeros(4, CRITIC_DIM)
    got = trainer._value(obs)
    assert torch.allclose(got, torch.full((4,), expected), atol=1e-6)

    actor_obs = torch.zeros(4, ACTOR_OBS_DIM)
    trainer.initialize(actor_obs, obs)
    trainer.act(actor_obs, obs)
    assert torch.allclose(
        trainer._rollout.old_values[0], torch.full((4,), expected), atol=1e-6
    )

    reward = torch.ones(4)
    timed_out = torch.tensor([True, False, True, False])
    bootstrapped = apply_timeout_bootstrap_rewards(
        reward,
        timed_out,
        obs,
        value_fn=trainer._value,
        critic_normalizer=None,
        gamma=0.9,
        device=torch.device("cpu"),
    )
    assert float(bootstrapped[0]) == pytest.approx(1.0 + 0.9 * expected)
    assert float(bootstrapped[1]) == pytest.approx(1.0)


def test_timeout_bootstrap_uses_value_callable():
    reward = torch.tensor([1.0, 2.0, 3.0])
    timed_out = torch.tensor([True, False, True])
    obs = torch.zeros(3, CRITIC_DIM)

    def value_fn(normalized):
        return torch.full((normalized.shape[0],), 4.0)

    out = apply_timeout_bootstrap_rewards(
        reward,
        timed_out,
        obs,
        value_fn=value_fn,
        critic_normalizer=None,
        gamma=0.5,
        device=torch.device("cpu"),
    )
    assert float(out[0]) == pytest.approx(1.0 + 0.5 * 4.0)
    assert float(out[1]) == pytest.approx(2.0)
    assert float(out[2]) == pytest.approx(3.0 + 0.5 * 4.0)


def test_value_max_grad_norm_defaults_to_max_grad_norm():
    trainer = _make_trainer(max_grad_norm=0.5)
    assert trainer.value_max_grad_norm == pytest.approx(0.5)
    split = _make_trainer(max_grad_norm=0.5, value_max_grad_norm=10.0)
    assert split.value_max_grad_norm == pytest.approx(10.0)
    assert split.max_grad_norm == pytest.approx(0.5)


def test_value_max_grad_norm_clips_critic_not_actor():
    trainer, frozen, states, _ = _collect_frozen_batch(
        seed=11, num_envs=NUM_ENVS, reward=torch.full((NUM_ENVS,), 5.0)
    )
    _restore(trainer, states)
    new_values = trainer.value_critic(frozen.critic_obs)
    loss = 0.5 * (new_values - frozen.returns).square().mean()
    trainer.value_critic.zero_grad(set_to_none=True)
    loss.backward()
    unclipped = float(
        nn.utils.clip_grad_norm_(trainer.value_critic.parameters(), 1e6)
    )
    assert unclipped > 0.5

    _restore(trainer, states)
    trainer.value_max_grad_norm = 10.0
    metrics_split = trainer.update(_clone_rollout(frozen))
    if 0.5 < metrics_split["value_grad_norm"] <= 10.0:
        assert metrics_split["value_grad_clip_fraction"] == pytest.approx(0.0)
    elif metrics_split["value_grad_norm"] > 10.0:
        assert metrics_split["value_grad_clip_fraction"] == pytest.approx(1.0)

    _restore(trainer, states)
    trainer.value_max_grad_norm = trainer.max_grad_norm
    metrics_shared = trainer.update(_clone_rollout(frozen))
    if metrics_shared["value_grad_norm"] > 0.5:
        assert metrics_shared["value_grad_clip_fraction"] == pytest.approx(1.0)

    trainer_hi, frozen_hi, states_hi, _ = _collect_frozen_batch(
        seed=12,
        num_envs=NUM_ENVS,
        reward=torch.full((NUM_ENVS,), 25.0),
        value_max_grad_norm=10.0,
    )
    _restore(trainer_hi, states_hi)
    new_hi = trainer_hi.value_critic(frozen_hi.critic_obs)
    loss_hi = 0.5 * (new_hi - frozen_hi.returns).square().mean()
    trainer_hi.value_critic.zero_grad(set_to_none=True)
    loss_hi.backward()
    pre = float(nn.utils.clip_grad_norm_(trainer_hi.value_critic.parameters(), 10.0))
    if pre > 10.0:
        post = torch.sqrt(
            sum(
                p.grad.detach().square().sum()
                for p in trainer_hi.value_critic.parameters()
                if p.grad is not None
            )
        )
        assert float(post) == pytest.approx(10.0, rel=1e-5)
        _restore(trainer_hi, states_hi)
        metrics = trainer_hi.update(_clone_rollout(frozen_hi))
        assert metrics["value_grad_norm"] > 10.0
        assert metrics["value_grad_clip_fraction"] == pytest.approx(1.0)
    assert trainer_hi.max_grad_norm == pytest.approx(0.5)


def test_huber_large_delta_reproduces_mse_exactly():
    torch.manual_seed(13)
    new_values = torch.randn(16)
    old_values = torch.randn(16)
    returns = torch.randn(16)
    mse, _ = value_prediction_loss(
        new_values, old_values, returns, value_clip=0.0, huber_delta=0.0
    )
    huber, _ = value_prediction_loss(
        new_values, old_values, returns, value_clip=0.0, huber_delta=1e8
    )
    assert torch.allclose(mse, huber, atol=0.0, rtol=0.0)

    trainer, frozen, states, _ = _collect_frozen_batch(
        seed=13, num_envs=1, value_clip=0, value_huber_delta=1e8
    )
    _restore(trainer, states)
    new_v = trainer.value_critic(frozen.critic_obs)
    expected = 0.5 * (new_v - frozen.returns).square().mean()
    metrics = trainer.update(_clone_rollout(frozen))
    assert metrics["value_loss"] == pytest.approx(float(expected.detach()), abs=0.0)


def test_huber_delta_one_bounds_per_row_grad_magnitude():
    new_values = torch.tensor([-8.0, -0.2, 0.0, 0.4, 9.0], requires_grad=True)
    returns = torch.zeros_like(new_values)
    per, _ = value_prediction_loss(
        new_values, new_values.detach(), returns, value_clip=0.0, huber_delta=1.0
    )
    per.sum().backward()
    assert float(new_values.grad.abs().max()) <= 1.0 + 1e-6


def test_resume_state_persists_value_norm(tmp_path):
    trainer = _make_trainer(value_normalization=True)
    trainer.value_normalizer.update(torch.arange(30.0).unsqueeze(-1))
    mean_before = trainer.value_normalizer.mean.clone()
    var_before = trainer.value_normalizer.var.clone()
    count_before = float(trainer.value_normalizer.count)
    models = ActorModels(actor=trainer.actor)
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))
    path = tmp_path / "resume.pt"
    save_resume_state(
        path,
        algorithm="ppo",
        models=models,
        trainer=trainer,
        actor_normalizer=actor_norm,
        critic_normalizer=critic_norm,
        env_transitions=128,
        gradient_updates=4,
        vector_ticks=16,
        sampled_replay_rows=128,
        device=torch.device("cpu"),
    )
    restored = _make_trainer(
        actor=copy.deepcopy(trainer.actor), value_normalization=True
    )
    load_resume_state(
        str(path),
        algorithm="ppo",
        models=ActorModels(actor=restored.actor),
        trainer=restored,
        actor_normalizer=ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu")),
        critic_normalizer=ObsNormalizer(CRITIC_DIM, torch.device("cpu")),
        device=torch.device("cpu"),
    )
    assert torch.equal(restored.value_normalizer.mean, mean_before)
    assert torch.equal(restored.value_normalizer.var, var_before)
    assert restored.value_normalizer.count == pytest.approx(count_before)


def test_validate_accepts_value_clip_zero_and_null():
    cfg = _tiny_cfg(value_clip=0)
    validate_algorithm_config(cfg, num_envs=8)
    cfg_none = _tiny_cfg(value_clip=None)
    validate_algorithm_config(cfg_none, num_envs=8)
    validate_config_patch({"ppo": {"value_clip": 0}})
    validate_config_patch({"ppo": {"value_clip": None}})
    validate_config_patch({"ppo": {"value_normalization": True}})
    validate_config_patch({"ppo": {"value_max_grad_norm": 10.0}})
    validate_config_patch({"ppo": {"value_huber_delta": 1.0}})


def test_validate_rejects_negative_value_clip():
    cfg = _tiny_cfg(value_clip=-0.1)
    with pytest.raises(ValueError, match="value_clip"):
        validate_algorithm_config(cfg, num_envs=8)


def test_build_ppo_models_wires_critic_knobs():
    cfg = _tiny_cfg(
        value_clip=0,
        value_normalization=True,
        value_max_grad_norm=10.0,
        value_huber_delta=1.0,
    )
    device = torch.device("cpu")
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, device)
    critic_norm = ObsNormalizer(int(cfg["obs"]["num_obs"]), device)
    _, trainer = build_ppo_models(
        cfg, device, actor_norm, critic_norm, num_envs=8
    )
    assert trainer.value_clip == pytest.approx(0.0)
    assert trainer.value_normalization is True
    assert trainer.value_normalizer is not None
    assert trainer.value_max_grad_norm == pytest.approx(10.0)
    assert trainer.value_huber_delta == pytest.approx(1.0)

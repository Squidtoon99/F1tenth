from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_policy import ObsNormalizer
from f1tenth_policy.actor import apply_policy_output_init, make_actor
from f1tenth_policy.layout import ACTOR_OBS_DIM, GRU_HIDDEN_DIM
from ppo import (
    PPOTrainer,
    adapt_kl_learning_rate,
    scheduled_learning_rate,
)
from standalone_trainer import (
    ActorModels,
    load_resume_state,
    save_resume_state,
    validate_algorithm_config,
)

CRITIC_DIM = 24
ACT_DIM = 2
ACTOR_LR = 3e-5
VALUE_LR = 1e-4


def _make_actor(seed, **kwargs):
    torch.manual_seed(seed)
    return make_actor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        **kwargs,
    )


def _make_trainer(seed, **kwargs):
    actor = _make_actor(seed)
    trainer_kwargs = dict(
        critic_obs_dim=CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=2,
        num_epochs=1,
        env_minibatch_size=1,
        actor_lr=ACTOR_LR,
        value_lr=VALUE_LR,
        action_clip=1.0,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    trainer_kwargs.update(kwargs)
    torch.manual_seed(seed + 1)
    return PPOTrainer(actor, **trainer_kwargs)


def test_default_init_is_bit_identical_to_today():
    baseline = _make_actor(0)
    post_apply = _make_actor(0)
    apply_policy_output_init(
        post_apply, policy_mean_init_scale=None, policy_init_log_std=None
    )
    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, post_apply.state_dict()[key]), key


def test_policy_mean_init_scale_and_log_std():
    baseline = _make_actor(7)
    scaled = _make_actor(7)
    apply_policy_output_init(
        scaled, policy_mean_init_scale=0.01, policy_init_log_std=-0.7
    )
    assert torch.equal(
        scaled.mu_layer.weight, baseline.mu_layer.weight * 0.01
    )
    assert torch.equal(scaled.mu_layer.bias, baseline.mu_layer.bias * 0.01)
    assert torch.equal(
        scaled.log_std_layer.weight,
        torch.zeros_like(scaled.log_std_layer.weight),
    )
    assert torch.equal(
        scaled.log_std_layer.bias,
        torch.full_like(scaled.log_std_layer.bias, -0.7),
    )
    for key, tensor in baseline.state_dict().items():
        if key.startswith("mu_layer.") or key.startswith("log_std_layer."):
            continue
        assert torch.equal(tensor, scaled.state_dict()[key]), key


def test_constant_schedule_keeps_initial_lrs():
    default = _make_trainer(0)
    constant = _make_trainer(0, lr_schedule="constant", lr_final=0.0)
    start_actor = default.current_actor_lr()
    start_value = default.current_value_lr()
    assert start_actor == pytest.approx(ACTOR_LR)
    assert start_value == pytest.approx(VALUE_LR)
    for transitions in (0, 1, 50, 99, 100, 10_000):
        default.set_schedule_transitions(transitions)
        constant.set_schedule_transitions(transitions)
        assert default.current_actor_lr() == start_actor
        assert default.current_value_lr() == start_value
        assert constant.current_actor_lr() == start_actor
        assert constant.current_value_lr() == start_value


def test_linear_and_cosine_lrs_at_start_mid_end():
    total = 100
    linear = _make_trainer(
        0, lr_schedule="linear", lr_final=0.0, total_transitions=total
    )
    cosine = _make_trainer(
        1, lr_schedule="cosine", lr_final=0.0, total_transitions=total
    )
    samples = {
        0: 0.0,
        50: 0.5,
        100: 1.0,
        150: 1.0,
    }
    for transitions, progress in samples.items():
        linear.set_schedule_transitions(transitions)
        cosine.set_schedule_transitions(transitions)
        expected_linear_actor = scheduled_learning_rate(
            progress, ACTOR_LR, 0.0, "linear"
        )
        expected_cosine_actor = scheduled_learning_rate(
            progress, ACTOR_LR, 0.0, "cosine"
        )
        expected_linear_value = scheduled_learning_rate(
            progress, VALUE_LR, 0.0, "linear"
        )
        expected_cosine_value = scheduled_learning_rate(
            progress, VALUE_LR, 0.0, "cosine"
        )
        assert linear.current_actor_lr() == pytest.approx(expected_linear_actor)
        assert linear.current_value_lr() == pytest.approx(expected_linear_value)
        assert cosine.current_actor_lr() == pytest.approx(expected_cosine_actor)
        assert cosine.current_value_lr() == pytest.approx(expected_cosine_value)

    assert scheduled_learning_rate(0.0, ACTOR_LR, 0.0, "linear") == ACTOR_LR
    assert scheduled_learning_rate(1.0, ACTOR_LR, 0.0, "linear") == 0.0
    assert scheduled_learning_rate(0.5, ACTOR_LR, 0.0, "linear") == pytest.approx(
        0.5 * ACTOR_LR
    )
    assert scheduled_learning_rate(0.5, ACTOR_LR, 0.0, "cosine") == pytest.approx(
        0.5 * ACTOR_LR
    )
    quarter_cos = 0.5 * ACTOR_LR * (1.0 + math.cos(math.pi * 0.25))
    assert scheduled_learning_rate(0.25, ACTOR_LR, 0.0, "cosine") == pytest.approx(
        quarter_cos
    )
    assert scheduled_learning_rate(0.25, ACTOR_LR, 0.0, "linear") == pytest.approx(
        0.75 * ACTOR_LR
    )


def test_adaptive_kl_matches_rudin():
    desired = 0.01
    lr = 3e-5
    assert adapt_kl_learning_rate(lr, 0.03, desired) == pytest.approx(lr / 1.5)
    assert adapt_kl_learning_rate(lr, 0.001, desired) == pytest.approx(lr * 1.5)
    assert adapt_kl_learning_rate(lr, 0.01, desired) == pytest.approx(lr)
    trainer = _make_trainer(0, lr_schedule="adaptive_kl", desired_kl=desired)
    trainer.adapt_kl_learning_rates(0.03)
    assert trainer.current_actor_lr() == pytest.approx(ACTOR_LR / 1.5)
    assert trainer.current_value_lr() == pytest.approx(VALUE_LR / 1.5)
    trainer.adapt_kl_learning_rates(0.001)
    assert trainer.current_actor_lr() == pytest.approx(ACTOR_LR)
    assert trainer.current_value_lr() == pytest.approx(VALUE_LR)


def _rollout_once(trainer, seed):
    torch.manual_seed(seed)
    actor_obs = torch.zeros(1, ACTOR_OBS_DIM)
    critic_obs = torch.zeros(1, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    metrics = None
    for _ in range(trainer.rollout_steps):
        trainer.act(actor_obs, critic_obs)
        metrics = trainer.observe(
            actor_obs,
            critic_obs,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
        )
    return metrics


def test_default_update_matches_explicit_constant_defaults():
    default = _make_trainer(3)
    explicit = _make_trainer(3, lr_schedule="constant", lr_final=0.0)
    torch.manual_seed(11)
    default_metrics = _rollout_once(default, 11)
    torch.manual_seed(11)
    explicit_metrics = _rollout_once(explicit, 11)
    assert default_metrics is not None
    assert explicit_metrics is not None
    for key, tensor in default.actor.state_dict().items():
        assert torch.equal(tensor, explicit.actor.state_dict()[key]), key
    for key, tensor in default.value_critic.state_dict().items():
        assert torch.equal(tensor, explicit.value_critic.state_dict()[key]), key
    assert default.current_actor_lr() == explicit.current_actor_lr() == ACTOR_LR
    assert default.current_value_lr() == explicit.current_value_lr() == VALUE_LR


def test_linear_schedule_survives_resume(tmp_path):
    total = 100
    trainer = _make_trainer(
        4, lr_schedule="linear", lr_final=0.0, total_transitions=total
    )
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))
    trainer.set_schedule_transitions(50)
    mid_actor = trainer.current_actor_lr()
    mid_value = trainer.current_value_lr()
    assert mid_actor == pytest.approx(0.5 * ACTOR_LR)
    assert mid_value == pytest.approx(0.5 * VALUE_LR)

    path = tmp_path / "resume_state.pt"
    save_resume_state(
        path,
        algorithm="ppo",
        models=ActorModels(actor=trainer.actor),
        trainer=trainer,
        actor_normalizer=actor_norm,
        critic_normalizer=critic_norm,
        env_transitions=50,
        gradient_updates=8,
        vector_ticks=50,
        sampled_replay_rows=16,
        device=torch.device("cpu"),
    )

    restored = _make_trainer(
        99, lr_schedule="linear", lr_final=0.0, total_transitions=total
    )
    assert restored.current_actor_lr() == pytest.approx(ACTOR_LR)
    payload = load_resume_state(
        str(path),
        algorithm="ppo",
        models=ActorModels(actor=restored.actor),
        trainer=restored,
        actor_normalizer=ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu")),
        critic_normalizer=ObsNormalizer(CRITIC_DIM, torch.device("cpu")),
        device=torch.device("cpu"),
    )
    assert payload["env_transitions"] == 50
    assert restored.current_actor_lr() == pytest.approx(mid_actor)
    assert restored.current_value_lr() == pytest.approx(mid_value)
    restored.set_schedule_transitions(100)
    assert restored.current_actor_lr() == pytest.approx(0.0)
    assert restored.current_value_lr() == pytest.approx(0.0)


def test_adaptive_kl_survives_resume(tmp_path):
    trainer = _make_trainer(5, lr_schedule="adaptive_kl", desired_kl=0.01)
    trainer.adapt_kl_learning_rates(0.05)
    saved_actor = trainer.current_actor_lr()
    saved_value = trainer.current_value_lr()
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))
    path = tmp_path / "resume_state.pt"
    save_resume_state(
        path,
        algorithm="ppo",
        models=ActorModels(actor=trainer.actor),
        trainer=trainer,
        actor_normalizer=actor_norm,
        critic_normalizer=critic_norm,
        env_transitions=256,
        gradient_updates=4,
        vector_ticks=2,
        sampled_replay_rows=8,
        device=torch.device("cpu"),
    )
    restored = _make_trainer(6, lr_schedule="adaptive_kl", desired_kl=0.01)
    load_resume_state(
        str(path),
        algorithm="ppo",
        models=ActorModels(actor=restored.actor),
        trainer=restored,
        actor_normalizer=ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu")),
        critic_normalizer=ObsNormalizer(CRITIC_DIM, torch.device("cpu")),
        device=torch.device("cpu"),
    )
    assert restored.current_actor_lr() == pytest.approx(saved_actor)
    assert restored.current_value_lr() == pytest.approx(saved_value)


def test_default_config_keys_validate():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    validate_algorithm_config(cfg, num_envs=32)
    assert cfg["ppo"]["lr_schedule"] == "constant"
    assert cfg["ppo"]["lr_final"] == 0.0
    assert cfg["ppo"]["policy_mean_init_scale"] is None
    assert cfg["ppo"]["policy_init_log_std"] is None
    cfg["ppo"]["lr_schedule"] = "bogus"
    with pytest.raises(ValueError, match="lr_schedule"):
        validate_algorithm_config(cfg, num_envs=32)

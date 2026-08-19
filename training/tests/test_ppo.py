from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from f1tenth_policy import ObsNormalizer, SquashedGaussianLidarGRUActor
from config import DEFAULT_CONFIG
from ppo import (
    PPOTrainer,
    ValueCritic,
    advantage_filter_keep_mask,
    apply_timeout_bootstrap_rewards,
    compute_gae,
    pure_timeout_mask,
)
from standalone_trainer import (
    build_ppo_models,
    initial_training_protocol_state,
    save_policy_artifact,
)

LIDAR_DIM = 64
PROPRIO_DIM = 3
ACTOR_OBS_DIM = LIDAR_DIM + PROPRIO_DIM
CRITIC_OBS_DIM = 7


def _actor(seed=0):
    torch.manual_seed(seed)
    return SquashedGaussianLidarGRUActor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=2,
        hidden_sizes=[8],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
        pool_bins=16,
        projection_dim=8,
        gru_hidden_dim=8,
    )


def _trainer(
    *,
    rollout_steps=4,
    action_clip=1.0,
    compile=False,
    compile_mode="default",
    advantage_filter_enabled=True,
    advantage_filter_discard_fraction=0.05,
):
    device = torch.device("cpu")
    actor = _actor()
    actor_normalizer = ObsNormalizer(ACTOR_OBS_DIM, device)
    critic_normalizer = ObsNormalizer(CRITIC_OBS_DIM, device)
    trainer = PPOTrainer(
        actor,
        CRITIC_OBS_DIM,
        actor_normalizer,
        critic_normalizer,
        device,
        value_hidden_sizes=(16,),
        rollout_steps=rollout_steps,
        num_epochs=2,
        env_minibatch_size=2,
        actor_lr=1e-3,
        value_lr=1e-3,
        max_grad_norm=1e-6,
        action_clip=action_clip,
        compile=compile,
        compile_mode=compile_mode,
        advantage_filter_enabled=advantage_filter_enabled,
        advantage_filter_discard_fraction=advantage_filter_discard_fraction,
    )
    return trainer


def test_sampled_and_stored_action_log_prob_match_without_state_change():
    actor = _actor()
    keys_before = tuple(actor.state_dict())
    batch, steps = 3, 4
    obs = torch.randn(batch, steps, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(batch)
    reset = torch.tensor(
        [
            [True, False, False, True],
            [True, False, True, False],
            [True, False, False, False],
        ]
    )

    torch.manual_seed(9)
    actions, sampled_logp, _ = actor.forward_sequence(
        obs, hidden, reset_mask=reset
    )
    evaluated_logp, _, _ = actor.evaluate_actions_sequence(
        obs, hidden, actions, reset_mask=reset
    )

    assert torch.allclose(sampled_logp, evaluated_logp, atol=1e-6, rtol=1e-6)
    assert tuple(actor.state_dict()) == keys_before


def test_pre_tanh_log_prob_remains_exact_for_saturated_actions():
    actor = _actor()
    with torch.no_grad():
        actor.mu_layer.weight.zero_()
        actor.mu_layer.bias.fill_(20.0)
        actor.log_std_layer.weight.zero_()
        actor.log_std_layer.bias.fill_(-20.0)
    obs = torch.zeros(1, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(1)
    reset = torch.ones(1, dtype=torch.bool)

    action, sampled_logp, _, pre_tanh = actor.step(
        obs,
        hidden,
        reset_mask=reset,
        deterministic=True,
        return_pre_tanh=True,
    )
    exact_logp, _, _ = actor.evaluate_actions_sequence(
        obs[:, None],
        hidden,
        action[:, None],
        reset_mask=reset[:, None],
        pre_tanh_actions=pre_tanh[:, None],
    )
    reconstructed_logp, _, _ = actor.evaluate_actions_sequence(
        obs[:, None],
        hidden,
        action[:, None],
        reset_mask=reset[:, None],
    )

    assert torch.equal(action, torch.ones_like(action))
    assert torch.allclose(sampled_logp, exact_logp[:, 0])
    assert not torch.allclose(sampled_logp, reconstructed_logp[:, 0])


def test_compute_gae_matches_hand_calculation_and_stops_at_terminal():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.6], [0.7]])
    dones = torch.tensor([[False], [True], [False]])
    bootstrap = torch.tensor([0.9])

    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        bootstrap,
        gamma=0.9,
        gae_lambda=0.8,
    )

    expected_advantages = torch.tensor([[2.048], [1.4], [3.11]])
    assert torch.allclose(advantages, expected_advantages, atol=1e-6)
    assert torch.allclose(returns, expected_advantages + values, atol=1e-6)


def test_lifecycle_rejects_ordering_misuse_and_narrow_action_clip():
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer = _trainer(rollout_steps=2)
    with pytest.raises(RuntimeError, match="initialize"):
        trainer.act(actor_obs, critic_obs)
    with pytest.raises(RuntimeError, match="initialize"):
        trainer.observe(
            actor_obs, critic_obs, torch.ones(2), torch.zeros(2, dtype=torch.bool)
        )

    trainer.initialize(actor_obs, critic_obs)
    with pytest.raises(RuntimeError, match="already initialized"):
        trainer.initialize(actor_obs, critic_obs)
    with pytest.raises(RuntimeError, match="act must"):
        trainer.observe(
            actor_obs, critic_obs, torch.ones(2), torch.zeros(2, dtype=torch.bool)
        )
    trainer.act(actor_obs, critic_obs)
    with pytest.raises(RuntimeError, match="observe must"):
        trainer.act(actor_obs, critic_obs)
    assert (
        trainer.observe(
            actor_obs, critic_obs, torch.ones(2), torch.zeros(2, dtype=torch.bool)
        )
        is None
    )

    with pytest.raises(ValueError, match="must equal"):
        _trainer(action_clip=0.9)


def test_incremental_boundary_updates_raw_preallocated_rollout_and_resets_hidden():
    trainer = _trainer(rollout_steps=3)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    actor_storage = trainer._rollout.actor_obs.data_ptr()
    critic_storage = trainer._rollout.critic_obs.data_ptr()
    actor_count = trainer.actor_normalizer.count
    critic_count = trainer.critic_normalizer.count
    raw_actor = []
    metrics = None

    for step in range(3):
        current_actor = torch.full_like(actor_obs, float(step + 4))
        current_critic = torch.full_like(critic_obs, float(step + 7))
        raw_actor.append(current_actor)
        action = trainer.act(current_actor, current_critic)
        assert action.shape == (2, 2)
        next_actor = torch.full_like(actor_obs, float(step + 5))
        next_critic = torch.full_like(critic_obs, float(step + 8))
        reward = torch.ones(2)
        done = torch.tensor([step == 0, step == 1])
        metrics = trainer.observe(next_actor, next_critic, reward, done)
        if step < 2:
            assert metrics is None
            assert trainer.actor_normalizer.count == actor_count
            assert trainer.critic_normalizer.count == critic_count

    rollout = trainer.last_rollout
    assert metrics is not None
    assert rollout.actor_obs.data_ptr() == actor_storage
    assert rollout.critic_obs.data_ptr() == critic_storage
    assert torch.equal(rollout.actor_obs, torch.stack(raw_actor))
    assert not hasattr(rollout, "raw_actor_obs")
    assert torch.equal(rollout.reset_masks[0], torch.tensor([True, True]))
    assert torch.equal(rollout.reset_masks[1], torch.tensor([True, False]))
    assert torch.equal(rollout.reset_masks[2], torch.tensor([False, True]))
    assert rollout.dones[0, 0]
    assert rollout.dones[1, 1]
    assert trainer.actor_normalizer.count == actor_count + 6
    assert trainer.critic_normalizer.count == critic_count + 6
    assert torch.count_nonzero(trainer._live_hidden) == 0
    assert trainer._step == 0
    assert not trainer._awaiting_observe
    assert metrics["actor_updates"] > 0
    assert metrics["value_updates"] > 0


def test_boundary_update_is_finite_and_reports_gradient_clipping():
    trainer = _trainer()
    actor_obs = torch.randn(4, ACTOR_OBS_DIM)
    critic_obs = torch.randn(4, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    actor_before = [
        parameter.detach().clone() for parameter in trainer.actor.parameters()
    ]
    critic_before = [
        parameter.detach().clone() for parameter in trainer.value_critic.parameters()
    ]
    metrics = None
    for step in range(4):
        actions = trainer.act(actor_obs, critic_obs)
        next_actor = actor_obs + 0.01 * (step + 1)
        next_critic = critic_obs + 0.02 * (step + 1)
        reward = 1.0 - actions.square().sum(dim=-1)
        done = torch.tensor([False, step == 1, False, step == 2])
        metrics = trainer.observe(next_actor, next_critic, reward, done)
        actor_obs = next_actor
        critic_obs = next_critic

    assert metrics is not None
    for value in metrics.values():
        if isinstance(value, float):
            assert math.isfinite(value)
    assert metrics["actor_grad_clip_fraction"] > 0.0
    assert metrics["value_grad_clip_fraction"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, trainer.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            critic_before, trainer.value_critic.parameters(), strict=True
        )
    )


def test_compiled_ppo_update_smoke():
    trainer = _trainer(rollout_steps=2, compile=True)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)

    metrics = None
    for _ in range(2):
        actions = trainer.act(actor_obs, critic_obs)
        next_actor = actor_obs + 0.01
        next_critic = critic_obs + 0.01
        metrics = trainer.observe(
            next_actor,
            next_critic,
            1.0 - actions.square().sum(dim=-1),
            torch.zeros(2, dtype=torch.bool),
        )
        actor_obs = next_actor
        critic_obs = next_critic

    assert trainer.compile
    assert metrics is not None
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["value_loss"])


def test_standalone_ppo_build_is_actor_only_and_artifact_compatible(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["actor_hidden_layers"] = [8]
    cfg["model"]["critic_hidden_layers"] = [8]
    cfg["model"]["lidar_pool_bins"] = 16
    device = torch.device("cpu")
    actor_normalizer = ObsNormalizer(cfg["obs"]["num_actor_obs"], device)
    critic_normalizer = ObsNormalizer(cfg["obs"]["num_obs"], device)

    models, trainer = build_ppo_models(
        cfg,
        device,
        actor_normalizer,
        critic_normalizer,
        num_envs=16,
    )

    assert tuple(vars(models)) == ("actor",)
    assert trainer.actor is models.actor
    assert isinstance(trainer.value_critic.net[1], nn.ReLU)
    assert trainer.env_minibatch_size == 1
    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert trainer.value_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    path = save_policy_artifact(
        models,
        2048,
        tmp_path,
        actor_normalizer,
        cfg,
        protocol_state=initial_training_protocol_state("ppo"),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert tuple(payload["actor"]) == tuple(models.actor.state_dict())
    assert payload["training_protocol"]["algorithm"] == "ppo"


def test_pure_timeout_mask_excludes_other_termination_flags():
    termination = {
        "time_out": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "out_of_bounds": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "not_moving": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "invalid_state": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "collision": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    }
    mask = pure_timeout_mask(termination)
    assert mask.tolist() == [True, False, False, False]


def test_timeout_bootstrap_adds_gamma_times_terminal_value_only():
    torch.manual_seed(0)
    critic = ValueCritic(7, (8,))
    reward = torch.tensor([1.0, 2.0, 3.0])
    timed_out = torch.tensor([True, False, True])
    terminal_obs = torch.randn(3, 7)
    with torch.no_grad():
        terminal_values = critic(terminal_obs)
    augmented = apply_timeout_bootstrap_rewards(
        reward,
        timed_out,
        terminal_obs,
        value_critic=critic,
        critic_normalizer=None,
        gamma=0.9,
        device=torch.device("cpu"),
    )
    assert augmented[0] == pytest.approx(reward[0] + 0.9 * terminal_values[0])
    assert augmented[1] == reward[1]
    assert augmented[2] == pytest.approx(reward[2] + 0.9 * terminal_values[2])


def test_observe_timeout_bootstrap_uses_terminal_not_post_reset_obs():
    trainer = _trainer(rollout_steps=1)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    terminal_obs = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    post_reset_obs = torch.tensor(
        [
            [-50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    reward = torch.tensor([1.0, 2.0])
    trainer.act(actor_obs, critic_obs)
    with torch.no_grad():
        normalized_terminal = trainer.critic_normalizer.normalize(terminal_obs[0:1])
        terminal_value = trainer.value_critic(normalized_terminal).squeeze(0)
        normalized_post = trainer.critic_normalizer.normalize(post_reset_obs[0:1])
        post_reset_value = trainer.value_critic(normalized_post).squeeze(0)
    assert not torch.allclose(terminal_value, post_reset_value)
    trainer.observe(
        actor_obs,
        post_reset_obs,
        reward,
        torch.tensor([True, False]),
        timed_out=torch.tensor([True, False]),
        timeout_critic_obs=terminal_obs,
    )
    expected = reward[0] + trainer.gamma * terminal_value
    wrong = reward[0] + trainer.gamma * post_reset_value
    assert trainer.last_rollout.rewards[0, 0] == pytest.approx(expected)
    assert trainer.last_rollout.rewards[0, 0] != pytest.approx(wrong)


def test_mixed_timeout_and_terminal_batch_gae_returns():
    trainer = _trainer(rollout_steps=2)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)

    terminal_obs = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    post_reset_obs = torch.tensor(
        [
            [-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    step0_reward = torch.tensor([1.0, 1.0])

    trainer.act(actor_obs, critic_obs)
    trainer.observe(
        actor_obs,
        post_reset_obs,
        step0_reward,
        torch.tensor([True, False]),
        timed_out=torch.tensor([True, False]),
        timeout_critic_obs=terminal_obs,
    )
    expected_step0 = apply_timeout_bootstrap_rewards(
        step0_reward,
        torch.tensor([True, False]),
        terminal_obs,
        value_critic=trainer.value_critic,
        critic_normalizer=trainer.critic_normalizer,
        gamma=trainer.gamma,
        device=trainer.device,
    )

    trainer.act(actor_obs, critic_obs)
    metrics = trainer.observe(
        actor_obs,
        post_reset_obs,
        torch.tensor([0.5, 0.5]),
        torch.tensor([False, True]),
        timed_out=torch.tensor([False, False]),
        timeout_critic_obs=terminal_obs,
    )
    assert metrics is not None
    rollout = trainer.last_rollout
    assert rollout.rewards[0, 0] == pytest.approx(expected_step0[0])
    assert rollout.rewards[0, 1] == pytest.approx(step0_reward[1])
    assert rollout.dones[1, 1]
    assert rollout.returns[1, 1] == pytest.approx(rollout.rewards[1, 1])


def test_advantage_filter_disabled_keeps_all_transitions_and_zero_discard():
    advantages = torch.tensor([[0.1, -0.2], [0.3, 0.0]])
    keep, _threshold = advantage_filter_keep_mask(
        advantages,
        enabled=False,
        discard_fraction=0.5,
    )
    assert keep.all()

    trainer = _trainer(rollout_steps=2, advantage_filter_enabled=False)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    metrics = None
    for _ in range(2):
        trainer.act(actor_obs, critic_obs)
        metrics = trainer.observe(
            actor_obs,
            critic_obs,
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
        )
    assert metrics is not None
    assert metrics["advantage_filter_discard_rate"] == pytest.approx(0.0)
    assert metrics["advantage_filter_retained"] == 4
    for value in metrics.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_advantage_filter_keep_mask_drops_bottom_fraction():
    advantages = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    keep, threshold = advantage_filter_keep_mask(
        advantages,
        enabled=True,
        discard_fraction=0.05,
    )
    assert int((~keep).sum()) == 1
    assert int(keep.sum()) == 19
    assert threshold == pytest.approx(0.0)


def test_advantage_filter_discard_rate_stable_across_advantage_scales():
    base = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
            ]
        ]
    )
    for scale in (0.01, 1.0, 100.0, 165.0, 0.65):
        advantages = base * scale
        keep, _threshold = advantage_filter_keep_mask(
            advantages,
            enabled=True,
            discard_fraction=0.05,
        )
        discard_rate = 1.0 - keep.sum().item() / keep.numel()
        assert discard_rate == pytest.approx(0.05)


def test_advantage_filter_tie_heavy_advantages_drop_at_most_one_fraction():
    advantages = torch.zeros(20, dtype=torch.float32).reshape(4, 5)
    keep, threshold = advantage_filter_keep_mask(
        advantages,
        enabled=True,
        discard_fraction=0.05,
    )
    assert int((~keep).sum()) == 1
    assert int(keep.sum()) == 19
    assert threshold == pytest.approx(0.0)


def test_advantage_filter_all_zero_advantages_does_not_crash():
    trainer = _trainer(
        rollout_steps=10,
        advantage_filter_enabled=True,
    )
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    for _ in range(10):
        trainer.act(actor_obs, critic_obs)
        trainer.observe(
            actor_obs,
            critic_obs,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
        )
    trainer._rollout.advantages.zero_()
    trainer._rollout.returns.copy_(trainer._rollout.old_values)
    trainer._rollout.optimized = False
    metrics = trainer.update(trainer._rollout)
    for key, value in metrics.items():
        if isinstance(value, float):
            assert math.isfinite(value)
    assert metrics["advantage_filter_retained"] == 19
    assert metrics["advantage_filter_discard_rate"] == pytest.approx(0.05)


def test_advantage_filter_enabled_end_to_end_update_is_finite():
    trainer = _trainer(rollout_steps=2, advantage_filter_enabled=True)
    actor_obs = torch.randn(2, ACTOR_OBS_DIM)
    critic_obs = torch.randn(2, CRITIC_OBS_DIM)
    trainer.initialize(actor_obs, critic_obs)
    metrics = None
    for _ in range(2):
        actions = trainer.act(actor_obs, critic_obs)
        next_actor = actor_obs + 0.01
        next_critic = critic_obs + 0.01
        metrics = trainer.observe(
            next_actor,
            next_critic,
            1.0 - actions.square().sum(dim=-1),
            torch.zeros(2, dtype=torch.bool),
        )
        actor_obs = next_actor
        critic_obs = next_critic

    assert metrics is not None
    assert metrics["advantage_filter_retained"] > 0
    assert 0.0 <= metrics["advantage_filter_discard_rate"] < 1.0
    for key in ("policy_loss", "value_loss", "advantage_filter_eta"):
        assert math.isfinite(metrics[key])

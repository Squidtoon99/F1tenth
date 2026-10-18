"""Focused recurrent PPO unit tests (no end-to-end trainer)."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from gigaflow_f1tenth.buffers import CompactRolloutBatch
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.critic import CriticOutput, CriticShapes
from gigaflow_f1tenth.model import ActorOutput, ActorShapes
from gigaflow_f1tenth.ppo import (
    AdaptiveFilterState,
    CollectEvaluateParityError,
    PreparedPPOInputs,
    adaptive_advantage_keep_mask,
    agents_per_minibatch,
    build_ppo,
    compute_gae,
    entropy_coefficient,
    evaluate_actions_sequence,
    export_resume_state,
    initial_filter_state,
    load_resume_state,
    orthogonal_init_,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


class TinyActor(nn.Module):
    """Minimal recurrent continuous actor satisfying PPO seams."""

    def __init__(self, obs_dim: int, cond_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self._shapes = ActorShapes(
            sensor_obs_dim=obs_dim,
            lidar_dim=obs_dim - 2,
            proprio_dim=2,
            condition_dim=cond_dim,
            action_dim=act_dim,
            gru_hidden_dim=hidden,
            cnn_projection_dim=8,
            mlp_sizes=(16,),
        )
        self.enc = nn.Linear(obs_dim + cond_dim, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def shapes(self) -> ActorShapes:
        return self._shapes

    def initial_hidden(self, batch_size: int, device: str) -> Tensor:
        return torch.zeros(
            batch_size, self._shapes.gru_hidden_dim, device=device
        )

    def forward(
        self,
        sensor_obs: Tensor,
        private_condition: Tensor,
        hidden: Tensor,
        reset_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> ActorOutput:
        if reset_mask is not None:
            hidden = hidden * (~reset_mask).unsqueeze(-1).to(hidden.dtype)
        x = torch.tanh(self.enc(torch.cat([sensor_obs, private_condition], dim=-1)))
        h = self.gru(x, hidden)
        mean = self.mu(h)
        std = self.log_std.exp().expand_as(mean)
        if deterministic:
            pre = mean
        else:
            pre = mean + std * torch.randn_like(mean)
        actions = torch.tanh(pre)
        # tanh-Gaussian logprob (summed)
        log_prob = (
            -0.5 * (((pre - mean) / (std + 1e-8)) ** 2 + 2 * self.log_std + 1.837877)
        ).sum(-1)
        log_prob = log_prob - torch.log(1.0 - actions.pow(2) + 1e-6).sum(-1)
        entropy = (0.5 + 0.5 * 1.837877 + self.log_std).sum().expand(actions.shape[0])
        return ActorOutput(
            actions=actions,
            log_prob=log_prob,
            entropy=entropy,
            hidden=h,
            pre_tanh=pre,
        )

    def evaluate_actions_sequence(
        self,
        sensor_obs: Tensor,
        private_condition: Tensor,
        hidden: Tensor,
        actions: Tensor,
        reset_mask: Tensor | None = None,
        pre_tanh: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # sensor_obs/actions: [B, T, ...]; condition: [B, C] or [B, T, C]
        b, t, _ = sensor_obs.shape
        if private_condition.dim() == 2:
            cond_seq = private_condition.unsqueeze(1).expand(b, t, -1)
        else:
            cond_seq = private_condition
        logps = []
        ents = []
        h = hidden
        for i in range(t):
            rm = None if reset_mask is None else reset_mask[:, i]
            if rm is not None:
                h = h * (~rm).unsqueeze(-1).to(h.dtype)
            x = torch.tanh(
                self.enc(torch.cat([sensor_obs[:, i], cond_seq[:, i]], dim=-1))
            )
            h = self.gru(x, h)
            mean = self.mu(h)
            std = self.log_std.exp().expand_as(mean)
            if pre_tanh is not None:
                pre = pre_tanh[:, i]
            else:
                a = actions[:, i].clamp(-0.999999, 0.999999)
                pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))
            a = torch.tanh(pre).clamp(-0.999999, 0.999999)
            log_prob = (
                -0.5
                * (((pre - mean) / (std + 1e-8)) ** 2 + 2 * self.log_std + 1.837877)
            ).sum(-1)
            log_prob = log_prob - torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
            ent = (0.5 + 0.5 * 1.837877 + self.log_std).sum().expand(b)
            logps.append(log_prob)
            ents.append(ent)
        return torch.stack(logps, dim=1), torch.stack(ents, dim=1), h


class TinyCritic(nn.Module):
    def __init__(self, ego_dim: int, el_dim: int, cond_dim: int, n_others: int):
        super().__init__()
        self._shapes = CriticShapes(
            ego_state_dim=ego_dim,
            element_dim=el_dim,
            condition_dim=cond_dim,
            max_other_agents=n_others,
            hidden_dim=16,
            mlp_sizes=(16,),
        )
        self.net = nn.Sequential(
            nn.Linear(ego_dim + el_dim + cond_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def shapes(self) -> CriticShapes:
        return self._shapes

    def forward(
        self,
        ego_state: Tensor,
        other_agents: Tensor,
        other_mask: Tensor,
        private_condition: Tensor,
    ) -> CriticOutput:
        masked = other_agents * other_mask.unsqueeze(-1).to(other_agents.dtype)
        pooled = masked.max(dim=1).values
        x = torch.cat([ego_state, pooled, private_condition], dim=-1)
        return CriticOutput(values=self.net(x).squeeze(-1))


class CrossSequenceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.register_buffer("log_std", torch.zeros(1))

    def initial_hidden(self, batch_size: int, device: str) -> Tensor:
        return torch.zeros(batch_size, 1, device=device)

    def evaluate_actions_sequence(
        self,
        sensor_obs,
        private_condition,
        hidden,
        actions,
        reset_mask=None,
        pre_tanh=None,
    ):
        del private_condition, actions, reset_mask
        assert pre_tanh is not None
        mean = self.weight * sensor_obs[..., :1]
        logp = -0.5 * (pre_tanh - mean).square().sum(dim=-1)
        entropy = (self.weight * 0.0 + 1.0).expand_as(logp)
        return logp, entropy, hidden


def _make_batch(
    t: int,
    s: int,
    a: int,
    h: int,
    c: int,
    state_dim: int,
    *,
    device: str = "cpu",
) -> CompactRolloutBatch:
    torch.manual_seed(0)
    valid = torch.ones(t, s, dtype=torch.bool, device=device)
    valid[:, -1] = False  # one inactive padding slot
    done = torch.zeros(t, s, dtype=torch.bool, device=device)
    timeout = torch.zeros(t, s, dtype=torch.bool, device=device)
    done[t // 2, 0] = True
    timeout[-1, 1] = True
    reset = done.clone()
    pre = torch.randn(t, s, a, device=device)
    return CompactRolloutBatch(
        state=torch.randn(t, s, state_dim, device=device),
        next_state=torch.randn(t, s, state_dim, device=device),
        actions=torch.tanh(pre),
        rewards=torch.randn(t, s, device=device),
        valid=valid,
        done=done,
        timeout=timeout,
        reset_mask=reset,
        track_id=torch.zeros(s, dtype=torch.int32, device=device),
        condition=torch.randn(t, s, c, device=device),
        sensor_noise_seed=torch.zeros(t, s, dtype=torch.int64, device=device),
        episode_id=torch.zeros(t, s, dtype=torch.int32, device=device),
        episode_step=torch.zeros(t, s, dtype=torch.int32, device=device),
        gru_start=torch.zeros(s, h, device=device),
        old_logp=torch.randn(t, s, device=device) * 0.01,
        obs_digest=torch.zeros(t, s, dtype=torch.int64, device=device),
        pre_tanh=pre,
    )


def _consistent_tiny_case(cfg):
    obs_dim = 8
    cond_dim = cfg.agents.condition_dim
    actor = TinyActor(obs_dim, cond_dim, 2, 8)
    critic = TinyCritic(obs_dim, 4, cond_dim, 1)
    learner = build_ppo(
        cfg,
        actor,
        critic,
        total_updates=cfg.ppo.total_updates,
        device="cpu",
    )
    t = cfg.ppo.rollout_length
    s = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    batch = _make_batch(t, s, 2, 8, cond_dim, obs_dim)
    with torch.no_grad():
        hidden = actor.initial_hidden(s, "cpu")
        batch.gru_start.copy_(hidden)
        for step in range(t):
            out = actor.forward(
                batch.state[step],
                batch.condition[step],
                hidden,
                reset_mask=batch.reset_mask[step],
            )
            batch.actions[step].copy_(out.actions)
            batch.old_logp[step].copy_(out.log_prob)
            batch.pre_tanh[step].copy_(out.pre_tanh)
            hidden = out.hidden
    prepared = PreparedPPOInputs(
        sensor_obs=batch.state,
        ego_state=batch.state,
        other_agents=torch.zeros(t, s, 1, 4),
        other_mask=torch.ones(t, s, 1, dtype=torch.bool),
        bootstrap_values=torch.zeros(t, s),
    )
    return learner, batch, prepared


def _assert_optimizer_state_equal(left, right):
    assert left.keys() == right.keys()
    assert left["state"].keys() == right["state"].keys()
    for key in left["state"]:
        assert left["state"][key].keys() == right["state"][key].keys()
        for name, value in left["state"][key].items():
            other = right["state"][key][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (key, name)
            else:
                assert value == other


def test_entropy_schedule_exact_values_and_resume_index():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            ent_coef=None,
            ent_coef_initial=0.005,
            ent_coef_final=0.0005,
            ent_anneal_updates=2000,
        ),
    )
    assert entropy_coefficient(cfg, 0) == pytest.approx(0.005)
    assert entropy_coefficient(cfg, 1000) == pytest.approx(0.00275)
    assert entropy_coefficient(cfg, 2000) == pytest.approx(0.0005)
    assert entropy_coefficient(cfg, 2500) == pytest.approx(0.0005)

    learner, _, _ = _consistent_tiny_case(cfg)
    learner.update_index = 1234
    payload = export_resume_state(learner)
    resumed, _, _ = _consistent_tiny_case(cfg)
    load_resume_state(resumed, payload)
    assert resumed.update_index == 1234
    assert entropy_coefficient(cfg, resumed.update_index) == pytest.approx(
        entropy_coefficient(cfg, learner.update_index)
    )


def test_actor_hard_kl_rollback_restores_adam_and_keeps_critic():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=2, max_agents_per_world=2),
        ppo=replace(
            cfg.ppo,
            rollout_length=4,
            num_epochs=2,
            minibatch_size=8,
            learning_rate=0.05,
            adaptive_filter_enabled=False,
            target_kl=0.0,
            actor_kl_soft=1.0e-12,
            actor_kl_hard=1.0e-10,
            actor_kl_warmup_updates=0,
            amp=False,
            total_updates=8,
        ),
        ablations=replace(cfg.ablations, adaptive_filter_enabled=False),
    )
    learner, batch, prepared = _consistent_tiny_case(cfg)

    learner.actor_optimizer.zero_grad(set_to_none=True)
    for parameter in learner.actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    learner.actor_optimizer.step()
    with torch.no_grad():
        hidden = learner.actor.initial_hidden(batch.rewards.shape[1], "cpu")
        for step in range(batch.rewards.shape[0]):
            out = learner.actor.forward(
                batch.state[step],
                batch.condition[step],
                hidden,
                reset_mask=batch.reset_mask[step],
            )
            batch.actions[step].copy_(out.actions)
            batch.old_logp[step].copy_(out.log_prob)
            batch.pre_tanh[step].copy_(out.pre_tanh)
            hidden = out.hidden

    actor_before = [
        parameter.detach().clone() for parameter in learner.actor.parameters()
    ]
    critic_before = [
        parameter.detach().clone() for parameter in learner.critic.parameters()
    ]
    optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
    scheduler_before = copy.deepcopy(learner.actor_scheduler.state_dict())
    gru_start_before = batch.gru_start.clone()
    reset_before = batch.reset_mask.clone()

    stats = learner.update(batch, prepared)
    assert stats.actor_rolled_back is True
    assert stats.candidate_kl > cfg.ppo.actor_kl_hard
    assert stats.actor_steps >= 1
    assert stats.critic_steps >= 1
    assert stats.actor_lr_safety_multiplier == pytest.approx(0.5)
    for before, after in zip(actor_before, learner.actor.parameters()):
        assert torch.equal(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, learner.critic.parameters())
    )
    _assert_optimizer_state_equal(
        optimizer_before,
        learner.actor_optimizer.state_dict(),
    )
    assert learner.actor_scheduler.state_dict() == scheduler_before
    assert torch.equal(batch.gru_start, gru_start_before)
    assert torch.equal(batch.reset_mask, reset_before)

    second = learner.update(batch, prepared)
    assert second.actor_rolled_back is True
    assert second.rollback_stop_requested is True
    assert second.consecutive_actor_rollbacks == 2
    for _ in range(4):
        learner.update(batch, prepared)
    assert learner.actor_lr_safety_multiplier == pytest.approx(0.125)


def test_final_full_rollout_kl_rejects_cross_minibatch_damage():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=1, max_agents_per_world=2),
        ppo=replace(
            cfg.ppo,
            rollout_length=1,
            num_epochs=1,
            minibatch_size=1,
            learning_rate=0.01,
            adaptive_filter_enabled=False,
            target_kl=0.0,
            actor_kl_soft=0.007,
            actor_kl_hard=0.0075,
            amp=False,
            total_updates=4,
        ),
        ablations=replace(cfg.ablations, adaptive_filter_enabled=False),
    )
    actor = CrossSequenceActor()
    critic = TinyCritic(1, 1, cfg.agents.condition_dim, 1)
    learner = build_ppo(cfg, actor, critic, total_updates=4, device="cpu")
    condition = torch.zeros(1, 2, cfg.agents.condition_dim)
    sensor = torch.tensor([[[10.0], [0.1]]])
    pre_tanh = torch.tensor([[[-1.0], [1.0]]])
    actions = torch.tanh(pre_tanh)
    with torch.no_grad():
        old_logp, _, _ = actor.evaluate_actions_sequence(
            sensor.transpose(0, 1),
            condition.transpose(0, 1),
            actor.initial_hidden(2, "cpu"),
            actions.transpose(0, 1),
            pre_tanh=pre_tanh.transpose(0, 1),
        )
    batch = CompactRolloutBatch(
        state=sensor,
        next_state=sensor.clone(),
        actions=actions,
        rewards=torch.tensor([[1.0, 2.0]]),
        valid=torch.ones(1, 2, dtype=torch.bool),
        done=torch.zeros(1, 2, dtype=torch.bool),
        timeout=torch.zeros(1, 2, dtype=torch.bool),
        reset_mask=torch.zeros(1, 2, dtype=torch.bool),
        track_id=torch.zeros(2, dtype=torch.int32),
        condition=condition,
        sensor_noise_seed=torch.zeros(1, 2, dtype=torch.int64),
        episode_id=torch.zeros(1, 2, dtype=torch.int32),
        episode_step=torch.zeros(1, 2, dtype=torch.int32),
        gru_start=actor.initial_hidden(2, "cpu"),
        old_logp=old_logp.transpose(0, 1).clone(),
        obs_digest=torch.zeros(1, 2, dtype=torch.int64),
        pre_tanh=pre_tanh,
    )
    prepared = PreparedPPOInputs(
        sensor_obs=sensor,
        ego_state=sensor,
        other_agents=torch.zeros(1, 2, 1, 1),
        other_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        old_values=torch.zeros(1, 2),
        bootstrap_values=torch.zeros(1, 2),
    )
    torch.manual_seed(0)
    actor_before = actor.weight.detach().clone()
    stats = learner.update(batch, prepared)
    assert stats.candidate_kl < cfg.ppo.actor_kl_hard
    assert stats.actor_rollback_full_rollout is True
    assert stats.actor_rolled_back is True
    assert torch.equal(actor.weight.detach(), actor_before)


def test_split_optimizer_resume_and_old_version_rejection():
    cfg = load_config(SMOKE)
    learner, _, _ = _consistent_tiny_case(cfg)
    for optimizer, module in (
        (learner.actor_optimizer, learner.actor),
        (learner.critic_optimizer, learner.critic),
    ):
        optimizer.zero_grad(set_to_none=True)
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    learner.actor_lr_safety_multiplier = 0.25
    learner.actor_rollback_count = 4
    learner.consecutive_actor_rollbacks = 1
    learner.rollback_updates = [10, 20, 30]
    learner.rollback_free_accepted_updates = 7
    learner._apply_actor_safety_lr()
    payload = export_resume_state(learner)

    resumed, _, _ = _consistent_tiny_case(cfg)
    load_resume_state(resumed, payload)
    _assert_optimizer_state_equal(
        learner.actor_optimizer.state_dict(),
        resumed.actor_optimizer.state_dict(),
    )
    _assert_optimizer_state_equal(
        learner.critic_optimizer.state_dict(),
        resumed.critic_optimizer.state_dict(),
    )
    assert resumed.actor_scheduler.state_dict() == learner.actor_scheduler.state_dict()
    assert resumed.critic_scheduler.state_dict() == learner.critic_scheduler.state_dict()
    assert resumed.actor_lr_safety_multiplier == pytest.approx(0.25)
    assert resumed.actor_rollback_count == 4
    assert resumed.consecutive_actor_rollbacks == 1
    assert resumed.rollback_updates == [10, 20, 30]
    assert resumed.rollback_free_accepted_updates == 7

    old = dict(payload)
    old["resume_state_version"] = 1
    with pytest.raises(ValueError, match="single-optimizer"):
        load_resume_state(resumed, old)


def test_gae_truncation_bootstraps_next_state_and_ignores_next_episode():
    """A truncation must use V(s'_t); values[t+1] is already a new episode."""
    gamma, lam = 0.9, 1.0
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    # values[2] belongs to the episode that respawned after the t=1 truncation.
    values = torch.tensor([[0.5], [1.0], [7.0]])
    next_values = torch.tensor([[0.0], [4.0], [0.0]])
    done = torch.tensor([[False], [True], [False]])
    timeout = torch.tensor([[False], [True], [False]])
    last = torch.tensor([9.0])
    adv, ret = compute_gae(
        rewards,
        values,
        done,
        timeout,
        last,
        gamma=gamma,
        gae_lambda=lam,
        next_values=next_values,
    )
    truncated_delta = 2.0 + gamma * 4.0 - 1.0
    assert adv[1, 0].item() == pytest.approx(truncated_delta, abs=1e-6)
    # The step before the cut still chains through its own next value.
    delta0 = 1.0 + gamma * 1.0 - 0.5
    assert adv[0, 0].item() == pytest.approx(
        delta0 + gamma * lam * truncated_delta, abs=1e-6
    )
    # The surviving tail bootstraps past the rollout edge.
    assert adv[2, 0].item() == pytest.approx(3.0 + gamma * 9.0 - 7.0, abs=1e-6)
    assert torch.allclose(ret, adv + values)

    # No post-respawn value may reach a pre-truncation advantage.
    poisoned = values.clone()
    poisoned[2, 0] = -1000.0
    adv_poisoned, _ = compute_gae(
        rewards,
        poisoned,
        done,
        timeout,
        last,
        gamma=gamma,
        gae_lambda=lam,
        next_values=next_values,
    )
    assert torch.allclose(adv_poisoned[:2], adv[:2], atol=1e-6)


def test_gae_final_step_truncation_bootstraps_next_state_not_own_value():
    """Final-step bootstrap must be V(s'_{T-1}), not the action-time value."""
    gamma = 0.9
    rewards = torch.zeros(2, 1)
    values = torch.tensor([[1.0], [2.0]])
    done = torch.tensor([[False], [True]])
    timeout = done.clone()
    next_values = torch.tensor([[0.0], [5.0]])
    adv, _ = compute_gae(
        rewards,
        values,
        done,
        timeout,
        values[-1].clone(),
        gamma=gamma,
        gae_lambda=1.0,
        next_values=next_values,
    )
    assert adv[1, 0].item() == pytest.approx(gamma * 5.0 - 2.0, abs=1e-6)
    # Pre-fix behaviour bootstrapped from old_values[-1].
    assert adv[1, 0].item() != pytest.approx(gamma * 2.0 - 2.0, abs=1e-6)


def test_gae_true_terminal_zeroes_bootstrap_and_cuts_trace():
    gamma = 0.9
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.ones(3, 1)
    done = torch.tensor([[False], [True], [False]])
    timeout = torch.zeros_like(done)
    next_values = torch.full((3, 1), 50.0)
    adv, _ = compute_gae(
        rewards,
        values,
        done,
        timeout,
        torch.tensor([10.0]),
        gamma=gamma,
        gae_lambda=1.0,
        next_values=next_values,
    )
    assert adv[1, 0].item() == pytest.approx(2.0 - 1.0, abs=1e-6)


def test_gae_requires_next_values_for_truncations():
    rewards = torch.zeros(2, 1)
    values = torch.zeros(2, 1)
    done = torch.tensor([[False], [True]])
    timeout = done.clone()
    with pytest.raises(ValueError, match="next_values"):
        compute_gae(
            rewards,
            values,
            done,
            timeout,
            torch.zeros(1),
            gamma=0.9,
            gae_lambda=0.95,
        )


def test_adaptive_filter_ewma_and_retained_norm_inputs():
    cfg = load_config(SMOKE)
    filt = initial_filter_state(cfg)
    assert filt.beta == 0.25
    assert filt.eta_scale == 0.01
    assert filt.initialized is False
    adv = torch.tensor([[0.0, 1.0], [0.001, 2.0]])
    valid = torch.ones_like(adv, dtype=torch.bool)
    keep, new_filt, eta = adaptive_advantage_keep_mask(adv, valid, filt)
    # First observation initializes EWMA from current max (not beta-blend with 0).
    assert new_filt.initialized is True
    assert new_filt.ewma_max_abs_adv == 2.0
    assert abs(eta - 0.01 * 2.0) < 1e-8
    assert bool(keep[0, 0]) is False  # |0| < eta=0.02
    assert bool(keep[0, 1]) is True  # |1| >= 0.02
    assert bool(keep[1, 0]) is False  # |0.001| < 0.02
    assert bool(keep[1, 1]) is True


def test_adaptive_filter_paper_ewma_beta_on_current_max():
    """Paper: ewma = beta*current_max + (1-beta)*previous; beta=0.25."""
    filt = AdaptiveFilterState(
        ewma_max_abs_adv=10.0, beta=0.25, eta_scale=0.01, initialized=True
    )
    adv = torch.tensor([[1.0, 4.0], [0.05, 0.0]])
    valid = torch.tensor([[True, True], [True, False]])
    keep, new_filt, eta = adaptive_advantage_keep_mask(adv, valid, filt)
    # max over valid+finite only: max(1,4,0.05)=4; invalid 0 ignored
    expected_ewma = 0.25 * 4.0 + 0.75 * 10.0
    assert abs(new_filt.ewma_max_abs_adv - expected_ewma) < 1e-8
    assert abs(eta - 0.01 * expected_ewma) < 1e-8
    # eta = 0.085 → keep |A| >= 0.085
    assert bool(keep[0, 0]) is True
    assert bool(keep[0, 1]) is True
    assert bool(keep[1, 0]) is False
    assert bool(keep[1, 1]) is False  # invalid masked out


def test_adaptive_filter_valid_mask_and_disabled():
    filt = AdaptiveFilterState(
        ewma_max_abs_adv=1.0, beta=0.25, eta_scale=0.5, initialized=True
    )
    adv = torch.tensor([[0.1, 10.0]])
    valid = torch.tensor([[False, True]])
    keep, new_filt, eta = adaptive_advantage_keep_mask(adv, valid, filt)
    assert new_filt.ewma_max_abs_adv == 0.25 * 10.0 + 0.75 * 1.0
    assert bool(keep[0, 0]) is False
    assert bool(keep[0, 1]) is True
    keep2, frozen, eta2 = adaptive_advantage_keep_mask(
        adv, torch.ones_like(adv, dtype=torch.bool), filt, enabled=False
    )
    assert frozen.ewma_max_abs_adv == filt.ewma_max_abs_adv
    assert eta2 == 0.0
    assert bool(keep2.all())


def test_orthogonal_init_zeros_bias():
    lin = nn.Linear(4, 4)
    nn.init.constant_(lin.bias, 3.0)
    orthogonal_init_(lin, gain=1.0)
    assert torch.allclose(lin.bias, torch.zeros_like(lin.bias))


def test_agents_per_minibatch_preserves_sequences():
    assert agents_per_minibatch(128, 64, 8) == 2
    assert agents_per_minibatch(10, 64, 8) == 1
    assert agents_per_minibatch(10_000, 64, 8) == 8


def test_build_ppo_update_resume_and_carry():
    cfg = load_config(SMOKE)
    obs_dim = 8
    cond_dim = cfg.agents.condition_dim
    act_dim = cfg.agents.action_dim
    # Shrink GRU for unit speed while keeping config rollout/epochs.
    hidden = 16
    ego_dim = 8
    el_dim = 4
    n_others = cfg.worlds.max_agents_per_world - 1
    actor = TinyActor(obs_dim, cond_dim, act_dim, hidden)
    critic = TinyCritic(ego_dim, el_dim, cond_dim, n_others)
    learner = build_ppo(
        cfg,
        actor,
        critic,
        total_updates=20,
        target_kl=None,
        orthogonal_init=True,
        device="cpu",
    )
    t = cfg.ppo.rollout_length
    s = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    # Patch actor hidden size vs config: buffer uses config gru — align batch.
    # Use learner-local hidden from actor shapes.
    h = actor.shapes().gru_hidden_dim
    batch = _make_batch(t, s, act_dim, h, cond_dim, ego_dim)
    # Collect-consistent old_logp via actor once.
    with torch.no_grad():
        h0 = actor.initial_hidden(s, "cpu")
        batch.gru_start.copy_(h0)
        for step in range(t):
            out = actor.forward(
                batch.state[step],  # reuse state as tiny sensor obs
                batch.condition[step],
                h0,
                reset_mask=batch.reset_mask[step],
            )
            batch.actions[step].copy_(out.actions)
            batch.old_logp[step].copy_(out.log_prob)
            if out.pre_tanh is not None:
                batch.pre_tanh[step].copy_(out.pre_tanh)
            h0 = out.hidden
            learner.advance_carry(
                h0, done=batch.done[step], reset_mask=batch.reset_mask[step]
            )
    carry_before = learner.carry_hidden.clone()
    n_el = max(n_others, 1)
    prepared = PreparedPPOInputs(
        sensor_obs=batch.state,
        ego_state=batch.state,
        other_agents=torch.zeros(t, s, n_el, el_dim),
        other_mask=torch.ones(t, s, n_el, dtype=torch.bool),
        bootstrap_values=torch.zeros(t, s),
    )
    stats = learner.update(batch, prepared)
    assert 0.0 <= stats.retention <= 1.0
    assert torch.isfinite(torch.tensor(stats.policy_loss))
    assert torch.isfinite(torch.tensor(stats.value_loss))
    assert stats.learning_rate <= cfg.ppo.learning_rate + 1e-12
    assert stats.epochs_completed >= 1
    # Carry must persist across update (not globally zeroed).
    assert learner.carry_hidden is not None
    assert learner.carry_hidden.shape == carry_before.shape
    # Exact resume round-trip
    payload = export_resume_state(learner)
    learner2 = build_ppo(
        cfg,
        TinyActor(obs_dim, cond_dim, act_dim, hidden),
        TinyCritic(ego_dim, el_dim, cond_dim, n_others),
        total_updates=20,
        target_kl=None,
        orthogonal_init=False,
        device="cpu",
    )
    load_resume_state(learner2, payload)
    assert learner2.update_index == learner.update_index
    assert abs(
        learner2.filter_state.ewma_max_abs_adv
        - learner.filter_state.ewma_max_abs_adv
    ) < 1e-8
    for p1, p2 in zip(learner.actor.parameters(), learner2.actor.parameters()):
        assert torch.allclose(p1, p2)
    assert torch.equal(learner2.carry_hidden.cpu(), learner.carry_hidden.cpu())


def test_inactive_slots_excluded_from_filter_valid():
    adv = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    valid = torch.tensor([[True, False], [True, False]])
    filt = initial_filter_state(load_config(SMOKE))
    keep, new_filt, _ = adaptive_advantage_keep_mask(adv, valid, filt)
    assert bool(keep[:, 1].any()) is False
    # First observation initializes from current valid max (inactive 0 ignored).
    assert new_filt.ewma_max_abs_adv == 5.0
    assert new_filt.initialized is True


def test_adaptive_filter_ignores_nonfinite_advantages():
    filt = AdaptiveFilterState(
        ewma_max_abs_adv=float("inf"),
        beta=0.25,
        eta_scale=0.01,
        initialized=False,
    )
    adv = torch.tensor([[float("inf"), 2.0], [float("nan"), 1.0]])
    valid = torch.ones_like(adv, dtype=torch.bool)
    keep, new_filt, eta = adaptive_advantage_keep_mask(adv, valid, filt)
    assert math.isfinite(new_filt.ewma_max_abs_adv)
    assert math.isfinite(eta)
    assert new_filt.ewma_max_abs_adv == 2.0
    assert bool(keep[0, 0]) is False
    assert bool(keep[1, 0]) is False
    assert bool(keep[0, 1])


def test_gae_sanitizes_nonfinite_rewards_values():
    t, s = 3, 2
    rewards = torch.tensor([[1.0, float("inf")], [float("nan"), 0.5], [0.0, 0.0]])
    values = torch.tensor([[0.0, 1.0], [1.0, float("inf")], [0.5, 0.5]])
    done = torch.zeros(t, s, dtype=torch.bool)
    timeout = torch.zeros(t, s, dtype=torch.bool)
    last = torch.tensor([1.0, float("nan")])
    adv, ret = compute_gae(
        rewards, values, done, timeout, last, gamma=0.99, gae_lambda=0.95
    )
    assert torch.isfinite(adv).all()
    assert torch.isfinite(ret).all()


def test_parity_gate_covers_slots_beyond_first_64():
    """Stale old_logp on slots >= 64 must fail-fast (not yield KL bombs)."""
    from dataclasses import replace

    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            rollout_length=8,
            num_epochs=1,
            minibatch_size=512,
            target_kl=0.0,
            amp=False,
            adaptive_filter_enabled=False,
        ),
        ablations=replace(cfg.ablations, adaptive_filter_enabled=False),
    )
    obs_dim = 8
    cond_dim = cfg.agents.condition_dim
    act_dim = 2
    hidden = 16
    ego_dim = 8
    el_dim = 4
    n_others = 1
    t, s = 8, 128
    actor = TinyActor(obs_dim, cond_dim, act_dim, hidden)
    critic = TinyCritic(ego_dim, el_dim, cond_dim, n_others)
    learner = build_ppo(
        cfg, actor, critic, total_updates=10, target_kl=None, device="cpu"
    )
    torch.manual_seed(0)
    with torch.no_grad():
        sensor = torch.randn(t, s, obs_dim)
        cond = torch.randn(t, s, cond_dim)
        reset = torch.zeros(t, s, dtype=torch.bool)
        reset[0] = True
        gru0 = actor.initial_hidden(s, "cpu")
        acts = []
        pres = []
        h = gru0
        for i in range(t):
            out = actor.forward(sensor[i], cond[i], h, reset_mask=reset[i])
            acts.append(out.actions)
            pres.append(out.pre_tanh)
            h = out.hidden
        actions = torch.stack(acts, 0)
        pre = torch.stack(pres, 0)
        logp_bt, _, _ = evaluate_actions_sequence(
            actor,
            sensor.transpose(0, 1).contiguous(),
            cond.transpose(0, 1).contiguous(),
            gru0,
            actions.transpose(0, 1).contiguous(),
            reset_mask=reset.transpose(0, 1).contiguous(),
            pre_tanh=pre.transpose(0, 1).contiguous(),
        )
        old_logp = logp_bt.transpose(0, 1).clone()
        # Corrupt only slots the old first-64 gate would miss.
        old_logp[:, 64:] = old_logp[:, 64:] - 20.0
        batch = CompactRolloutBatch(
            state=torch.randn(t, s, ego_dim),
            next_state=torch.randn(t, s, ego_dim),
            actions=actions,
            rewards=torch.randn(t, s),
            valid=torch.ones(t, s, dtype=torch.bool),
            done=torch.zeros(t, s, dtype=torch.bool),
            timeout=torch.zeros(t, s, dtype=torch.bool),
            reset_mask=reset,
            track_id=torch.zeros(s, dtype=torch.int32),
            condition=cond,
            sensor_noise_seed=torch.zeros(t, s, dtype=torch.int64),
            episode_id=torch.zeros(t, s, dtype=torch.int32),
            episode_step=torch.zeros(t, s, dtype=torch.int32),
            gru_start=gru0,
            old_logp=old_logp,
            obs_digest=torch.zeros(t, s, dtype=torch.int64),
            pre_tanh=pre,
        )
        prepared = PreparedPPOInputs(
            sensor_obs=sensor,
            ego_state=torch.randn(t, s, ego_dim),
            other_agents=torch.randn(t, s, n_others, el_dim),
            other_mask=torch.ones(t, s, n_others),
        )
    with pytest.raises(CollectEvaluateParityError):
        learner.update(batch, prepared)


def test_kept_nonfinite_old_logp_fail_fast():
    from dataclasses import replace

    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            rollout_length=4,
            num_epochs=1,
            minibatch_size=32,
            amp=False,
            adaptive_filter_enabled=False,
        ),
        ablations=replace(cfg.ablations, adaptive_filter_enabled=False),
    )
    actor = TinyActor(8, cfg.agents.condition_dim, 2, 8)
    critic = TinyCritic(8, 4, cfg.agents.condition_dim, 1)
    learner = build_ppo(cfg, actor, critic, total_updates=4, device="cpu")
    batch = _make_batch(4, 4, 2, 8, cfg.agents.condition_dim, 8)
    with torch.no_grad():
        h0 = actor.initial_hidden(4, "cpu")
        batch.gru_start.copy_(h0)
        for step in range(4):
            out = actor.forward(
                batch.state[step],
                batch.condition[step],
                h0,
                reset_mask=batch.reset_mask[step],
            )
            batch.actions[step].copy_(out.actions)
            batch.old_logp[step].copy_(out.log_prob)
            batch.pre_tanh[step].copy_(out.pre_tanh)
            h0 = out.hidden
    batch.old_logp[0, 0] = float("nan")
    prepared = PreparedPPOInputs(
        sensor_obs=batch.state,
        ego_state=batch.state,
        other_agents=torch.zeros(4, 4, 1, 4),
        other_mask=torch.ones(4, 4, 1, dtype=torch.bool),
        bootstrap_values=torch.zeros(4, 4),
    )
    with pytest.raises(CollectEvaluateParityError):
        learner.update(batch, prepared)


def test_cosine_lr_decreases():
    cfg = load_config(SMOKE)
    actor = TinyActor(4, cfg.agents.condition_dim, 2, 8)
    critic = TinyCritic(4, 2, cfg.agents.condition_dim, 1)
    learner = build_ppo(
        cfg, actor, critic, total_updates=5, target_kl=None, device="cpu"
    )
    lrs = [learner.optimizer.param_groups[0]["lr"]]
    # Pair dummy optimizer steps with scheduler steps (PyTorch ordering).
    for _ in range(4):
        learner.optimizer.step()
        learner.scheduler.step()
        lrs.append(learner.optimizer.param_groups[0]["lr"])
    assert lrs[-1] < lrs[0]

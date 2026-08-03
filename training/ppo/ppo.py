from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.optim import Adam


class ValueCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        sizes = [int(obs_dim), *[int(size) for size in hidden_sizes], 1]
        layers = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.obs_dim = int(obs_dim)

    def forward(self, obs):
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"critic obs shape={tuple(obs.shape)}; expected last dim {self.obs_dim}"
            )
        return self.net(obs).squeeze(-1)


@dataclass
class Rollout:
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    pre_tanh_actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    reset_masks: torch.Tensor
    old_logp: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rollout_start_hidden: torch.Tensor
    optimized: bool = False


def compute_gae(
    rewards,
    values,
    dones,
    bootstrap_value,
    *,
    gamma=0.99,
    gae_lambda=0.95,
):
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have identical (T, N) shapes")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError(
            f"bootstrap_value shape={tuple(bootstrap_value.shape)}; "
            f"expected {tuple(rewards.shape[1:])}"
        )
    advantages = torch.zeros_like(rewards)
    next_value = bootstrap_value
    next_advantage = torch.zeros_like(bootstrap_value)
    for step in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[step].to(values.dtype)
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[step] = next_advantage
        next_value = values[step]
    return advantages, advantages + values


def advantage_filter_keep_mask(
    advantages,
    *,
    enabled,
    discard_fraction,
):
    abs_adv = advantages.abs().reshape(-1)
    n = abs_adv.numel()
    if not enabled:
        keep = torch.ones_like(advantages, dtype=torch.bool)
        threshold = float(abs_adv.min()) if n > 0 else 0.0
        return keep, threshold
    num_drop = int(n * discard_fraction)
    if num_drop <= 0:
        keep = torch.ones_like(advantages, dtype=torch.bool)
        threshold = float(abs_adv.min()) if n > 0 else 0.0
        return keep, threshold
    if num_drop >= n:
        keep = torch.zeros_like(advantages, dtype=torch.bool)
        return keep, float(abs_adv.max())
    sorted_idx = torch.argsort(abs_adv, stable=True)
    drop = torch.zeros(n, dtype=torch.bool, device=abs_adv.device)
    drop[sorted_idx[:num_drop]] = True
    keep = (~drop).reshape(advantages.shape)
    threshold = float(abs_adv[drop].max())
    return keep, threshold


def pure_timeout_mask(termination):
    timed_out = termination["time_out"].bool()
    if not bool(timed_out.any()):
        return timed_out
    mask = timed_out
    for key in ("out_of_bounds", "not_moving", "invalid_state", "collision"):
        if key in termination:
            mask = mask & ~termination[key].bool()
    return mask


def apply_timeout_bootstrap_rewards(
    reward,
    timed_out,
    timeout_critic_obs,
    *,
    value_critic,
    critic_normalizer,
    gamma,
    device,
):
    if timed_out is None or not bool(timed_out.any()):
        return reward
    if timeout_critic_obs is None:
        raise ValueError("timeout_critic_obs is required when timed_out is set")
    timeout_critic_obs = timeout_critic_obs.to(device=device, dtype=torch.float32)
    if timeout_critic_obs.ndim != 2 or timeout_critic_obs.shape[0] != reward.shape[0]:
        raise ValueError(
            f"timeout_critic_obs shape={tuple(timeout_critic_obs.shape)}; "
            f"expected ({reward.shape[0]}, critic_dim)"
        )
    with torch.no_grad():
        normalized = timeout_critic_obs
        if critic_normalizer is not None:
            normalized = critic_normalizer.normalize(timeout_critic_obs)
        terminal_values = value_critic(normalized)
        augmented = reward.clone()
        augmented[timed_out] = reward[timed_out] + gamma * terminal_values[timed_out]
    return augmented


class PPOTrainer:
    def __init__(
        self,
        actor,
        critic_obs_dim,
        actor_normalizer=None,
        critic_normalizer=None,
        device="cpu",
        *,
        value_hidden_sizes=(256, 256),
        rollout_steps=128,
        num_epochs=4,
        env_minibatch_size=32,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip=0.2,
        actor_lr=3e-5,
        value_lr=1e-4,
        max_grad_norm=0.5,
        entropy_coef=0.01,
        action_clip=1.0,
        compile=False,
        compile_mode="default",
        advantage_filter_enabled=True,
        advantage_filter_discard_fraction=0.05,
    ):
        if not hasattr(actor, "step") or not hasattr(
            actor, "evaluate_actions_sequence"
        ):
            raise TypeError("PPOTrainer requires a recurrent actor")
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        if isinstance(critic_obs_dim, nn.Module):
            self.value_critic = critic_obs_dim.to(self.device)
            critic_obs_dim = self.value_critic.obs_dim
        else:
            self.value_critic = ValueCritic(
                critic_obs_dim, value_hidden_sizes
            ).to(self.device)
        self.actor_normalizer = actor_normalizer
        self.critic_normalizer = critic_normalizer
        self.actor_obs_dim = int(actor.obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.act_dim = int(actor.act_dim)
        self.hidden_dim = int(actor.gru_hidden_dim)
        self.rollout_steps = int(rollout_steps)
        self.num_epochs = int(num_epochs)
        self.env_minibatch_size = int(env_minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_clip = float(value_clip)
        self.max_grad_norm = float(max_grad_norm)
        self.entropy_coef = float(entropy_coef)
        self.action_clip = float(action_clip)
        self.compile = bool(compile)
        self.compile_mode = str(compile_mode)
        self.advantage_filter_enabled = bool(advantage_filter_enabled)
        self.advantage_filter_discard_fraction = float(
            advantage_filter_discard_fraction
        )
        if (
            self.rollout_steps <= 0
            or self.num_epochs <= 0
            or self.env_minibatch_size <= 0
        ):
            raise ValueError("rollout_steps, num_epochs, and env_minibatch_size must be > 0")
        actor_limit = float(actor.act_limit)
        if not math.isclose(
            self.action_clip, actor_limit, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                f"action_clip={self.action_clip} must equal "
                f"actor.act_limit={actor_limit}"
            )

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.value_optimizer = Adam(self.value_critic.parameters(), lr=value_lr)
        self._evaluate_actions_sequence = self.actor.evaluate_actions_sequence
        self._update_value_forward = self.value_critic
        if self.compile:
            self._evaluate_actions_sequence = torch.compile(
                self._evaluate_actions_sequence, mode=self.compile_mode
            )
            self._update_value_forward = torch.compile(
                self._update_value_forward, mode=self.compile_mode
            )
        self._live_hidden = None
        self._reset_mask = None
        self._rollout = None
        self._num_envs = None
        self._step = 0
        self._initialized = False
        self._awaiting_observe = False
        self.last_rollout = None

    def _normalize(self, normalizer, obs):
        obs = obs.to(device=self.device, dtype=torch.float32)
        return normalizer.normalize(obs) if normalizer is not None else obs

    @property
    def rollout_position(self):
        return self._step

    def _validate_observations(self, actor_obs, critic_obs):
        if actor_obs.ndim != 2 or actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(
                f"actor_obs shape={tuple(actor_obs.shape)}; "
                f"expected (N, {self.actor_obs_dim})"
            )
        expected_critic = (actor_obs.shape[0], self.critic_obs_dim)
        if tuple(critic_obs.shape) != expected_critic:
            raise ValueError(
                f"critic_obs shape={tuple(critic_obs.shape)}; "
                f"expected {expected_critic}"
            )

    @torch.no_grad()
    def initialize(self, actor_obs, critic_obs):
        if self._initialized:
            raise RuntimeError("PPOTrainer is already initialized")
        actor_obs = actor_obs.to(device=self.device, dtype=torch.float32)
        critic_obs = critic_obs.to(device=self.device, dtype=torch.float32)
        self._validate_observations(actor_obs, critic_obs)
        self._num_envs = int(actor_obs.shape[0])
        time_env = (self.rollout_steps, self._num_envs)
        self._live_hidden = self.actor.initial_hidden(
            self._num_envs, device=self.device, dtype=torch.float32
        )
        self._reset_mask = torch.ones(
            self._num_envs, device=self.device, dtype=torch.bool
        )
        self._rollout = Rollout(
            actor_obs=torch.empty(
                *time_env,
                self.actor_obs_dim,
                device=self.device,
                dtype=torch.float32,
            ),
            critic_obs=torch.empty(
                *time_env,
                self.critic_obs_dim,
                device=self.device,
                dtype=torch.float32,
            ),
            actions=torch.empty(
                *time_env, self.act_dim, device=self.device, dtype=torch.float32
            ),
            pre_tanh_actions=torch.empty(
                *time_env, self.act_dim, device=self.device, dtype=torch.float32
            ),
            rewards=torch.empty(*time_env, device=self.device, dtype=torch.float32),
            dones=torch.empty(*time_env, device=self.device, dtype=torch.bool),
            reset_masks=torch.empty(*time_env, device=self.device, dtype=torch.bool),
            old_logp=torch.empty(*time_env, device=self.device, dtype=torch.float32),
            old_values=torch.empty(*time_env, device=self.device, dtype=torch.float32),
            returns=torch.empty(*time_env, device=self.device, dtype=torch.float32),
            advantages=torch.empty(*time_env, device=self.device, dtype=torch.float32),
            rollout_start_hidden=torch.zeros(
                self._num_envs,
                self.hidden_dim,
                device=self.device,
                dtype=torch.float32,
            ),
        )
        self._initialized = True

    @torch.no_grad()
    def act(self, actor_obs, critic_obs):
        if not self._initialized:
            raise RuntimeError("initialize must be called before act")
        if self._awaiting_observe:
            raise RuntimeError("observe must be called before the next act")
        actor_obs = actor_obs.to(device=self.device, dtype=torch.float32)
        critic_obs = critic_obs.to(device=self.device, dtype=torch.float32)
        self._validate_observations(actor_obs, critic_obs)
        if actor_obs.shape[0] != self._num_envs:
            raise ValueError(
                f"observation batch has {actor_obs.shape[0]} envs; "
                f"expected {self._num_envs}"
            )
        if self._step == 0:
            self._rollout.optimized = False
            self._rollout.rollout_start_hidden.copy_(self._live_hidden)
        self._rollout.actor_obs[self._step].copy_(actor_obs)
        self._rollout.critic_obs[self._step].copy_(critic_obs)
        self._rollout.reset_masks[self._step].copy_(self._reset_mask)
        normalized_actor = self._normalize(self.actor_normalizer, actor_obs)
        normalized_critic = self._normalize(self.critic_normalizer, critic_obs)
        action, logp, next_hidden, pre_tanh = self.actor.step(
            normalized_actor,
            self._live_hidden,
            reset_mask=self._reset_mask,
            deterministic=False,
            with_logprob=True,
            return_pre_tanh=True,
        )
        self._rollout.actions[self._step].copy_(action)
        self._rollout.pre_tanh_actions[self._step].copy_(pre_tanh)
        self._rollout.old_logp[self._step].copy_(logp)
        self._rollout.old_values[self._step].copy_(
            self.value_critic(normalized_critic)
        )
        self._live_hidden.copy_(next_hidden)
        self._awaiting_observe = True
        return action

    def observe(
        self,
        next_actor_obs,
        next_critic_obs,
        reward,
        done,
        *,
        timed_out=None,
        timeout_critic_obs=None,
    ):
        if not self._initialized:
            raise RuntimeError("initialize must be called before observe")
        if not self._awaiting_observe:
            raise RuntimeError("act must be called before observe")
        next_actor_obs = next_actor_obs.to(device=self.device, dtype=torch.float32)
        next_critic_obs = next_critic_obs.to(
            device=self.device, dtype=torch.float32
        )
        self._validate_observations(next_actor_obs, next_critic_obs)
        if next_actor_obs.shape[0] != self._num_envs:
            raise ValueError(
                f"observation batch has {next_actor_obs.shape[0]} envs; "
                f"expected {self._num_envs}"
            )
        reward = reward.to(device=self.device, dtype=torch.float32)
        done = done.to(device=self.device, dtype=torch.bool)
        if reward.shape != (self._num_envs,) or done.shape != (self._num_envs,):
            raise ValueError("reward and done must have shape (N,)")
        step_reward = apply_timeout_bootstrap_rewards(
            reward,
            timed_out,
            timeout_critic_obs,
            value_critic=self.value_critic,
            critic_normalizer=self.critic_normalizer,
            gamma=self.gamma,
            device=self.device,
        )
        self._rollout.rewards[self._step].copy_(step_reward)
        self._rollout.dones[self._step].copy_(done)
        self._live_hidden.masked_fill_(done.unsqueeze(-1), 0.0)
        self._reset_mask.copy_(done)
        self._step += 1
        self._awaiting_observe = False
        if self._step < self.rollout_steps:
            return None

        with torch.no_grad():
            normalized_next_critic = self._normalize(
                self.critic_normalizer, next_critic_obs
            )
            bootstrap_value = self.value_critic(normalized_next_critic)
            bootstrap_value.masked_fill_(done, 0.0)
            advantages, returns = compute_gae(
                self._rollout.rewards,
                self._rollout.old_values,
                self._rollout.dones,
                bootstrap_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            self._rollout.advantages.copy_(advantages)
            self._rollout.returns.copy_(returns)
        metrics = self.update(self._rollout)
        self.last_rollout = self._rollout
        self._step = 0
        self._live_hidden.zero_()
        self._reset_mask.fill_(True)
        return metrics

    def update(self, rollout):
        if rollout.optimized:
            raise ValueError("rollout has already been optimized")
        steps, num_envs = rollout.rewards.shape
        if steps != self.rollout_steps:
            raise ValueError(
                f"rollout has {steps} steps; expected fixed length {self.rollout_steps}"
            )
        raw_advantages = rollout.advantages
        keep_mask, eta = advantage_filter_keep_mask(
            raw_advantages,
            enabled=self.advantage_filter_enabled,
            discard_fraction=self.advantage_filter_discard_fraction,
        )
        retained = int(keep_mask.sum().item())
        total = keep_mask.numel()
        discard_rate = 1.0 - (retained / total)
        if self.advantage_filter_enabled and retained == 0:
            rollout.optimized = True
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "negative_log_prob": 0.0,
                "approx_kl": 0.0,
                "policy_entropy": 0.0,
                "policy_clip_fraction": 0.0,
                "value_clip_fraction": 0.0,
                "actor_grad_norm": 0.0,
                "value_grad_norm": 0.0,
                "actor_grad_clip_fraction": 0.0,
                "value_grad_clip_fraction": 0.0,
                "epochs": 0,
                "minibatches": 0,
                "actor_updates": 0,
                "value_updates": 0,
                "advantage_filter_discard_rate": 1.0,
                "advantage_filter_retained": 0,
                "advantage_filter_eta": float(eta),
            }
        if self.advantage_filter_enabled:
            kept_adv = raw_advantages[keep_mask]
            adv_mean = kept_adv.mean()
            adv_std = kept_adv.std(unbiased=False)
        else:
            adv_mean = raw_advantages.mean()
            adv_std = raw_advantages.std(unbiased=False)
        advantages = (raw_advantages - adv_mean) / (adv_std + 1e-8)
        keep_weights = keep_mask.to(dtype=advantages.dtype)
        metric_sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "negative_log_prob": 0.0,
            "approx_kl": 0.0,
            "policy_entropy": 0.0,
            "policy_clip_fraction": 0.0,
            "value_clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "value_grad_norm": 0.0,
            "actor_grad_clip_fraction": 0.0,
            "value_grad_clip_fraction": 0.0,
        }
        updates = 0
        actor_updates = 0
        value_updates = 0
        epochs_completed = 0

        for epoch in range(self.num_epochs):
            permutation = torch.randperm(num_envs, device=self.device)
            for start in range(0, num_envs, self.env_minibatch_size):
                if self.compile and self.compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                env_index = permutation[start : start + self.env_minibatch_size]
                actor_raw = rollout.actor_obs[:, env_index]
                critic_raw = rollout.critic_obs[:, env_index]
                actor_seq = self._normalize(
                    self.actor_normalizer, actor_raw
                ).transpose(0, 1)
                action_seq = rollout.actions[:, env_index].transpose(0, 1)
                pre_tanh_seq = rollout.pre_tanh_actions[:, env_index].transpose(
                    0, 1
                )
                reset_seq = rollout.reset_masks[:, env_index].transpose(0, 1)
                hidden = rollout.rollout_start_hidden[env_index]
                new_logp, _, new_entropy = self._evaluate_actions_sequence(
                    actor_seq,
                    hidden,
                    action_seq,
                    reset_mask=reset_seq,
                    pre_tanh_actions=pre_tanh_seq,
                )
                new_logp = new_logp.transpose(0, 1)
                new_entropy = new_entropy.transpose(0, 1)
                old_logp = rollout.old_logp[:, env_index]
                mb_advantages = advantages[:, env_index]
                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                unclipped = ratio * mb_advantages
                clipped = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * mb_advantages
                mb_keep = keep_weights[:, env_index]
                keep_denom = mb_keep.sum().clamp(min=1.0)
                policy_loss_per = -torch.minimum(unclipped, clipped)
                policy_loss = (policy_loss_per * mb_keep).sum() / keep_denom
                policy_entropy = (new_entropy * mb_keep).sum() / keep_denom
                policy_loss = policy_loss - self.entropy_coef * policy_entropy
                approx_kl = (((ratio - 1.0) - log_ratio) * mb_keep).sum() / keep_denom

                critic_obs = self._normalize(self.critic_normalizer, critic_raw)
                new_values = self._update_value_forward(critic_obs)
                old_values = rollout.old_values[:, env_index]
                mb_returns = rollout.returns[:, env_index]
                clipped_values = old_values + (new_values - old_values).clamp(
                    -self.value_clip, self.value_clip
                )
                value_loss_unclipped = (new_values - mb_returns).square()
                value_loss_clipped = (clipped_values - mb_returns).square()
                value_loss_per = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                )
                value_loss = (value_loss_per * mb_keep).sum() / keep_denom

                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad_norm = nn.utils.clip_grad_norm_(
                    self.value_critic.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()
                value_updates += 1

                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                actor_updates += 1
                actor_grad = float(actor_grad_norm)
                metric_sums["actor_grad_norm"] += actor_grad
                metric_sums["actor_grad_clip_fraction"] += float(
                    actor_grad > self.max_grad_norm
                )

                metric_sums["policy_loss"] += float(policy_loss.detach())
                metric_sums["value_loss"] += float(value_loss.detach())
                metric_sums["negative_log_prob"] += float(
                    ((-new_logp) * mb_keep).sum().div(keep_denom).detach()
                )
                metric_sums["approx_kl"] += float(approx_kl.detach())
                metric_sums["policy_entropy"] += float(policy_entropy.detach())
                metric_sums["policy_clip_fraction"] += float(
                    (
                        ((ratio - 1.0).abs() > self.clip_ratio).float() * mb_keep
                    )
                    .sum()
                    .div(keep_denom)
                    .detach()
                )
                metric_sums["value_clip_fraction"] += float(
                    (
                        ((new_values - old_values).abs() > self.value_clip).float()
                        * mb_keep
                    )
                    .sum()
                    .div(keep_denom)
                    .detach()
                )
                value_grad = float(value_grad_norm)
                metric_sums["value_grad_norm"] += value_grad
                metric_sums["value_grad_clip_fraction"] += float(
                    value_grad > self.max_grad_norm
                )
                updates += 1
            epochs_completed = epoch + 1

        rollout.optimized = True
        if self.actor_normalizer is not None:
            self.actor_normalizer.update(
                rollout.actor_obs.reshape(-1, self.actor_obs_dim)
            )
        if self.critic_normalizer is not None:
            self.critic_normalizer.update(
                rollout.critic_obs.reshape(-1, self.critic_obs_dim)
            )
        metrics = {key: value / updates for key, value in metric_sums.items()}
        metrics.update(
            {
                "epochs": epochs_completed,
                "minibatches": updates,
                "actor_updates": actor_updates,
                "value_updates": value_updates,
                "advantage_filter_discard_rate": discard_rate,
                "advantage_filter_retained": retained,
                "advantage_filter_eta": float(eta),
            }
        )
        return metrics

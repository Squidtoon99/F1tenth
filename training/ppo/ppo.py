from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from f1tenth_policy.layout import executed_plant_action
from f1tenth_policy.normalizer import ObsNormalizer


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


def annealed_entropy_coef(start, end, horizon_transitions, transitions):
    start = float(start)
    end = float(end)
    horizon = int(horizon_transitions)
    if horizon <= 0:
        return start
    t = min(max(int(transitions), 0), horizon)
    return start + (end - start) * (t / horizon)


_GAUSSIAN_ENTROPY_OFFSET = 0.5 * (1.0 + math.log(2.0 * math.pi))


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


def caps_temporal_loss(mean_actions, reset_mask):
    if mean_actions.shape[1] < 2:
        return mean_actions.new_zeros(())
    delta = mean_actions[:, 1:] - mean_actions[:, :-1]
    valid = (~reset_mask[:, 1:]).to(dtype=delta.dtype)
    return (delta.square().sum(dim=-1) * valid).sum() / valid.sum().clamp(min=1.0)


def caps_spatial_loss(mean_actions, perturbed_mean):
    delta = mean_actions - perturbed_mean
    return delta.square().sum(dim=-1).mean()


def grad_caps_temporal_loss(mean_actions, reset_mask, eps=1e-6):
    if mean_actions.shape[1] < 3:
        return mean_actions.new_zeros(())
    delta = mean_actions[:, 1:] - mean_actions[:, :-1]
    second = (delta[:, 1:] - delta[:, :-1]).norm(dim=-1)
    displacement = (mean_actions[:, 2:] - mean_actions[:, :-2]).norm(dim=-1)
    scale = torch.tanh(1.0 / (displacement + eps))
    valid = ((~reset_mask[:, 1:-1]) & (~reset_mask[:, 2:])).to(dtype=second.dtype)
    return (second * scale * valid).sum() / valid.sum().clamp(min=1.0)


def _channel_weight_tensor(actions, channel_weights):
    weights = torch.as_tensor(
        channel_weights, device=actions.device, dtype=actions.dtype
    ).reshape(-1)
    packed = actions.new_zeros(actions.shape[-1])
    n = min(weights.numel(), packed.numel())
    if n:
        packed[:n] = weights[:n]
    return packed


def action_reversal_penalty(
    actions,
    reset_mask,
    *,
    channel_weights,
    deadband=0.0,
    lag2=False,
):
    zero = actions.new_zeros(())
    rates = actions.new_zeros(actions.shape[-1])
    weights = _channel_weight_tensor(actions, channel_weights)
    deadband = float(deadband)
    if actions.shape[1] < 3:
        return zero, rates
    delta = actions[:, 1:] - actions[:, :-1]
    prod = delta[:, 1:] * delta[:, :-1]
    rev = F.relu(-prod - deadband)
    valid = ((~reset_mask[:, 1:-1]) & (~reset_mask[:, 2:])).to(dtype=actions.dtype)
    denom = valid.sum().clamp(min=1.0)
    penalty = (rev * weights * valid.unsqueeze(-1)).sum() / denom
    flipped = (prod < -deadband).to(dtype=actions.dtype) * valid.unsqueeze(-1)
    rates = flipped.sum(dim=(0, 1)) / denom
    if lag2 and actions.shape[1] >= 4:
        prod2 = delta[:, 2:] * delta[:, :-2]
        rev2 = F.relu(-prod2 - deadband)
        valid2 = (
            (~reset_mask[:, 1:-2])
            & (~reset_mask[:, 2:-1])
            & (~reset_mask[:, 3:])
        ).to(dtype=actions.dtype)
        denom2 = valid2.sum().clamp(min=1.0)
        penalty = penalty + (rev2 * weights * valid2.unsqueeze(-1)).sum() / denom2
    return penalty, rates


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
    value_fn,
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
        terminal_values = value_fn(normalized)
        augmented = reward.clone()
        augmented[timed_out] = reward[timed_out] + gamma * terminal_values[timed_out]
    return augmented


def explained_variance(returns, values):
    var_r = returns.var(unbiased=False)
    return 1.0 - (returns - values).var(unbiased=False) / var_r.clamp_min(1e-8)


def value_prediction_loss(
    new_values,
    old_values,
    returns,
    *,
    value_clip,
    huber_delta,
):
    if huber_delta > 0.0:
        unclipped = F.huber_loss(
            new_values, returns, reduction="none", delta=huber_delta
        )
    else:
        unclipped = 0.5 * (new_values - returns).square()
    if value_clip > 0.0:
        clipped_values = old_values + (new_values - old_values).clamp(
            -value_clip, value_clip
        )
        if huber_delta > 0.0:
            clipped = F.huber_loss(
                clipped_values, returns, reduction="none", delta=huber_delta
            )
        else:
            clipped = 0.5 * (clipped_values - returns).square()
        per_row = torch.maximum(unclipped, clipped)
        active = clipped > unclipped
    else:
        per_row = unclipped
        active = torch.zeros_like(unclipped, dtype=torch.bool)
    return per_row, active


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
        value_max_grad_norm=None,
        value_normalization=False,
        value_huber_delta=0.0,
        entropy_coef=0.01,
        entropy_from_squashed=False,
        action_clip=1.0,
        compile=False,
        compile_mode="default",
        advantage_filter_enabled=True,
        advantage_filter_discard_fraction=0.05,
        lidar_aug_enabled=False,
        lidar_aug_max_shift_beams=0,
        lr_schedule="constant",
        lr_final=0.0,
        total_transitions=0,
        desired_kl=0.01,
        caps_coef=0.0,
        caps_spatial_coef=0.0,
        caps_spatial_std=0.05,
        grad_caps_coef=0.0,
        reversal_coef_throttle=0.0,
        reversal_coef_steer=0.0,
        reversal_deadband=0.0,
        reversal_lag2=False,
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
        self.value_clip = 0.0 if value_clip is None else float(value_clip)
        if self.value_clip < 0.0:
            raise ValueError(f"value_clip must be >= 0 or None, got {value_clip!r}")
        self.max_grad_norm = float(max_grad_norm)
        if value_max_grad_norm is None:
            self.value_max_grad_norm = self.max_grad_norm
        else:
            self.value_max_grad_norm = float(value_max_grad_norm)
        self.value_normalization = bool(value_normalization)
        self.value_huber_delta = float(value_huber_delta)
        if self.value_huber_delta < 0.0:
            raise ValueError(
                f"value_huber_delta must be >= 0, got {value_huber_delta!r}"
            )
        self.entropy_coef = float(entropy_coef)
        self.entropy_from_squashed = bool(entropy_from_squashed)
        self.action_clip = float(action_clip)
        self.compile = bool(compile)
        self.compile_mode = str(compile_mode)
        self.advantage_filter_enabled = bool(advantage_filter_enabled)
        self.advantage_filter_discard_fraction = float(
            advantage_filter_discard_fraction
        )
        self.lidar_aug_enabled = bool(lidar_aug_enabled)
        self.lidar_aug_max_shift_beams = int(lidar_aug_max_shift_beams)
        self.caps_coef = float(caps_coef)
        self.caps_spatial_coef = float(caps_spatial_coef)
        self.caps_spatial_std = float(caps_spatial_std)
        self.grad_caps_coef = float(grad_caps_coef)
        for name, value in (
            ("caps_coef", self.caps_coef),
            ("caps_spatial_coef", self.caps_spatial_coef),
            ("caps_spatial_std", self.caps_spatial_std),
            ("grad_caps_coef", self.grad_caps_coef),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be a finite number >= 0, got {value!r}"
                )
        self.reversal_coef_throttle = float(reversal_coef_throttle)
        self.reversal_coef_steer = float(reversal_coef_steer)
        self.reversal_deadband = float(reversal_deadband)
        self.reversal_lag2 = bool(reversal_lag2)
        for name, value in (
            ("reversal_coef_throttle", self.reversal_coef_throttle),
            ("reversal_coef_steer", self.reversal_coef_steer),
            ("reversal_deadband", self.reversal_deadband),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be a finite number >= 0, got {value!r}"
                )
        self._need_action_mean = (
            self.caps_coef > 0.0
            or self.caps_spatial_coef > 0.0
            or self.grad_caps_coef > 0.0
            or self.reversal_coef_throttle > 0.0
            or self.reversal_coef_steer > 0.0
        )
        if self.lidar_aug_max_shift_beams < 0:
            raise ValueError(
                "lidar_aug_max_shift_beams must be a non-negative int, got "
                f"{lidar_aug_max_shift_beams!r}"
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
        self.value_normalizer = (
            ObsNormalizer(1, self.device) if self.value_normalization else None
        )
        self._evaluate_actions_sequence = self.actor.evaluate_actions_sequence
        self._update_value_forward = self.value_critic
        if self.compile and hasattr(self.actor, "use_segment_gru"):
            self.actor.use_segment_gru = False
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
        self.actor_lr0 = float(actor_lr)
        self.value_lr0 = float(value_lr)
        self.lr_schedule = str(lr_schedule)
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {sorted(LR_SCHEDULES)}, got "
                f"{lr_schedule!r}"
            )
        self.lr_final = float(lr_final)
        if not math.isfinite(self.lr_final) or self.lr_final < 0.0:
            raise ValueError(
                f"lr_final must be a finite number >= 0, got {lr_final!r}"
            )
        self.total_transitions = int(total_transitions)
        if self.total_transitions < 0:
            raise ValueError(
                "total_transitions must be a non-negative int, got "
                f"{total_transitions!r}"
            )
        if (
            self.lr_schedule in ("linear", "cosine")
            and self.total_transitions <= 0
        ):
            raise ValueError(
                f"lr_schedule={self.lr_schedule!r} requires total_transitions > 0"
            )
        self.desired_kl = float(desired_kl)
        if not math.isfinite(self.desired_kl) or self.desired_kl <= 0.0:
            raise ValueError(
                f"desired_kl must be a positive finite number, got {desired_kl!r}"
            )
        self._schedule_transitions = 0

    def current_actor_lr(self):
        return _optimizer_lr(self.actor_optimizer)

    def current_value_lr(self):
        return _optimizer_lr(self.value_optimizer)

    def set_schedule_transitions(self, env_transitions):
        self._schedule_transitions = int(env_transitions)
        if self.lr_schedule in ("linear", "cosine"):
            self.apply_progress_schedule()

    def apply_progress_schedule(self):
        if self.lr_schedule not in ("linear", "cosine"):
            return
        progress = (
            self._schedule_transitions / self.total_transitions
            if self.total_transitions > 0
            else 0.0
        )
        _set_optimizer_lr(
            self.actor_optimizer,
            scheduled_learning_rate(
                progress, self.actor_lr0, self.lr_final, self.lr_schedule
            ),
        )
        _set_optimizer_lr(
            self.value_optimizer,
            scheduled_learning_rate(
                progress, self.value_lr0, self.lr_final, self.lr_schedule
            ),
        )

    def adapt_kl_learning_rates(self, approx_kl):
        _set_optimizer_lr(
            self.actor_optimizer,
            adapt_kl_learning_rate(
                self.current_actor_lr(), approx_kl, self.desired_kl
            ),
        )
        _set_optimizer_lr(
            self.value_optimizer,
            adapt_kl_learning_rate(
                self.current_value_lr(), approx_kl, self.desired_kl
            ),
        )

    def lr_schedule_state(self):
        return {
            "actor_lr": self.current_actor_lr(),
            "value_lr": self.current_value_lr(),
            "schedule_transitions": int(self._schedule_transitions),
        }

    def load_lr_schedule_state(self, state):
        if not state:
            return
        actor_lr = state.get("actor_lr")
        value_lr = state.get("value_lr")
        if actor_lr is not None:
            _set_optimizer_lr(self.actor_optimizer, actor_lr)
        if value_lr is not None:
            _set_optimizer_lr(self.value_optimizer, value_lr)
        if "schedule_transitions" in state:
            self._schedule_transitions = int(state["schedule_transitions"])

    def _normalize(self, normalizer, obs):
        obs = obs.to(device=self.device, dtype=torch.float32)
        return normalizer.normalize(obs) if normalizer is not None else obs

    def _value_mu_sigma(self):
        norm = self.value_normalizer
        mu = norm.mean.reshape(())
        sigma = torch.sqrt(norm.var.reshape(()) + norm.eps)
        return mu, sigma

    def _denormalize_value(self, normalized_pred):
        if self.value_normalizer is None:
            return normalized_pred
        mu, sigma = self._value_mu_sigma()
        return normalized_pred * sigma + mu

    def _value(self, normalized_obs):
        return self._denormalize_value(self.value_critic(normalized_obs))

    def _popart_rescale(self, mu_old, sigma_old, mu_new, sigma_new):
        last = self.value_critic.net[-1]
        with torch.no_grad():
            last.weight.mul_((sigma_old / sigma_new).reshape(()))
            last.bias.copy_(
                (sigma_old * last.bias + mu_old - mu_new) / sigma_new
            )

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
            self._value(normalized_critic)
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
            value_fn=self._value,
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
            bootstrap_value = self._value(normalized_next_critic)
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
        if self.lr_schedule == "adaptive_kl":
            self.adapt_kl_learning_rates(metrics["approx_kl"])
        metrics["actor_lr"] = self.current_actor_lr()
        metrics["value_lr"] = self.current_value_lr()
        self.last_rollout = self._rollout
        self._step = 0
        self._live_hidden.zero_()
        self._reset_mask.fill_(True)
        return metrics

    def update(self, rollout):
        if self.lr_schedule in ("linear", "cosine"):
            self.apply_progress_schedule()
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
                "policy_gaussian_entropy": 0.0,
                "policy_log_std": 0.0,
                "entropy_coef": self.entropy_coef,
                "policy_clip_fraction": 0.0,
                "value_clip_fraction": 0.0,
                "actor_grad_norm": 0.0,
                "value_grad_norm": 0.0,
                "actor_grad_clip_fraction": 0.0,
                "value_grad_clip_fraction": 0.0,
                "caps_loss": 0.0,
                "caps_spatial_loss": 0.0,
                "grad_caps_loss": 0.0,
                "reversal_loss": 0.0,
                "reversal_rate_throttle": 0.0,
                "reversal_rate_steer": 0.0,
                "epochs": 0,
                "minibatches": 0,
                "actor_updates": 0,
                "value_updates": 0,
                "advantage_filter_discard_rate": 1.0,
                "advantage_filter_retained": 0,
                "advantage_filter_eta": float(eta),
                "value_clip_active_fraction": 0.0,
                "explained_variance": 0.0,
                "explained_variance_post": 0.0,
                "return_mean": 0.0,
                "return_std": 0.0,
                "value_mean": 0.0,
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
        kept_returns = rollout.returns[keep_mask]
        kept_old_values = rollout.old_values[keep_mask]
        with torch.no_grad():
            ev_pre = explained_variance(kept_returns, kept_old_values)
            return_mean = kept_returns.mean()
            return_std = kept_returns.std(unbiased=False)
            value_mean = kept_old_values.mean()
        if self.value_normalizer is not None:
            mu_old, sigma_old = self._value_mu_sigma()
            self.value_normalizer.update(rollout.returns.reshape(-1, 1))
            mu_new, sigma_new = self._value_mu_sigma()
            self._popart_rescale(mu_old, sigma_old, mu_new, sigma_new)
        post_values = torch.empty_like(rollout.old_values)
        metric_sums = {
            key: torch.zeros((), device=self.device, dtype=torch.float32)
            for key in (
                "policy_loss",
                "value_loss",
                "negative_log_prob",
                "approx_kl",
                "policy_entropy",
                "policy_gaussian_entropy",
                "policy_log_std",
                "policy_clip_fraction",
                "value_clip_fraction",
                "actor_grad_norm",
                "value_grad_norm",
                "actor_grad_clip_fraction",
                "value_grad_clip_fraction",
                "caps_loss",
                "caps_spatial_loss",
                "grad_caps_loss",
                "reversal_loss",
                "reversal_rate_throttle",
                "reversal_rate_steer",
            )
        }
        clip_active_sum = 0.0
        clip_frac_sum = 0.0
        value_grad_clip_sum = 0.0
        updates = 0
        actor_updates = 0
        value_updates = 0
        epochs_completed = 0

        if self.lidar_aug_enabled and self.lidar_aug_max_shift_beams > 0:
            from standalone_trainer import augment_actor_lidar_beam_shift
        else:
            augment_actor_lidar_beam_shift = None

        for epoch in range(self.num_epochs):
            permutation = torch.randperm(num_envs, device=self.device)
            for start in range(0, num_envs, self.env_minibatch_size):
                if self.compile and self.compile_mode == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                env_index = permutation[start : start + self.env_minibatch_size]
                actor_raw = rollout.actor_obs[:, env_index]
                if augment_actor_lidar_beam_shift is not None:
                    actor_raw = augment_actor_lidar_beam_shift(
                        actor_raw.transpose(0, 1),
                        max_shift_beams=self.lidar_aug_max_shift_beams,
                    ).transpose(0, 1)
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
                eval_out = self._evaluate_actions_sequence(
                    actor_seq,
                    hidden,
                    action_seq,
                    reset_mask=reset_seq,
                    pre_tanh_actions=pre_tanh_seq,
                    return_mean=self._need_action_mean,
                )
                if self._need_action_mean:
                    new_logp, _, new_entropy, new_mean = eval_out
                else:
                    new_logp, _, new_entropy = eval_out
                    new_mean = None
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
                gaussian_entropy = (new_entropy * mb_keep).sum() / keep_denom
                squashed_entropy = ((-new_logp) * mb_keep).sum() / keep_denom
                policy_entropy = (
                    squashed_entropy
                    if self.entropy_from_squashed
                    else gaussian_entropy
                )
                policy_loss = policy_loss - self.entropy_coef * policy_entropy
                mean_log_std = (
                    gaussian_entropy / self.act_dim - _GAUSSIAN_ENTROPY_OFFSET
                )
                caps_loss = policy_loss.new_zeros(())
                caps_spatial = policy_loss.new_zeros(())
                grad_caps_loss = policy_loss.new_zeros(())
                reversal_loss = policy_loss.new_zeros(())
                reversal_rate = policy_loss.new_zeros((self.act_dim,))
                if new_mean is not None:
                    plant_mean = executed_plant_action(new_mean)
                    caps_loss = caps_temporal_loss(plant_mean, reset_seq)
                    policy_loss = policy_loss + self.caps_coef * caps_loss
                    grad_caps_loss = grad_caps_temporal_loss(plant_mean, reset_seq)
                    policy_loss = policy_loss + self.grad_caps_coef * grad_caps_loss
                    if self.caps_spatial_coef > 0.0:
                        noise = torch.randn_like(actor_seq) * self.caps_spatial_std
                        spat_out = self._evaluate_actions_sequence(
                            actor_seq + noise,
                            hidden,
                            action_seq,
                            reset_mask=reset_seq,
                            pre_tanh_actions=pre_tanh_seq,
                            return_mean=True,
                        )
                        caps_spatial = caps_spatial_loss(
                            plant_mean, executed_plant_action(spat_out[3])
                        )
                        policy_loss = (
                            policy_loss + self.caps_spatial_coef * caps_spatial
                        )
                    reversal_loss, reversal_rate = action_reversal_penalty(
                        plant_mean,
                        reset_seq,
                        channel_weights=(
                            self.reversal_coef_throttle,
                            self.reversal_coef_steer,
                        ),
                        deadband=self.reversal_deadband,
                        lag2=self.reversal_lag2,
                    )
                    policy_loss = policy_loss + reversal_loss
                approx_kl = (((ratio - 1.0) - log_ratio) * mb_keep).sum() / keep_denom

                critic_obs = self._normalize(self.critic_normalizer, critic_raw)
                new_values = self._update_value_forward(critic_obs)
                old_values = rollout.old_values[:, env_index]
                mb_returns = rollout.returns[:, env_index]
                if self.value_normalizer is not None:
                    mu, sigma = self._value_mu_sigma()
                    old_n = (old_values - mu) / sigma
                    ret_n = (mb_returns - mu) / sigma
                else:
                    old_n = old_values
                    ret_n = mb_returns
                value_loss_per, clip_active = value_prediction_loss(
                    new_values,
                    old_n,
                    ret_n,
                    value_clip=self.value_clip,
                    huber_delta=self.value_huber_delta,
                )
                value_loss = (value_loss_per * mb_keep).sum() / keep_denom
                if epoch == self.num_epochs - 1:
                    post_raw = new_values.detach()
                    if self.value_normalizer is not None:
                        post_raw = post_raw * sigma + mu
                    post_values[:, env_index] = post_raw
                if self.value_clip > 0.0:
                    value_clip_hit = (
                        (new_values - old_n).abs() > self.value_clip
                    ).to(dtype=mb_keep.dtype)
                else:
                    value_clip_hit = torch.zeros_like(mb_keep)
                clip_frac_sum += float(
                    (value_clip_hit * mb_keep).sum().div(keep_denom).detach()
                )
                clip_active_sum += float(
                    (clip_active.to(dtype=mb_keep.dtype) * mb_keep)
                    .sum()
                    .div(keep_denom)
                    .detach()
                )

                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad_norm = nn.utils.clip_grad_norm_(
                    self.value_critic.parameters(), self.value_max_grad_norm
                )
                self.value_optimizer.step()
                value_updates += 1
                value_grad_clip_sum += float(
                    float(value_grad_norm) > self.value_max_grad_norm
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                actor_updates += 1
                actor_grad = actor_grad_norm.detach()
                metric_sums["actor_grad_norm"] += actor_grad
                metric_sums["actor_grad_clip_fraction"] += (
                    actor_grad > self.max_grad_norm
                ).to(dtype=torch.float32)

                metric_sums["policy_loss"] += policy_loss.detach()
                metric_sums["value_loss"] += value_loss.detach()
                metric_sums["negative_log_prob"] += (
                    ((-new_logp) * mb_keep).sum().div(keep_denom).detach()
                )
                metric_sums["approx_kl"] += approx_kl.detach()
                metric_sums["policy_entropy"] += policy_entropy.detach()
                metric_sums["policy_gaussian_entropy"] += gaussian_entropy.detach()
                metric_sums["policy_log_std"] += mean_log_std.detach()
                metric_sums["caps_loss"] += caps_loss.detach()
                metric_sums["caps_spatial_loss"] += caps_spatial.detach()
                metric_sums["grad_caps_loss"] += grad_caps_loss.detach()
                metric_sums["reversal_loss"] += reversal_loss.detach()
                metric_sums["reversal_rate_throttle"] += reversal_rate[0].detach()
                if reversal_rate.numel() > 1:
                    metric_sums["reversal_rate_steer"] += reversal_rate[1].detach()
                metric_sums["policy_clip_fraction"] += (
                    (
                        ((ratio - 1.0).abs() > self.clip_ratio).to(
                            dtype=torch.float32
                        )
                        * mb_keep
                    )
                    .sum()
                    .div(keep_denom)
                    .detach()
                )
                metric_sums["value_clip_fraction"] += (
                    (
                        ((new_values - old_values).abs() > self.value_clip).to(
                            dtype=torch.float32
                        )
                        * mb_keep
                    )
                    .sum()
                    .div(keep_denom)
                    .detach()
                )
                value_grad = value_grad_norm.detach()
                metric_sums["value_grad_norm"] += value_grad
                metric_sums["value_grad_clip_fraction"] += (
                    value_grad > self.max_grad_norm
                ).to(dtype=torch.float32)
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
        metrics = {key: float(value / updates) for key, value in metric_sums.items()}
        metrics.update(
            {
                "epochs": epochs_completed,
                "minibatches": updates,
                "actor_updates": actor_updates,
                "value_updates": value_updates,
                "advantage_filter_discard_rate": discard_rate,
                "advantage_filter_retained": retained,
                "advantage_filter_eta": float(eta),
                "entropy_coef": self.entropy_coef,
            }
        )
        with torch.no_grad():
            ev_post = explained_variance(kept_returns, post_values[keep_mask])
        metrics["value_clip_fraction"] = clip_frac_sum / updates
        metrics["value_grad_clip_fraction"] = value_grad_clip_sum / updates
        metrics["value_clip_active_fraction"] = clip_active_sum / updates
        metrics["explained_variance"] = float(ev_pre)
        metrics["explained_variance_post"] = float(ev_post)
        metrics["return_mean"] = float(return_mean)
        metrics["return_std"] = float(return_std)
        metrics["value_mean"] = float(value_mean)
        return metrics


LR_SCHEDULES = ("constant", "linear", "cosine", "adaptive_kl")
ADAPTIVE_KL_LR_MIN = 1e-5
ADAPTIVE_KL_LR_MAX = 1e-2


def scheduled_learning_rate(progress, lr0, lr_final, schedule):
    if schedule == "constant":
        return float(lr0)
    if schedule not in ("linear", "cosine"):
        raise ValueError(
            f"schedule must be 'constant', 'linear', or 'cosine', got {schedule!r}"
        )
    progress = min(max(float(progress), 0.0), 1.0)
    lr0 = float(lr0)
    lr_final = float(lr_final)
    if schedule == "linear":
        return lr0 + (lr_final - lr0) * progress
    return lr_final + 0.5 * (lr0 - lr_final) * (1.0 + math.cos(math.pi * progress))


def adapt_kl_learning_rate(
    lr,
    approx_kl,
    desired_kl,
    *,
    lo=ADAPTIVE_KL_LR_MIN,
    hi=ADAPTIVE_KL_LR_MAX,
):
    kl = float(approx_kl)
    target = float(desired_kl)
    lr = float(lr)
    if kl > 2.0 * target:
        return max(float(lo), lr / 1.5)
    if kl < 0.5 * target:
        return min(float(hi), lr * 1.5)
    return lr


def _optimizer_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(optimizer, lr):
    lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr

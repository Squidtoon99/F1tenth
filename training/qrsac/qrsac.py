import torch
import torch.nn as nn
from torch.optim import Adam

from .spinningup.core import GRU_HIDDEN_DIM
from .quantile_critic import QuantileCritic

from dataclasses import dataclass

DEFAULT_BURN_IN = 16
DEFAULT_TRAIN_LEN = 32


def quantile_huber_loss(pred, target, kappa=1.0, taus=None):
    _, M = pred.shape

    pred_expanded = pred.unsqueeze(1)
    target_expanded = target.unsqueeze(2)
    diff = target_expanded - pred_expanded

    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff.pow(2),
        kappa * (abs_diff - 0.5 * kappa),
    )

    if taus is None:
        taus = (torch.arange(M, device=pred.device, dtype=pred.dtype) + 0.5) / M
    taus = taus.to(device=pred.device, dtype=pred.dtype).view(1, 1, M)

    indicator = (diff.detach() < 0).float()
    loss = torch.abs(taus - indicator) * huber
    return loss.sum(dim=2).mean(dim=1).mean()


def select_min_quantiles(
    q1_quantiles: torch.Tensor, q2_quantiles: torch.Tensor
) -> torch.Tensor:
    q1_mean = q1_quantiles.mean(dim=-1, keepdim=True)
    q2_mean = q2_quantiles.mean(dim=-1, keepdim=True)
    use_q1 = q1_mean <= q2_mean
    return torch.where(use_q1, q1_quantiles, q2_quantiles)


def _policy_phase(
    actor,
    critic1,
    critic2,
    critic1_target,
    critic2_target,
    actor_obs,
    critic_obs,
    next_actor_obs,
    next_critic_obs,
    reward,
    done,
    discount,
    alpha,
):
    """Targets (no_grad) + policy loss in one compiled region."""
    with torch.no_grad():
        actions_next, log_prob_next = actor(next_actor_obs)
        q1_quantile_next = critic1_target(next_critic_obs, actions_next)
        q2_quantile_next = critic2_target(next_critic_obs, actions_next)
        min_q_quantile_next = select_min_quantiles(
            q1_quantile_next, q2_quantile_next
        )
        target_quantiles = reward.unsqueeze(-1) + discount * (
            1.0 - done.unsqueeze(-1)
        ) * (min_q_quantile_next - alpha * log_prob_next.unsqueeze(-1))

    sampled_actions, log_prob = actor(actor_obs)
    q1_sampled = critic1(critic_obs, sampled_actions)
    q2_sampled = critic2(critic_obs, sampled_actions)
    q_sampled = torch.minimum(
        q1_sampled.mean(dim=-1, keepdim=True),
        q2_sampled.mean(dim=-1, keepdim=True),
    )
    policy_loss = (alpha * log_prob.unsqueeze(-1) - q_sampled).mean()
    return policy_loss, target_quantiles


def _critic_phase(
    critic1,
    critic2,
    critic_obs,
    action,
    target_quantiles,
    kappa,
    taus,
):
    """Critic forwards + quantile-Huber in one compiled region."""
    q1_quantile_observed = critic1(critic_obs, action)
    q2_quantile_observed = critic2(critic_obs, action)
    return quantile_huber_loss(
        q1_quantile_observed,
        target_quantiles,
        kappa=kappa,
        taus=taus,
    ) + quantile_huber_loss(
        q2_quantile_observed,
        target_quantiles,
        kappa=kappa,
        taus=taus,
    )


def _sequence_policy_phase(
    actor,
    critic1,
    critic2,
    critic1_target,
    critic2_target,
    burn_actor_obs,
    burn_reset,
    post_actor,
    post_reset,
    train_actor,
    train_critic,
    train_reset,
    hidden,
    bootstrap_critic_obs,
    n_step_reward,
    n_step_done,
    discount,
    alpha,
    n_step,
    act_dim,
    critic_obs_dim,
):
    """Sequence burn-in + targets + policy loss in one compiled region.

    All sequence lengths are fixed by the caller (burn-in / train / train+n_step)
    so ``torch.compile`` sees static shapes. Modules stay eager; only this phase
    function is compiled.
    """
    train_len = train_actor.shape[1]
    rows = train_actor.shape[0] * train_len
    with torch.no_grad():
        if burn_actor_obs.shape[1] > 0:
            _, _, h_burn = actor.forward_sequence(
                burn_actor_obs,
                hidden,
                reset_mask=burn_reset,
                deterministic=True,
                with_logprob=False,
            )
        else:
            h_burn = hidden
        h_burn = h_burn.detach()

        boot_actions_seq, boot_logp_seq, _ = actor.forward_sequence(
            post_actor,
            h_burn,
            reset_mask=post_reset,
            deterministic=False,
            with_logprob=True,
        )
        actions_next = boot_actions_seq[:, n_step : n_step + train_len].reshape(
            rows, act_dim
        )
        log_prob_next = boot_logp_seq[:, n_step : n_step + train_len].reshape(rows)
        next_critic = bootstrap_critic_obs.reshape(rows, critic_obs_dim)
        reward_flat = n_step_reward.reshape(rows)
        done_flat = n_step_done.reshape(rows)

        q1_quantile_next = critic1_target(next_critic, actions_next)
        q2_quantile_next = critic2_target(next_critic, actions_next)
        min_q_quantile_next = select_min_quantiles(
            q1_quantile_next, q2_quantile_next
        )
        target_quantiles = reward_flat.unsqueeze(-1) + discount * (
            1.0 - done_flat.unsqueeze(-1)
        ) * (min_q_quantile_next - alpha * log_prob_next.unsqueeze(-1))

    sampled_actions, log_prob, _ = actor.forward_sequence(
        train_actor,
        h_burn,
        reset_mask=train_reset,
        deterministic=False,
        with_logprob=True,
    )
    sampled_flat = sampled_actions.reshape(rows, act_dim)
    log_prob_flat = log_prob.reshape(rows)
    critic_flat = train_critic.reshape(rows, critic_obs_dim)
    q1_sampled = critic1(critic_flat, sampled_flat)
    q2_sampled = critic2(critic_flat, sampled_flat)
    q_sampled = torch.minimum(
        q1_sampled.mean(dim=-1, keepdim=True),
        q2_sampled.mean(dim=-1, keepdim=True),
    )
    policy_loss = (alpha * log_prob_flat.unsqueeze(-1) - q_sampled).mean()
    return policy_loss, target_quantiles


_ASYMMETRIC_OBS_KEYS = (
    "actor_obs",
    "critic_obs",
    "next_actor_obs",
    "next_critic_obs",
)
_LEGACY_OBS_KEYS = ("obs", "next_obs")
_IID_ONLY_KEYS = ("next_actor_obs", "next_critic_obs", "obs", "next_obs")
_SEQUENCE_REQUIRED_KEYS = (
    "actor_obs",
    "critic_obs",
    "action",
    "reset",
    "hidden",
    "n_step_reward",
    "n_step_done",
    "bootstrap_actor_obs",
    "bootstrap_critic_obs",
)


def _require_asymmetric_batch(batch) -> None:
    missing = [key for key in _ASYMMETRIC_OBS_KEYS if key not in batch]
    legacy = [key for key in _LEGACY_OBS_KEYS if key in batch]
    if missing or legacy:
        parts = []
        if missing:
            parts.append(f"missing required keys {missing}")
        if legacy:
            parts.append(
                f"legacy symmetric keys {legacy} are not supported "
                "(use actor_obs/critic_obs/next_actor_obs/next_critic_obs)"
            )
        raise KeyError("Asymmetric QR-SAC batch invalid: " + "; ".join(parts))


def _require_sequence_batch(
    batch,
    *,
    burn_in: int,
    train_len: int,
    n_step: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    hidden_dim: int,
) -> int:
    """Validate a trajectory batch; return the sequence count B."""
    iid = [key for key in _IID_ONLY_KEYS if key in batch]
    if iid:
        raise KeyError(
            "Sequence QR-SAC rejects IID/legacy keys "
            f"{iid}; use TrajectoryReplayBuffer.sample windows"
        )
    missing = [key for key in _SEQUENCE_REQUIRED_KEYS if key not in batch]
    if missing:
        raise KeyError(
            f"Sequence QR-SAC batch missing required keys {missing}"
        )

    seq_len = burn_in + train_len + n_step
    actor_obs = batch["actor_obs"]
    if actor_obs.ndim != 3:
        raise ValueError(
            f"actor_obs must be rank-3 (B, T, D); got shape={tuple(actor_obs.shape)}"
        )
    num_seq = int(actor_obs.shape[0])
    if num_seq <= 0:
        raise ValueError("sequence batch must contain at least one trajectory")

    expected = {
        "actor_obs": (num_seq, seq_len, actor_obs_dim),
        "critic_obs": (num_seq, seq_len, critic_obs_dim),
        "action": (num_seq, seq_len, None),
        "reset": (num_seq, seq_len),
        "hidden": (num_seq, hidden_dim),
        "n_step_reward": (num_seq, train_len),
        "n_step_done": (num_seq, train_len),
        "bootstrap_actor_obs": (num_seq, train_len, actor_obs_dim),
        "bootstrap_critic_obs": (num_seq, train_len, critic_obs_dim),
    }
    for key, shape in expected.items():
        tensor = batch[key]
        if key == "action":
            if (
                tensor.ndim != 3
                or tensor.shape[0] != num_seq
                or tensor.shape[1] != seq_len
            ):
                raise ValueError(
                    f"{key} shape={tuple(tensor.shape)}; expected "
                    f"({num_seq}, {seq_len}, act_dim)"
                )
            continue
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{key} shape={tuple(tensor.shape)}; expected {shape}"
            )
    return num_seq


@dataclass
class Models:
    actor: nn.Module
    critic1: QuantileCritic
    critic2: QuantileCritic
    critic1_target: QuantileCritic
    critic2_target: QuantileCritic


@dataclass
class Losses:
    policy_loss: torch.Tensor
    critic_loss: torch.Tensor


def _reset_module_parameters(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


class QRSACTrainer:
    def __init__(
        self,
        models: Models,
        device: torch.device,
        gamma: float = 0.99,
        n_step: int = 7,
        alpha: float = 0.2,
        smooth_factor: float = 0.005,
        kappa: float = 1.0,
        burn_in: int = DEFAULT_BURN_IN,
        train_len: int = DEFAULT_TRAIN_LEN,
        compile: bool = False,
        compile_mode: str = "default",
        assert_bootstrap_alignment: bool = False,
        actor_lr: float = 2.5e-5,
        critic_lr: float = 2.5e-5,
    ):
        self.device = device
        self.actor = models.actor
        self.critic1 = models.critic1
        self.critic2 = models.critic2
        self.critic1_target = models.critic1_target
        self.critic2_target = models.critic2_target
        self.actor_obs_dim = int(
            getattr(models.actor, "obs_dim", models.actor.net[0].weight.shape[1])
        )
        act_dim = int(models.actor.mu_layer.out_features)
        self.act_dim = act_dim
        self.critic_obs_dim = int(
            models.critic1.backbone[0].weight.shape[1] - act_dim
        )
        self.hidden_dim = int(
            getattr(models.actor, "gru_hidden_dim", GRU_HIDDEN_DIM)
        )
        self.burn_in = int(burn_in)
        self.train_len = int(train_len)
        if self.burn_in < 0 or self.train_len <= 0:
            raise ValueError(
                f"burn_in={self.burn_in} must be >= 0 and train_len={self.train_len} > 0"
            )
        self.seq_len = self.burn_in + self.train_len + int(n_step)
        self.assert_bootstrap_alignment = bool(assert_bootstrap_alignment)

        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError(
                f"actor_lr={self.actor_lr} and critic_lr={self.critic_lr} must be > 0"
            )
        self._adam_fused = bool(compile and device.type == "cuda")
        self._adam_capturable = bool(
            compile and compile_mode == "reduce-overhead" and device.type == "cuda"
        )
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            fused=self._adam_fused,
            capturable=self._adam_capturable,
        )
        self.critic_optimizer = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.critic_lr,
            fused=self._adam_fused,
            capturable=self._adam_capturable,
        )

        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.actor_frozen = False
        self.smooth_factor = smooth_factor
        self.kappa = kappa
        self.critic_params = tuple(self.critic1.parameters()) + tuple(
            self.critic2.parameters()
        )
        self._c1_target_params = list(self.critic1_target.parameters())
        self._c1_source_params = list(self.critic1.parameters())
        self._c2_target_params = list(self.critic2_target.parameters())
        self._c2_source_params = list(self.critic2.parameters())
        num_quantiles = self.critic1.head.out_features
        self.num_quantiles = int(num_quantiles)
        self.quantile_fractions = (
            torch.arange(num_quantiles, device=device, dtype=torch.float32) + 0.5
        ) / num_quantiles

        self._full_sequence_graph = self._adam_capturable
        self._sequence_graph = None
        self._sequence_graph_warmed = False
        self._sequence_static_batch = None
        self._sequence_graph_losses = None
        self._policy_phase = _policy_phase
        self._sequence_policy_phase = _sequence_policy_phase
        self._critic_phase = _critic_phase
        if compile:
            phase_mode = "default" if self._full_sequence_graph else compile_mode
            self._policy_phase = torch.compile(_policy_phase, mode=phase_mode)
            self._sequence_policy_phase = torch.compile(
                _sequence_policy_phase, mode=phase_mode
            )
            self._critic_phase = torch.compile(_critic_phase, mode=phase_mode)

    def _assert_obs_routing(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
    ) -> None:
        for name, tensor, expected in (
            ("actor_obs", actor_obs, self.actor_obs_dim),
            ("next_actor_obs", next_actor_obs, self.actor_obs_dim),
            ("critic_obs", critic_obs, self.critic_obs_dim),
            ("next_critic_obs", next_critic_obs, self.critic_obs_dim),
        ):
            if tensor.ndim != 2 or tensor.shape[-1] != expected:
                raise ValueError(
                    f"{name} shape={tuple(tensor.shape)}; expected (*, {expected}). "
                    "Actor and critic observation streams must not be swapped."
                )

    def update(self, batch) -> Losses:
        _require_asymmetric_batch(batch)
        actor_obs = batch["actor_obs"].to(self.device)
        critic_obs = batch["critic_obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_actor_obs = batch["next_actor_obs"].to(self.device)
        next_critic_obs = batch["next_critic_obs"].to(self.device)
        done = batch["done"].to(self.device)
        self._assert_obs_routing(
            actor_obs, critic_obs, next_actor_obs, next_critic_obs
        )

        discount = self.gamma**self.n_step

        for p in self.critic_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss, target_quantiles = self._policy_phase(
            self.actor,
            self.critic1,
            self.critic2,
            self.critic1_target,
            self.critic2_target,
            actor_obs,
            critic_obs,
            next_actor_obs,
            next_critic_obs,
            reward,
            done,
            discount,
            self.alpha,
        )
        if not self.actor_frozen:
            policy_loss.backward()
            self.actor_optimizer.step()

        for p in self.critic_params:
            p.requires_grad = True

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = self._critic_phase(
            self.critic1,
            self.critic2,
            critic_obs,
            action,
            target_quantiles,
            self.kappa,
            self.quantile_fractions,
        )
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.critic_params, max_norm=10.0)

        self.critic_optimizer.step()
        self._polyak_update()

        return Losses(
            policy_loss=policy_loss.detach(),
            critic_loss=critic_loss.detach(),
        )

    def _polyak_update(self) -> None:
        with torch.no_grad():
            torch._foreach_mul_(self._c1_target_params, 1.0 - self.smooth_factor)
            torch._foreach_add_(
                self._c1_target_params,
                self._c1_source_params,
                alpha=self.smooth_factor,
            )
            torch._foreach_mul_(self._c2_target_params, 1.0 - self.smooth_factor)
            torch._foreach_add_(
                self._c2_target_params,
                self._c2_source_params,
                alpha=self.smooth_factor,
            )

    def reinitialize_networks(self) -> None:
        """Replay-full reinit: fresh nets + Adam; targets hard-copied from online."""
        self.actor.apply(_reset_module_parameters)
        self.critic1.apply(_reset_module_parameters)
        self.critic2.apply(_reset_module_parameters)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            fused=self._adam_fused,
            capturable=self._adam_capturable,
        )
        self.critic_optimizer = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.critic_lr,
            fused=self._adam_fused,
            capturable=self._adam_capturable,
        )
        self._sequence_graph = None
        self._sequence_graph_warmed = False
        self._sequence_static_batch = None
        self._sequence_graph_losses = None

    def update_from_sequences(self, batch) -> Losses:
        if not hasattr(self.actor, "forward_sequence") or not hasattr(
            self.actor, "step"
        ):
            raise TypeError(
                "update_from_sequences requires a recurrent actor with "
                "step/forward_sequence (e.g. lidar_cnn_gru)"
            )
        _require_sequence_batch(
            batch,
            burn_in=self.burn_in,
            train_len=self.train_len,
            n_step=self.n_step,
            actor_obs_dim=self.actor_obs_dim,
            critic_obs_dim=self.critic_obs_dim,
            hidden_dim=self.hidden_dim,
        )
        if not self._full_sequence_graph:
            return self._update_from_sequences_impl(batch)

        if self._sequence_static_batch is None:
            self._sequence_static_batch = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
        else:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    self._sequence_static_batch[key].copy_(value)
                else:
                    self._sequence_static_batch[key] = value

        if not self._sequence_graph_warmed:
            losses = self._update_from_sequences_impl(self._sequence_static_batch)
            self._sequence_graph_warmed = True
            return losses

        if self.actor_frozen:
            return self._update_from_sequences_impl(self._sequence_static_batch)

        if self._sequence_graph is None:
            self._sequence_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._sequence_graph):
                self._sequence_graph_losses = self._update_from_sequences_impl(
                    self._sequence_static_batch
                )
            self._sequence_graph.replay()
        else:
            self._sequence_graph.replay()
        return Losses(
            policy_loss=self._sequence_graph_losses.policy_loss.detach().clone(),
            critic_loss=self._sequence_graph_losses.critic_loss.detach().clone(),
        )

    def _update_from_sequences_impl(self, batch) -> Losses:
        """Recurrent QR-SAC update from contiguous trajectory windows.

        Burn-in reconstructs actor state from the stored hidden checkpoint with
        no loss and no gradient. The following ``train_len`` steps contribute
        policy/critic losses (flattened to ``B * train_len`` rows). ``n_step``
        lookahead steps build sequence-consistent bootstrap actor actions for
        the existing n-step QR-SAC targets. The feed-forward critic never sees
        sequence structure.
        """
        burn = self.burn_in
        train = self.train_len
        n_step = self.n_step
        post_len = train + n_step

        actor_obs = batch["actor_obs"].to(self.device)
        critic_obs = batch["critic_obs"].to(self.device)
        action = batch["action"].to(self.device)
        reset = batch["reset"].to(self.device)
        hidden = batch["hidden"].to(self.device)
        n_step_reward = batch["n_step_reward"].to(self.device)
        n_step_done = batch["n_step_done"].to(self.device)
        bootstrap_critic_obs = batch["bootstrap_critic_obs"].to(self.device)
        if self.assert_bootstrap_alignment:
            bootstrap_actor_obs = batch["bootstrap_actor_obs"].to(self.device)
            expected_boot = actor_obs[:, burn + n_step : burn + train + n_step]
            if bootstrap_actor_obs.shape != expected_boot.shape or (
                bootstrap_actor_obs is not expected_boot
                and not torch.equal(bootstrap_actor_obs, expected_boot)
            ):
                raise ValueError(
                    "bootstrap_actor_obs must equal actor_obs[:, burn_in+n_step:"
                    "burn_in+train_len+n_step] for sequence-consistent targets"
                )

        burn_actor = actor_obs[:, :burn]
        burn_reset = reset[:, :burn]
        train_actor = actor_obs[:, burn : burn + train]
        train_critic = critic_obs[:, burn : burn + train]
        train_action = action[:, burn : burn + train]
        train_reset = reset[:, burn : burn + train]
        post_actor = actor_obs[:, burn : burn + post_len]
        post_reset = reset[:, burn : burn + post_len]
        discount = self.gamma**self.n_step

        for p in self.critic_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad(set_to_none=not self._full_sequence_graph)
        policy_loss, target_quantiles = self._sequence_policy_phase(
            self.actor,
            self.critic1,
            self.critic2,
            self.critic1_target,
            self.critic2_target,
            burn_actor,
            burn_reset,
            post_actor,
            post_reset,
            train_actor,
            train_critic,
            train_reset,
            hidden,
            bootstrap_critic_obs,
            n_step_reward,
            n_step_done,
            discount,
            self.alpha,
            n_step,
            self.act_dim,
            self.critic_obs_dim,
        )
        if not self.actor_frozen:
            policy_loss.backward()
            self.actor_optimizer.step()

        for p in self.critic_params:
            p.requires_grad = True

        self.critic_optimizer.zero_grad(set_to_none=not self._full_sequence_graph)
        rows = train_actor.shape[0] * train
        action_flat = train_action.reshape(rows, self.act_dim)
        critic_flat = train_critic.reshape(rows, self.critic_obs_dim)
        critic_loss = self._critic_phase(
            self.critic1,
            self.critic2,
            critic_flat,
            action_flat,
            target_quantiles,
            self.kappa,
            self.quantile_fractions,
        )
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, max_norm=10.0)
        self.critic_optimizer.step()
        self._polyak_update()

        return Losses(
            policy_loss=policy_loss.detach(),
            critic_loss=critic_loss.detach(),
        )

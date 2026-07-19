import torch
import torch.nn as nn
from torch.optim import Adam

from .spinningup.core import SquashedGaussianMLPActor
from .quantile_critic import QuantileCritic

from dataclasses import dataclass


def quantile_huber_loss(pred, target, kappa=1.0, taus=None):
    # pred, target: (B, M)
    _, M = pred.shape

    pred_expanded = pred.unsqueeze(1)  # (B, 1, M)
    target_expanded = target.unsqueeze(2)  # (B, M, 1)
    diff = target_expanded - pred_expanded  # (B, M, M)

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
    # q1_quantiles, q2_quantiles: (B, M)
    q1_mean = q1_quantiles.mean(dim=-1, keepdim=True)  # (B, 1)
    q2_mean = q2_quantiles.mean(dim=-1, keepdim=True)  # (B, 1)
    use_q1 = q1_mean <= q2_mean  # (B, 1)
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


_ASYMMETRIC_OBS_KEYS = (
    "actor_obs",
    "critic_obs",
    "next_actor_obs",
    "next_critic_obs",
)
_LEGACY_OBS_KEYS = ("obs", "next_obs")


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


@dataclass
class Models:
    actor: SquashedGaussianMLPActor
    critic1: QuantileCritic
    critic2: QuantileCritic
    critic1_target: QuantileCritic
    critic2_target: QuantileCritic


@dataclass
class Losses:
    policy_loss: torch.Tensor
    critic_loss: torch.Tensor


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
        compile: bool = False,
        compile_mode: str = "default",
    ):
        self.device = device
        self.actor = models.actor
        self.critic1 = models.critic1
        self.critic2 = models.critic2
        self.critic1_target = models.critic1_target
        self.critic2_target = models.critic2_target
        # Capture dims before compile wraps callables.
        self.actor_obs_dim = int(
            getattr(models.actor, "obs_dim", models.actor.net[0].weight.shape[1])
        )
        act_dim = int(models.actor.mu_layer.out_features)
        self.critic_obs_dim = int(
            models.critic1.backbone[0].weight.shape[1] - act_dim
        )

        fused = compile and device.type == "cuda"
        self.actor_optimizer = Adam(self.actor.parameters(), lr=2.5e-5, fused=fused)
        self.critic_optimizer = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=5e-5,
            fused=fused,
        )

        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.smooth_factor = smooth_factor
        self.kappa = kappa
        self.critic_params = tuple(self.critic1.parameters()) + tuple(
            self.critic2.parameters()
        )
        # Polyak param lists captured once; torch.compile shares parameter objects.
        self._c1_target_params = list(self.critic1_target.parameters())
        self._c1_source_params = list(self.critic1.parameters())
        self._c2_target_params = list(self.critic2_target.parameters())
        self._c2_source_params = list(self.critic2.parameters())
        num_quantiles = self.critic1.head.out_features
        self.quantile_fractions = (
            torch.arange(num_quantiles, device=device, dtype=torch.float32) + 0.5
        ) / num_quantiles

        # Two compiled regions (policy phase + critic phase) instead of six
        # per-module graphs. Cross-region tensors stay live until the producer
        # region runs again next update, so per-tensor clone guards are unnecessary.
        self._cudagraph = compile and compile_mode == "reduce-overhead"
        self._policy_phase = _policy_phase
        self._critic_phase = _critic_phase
        if compile:
            # Compiling the fused phases lets AOTAutograd fuse forward AND backward
            # across actor/critics/huber inside each region. mode="reduce-overhead"
            # captures one CUDA graph per phase instead of one per module.
            self._policy_phase = torch.compile(_policy_phase, mode=compile_mode)
            self._critic_phase = torch.compile(_critic_phase, mode=compile_mode)

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
        if self._cudagraph:
            torch.compiler.cudagraph_mark_step_begin()
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

        # Gradient clipping for stability (gt sophy uses it)
        nn.utils.clip_grad_norm_(self.critic_params, max_norm=10.0)

        self.critic_optimizer.step()

        # Target Update (polyak) via foreach: mul target by (1-tau), add tau*source.
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

        return Losses(
            policy_loss=policy_loss.detach(),
            critic_loss=critic_loss.detach(),
        )

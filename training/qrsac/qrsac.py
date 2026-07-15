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

        # Under CUDA graphs (reduce-overhead) each compiled region reuses a static
        # output buffer, so a tensor produced by one compiled call is overwritten by
        # the next. Clone tensors that cross between separately compiled regions.
        self._cudagraph = compile and compile_mode == "reduce-overhead"

        self._quantile_huber_loss = quantile_huber_loss
        if compile:
            # Compiling the modules lets AOTAutograd fuse the forward AND backward;
            # compiling the loss fuses the (B, M, M) quantile-huber pointwise ops.
            # mode="reduce-overhead" additionally captures CUDA graphs to collapse
            # per-kernel launch overhead.
            self.actor = torch.compile(self.actor, mode=compile_mode)
            self.critic1 = torch.compile(self.critic1, mode=compile_mode)
            self.critic2 = torch.compile(self.critic2, mode=compile_mode)
            self.critic1_target = torch.compile(self.critic1_target, mode=compile_mode)
            self.critic2_target = torch.compile(self.critic2_target, mode=compile_mode)
            self._quantile_huber_loss = torch.compile(
                quantile_huber_loss, mode=compile_mode
            )

    def _guard(self, t: torch.Tensor) -> torch.Tensor:
        return t.clone() if self._cudagraph else t

    def update(self, batch) -> Losses:
        if self._cudagraph:
            torch.compiler.cudagraph_mark_step_begin()
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Values used for target construction should not backpropagate through target networks.
        with torch.no_grad():
            actions_next, log_prob_next = self.actor(next_obs)
            actions_next = self._guard(actions_next)
            log_prob_next = self._guard(log_prob_next)
            q1_quantile_next = self._guard(self.critic1_target(next_obs, actions_next))
            q2_quantile_next = self._guard(self.critic2_target(next_obs, actions_next))
            min_q_quantile_next = select_min_quantiles(
                q1_quantile_next, q2_quantile_next
            )

            reward = reward.unsqueeze(-1)
            done = done.unsqueeze(-1)
            discount = self.gamma**self.n_step
            target_quantiles = reward + discount * (1.0 - done) * (
                min_q_quantile_next - self.alpha * log_prob_next.unsqueeze(-1)
            )

        for p in self.critic_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad(set_to_none=True)
        sampled_actions, log_prob = self.actor(obs)
        sampled_actions = self._guard(sampled_actions)
        log_prob = self._guard(log_prob)
        q1_sampled = self._guard(self.critic1(obs, sampled_actions))
        q2_sampled = self._guard(self.critic2(obs, sampled_actions))
        q1_mean = q1_sampled.mean(dim=-1, keepdim=True)
        q2_mean = q2_sampled.mean(dim=-1, keepdim=True)
        q_sampled = torch.minimum(q1_mean, q2_mean)
        policy_loss = (self.alpha * log_prob.unsqueeze(-1) - q_sampled).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic_params:
            p.requires_grad = True

        # Critic Update
        self.critic_optimizer.zero_grad(set_to_none=True)
        q1_quantile_observed = self._guard(self.critic1(obs, action))
        q2_quantile_observed = self._guard(self.critic2(obs, action))
        critic_loss = self._quantile_huber_loss(
            q1_quantile_observed,
            target_quantiles,
            kappa=self.kappa,
            taus=self.quantile_fractions,
        ) + self._quantile_huber_loss(
            q2_quantile_observed,
            target_quantiles,
            kappa=self.kappa,
            taus=self.quantile_fractions,
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
            policy_loss=self._guard(policy_loss.detach()),
            critic_loss=self._guard(critic_loss.detach()),
        )

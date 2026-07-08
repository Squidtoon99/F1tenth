import torch
import torch.nn as nn
from torch.optim import Adam

from .spinningup.core import SquashedGaussianMLPActor
from .quantile_critic import QuantileCritic

from dataclasses import dataclass


def quantile_huber_loss(pred, target, kappa=1.0):
    # pred, target: (B, M)
    B, M = pred.shape

    pred_expanded = pred.unsqueeze(1)  # (B, 1, M)
    target_expanded = target.unsqueeze(2)  # (B, M, 1)
    diff = target_expanded - pred_expanded  # (B, M, M)

    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff.pow(2),
        kappa * (abs_diff - 0.5 * kappa),
    )

    taus = (torch.arange(M, device=pred.device, dtype=torch.float32) + 0.5) / M
    taus = taus.view(1, 1, M)

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
    policy_loss: float
    critic_loss: float


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
    ):
        self.device = device
        self.actor = models.actor
        self.critic1 = models.critic1
        self.critic2 = models.critic2
        self.critic1_target = models.critic1_target
        self.critic2_target = models.critic2_target

        self.actor_optimizer = Adam(self.actor.parameters(), lr=2.5e-5)
        self.critic_optimizer = Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=5e-5,
        )

        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.smooth_factor = smooth_factor
        self.kappa = kappa

    def update(self, batch) -> Losses:
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        if action is None or reward is None or next_obs is None:
            raise KeyError(
                "Batch must contain obs, action, reward, next_obs, and done."
            )

        critic_params = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        # Values used for target construction should not backpropagate through target networks.
        with torch.no_grad():
            actions_next, log_prob_next = self.actor(next_obs)
            q1_quantile_next = self.critic1_target(next_obs, actions_next)
            q2_quantile_next = self.critic2_target(next_obs, actions_next)
            min_q_quantile_next = select_min_quantiles(
                q1_quantile_next, q2_quantile_next
            )

            reward = reward.unsqueeze(-1)
            done = done.unsqueeze(-1)
            discount = self.gamma**self.n_step
            target_quantiles = reward + discount * (1.0 - done) * (
                min_q_quantile_next - self.alpha * log_prob_next.unsqueeze(-1)
            )

        for p in critic_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad(set_to_none=True)
        sampled_actions, log_prob = self.actor(obs)
        q1_sampled = self.critic1(obs, sampled_actions)
        q2_sampled = self.critic2(obs, sampled_actions)
        q1_mean = q1_sampled.mean(dim=-1, keepdim=True)
        q2_mean = q2_sampled.mean(dim=-1, keepdim=True)
        q_sampled = torch.minimum(q1_mean, q2_mean)
        policy_loss = (self.alpha * log_prob - q_sampled).mean()
        policy_loss.backward()
        # actor_grad_norm = _grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        for p in critic_params:
            p.requires_grad = True

        # Critic Update
        self.critic_optimizer.zero_grad(set_to_none=True)
        q1_quantile_observed = self.critic1(obs, action)
        q2_quantile_observed = self.critic2(obs, action)
        critic_loss = quantile_huber_loss(
            q1_quantile_observed, target_quantiles, kappa=self.kappa
        ) + quantile_huber_loss(
            q2_quantile_observed, target_quantiles, kappa=self.kappa
        )
        critic_loss.backward()

        # Gradient clipping for stability (gt sophy uses it)
        nn.utils.clip_grad_norm_(critic_params, max_norm=10.0)

        self.critic_optimizer.step()

        # Target Update
        with torch.no_grad():
            for pred_param, target_param in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                target_param.data.mul_(1.0 - self.smooth_factor)
                target_param.data.add_(self.smooth_factor * pred_param.data)

            for pred_param, target_param in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                target_param.data.mul_(1.0 - self.smooth_factor)
                target_param.data.add_(self.smooth_factor * pred_param.data)

        return Losses(
            policy_loss=policy_loss.detach().item(),
            critic_loss=critic_loss.detach().item(),
        )

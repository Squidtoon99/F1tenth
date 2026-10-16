from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl_divergence

from f1tenth_policy.actor import mlp
from f1tenth_policy.dreamer import LidarDreamerActor


class WorldModelHeads(nn.Module):
    """Training-only RSSM prior, LiDAR decoder, reward, and continue heads."""

    def __init__(self, actor: LidarDreamerActor, hidden_sizes=(256, 256)):
        super().__init__()
        self.deter_dim = int(actor.deter_dim)
        self.stoch_dim = int(actor.stoch_dim)
        self.act_dim = int(actor.act_dim)
        self.lidar_dim = int(actor.lidar_dim)
        feat_dim = self.deter_dim + self.stoch_dim
        sizes = list(int(s) for s in hidden_sizes)
        self.prior_mean = nn.Linear(self.deter_dim, self.stoch_dim)
        self.prior_std = nn.Linear(self.deter_dim, self.stoch_dim)
        self.decoder = nn.Sequential(
            mlp([feat_dim] + sizes, nn.ReLU, nn.ReLU),
            nn.Linear(sizes[-1], self.lidar_dim),
        )
        self.reward_head = nn.Sequential(
            mlp([feat_dim] + sizes, nn.ReLU, nn.ReLU),
            nn.Linear(sizes[-1], 1),
        )
        self.continue_head = nn.Sequential(
            mlp([feat_dim] + sizes, nn.ReLU, nn.ReLU),
            nn.Linear(sizes[-1], 1),
        )

    def prior_stats(self, h):
        mean = self.prior_mean(h)
        std = F.softplus(self.prior_std(h)) + 1e-4
        return mean, std

    def imagine_step(
        self, actor: LidarDreamerActor, h, z, action, *, deterministic=False
    ):
        h, _ = actor.ls3c(torch.cat((z, action), dim=-1), h)
        prior_mean, prior_std = self.prior_stats(h)
        z = actor.sample_posterior(prior_mean, prior_std, deterministic)
        return h, z, prior_mean, prior_std

    def decode_lidar(self, h, z):
        return self.decoder(torch.cat((h, z), dim=-1))

    def reward(self, h, z):
        return self.reward_head(torch.cat((h, z), dim=-1)).squeeze(-1)

    def continue_pred(self, h, z):
        return self.continue_head(torch.cat((h, z), dim=-1)).squeeze(-1)


def diagonal_gaussian_kl(post_mean, post_std, prior_mean, prior_std):
    post = Independent(Normal(post_mean, post_std), 1)
    prior = Independent(Normal(prior_mean, prior_std), 1)
    return kl_divergence(post, prior)


def world_model_losses(
    actor: LidarDreamerActor,
    world: WorldModelHeads,
    obs,
    actions,
    rewards,
    dones,
    hidden,
    reset_mask=None,
    sample_latents=True,
):
    """One real sequence (B, T, ...) → reconstruction, reward, KL, continue."""
    if obs.dim() != 3:
        raise ValueError(f"obs must be (B, T, D), got {tuple(obs.shape)}")
    batch, steps, _ = obs.shape
    h, z, prev_a = actor.unpack_hidden(hidden)
    recon = []
    reward_pred = []
    cont_pred = []
    kl = []
    for t in range(steps):
        if reset_mask is not None:
            mask = reset_mask[:, t].to(dtype=h.dtype).unsqueeze(-1)
            h = h * (1.0 - mask)
            z = z * (1.0 - mask)
            prev_a = prev_a * (1.0 - mask)
        embed, proprio = actor.encode_obs(obs[:, t])
        h, _ = actor.ls3c(torch.cat((z, prev_a), dim=-1), h)
        prior_mean, prior_std = world.prior_stats(h)
        post_mean, post_std = actor.posterior_stats(h, embed, proprio)
        z = actor.sample_posterior(
            post_mean, post_std, deterministic=not sample_latents
        )
        recon.append(world.decode_lidar(h, z))
        reward_pred.append(world.reward(h, z))
        cont_pred.append(world.continue_pred(h, z))
        kl.append(diagonal_gaussian_kl(post_mean, post_std, prior_mean, prior_std))
        prev_a = actions[:, t]
    recon = torch.stack(recon, dim=1)
    lidar = obs[..., : actor.lidar_dim]
    recon_loss = (recon - lidar).square().mean()
    reward_loss = (torch.stack(reward_pred, dim=1) - rewards).square().mean()
    continue_target = 1.0 - dones.to(dtype=rewards.dtype)
    continue_loss = F.binary_cross_entropy_with_logits(
        torch.stack(cont_pred, dim=1), continue_target
    )
    kl_loss = torch.stack(kl, dim=1).mean()
    return {
        "recon_loss": recon_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "world_loss": recon_loss + reward_loss + continue_loss + kl_loss,
        "posterior_hidden": actor.pack_hidden(h, z, prev_a),
    }


def imagine_actor_critic(
    actor: LidarDreamerActor,
    world: WorldModelHeads,
    critic: nn.Module,
    start_hidden,
    *,
    horizon,
    gamma,
    lambda_gae,
    entropy_coef,
    sample_latents=True,
):
    """Train the actor on imagined RSSM rollouts, not on real PPO advantages."""
    h, z, _prev_a = actor.unpack_hidden(start_hidden)
    h = h.detach()
    z = z.detach()
    rewards = []
    values = []
    logps = []
    entropies = []
    continues = []
    for _ in range(int(horizon)):
        action, logp = actor.policy_from_state(h, z, deterministic=False)
        entropy = actor.entropy_from_state(h, z)
        with torch.no_grad():
            h, z, _mean, _std = world.imagine_step(
                actor, h, z, action, deterministic=not sample_latents
            )
            rewards.append(world.reward(h, z))
            continues.append(torch.sigmoid(world.continue_pred(h, z)))
        values.append(critic(actor.features(h, z)))
        logps.append(logp)
        entropies.append(entropy)
    rewards = torch.stack(rewards, dim=1)
    values = torch.stack(values, dim=1)
    logps = torch.stack(logps, dim=1)
    entropies = torch.stack(entropies, dim=1)
    continues = torch.stack(continues, dim=1)
    bootstrap = values[:, -1].detach()
    returns = _lambda_return(rewards, values, continues, bootstrap, gamma, lambda_gae)
    advantage = (returns - values).detach()
    actor_loss = -(logps * advantage + float(entropy_coef) * entropies).mean()
    critic_loss = 0.5 * (values - returns.detach()).square().mean()
    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "imagined_return": returns.mean().detach(),
        "policy_entropy": entropies.mean().detach(),
        "clip_ratio": values.new_zeros(()),
    }


def _lambda_return(rewards, values, continues, bootstrap, gamma, lam):
    gamma = float(gamma)
    lam = float(lam)
    steps = rewards.shape[1]
    acc = bootstrap
    outs = []
    for t in range(steps - 1, -1, -1):
        cont = continues[:, t]
        next_value = acc if t == steps - 1 else values[:, t + 1]
        acc = rewards[:, t] + gamma * cont * (
            (1.0 - lam) * next_value + lam * acc
        )
        outs.append(acc)
    outs.reverse()
    return torch.stack(outs, dim=1)

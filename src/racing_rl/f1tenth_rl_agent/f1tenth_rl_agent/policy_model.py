"""Dependency-free copy of the trained actor network.

This mirrors ``qrsac/spinningup/core.py`` ``SquashedGaussianMLPActor`` exactly so the
saved ``actor`` state_dict loads with identical architecture, without pulling in the
genesis-heavy ``qrsac`` package. Keep this in sync with the training definition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from f1tenth_contract import validate_policy_artifact

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (
                2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))
            ).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi


def load_actor(
    checkpoint_path: str,
    obs_dim: int,
    act_dim: int,
    hidden_sizes,
    act_limit: float,
    state_dict_key: str,
    device: torch.device,
) -> SquashedGaussianMLPActor:
    """Build the actor and load weights from a checkpoint.

    Accepts either a raw actor ``state_dict`` or a training checkpoint dict that
    contains the actor under ``state_dict_key`` (e.g. ``standalone_trainer`` saves
    ``{"step", "actor", "critic1", "critic2"}``).
    """
    actor = SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=list(hidden_sizes),
        activation=nn.ReLU,
        act_limit=act_limit,
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_policy_artifact(
        payload, expected_obs_dim=obs_dim, expected_action_dim=act_dim
    )
    if isinstance(payload, dict) and state_dict_key in payload:
        state_dict = payload[state_dict_key]
    else:
        raise ValueError(
            f"Checkpoint missing '{state_dict_key}' state_dict under the "
            "current/force policy artifact format."
        )
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor


class ObsNormalizer:
    """Apply the trainer's observation standardization at inference.

    Mirrors ``standalone_trainer.ObsNormalizer.normalize``: standardize each feature
    by the running mean/variance accumulated during training, then clamp. The policy
    was trained on normalized observations, so deploy MUST apply the same transform
    before the actor sees the raw observation (384 solo / 390 with the opponent block).
    """

    def __init__(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        eps: float,
        clip: float,
        device: torch.device,
    ):
        self.device = device
        self.mean = mean.to(device=device, dtype=torch.float32)
        self.var = var.to(device=device, dtype=torch.float32)
        self.eps = float(eps)
        self.clip = float(clip)

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(torch.float32) - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)


def load_obs_norm(
    checkpoint_path: str,
    obs_dim: int,
    device: torch.device,
    eps: float,
    clip: float,
) -> ObsNormalizer | None:
    """Load ObsNormalizer stats from a training checkpoint, or None if absent.

    ``standalone_trainer.save_policy_artifact`` stores the running statistics under
    the ``obs_norm`` key (``{"mean", "var", "count"}``).
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_policy_artifact(payload, expected_obs_dim=obs_dim)
    stats = payload.get("obs_norm")
    if not isinstance(stats, dict) or "mean" not in stats or "var" not in stats:
        return None
    return ObsNormalizer(
        mean=torch.as_tensor(stats["mean"]),
        var=torch.as_tensor(stats["var"]),
        eps=eps,
        clip=clip,
        device=device,
    )

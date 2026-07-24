
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    ALLOWED_LIDAR_POOL_BINS,
    CNN_PROJECTION_DIM,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    PROPRIO_DIM,
)

LOG_STD_MAX = 2
LOG_STD_MIN = -20
CNN_CONV_CHANNELS = (32, 64, 64)
CNN_KERNELS = (7, 5, 3)
CNN_STRIDES = (3, 2, 2)
CNN_PADDING = (3, 2, 1)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def _squashed_gaussian_forward(
    net,
    mu_layer,
    log_std_layer,
    act_limit,
    obs,
    deterministic,
    with_logprob,
    standard_normal=None,
):
    net_out = net(obs)
    mu = mu_layer(net_out)
    log_std = log_std_layer(net_out)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)

    pi_distribution = Normal(mu, std)
    if deterministic:
        pi_action = mu
    elif standard_normal is not None:
        pi_action = mu + std * standard_normal
    else:
        pi_action = pi_distribution.rsample()

    if with_logprob:
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)  # type: ignore
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
            axis=1
        )
    else:
        logp_pi = None

    pi_action = torch.tanh(pi_action)
    pi_action = act_limit * pi_action
    return pi_action, logp_pi


@torch.compiler.disable
def _partition_invariant_standard_normal(reference):
    eps = torch.finfo(reference.dtype).eps
    uniform = torch.rand_like(reference).clamp_(eps, 1.0 - eps)
    return torch.erfinv(uniform.mul_(2.0).sub_(1.0)).mul_(np.sqrt(2.0))


def normalize_actor_architecture(architecture: Mapping[str, Any] | dict) -> dict:
    return json.loads(
        json.dumps(
            architecture,
            default=lambda v: int(v) if hasattr(v, "item") else v,
        )
    )


def architectures_match(left, right) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    return normalize_actor_architecture(left) == normalize_actor_architecture(right)


class LidarCNNEncoder(nn.Module):
    def __init__(
        self,
        lidar_dim=LIDAR_DIM,
        pool_bins=32,
        projection_dim=CNN_PROJECTION_DIM,
    ):
        super().__init__()
        pool_bins = int(pool_bins)
        if pool_bins not in ALLOWED_LIDAR_POOL_BINS:
            raise ValueError(
                f"lidar_pool_bins must be one of {sorted(ALLOWED_LIDAR_POOL_BINS)}, "
                f"got {pool_bins}"
            )
        self.lidar_dim = int(lidar_dim)
        self.pool_bins = pool_bins
        self.projection_dim = int(projection_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(
                1,
                CNN_CONV_CHANNELS[0],
                kernel_size=CNN_KERNELS[0],
                stride=CNN_STRIDES[0],
                padding=CNN_PADDING[0],
            ),
            nn.ReLU(),
            nn.Conv1d(
                CNN_CONV_CHANNELS[0],
                CNN_CONV_CHANNELS[1],
                kernel_size=CNN_KERNELS[1],
                stride=CNN_STRIDES[1],
                padding=CNN_PADDING[1],
            ),
            nn.ReLU(),
            nn.Conv1d(
                CNN_CONV_CHANNELS[1],
                CNN_CONV_CHANNELS[2],
                kernel_size=CNN_KERNELS[2],
                stride=CNN_STRIDES[2],
                padding=CNN_PADDING[2],
            ),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.projection = nn.Linear(
            CNN_CONV_CHANNELS[2] * self.pool_bins, self.projection_dim
        )

    def forward_features(self, lidar):
        x = lidar.unsqueeze(1)
        x = self.conv(x)
        pooled = self.pool(x)
        flat = pooled.flatten(1)
        return F.relu(self.projection(flat)), pooled

    def forward(self, lidar):
        features, _ = self.forward_features(lidar)
        return features


class SquashedGaussianLidarGRUActor(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation,
        act_limit,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
        pool_bins=32,
        projection_dim=CNN_PROJECTION_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.lidar_dim = int(lidar_dim)
        self.proprio_dim = int(proprio_dim)
        self.pool_bins = int(pool_bins)
        self.projection_dim = int(projection_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.hidden_sizes = tuple(int(s) for s in hidden_sizes)
        if self.obs_dim != self.lidar_dim + self.proprio_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} must equal lidar_dim+proprio_dim="
                f"{self.lidar_dim + self.proprio_dim}"
            )
        if self.gru_hidden_dim <= 0:
            raise ValueError(
                f"gru_hidden_dim must be positive, got {self.gru_hidden_dim}"
            )
        self.encoder = LidarCNNEncoder(
            lidar_dim=self.lidar_dim,
            pool_bins=self.pool_bins,
            projection_dim=self.projection_dim,
        )
        gru_in = self.projection_dim + self.proprio_dim
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
        )
        self.net = mlp(
            [self.gru_hidden_dim] + list(hidden_sizes),
            activation,
            activation,  # type: ignore
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.actor_architecture = {
            "name": ACTOR_ARCHITECTURE_NAME,
            "version": 1,
            "obs_dim": self.obs_dim,
            "lidar_dim": self.lidar_dim,
            "proprio_dim": self.proprio_dim,
            "conv_channels": list(CNN_CONV_CHANNELS),
            "kernels": list(CNN_KERNELS),
            "strides": list(CNN_STRIDES),
            "padding": list(CNN_PADDING),
            "pool": "adaptive_avg_1d",
            "pool_bins": self.pool_bins,
            "projection_dim": self.projection_dim,
            "gru_hidden_dim": self.gru_hidden_dim,
            "hidden_layers": list(self.hidden_sizes),
            "activation": "relu",
            "action_dim": self.act_dim,
        }

    def initial_hidden(self, batch_size, *, device=None, dtype=None):
        return torch.zeros(
            int(batch_size),
            self.gru_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def encode(self, obs):
        lidar = obs[..., : self.lidar_dim]
        proprio = obs[..., self.lidar_dim :]
        features, pooled = self.encoder.forward_features(lidar)
        trunk_in = torch.cat([features, proprio], dim=-1)
        return trunk_in, pooled

    def _apply_reset_mask(self, hidden, reset_mask):
        if reset_mask is None:
            return hidden
        if reset_mask.shape != (hidden.shape[0],):
            raise ValueError(
                f"reset_mask shape={tuple(reset_mask.shape)}; "
                f"expected ({hidden.shape[0]},)"
            )
        mask = reset_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return hidden * (1.0 - mask)

    def step(
        self,
        obs,
        hidden,
        reset_mask=None,
        deterministic=False,
        with_logprob=True,
    ):
        if obs.dim() != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"step expects obs shape (B, {self.obs_dim}), got {tuple(obs.shape)}"
            )
        if (
            hidden.dim() != 2
            or hidden.shape[0] != obs.shape[0]
            or hidden.shape[-1] != self.gru_hidden_dim
        ):
            raise ValueError(
                f"step expects hidden shape ({obs.shape[0]}, {self.gru_hidden_dim}), "
                f"got {tuple(hidden.shape)}"
            )
        hidden = self._apply_reset_mask(hidden, reset_mask)
        features, _ = self.encode(obs)
        out, h_n = self.gru(features.unsqueeze(1), hidden.unsqueeze(0).contiguous())
        next_hidden = h_n.squeeze(0)
        pi_action, logp_pi = _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            out.squeeze(1),
            deterministic,
            with_logprob,
            standard_normal=(
                None
                if deterministic
                else _partition_invariant_standard_normal(
                    out.new_empty(out.shape[0], self.act_dim)
                )
            ),
        )
        return pi_action, logp_pi, next_hidden

    def _gru_sequence(self, features, hidden, reset_mask):
        batch, steps, _ = features.shape
        if steps == 0:
            empty = features.new_zeros(batch, 0, self.gru_hidden_dim)
            return empty, hidden

        if reset_mask is None:
            out, h_n = self.gru(features, hidden.unsqueeze(0).contiguous())
            return out, h_n.squeeze(0)

        outs = []
        h = hidden
        for t in range(steps):
            h = self._apply_reset_mask(h, reset_mask[:, t])
            h = torch._VF.gru_cell(
                features[:, t],
                h,
                self.gru.weight_ih_l0,
                self.gru.weight_hh_l0,
                self.gru.bias_ih_l0,
                self.gru.bias_hh_l0,
            )
            outs.append(h)
        return torch.stack(outs, dim=1), h

    def forward_sequence(
        self,
        obs,
        hidden,
        reset_mask=None,
        deterministic=False,
        with_logprob=True,
    ):
        if obs.dim() != 3 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"forward_sequence expects obs shape (B, T, {self.obs_dim}), "
                f"got {tuple(obs.shape)}"
            )
        batch, steps, _ = obs.shape
        if (
            hidden.dim() != 2
            or hidden.shape[0] != batch
            or hidden.shape[-1] != self.gru_hidden_dim
        ):
            raise ValueError(
                f"forward_sequence expects hidden shape ({batch}, "
                f"{self.gru_hidden_dim}), got {tuple(hidden.shape)}"
            )
        if reset_mask is not None and reset_mask.shape != (batch, steps):
            raise ValueError(
                f"reset_mask shape={tuple(reset_mask.shape)}; "
                f"expected ({batch}, {steps})"
            )
        if steps == 0:
            empty_act = obs.new_zeros(batch, 0, self.act_dim)
            empty_logp = obs.new_zeros(batch, 0) if with_logprob else None
            return empty_act, empty_logp, hidden

        flat_obs = obs.reshape(batch * steps, self.obs_dim)
        features, _ = self.encode(flat_obs)
        features = features.view(batch, steps, -1)
        gru_out, h = self._gru_sequence(features, hidden, reset_mask)
        flat_h = gru_out.transpose(0, 1).reshape(
            steps * batch, self.gru_hidden_dim
        )
        pi_action, logp_pi = _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            flat_h,
            deterministic,
            with_logprob,
            standard_normal=(
                None
                if deterministic
                else _partition_invariant_standard_normal(
                    flat_h.new_empty(flat_h.shape[0], self.act_dim)
                )
            ),
        )
        stacked_actions = pi_action.view(steps, batch, self.act_dim).transpose(0, 1)
        stacked_logp = (
            None
            if logp_pi is None
            else logp_pi.view(steps, batch).transpose(0, 1)
        )
        return stacked_actions, stacked_logp, h

    def forward(self, obs, deterministic=False, with_logprob=True):
        if obs.dim() != 2:
            raise ValueError(
                f"forward expects obs shape (B, {self.obs_dim}); "
                f"use step or forward_sequence for recurrent calls, "
                f"got {tuple(obs.shape)}"
            )
        hidden = self.initial_hidden(
            obs.shape[0], device=obs.device, dtype=obs.dtype
        )
        action, logp, _ = self.step(
            obs,
            hidden,
            reset_mask=None,
            deterministic=deterministic,
            with_logprob=with_logprob,
        )
        return action, logp


def make_actor(
    obs_dim,
    act_dim,
    hidden_sizes,
    activation=nn.ReLU,
    act_limit=1.0,
    lidar_pool_bins=32,
    lidar_dim=LIDAR_DIM,
    proprio_dim=PROPRIO_DIM,
    gru_hidden_dim=GRU_HIDDEN_DIM,
    actor_type=ACTOR_ARCHITECTURE_NAME,
):
    if actor_type != ACTOR_ARCHITECTURE_NAME:
        raise ValueError(
            f"Unsupported actor_type={actor_type!r}; "
            f"expected {ACTOR_ARCHITECTURE_NAME!r}"
        )
    return SquashedGaussianLidarGRUActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        act_limit=act_limit,
        lidar_dim=lidar_dim,
        proprio_dim=proprio_dim,
        pool_bins=lidar_pool_bins,
        gru_hidden_dim=gru_hidden_dim,
    )


def actor_from_architecture(architecture: Mapping[str, Any]):
    arch = normalize_actor_architecture(architecture)
    if arch.get("name") != ACTOR_ARCHITECTURE_NAME:
        raise ValueError(
            f"Unsupported actor_architecture.name={arch.get('name')!r}; "
            f"expected {ACTOR_ARCHITECTURE_NAME!r}"
        )
    return make_actor(
        obs_dim=int(arch["obs_dim"]),
        act_dim=int(arch["action_dim"]),
        hidden_sizes=list(arch["hidden_layers"]),
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=int(arch.get("pool_bins", 32)),
        lidar_dim=int(arch.get("lidar_dim", LIDAR_DIM)),
        proprio_dim=int(arch.get("proprio_dim", PROPRIO_DIM)),
        gru_hidden_dim=int(arch.get("gru_hidden_dim", GRU_HIDDEN_DIM)),
    )


def actor_architecture_from_module(actor) -> dict:
    return normalize_actor_architecture(dict(actor.actor_architecture))

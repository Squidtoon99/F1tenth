from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from f1tenth_policy.actor import (
    CNN_CONV_CHANNELS,
    CNN_KERNELS,
    CNN_PADDING,
    CNN_STRIDES,
    LOG_STD_MAX,
    LOG_STD_MIN,
    _squashed_gaussian_distribution,
    _squashed_gaussian_forward,
    _squashed_gaussian_log_prob,
    _partition_invariant_standard_normal,
    mlp,
)
from f1tenth_policy.layout import (
    CNN_PROJECTION_DIM,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_DREAMER_ARCHITECTURE_NAME,
    NUM_ACTIONS,
    PROPRIO_DIM,
)

DREAMER_DETER_DIM = 256
DREAMER_STOCH_DIM = 32
DREAMER_EMBED_DIM = CNN_PROJECTION_DIM
DREAMER_DEPLOY_PREFIXES = (
    "encoder.",
    "ls3c.",
    "post_mean.",
    "post_std.",
    "net.",
    "mu_layer.",
    "log_std_layer.",
)


def cartesian_polar_lidar(
    ranges,
    *,
    angle_min=LIDAR_ANGLE_MIN,
    angle_increment=LIDAR_ANGLE_INCREMENT,
):
    """Stack polar range with Cartesian hit coordinates, shape (..., 3, L)."""
    if ranges.dim() < 1:
        raise ValueError("ranges must include a beam axis")
    beams = ranges.shape[-1]
    angles = ranges.new_tensor(float(angle_min)) + ranges.new_tensor(
        float(angle_increment)
    ) * torch.arange(beams, device=ranges.device, dtype=ranges.dtype)
    x = ranges * torch.cos(angles)
    y = ranges * torch.sin(angles)
    return torch.stack((ranges, x, y), dim=-2)


class CartesianPolarEncoder(nn.Module):
    def __init__(self, lidar_dim=LIDAR_DIM, embed_dim=DREAMER_EMBED_DIM):
        super().__init__()
        self.lidar_dim = int(lidar_dim)
        self.embed_dim = int(embed_dim)
        channels = list(CNN_CONV_CHANNELS)
        layers = []
        in_ch = 3
        for out_ch, kernel, stride, padding in zip(
            channels, CNN_KERNELS, CNN_STRIDES, CNN_PADDING
        ):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    int(out_ch),
                    kernel_size=int(kernel),
                    stride=int(stride),
                    padding=int(padding),
                )
            )
            layers.append(nn.ReLU())
            in_ch = int(out_ch)
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.proj = nn.Linear(in_ch * 32, self.embed_dim)

    def forward(self, lidar):
        if lidar.shape[-1] != self.lidar_dim:
            raise ValueError(
                f"lidar last dim={lidar.shape[-1]}; expected {self.lidar_dim}"
            )
        chart = cartesian_polar_lidar(lidar)
        flat = chart.reshape(-1, 3, self.lidar_dim)
        pooled = self.pool(self.conv(flat))
        return self.proj(pooled.reshape(pooled.shape[0], -1)).view(
            *lidar.shape[:-1], self.embed_dim
        )


class LightweightStateSpaceCell(nn.Module):
    """Diagonal gated state-space cell (LS3C) used in place of a GRU RSSM."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_decay = nn.Parameter(torch.full((self.hidden_dim,), -1.0))
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.skip = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.gate = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

    def forward(self, inputs, hidden):
        if inputs.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(
                f"LS3C batch mismatch: inputs={tuple(inputs.shape)} "
                f"hidden={tuple(hidden.shape)}"
            )
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"LS3C input dim={inputs.shape[-1]}; expected {self.input_dim}"
            )
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"LS3C hidden dim={hidden.shape[-1]}; expected {self.hidden_dim}"
            )
        decay = torch.exp(-F.softplus(self.log_decay))
        state = decay * hidden + self.in_proj(inputs)
        mixed = self.out_proj(state) + self.skip(inputs)
        gate = torch.sigmoid(self.gate(torch.cat((inputs, hidden), dim=-1)))
        return gate * mixed + (1.0 - gate) * hidden, state


class LidarDreamerActor(nn.Module):
    """Deploy encoder + LS3C RSSM posterior + Gaussian actor (no decoder)."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
        embed_dim=DREAMER_EMBED_DIM,
        deter_dim=DREAMER_DETER_DIM,
        stoch_dim=DREAMER_STOCH_DIM,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        if self.act_dim != NUM_ACTIONS:
            raise ValueError(
                f"LiDAR Dreamer act_dim={self.act_dim} must equal "
                f"plant NUM_ACTIONS={NUM_ACTIONS}"
            )
        self.action_horizon = 1
        log_std_min = float(log_std_min)
        log_std_max = float(log_std_max)
        if not math.isfinite(log_std_min) or not math.isfinite(log_std_max):
            raise ValueError("log_std_min and log_std_max must be finite")
        if log_std_min >= log_std_max:
            raise ValueError(
                f"log_std_min={log_std_min} must be < log_std_max={log_std_max}"
            )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lidar_dim = int(lidar_dim)
        self.proprio_dim = int(proprio_dim)
        self.embed_dim = int(embed_dim)
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.hidden_sizes = tuple(int(s) for s in hidden_sizes)
        if self.obs_dim != self.lidar_dim + self.proprio_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} must equal lidar_dim+proprio_dim="
                f"{self.lidar_dim + self.proprio_dim}"
            )
        self.encoder = CartesianPolarEncoder(
            lidar_dim=self.lidar_dim, embed_dim=self.embed_dim
        )
        rssm_in = self.stoch_dim + self.act_dim
        self.ls3c = LightweightStateSpaceCell(rssm_in, self.deter_dim)
        post_in = self.deter_dim + self.embed_dim + self.proprio_dim
        self.post_mean = nn.Linear(post_in, self.stoch_dim)
        self.post_std = nn.Linear(post_in, self.stoch_dim)
        feat_dim = self.deter_dim + self.stoch_dim
        self.net = mlp(
            [feat_dim] + list(self.hidden_sizes),
            activation,
            activation,  # type: ignore
        )
        self.mu_layer = nn.Linear(self.hidden_sizes[-1], self.act_dim)
        self.log_std_layer = nn.Linear(self.hidden_sizes[-1], self.act_dim)
        self.act_limit = act_limit
        self.gru_hidden_dim = (
            self.deter_dim + self.stoch_dim + self.act_dim
        )
        self.actor_architecture = {
            "name": LIDAR_DREAMER_ARCHITECTURE_NAME,
            "version": 1,
            "obs_dim": self.obs_dim,
            "lidar_dim": self.lidar_dim,
            "proprio_dim": self.proprio_dim,
            "encoding": "cartesian_polar",
            "recurrent": "ls3c",
            "embed_dim": self.embed_dim,
            "deter_dim": self.deter_dim,
            "stoch_dim": self.stoch_dim,
            "hidden_layers": list(self.hidden_sizes),
            "activation": "relu",
            "action_dim": self.act_dim,
            "deploy": "encoder_actor",
        }

    def initial_hidden(self, batch_size, *, device=None, dtype=None):
        return torch.zeros(
            int(batch_size),
            self.gru_hidden_dim,
            device=device,
            dtype=dtype,
        )

    def unpack_hidden(self, hidden):
        h = hidden[..., : self.deter_dim]
        z = hidden[..., self.deter_dim : self.deter_dim + self.stoch_dim]
        prev_a = hidden[..., self.deter_dim + self.stoch_dim :]
        return h, z, prev_a

    def pack_hidden(self, h, z, prev_a):
        return torch.cat((h, z, prev_a), dim=-1)

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

    def encode_obs(self, obs):
        lidar = obs[..., : self.lidar_dim]
        proprio = obs[..., self.lidar_dim :]
        return self.encoder(lidar), proprio

    def posterior_stats(self, h, embed, proprio):
        inp = torch.cat((h, embed, proprio), dim=-1)
        mean = self.post_mean(inp)
        std = F.softplus(self.post_std(inp)) + 1e-4
        return mean, std

    def sample_posterior(self, mean, std, deterministic):
        if deterministic:
            return mean
        eps = _partition_invariant_standard_normal(mean)
        return mean + std * eps

    def features(self, h, z):
        return torch.cat((h, z), dim=-1)

    def rssm_observe(self, obs, hidden, *, deterministic=True):
        h, z, prev_a = self.unpack_hidden(hidden)
        embed, proprio = self.encode_obs(obs)
        h, _ = self.ls3c(torch.cat((z, prev_a), dim=-1), h)
        post_mean, post_std = self.posterior_stats(h, embed, proprio)
        z = self.sample_posterior(post_mean, post_std, deterministic)
        return h, z, post_mean, post_std, embed

    def step(
        self,
        obs,
        hidden,
        reset_mask=None,
        deterministic=False,
        with_logprob=True,
        return_pre_tanh=False,
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
                f"step expects hidden shape ({obs.shape[0]}, "
                f"{self.gru_hidden_dim}), got {tuple(hidden.shape)}"
            )
        hidden = self._apply_reset_mask(hidden, reset_mask)
        h, z, post_mean, post_std, _embed = self.rssm_observe(
            obs, hidden, deterministic=True
        )
        del post_mean, post_std
        feat = self.features(h, z)
        result = _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            feat,
            deterministic,
            with_logprob,
            standard_normal=(
                None
                if deterministic
                else _partition_invariant_standard_normal(
                    feat.new_empty(feat.shape[0], self.act_dim)
                )
            ),
            return_pre_tanh=return_pre_tanh,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        if return_pre_tanh:
            action, logp, pre_tanh = result
            next_hidden = self.pack_hidden(h, z, action.detach())
            return action, logp, next_hidden, pre_tanh
        action, logp = result
        next_hidden = self.pack_hidden(h, z, action.detach())
        return action, logp, next_hidden

    def policy_from_state(
        self, h, z, *, deterministic=False, with_logprob=True
    ):
        feat = self.features(h, z)
        return _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            feat,
            deterministic,
            with_logprob,
            standard_normal=(
                None
                if deterministic
                else _partition_invariant_standard_normal(
                    feat.new_empty(feat.shape[0], self.act_dim)
                )
            ),
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )

    def log_prob_from_state(self, h, z, actions):
        feat = self.features(h, z)
        distribution = _squashed_gaussian_distribution(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            feat,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        return _squashed_gaussian_log_prob(distribution, actions, self.act_limit)

    def entropy_from_state(self, h, z):
        feat = self.features(h, z)
        distribution = _squashed_gaussian_distribution(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            feat,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        return distribution.entropy().sum(dim=-1)

    def forward(self, obs, deterministic=False, with_logprob=True):
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


def deploy_actor_state_dict(actor: nn.Module) -> dict[str, torch.Tensor]:
    """Encoder + actor weights only; world-model heads stay in the trainer."""
    filtered = {
        key: tensor
        for key, tensor in actor.state_dict().items()
        if any(key.startswith(prefix) for prefix in DREAMER_DEPLOY_PREFIXES)
    }
    if not filtered:
        raise ValueError("LidarDreamerActor state_dict has no deploy prefixes")
    return filtered


def rssm_parameter_count(actor: LidarDreamerActor) -> int:
    modules = (actor.ls3c, actor.post_mean, actor.post_std)
    return sum(int(p.numel()) for m in modules for p in m.parameters())


def gru_rssm_parameter_count(input_dim: int, hidden_dim: int) -> int:
    gru = nn.GRU(input_size=int(input_dim), hidden_size=int(hidden_dim))
    return sum(int(p.numel()) for p in gru.parameters())

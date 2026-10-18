"""Decentralized CNN-GRU actor with private reward/dynamics conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from gigaflow_f1tenth.config import (
    ACTION_DIM,
    CNN_PROJECTION_DIM,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    PROPRIO_DIM,
    SENSOR_OBS_DIM,
    ExperimentConfig,
)
from gigaflow_f1tenth.normalization import SensorNormalizer
from gigaflow_f1tenth.rewards import CONDITION_DIM

CNN_CONV_CHANNELS = (32, 64, 64)
CNN_KERNELS = (7, 5, 3)
CNN_STRIDES = (3, 2, 2)
CNN_PADDING = (3, 2, 1)
DEFAULT_ACTOR_MLP = (1024, 1024, 1024)
ACTOR_ARCHITECTURE_NAME = "lidar_cnn_gru_conditioned"
CONDITION_EMBED_DIM = 64
LIDAR_POOL_BINS = 32
# Tight upper bound: std≫1 saturates tanh; atanh(action) then disagrees with the
# sampled pre_tanh and PPO ratios / pre-update KL explode.
LOG_STD_MAX = 0.5
LOG_STD_MIN = -5.0
ACT_LIMIT = 1.0

ActorVariantName = Literal["gru", "feedforward", "frame_stack"]


@dataclass(frozen=True)
class ActorShapes:
    sensor_obs_dim: int = SENSOR_OBS_DIM
    lidar_dim: int = LIDAR_DIM
    proprio_dim: int = PROPRIO_DIM
    condition_dim: int = CONDITION_DIM
    action_dim: int = ACTION_DIM
    gru_hidden_dim: int = GRU_HIDDEN_DIM
    cnn_projection_dim: int = CNN_PROJECTION_DIM
    mlp_sizes: tuple[int, ...] = DEFAULT_ACTOR_MLP


def actor_shapes(cfg: ExperimentConfig) -> ActorShapes:
    return ActorShapes(
        sensor_obs_dim=cfg.agents.sensor_obs_dim,
        lidar_dim=cfg.agents.lidar_dim,
        proprio_dim=cfg.agents.proprio_dim,
        condition_dim=cfg.agents.condition_dim,
        action_dim=cfg.agents.action_dim,
        gru_hidden_dim=cfg.agents.gru_hidden_dim,
        cnn_projection_dim=cfg.agents.cnn_projection_dim,
        mlp_sizes=cfg.agents.actor_mlp_sizes,
    )


@dataclass
class ActorOutput:
    actions: Any  # [B, A] tanh-squashed continuous force / steer-delta
    log_prob: Any  # [B]
    entropy: Any  # [B]
    hidden: Any  # [B, H]
    pre_tanh: Any = None  # [B, A] Gaussian sample before squash (PPO rescoring)
    values_unused: None = None  # actor does not estimate value


@runtime_checkable
class ConditionedActor(Protocol):
    """Shared live actor for every active car; decentralized / occluded."""

    def shapes(self) -> ActorShapes:
        ...

    def initial_hidden(self, batch_size: int, device: str) -> Any:
        ...

    def forward(
        self,
        sensor_obs: Any,
        private_condition: Any,
        hidden: Any,
        reset_mask: Any | None = None,
        deterministic: bool = False,
    ) -> ActorOutput:
        """
        sensor_obs: [B, 1097] = LiDAR[1081] + proprio[16]
        private_condition: [B, C] normalized side channel (not part of sensor contract)
        hidden: [B, 512]
        reset_mask: optional [B] bool — clear GRU state for reset rows only
        """


def mlp(sizes: list[int], activation: type[nn.Module], output_activation=nn.Identity):
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def orthogonal_init_(module: nn.Module, gain: float = np.sqrt(2.0)) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.zeros_(param)


@torch.compiler.disable
def _partition_invariant_standard_normal(reference: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(reference.dtype).eps
    uniform = torch.rand_like(reference).clamp_(eps, 1.0 - eps)
    return torch.erfinv(uniform.mul_(2.0).sub_(1.0)).mul_(np.sqrt(2.0))


def _tanh_log_prob_correction(pre_tanh: torch.Tensor) -> torch.Tensor:
    return (2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)


def _squashed_gaussian(
    net: nn.Module,
    mu_layer: nn.Linear,
    log_std_layer: nn.Linear,
    features: torch.Tensor,
    deterministic: bool,
    act_limit: float = ACT_LIMIT,
    standard_normal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Head + Gaussian sample/log-prob in FP32 even when trunk ran under BF16.
    with torch.amp.autocast(device_type=features.device.type, enabled=False):
        feats = features.to(dtype=torch.float32)
        net_out = net(feats)
        mu = mu_layer(net_out)
        log_std = torch.clamp(log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            pre_tanh = mu
        elif standard_normal is not None:
            pre_tanh = mu + std * standard_normal.to(dtype=torch.float32)
        else:
            pre_tanh = dist.rsample()
        action = act_limit * torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh).sum(dim=-1) - _tanh_log_prob_correction(
            pre_tanh
        )
        entropy = dist.entropy().sum(dim=-1)
    return action, logp, entropy, pre_tanh


def _log_prob_pre_tanh(
    net: nn.Module,
    mu_layer: nn.Linear,
    log_std_layer: nn.Linear,
    features: torch.Tensor,
    pre_tanh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact log-prob using the collection-time pre-tanh Gaussian sample."""
    # Trunk/GRU may run under BF16 autocast; force FP32 for Gaussian log-prob.
    with torch.amp.autocast(device_type=features.device.type, enabled=False):
        feats = features.to(dtype=torch.float32)
        net_out = net(feats)
        mu = mu_layer(net_out)
        log_std = torch.clamp(log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pre = pre_tanh.to(dtype=torch.float32)
        dist = Normal(mu, std)
        logp = dist.log_prob(pre).sum(dim=-1) - _tanh_log_prob_correction(pre)
        entropy = dist.entropy().sum(dim=-1)
    return logp, entropy


def _log_prob_actions(
    net: nn.Module,
    mu_layer: nn.Linear,
    log_std_layer: nn.Linear,
    features: torch.Tensor,
    actions: torch.Tensor,
    act_limit: float = ACT_LIMIT,
    pre_tanh: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob / entropy; prefer stored pre_tanh when available."""
    if pre_tanh is not None:
        return _log_prob_pre_tanh(
            net, mu_layer, log_std_layer, features, pre_tanh
        )
    with torch.amp.autocast(device_type=features.device.type, enabled=False):
        feats = features.to(dtype=torch.float32)
        net_out = net(feats)
        mu = mu_layer(net_out)
        log_std = torch.clamp(log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        scaled = torch.clamp(
            actions.to(dtype=torch.float32) / act_limit, -1.0 + 1e-6, 1.0 - 1e-6
        )
        recovered = 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))
        logp = dist.log_prob(recovered).sum(dim=-1) - _tanh_log_prob_correction(
            recovered
        )
        entropy = dist.entropy().sum(dim=-1)
    return logp, entropy


class LidarCNNEncoder(nn.Module):
    def __init__(
        self,
        lidar_dim: int = LIDAR_DIM,
        pool_bins: int = LIDAR_POOL_BINS,
        projection_dim: int = CNN_PROJECTION_DIM,
        in_channels: int = 1,
    ):
        super().__init__()
        self.lidar_dim = int(lidar_dim)
        self.pool_bins = int(pool_bins)
        self.projection_dim = int(projection_dim)
        self.in_channels = int(in_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
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

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        # lidar: [B, L] or [B, K, L] stacked frames
        if lidar.dim() == 2:
            x = lidar.unsqueeze(1)
        else:
            x = lidar
        x = self.conv(x)
        flat = self.pool(x).flatten(1)
        return F.relu(self.projection(flat))


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim: int, embed_dim: int = CONDITION_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class ConditionedLidarGRUActor(nn.Module):
    """Primary racing actor: ordered-LiDAR CNN + condition MLP + 512 GRU + MLP head."""

    def __init__(self, shapes: ActorShapes, condition_embed_dim: int = CONDITION_EMBED_DIM):
        super().__init__()
        if shapes.sensor_obs_dim != shapes.lidar_dim + shapes.proprio_dim:
            raise ValueError("sensor_obs_dim must equal lidar_dim + proprio_dim")
        if shapes.condition_dim != CONDITION_DIM:
            raise ValueError(
                f"condition_dim={shapes.condition_dim} != schema {CONDITION_DIM}"
            )
        self._shapes = shapes
        self.condition_embed_dim = int(condition_embed_dim)
        self.sensor_normalizer = SensorNormalizer(shapes.sensor_obs_dim)
        self.encoder = LidarCNNEncoder(
            lidar_dim=shapes.lidar_dim,
            projection_dim=shapes.cnn_projection_dim,
            in_channels=1,
        )
        self.condition_encoder = ConditionEncoder(
            shapes.condition_dim, self.condition_embed_dim
        )
        gru_in = (
            shapes.cnn_projection_dim + shapes.proprio_dim + self.condition_embed_dim
        )
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=shapes.gru_hidden_dim,
            batch_first=True,
        )
        self.net = mlp(
            [shapes.gru_hidden_dim] + list(shapes.mlp_sizes),
            nn.ReLU,
            nn.ReLU,
        )
        self.mu_layer = nn.Linear(shapes.mlp_sizes[-1], shapes.action_dim)
        self.log_std_layer = nn.Linear(shapes.mlp_sizes[-1], shapes.action_dim)
        orthogonal_init_(self)
        nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)

    def shapes(self) -> ActorShapes:
        return self._shapes

    def initial_hidden(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(
            int(batch_size),
            self._shapes.gru_hidden_dim,
            device=device,
            dtype=torch.float32,
        )

    def _split_obs(self, sensor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.sensor_normalizer(sensor_obs)
        return (
            normed[..., : self._shapes.lidar_dim],
            normed[..., self._shapes.lidar_dim :],
        )

    def _apply_reset_mask(
        self, hidden: torch.Tensor, reset_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if reset_mask is None:
            return hidden
        mask = reset_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return hidden * (1.0 - mask)

    def encode_trunk(
        self, sensor_obs: torch.Tensor, private_condition: torch.Tensor
    ) -> torch.Tensor:
        lidar, proprio = self._split_obs(sensor_obs)
        features = self.encoder(lidar)
        cond = self.condition_encoder(private_condition)
        return torch.cat([features, proprio, cond], dim=-1)

    def forward(
        self,
        sensor_obs: torch.Tensor,
        private_condition: torch.Tensor,
        hidden: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ActorOutput:
        if sensor_obs.dim() != 2 or sensor_obs.shape[-1] != self._shapes.sensor_obs_dim:
            raise ValueError(
                f"sensor_obs shape {tuple(sensor_obs.shape)}; "
                f"expected (B, {self._shapes.sensor_obs_dim})"
            )
        if private_condition.shape[-1] != self._shapes.condition_dim:
            raise ValueError(
                f"private_condition dim {private_condition.shape[-1]} != "
                f"{self._shapes.condition_dim}"
            )
        hidden = self._apply_reset_mask(hidden, reset_mask)
        trunk = self.encode_trunk(sensor_obs, private_condition)
        out, h_n = self.gru(trunk.unsqueeze(1), hidden.unsqueeze(0).contiguous())
        next_hidden = h_n.squeeze(0)
        noise = None
        if not deterministic:
            noise = _partition_invariant_standard_normal(
                out.new_empty(out.shape[0], self._shapes.action_dim)
            )
        actions, log_prob, entropy, pre_tanh = _squashed_gaussian(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            out.squeeze(1),
            deterministic=deterministic,
            standard_normal=noise,
        )
        return ActorOutput(
            actions=actions,
            log_prob=log_prob,
            entropy=entropy,
            hidden=next_hidden,
            pre_tanh=pre_tanh,
        )

    def _gru_sequence(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match collection ``forward``: per-step ``nn.GRU`` (not VF gru_cell).

        cuDNN sequence GRU / ``gru_cell`` diverge enough on CUDA to trip the
        collect/evaluate logp parity gate even with stored ``pre_tanh``.
        """
        batch, steps, _ = features.shape
        if steps == 0:
            empty = features.new_zeros(batch, 0, self._shapes.gru_hidden_dim)
            return empty, hidden
        outs = []
        h = hidden
        for t in range(steps):
            if reset_mask is not None:
                h = self._apply_reset_mask(h, reset_mask[:, t])
            out, h_n = self.gru(
                features[:, t : t + 1], h.unsqueeze(0).contiguous()
            )
            h = h_n.squeeze(0)
            outs.append(out.squeeze(1))
        return torch.stack(outs, dim=1), h

    def evaluate_actions_sequence(
        self,
        sensor_obs: torch.Tensor,
        private_condition: torch.Tensor,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        pre_tanh: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Recurrent PPO seam: return (log_prob[B,T], entropy[B,T], hidden[B,H])."""
        if sensor_obs.dim() != 3 or sensor_obs.shape[-1] != self._shapes.sensor_obs_dim:
            raise ValueError(
                f"evaluate_actions_sequence expects obs [B,T,{self._shapes.sensor_obs_dim}], "
                f"got {tuple(sensor_obs.shape)}"
            )
        batch, steps, _ = sensor_obs.shape
        if private_condition.dim() == 2:
            if private_condition.shape != (batch, self._shapes.condition_dim):
                raise ValueError(
                    f"private_condition shape {tuple(private_condition.shape)}; "
                    f"expected ({batch}, {self._shapes.condition_dim})"
                )
            cond = private_condition.unsqueeze(1).expand(batch, steps, -1)
        elif private_condition.dim() == 3:
            if private_condition.shape != (
                batch,
                steps,
                self._shapes.condition_dim,
            ):
                raise ValueError(
                    f"private_condition shape {tuple(private_condition.shape)}; "
                    f"expected ({batch}, {steps}, {self._shapes.condition_dim})"
                )
            cond = private_condition
        else:
            raise ValueError(
                f"private_condition shape {tuple(private_condition.shape)} unsupported"
            )
        if reset_mask is not None and reset_mask.shape != (batch, steps):
            raise ValueError(
                f"reset_mask shape {tuple(reset_mask.shape)}; expected ({batch}, {steps})"
            )
        if steps == 0:
            empty = sensor_obs.new_zeros(batch, 0)
            return empty, empty, hidden

        flat_obs = sensor_obs.reshape(batch * steps, self._shapes.sensor_obs_dim)
        flat_cond = cond.reshape(batch * steps, self._shapes.condition_dim)
        trunk = self.encode_trunk(flat_obs, flat_cond).view(batch, steps, -1)
        gru_out, h = self._gru_sequence(trunk, hidden, reset_mask)
        flat_h = gru_out.reshape(batch * steps, self._shapes.gru_hidden_dim)
        flat_act = actions.reshape(batch * steps, self._shapes.action_dim)
        flat_pre = None
        if pre_tanh is not None:
            flat_pre = pre_tanh.reshape(batch * steps, self._shapes.action_dim)
        logp, ent = _log_prob_actions(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            flat_h,
            flat_act,
            pre_tanh=flat_pre,
        )
        result = (
            logp.view(batch, steps),
            ent.view(batch, steps),
            h,
        )
        if not return_diagnostics:
            return result
        with torch.amp.autocast(device_type=flat_h.device.type, enabled=False):
            log_std = torch.clamp(
                self.log_std_layer(self.net(flat_h.to(dtype=torch.float32))),
                LOG_STD_MIN,
                LOG_STD_MAX,
            )
        return (*result, log_std.view(batch, steps, self._shapes.action_dim))


class ConditionedLidarFeedForwardActor(nn.Module):
    """Matched-parameter ablation: CNN + condition, no GRU recurrence."""

    def __init__(
        self,
        shapes: ActorShapes,
        condition_embed_dim: int = CONDITION_EMBED_DIM,
        frame_stack: int = 1,
    ):
        super().__init__()
        self._shapes = shapes
        self.frame_stack = int(frame_stack)
        self.condition_embed_dim = int(condition_embed_dim)
        self.sensor_normalizer = SensorNormalizer(shapes.sensor_obs_dim)
        self.encoder = LidarCNNEncoder(
            lidar_dim=shapes.lidar_dim,
            projection_dim=shapes.cnn_projection_dim,
            in_channels=self.frame_stack,
        )
        self.condition_encoder = ConditionEncoder(
            shapes.condition_dim, self.condition_embed_dim
        )
        trunk_dim = (
            shapes.cnn_projection_dim + shapes.proprio_dim + self.condition_embed_dim
        )
        # Absorb GRU capacity into a linear projection so param count stays comparable.
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, shapes.gru_hidden_dim),
            nn.ReLU(),
        )
        self.net = mlp(
            [shapes.gru_hidden_dim] + list(shapes.mlp_sizes),
            nn.ReLU,
            nn.ReLU,
        )
        self.mu_layer = nn.Linear(shapes.mlp_sizes[-1], shapes.action_dim)
        self.log_std_layer = nn.Linear(shapes.mlp_sizes[-1], shapes.action_dim)
        orthogonal_init_(self)
        nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self.log_std_layer.bias)

    def shapes(self) -> ActorShapes:
        return self._shapes

    def initial_hidden(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        # Unused carry; kept so callers share one interface with the GRU actor.
        return torch.zeros(
            int(batch_size),
            self._shapes.gru_hidden_dim,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        sensor_obs: torch.Tensor,
        private_condition: torch.Tensor,
        hidden: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ActorOutput:
        del reset_mask  # no recurrent state
        b = sensor_obs.shape[0]
        trunk = self._trunk_features(sensor_obs, private_condition)
        noise = None
        if not deterministic:
            noise = _partition_invariant_standard_normal(
                trunk.new_empty(b, self._shapes.action_dim)
            )
        actions, log_prob, entropy, pre_tanh = _squashed_gaussian(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            trunk,
            deterministic=deterministic,
            standard_normal=noise,
        )
        return ActorOutput(
            actions=actions,
            log_prob=log_prob,
            entropy=entropy,
            hidden=hidden,
            pre_tanh=pre_tanh,
        )

    def _trunk_features(
        self, sensor_obs: torch.Tensor, private_condition: torch.Tensor
    ) -> torch.Tensor:
        sensor_obs = self.sensor_normalizer(sensor_obs)
        if sensor_obs.dim() == 2:
            if self.frame_stack != 1:
                raise ValueError(
                    f"frame_stack={self.frame_stack} requires stacked obs [B,K,D]"
                )
            lidar_in = sensor_obs[..., : self._shapes.lidar_dim].unsqueeze(1)
            proprio = sensor_obs[..., self._shapes.lidar_dim :]
        elif sensor_obs.dim() == 3:
            if sensor_obs.shape[-1] != self._shapes.sensor_obs_dim:
                raise ValueError(
                    f"unexpected stacked obs last-dim {sensor_obs.shape[-1]}"
                )
            if sensor_obs.shape[1] != self.frame_stack:
                raise ValueError(
                    f"stack length {sensor_obs.shape[1]} != {self.frame_stack}"
                )
            lidar_in = sensor_obs[..., : self._shapes.lidar_dim]
            proprio = sensor_obs[:, -1, self._shapes.lidar_dim :]
        else:
            raise ValueError(f"unexpected sensor_obs shape {tuple(sensor_obs.shape)}")
        features = self.encoder(lidar_in)
        cond = self.condition_encoder(private_condition)
        return self.trunk(torch.cat([features, proprio, cond], dim=-1))

    def evaluate_actions_sequence(
        self,
        sensor_obs: torch.Tensor,
        private_condition: torch.Tensor,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
        pre_tanh: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple:
        """Feed-forward / frame-stack PPO seam (ignores reset_mask / hidden carry)."""
        del reset_mask
        if sensor_obs.dim() == 2:
            sensor_obs = sensor_obs.unsqueeze(1)
            actions = actions.unsqueeze(1)
            if pre_tanh is not None and pre_tanh.dim() == 2:
                pre_tanh = pre_tanh.unsqueeze(1)
        batch, steps, _ = sensor_obs.shape
        if private_condition.dim() == 2:
            cond_seq = private_condition.unsqueeze(1).expand(batch, steps, -1)
        elif private_condition.dim() == 3:
            cond_seq = private_condition
        else:
            raise ValueError(
                f"private_condition shape {tuple(private_condition.shape)} unsupported"
            )
        logps = []
        ents = []
        log_stds = []
        for t in range(steps):
            step = sensor_obs[:, t]
            if self.frame_stack > 1:
                step = step.unsqueeze(1).expand(batch, self.frame_stack, -1)
            trunk = self._trunk_features(step, cond_seq[:, t])
            pre_t = None if pre_tanh is None else pre_tanh[:, t]
            logp, ent = _log_prob_actions(
                self.net,
                self.mu_layer,
                self.log_std_layer,
                trunk,
                actions[:, t],
                pre_tanh=pre_t,
            )
            logps.append(logp)
            ents.append(ent)
            if return_diagnostics:
                with torch.amp.autocast(device_type=trunk.device.type, enabled=False):
                    log_stds.append(
                        torch.clamp(
                            self.log_std_layer(
                                self.net(trunk.to(dtype=torch.float32))
                            ),
                            LOG_STD_MIN,
                            LOG_STD_MAX,
                        )
                    )
        result = torch.stack(logps, dim=1), torch.stack(ents, dim=1), hidden
        if not return_diagnostics:
            return result
        return (*result, torch.stack(log_stds, dim=1))


def build_actor(
    cfg: ExperimentConfig,
    variant: ActorVariantName = "gru",
    frame_stack: int = 1,
) -> ConditionedActor:
    shapes = actor_shapes(cfg)
    if shapes.condition_dim != CONDITION_DIM:
        raise ValueError(
            f"cfg.agents.condition_dim={shapes.condition_dim} != "
            f"rewards.CONDITION_DIM={CONDITION_DIM}"
        )
    if variant == "gru":
        if frame_stack != 1:
            raise ValueError("gru variant does not use frame_stack")
        return ConditionedLidarGRUActor(shapes)
    if variant == "feedforward":
        return ConditionedLidarFeedForwardActor(shapes, frame_stack=1)
    if variant == "frame_stack":
        if frame_stack not in (4, 8):
            raise ValueError("frame_stack ablation expects 4 or 8 frames")
        return ConditionedLidarFeedForwardActor(shapes, frame_stack=frame_stack)
    raise ValueError(f"unknown actor variant {variant!r}")


def architecture_metadata(cfg: ExperimentConfig) -> dict[str, Any]:
    shapes = actor_shapes(cfg)
    return {
        "name": ACTOR_ARCHITECTURE_NAME,
        "sensor_obs_dim": shapes.sensor_obs_dim,
        "lidar_dim": shapes.lidar_dim,
        "proprio_dim": shapes.proprio_dim,
        "condition_dim": shapes.condition_dim,
        "condition_embed_dim": CONDITION_EMBED_DIM,
        "action_dim": shapes.action_dim,
        "gru_hidden_dim": shapes.gru_hidden_dim,
        "cnn_projection_dim": shapes.cnn_projection_dim,
        "cnn_conv_channels": list(CNN_CONV_CHANNELS),
        "cnn_kernels": list(CNN_KERNELS),
        "cnn_strides": list(CNN_STRIDES),
        "cnn_padding": list(CNN_PADDING),
        "pool_bins": LIDAR_POOL_BINS,
        "mlp_sizes": list(shapes.mlp_sizes),
        "steering_action_mode": "delta",
        "longitudinal_mode": "force",
        "ablation_variants": ["gru", "feedforward", "frame_stack"],
    }


def actor_architecture_from_module(actor: nn.Module) -> dict[str, Any]:
    """Snapshot deployable architecture fields from a built actor module."""
    if not hasattr(actor, "shapes"):
        raise TypeError("actor must expose shapes()")
    shapes = actor.shapes()
    meta = {
        "name": ACTOR_ARCHITECTURE_NAME,
        "sensor_obs_dim": shapes.sensor_obs_dim,
        "lidar_dim": shapes.lidar_dim,
        "proprio_dim": shapes.proprio_dim,
        "condition_dim": shapes.condition_dim,
        "condition_embed_dim": getattr(actor, "condition_embed_dim", CONDITION_EMBED_DIM),
        "action_dim": shapes.action_dim,
        "gru_hidden_dim": shapes.gru_hidden_dim,
        "cnn_projection_dim": shapes.cnn_projection_dim,
        "cnn_conv_channels": list(CNN_CONV_CHANNELS),
        "cnn_kernels": list(CNN_KERNELS),
        "cnn_strides": list(CNN_STRIDES),
        "cnn_padding": list(CNN_PADDING),
        "pool_bins": LIDAR_POOL_BINS,
        "mlp_sizes": list(shapes.mlp_sizes),
        "steering_action_mode": "delta",
        "longitudinal_mode": "force",
    }
    if isinstance(actor, ConditionedLidarFeedForwardActor):
        meta["name"] = (
            "lidar_cnn_frame_stack_conditioned"
            if actor.frame_stack > 1
            else "lidar_cnn_ff_conditioned"
        )
        meta["frame_stack"] = actor.frame_stack
        meta["recurrent"] = False
    else:
        meta["recurrent"] = True
        meta["frame_stack"] = 1
    return meta

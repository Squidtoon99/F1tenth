"""Training-only centralized Deep Sets value critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from gigaflow_f1tenth.buffers import STATE_INDEX
from gigaflow_f1tenth.config import (
    AGENT_STATE_DIM,
    CRITIC_ELEMENT_DIM,
    ExperimentConfig,
)
from gigaflow_f1tenth.model import ConditionEncoder, mlp, orthogonal_init_
from gigaflow_f1tenth.rewards import CONDITION_DIM
from gigaflow_f1tenth.tracks import (
    PackedTrackAtlasView,
    TRACK_PREVIEW_DIM,
    sample_track_lookahead,
)

DEFAULT_CRITIC_MLP = (1024, 1024, 1024)
CRITIC_ARCHITECTURE_NAME = "deep_sets_central_value"
CRITIC_ELEMENT_HIDDEN = 256
CRITIC_CONDITION_EMBED = 64
# Ego branch takes the compact state plus an ordered ego-relative track preview
# (never pooled: it is a concat, not a Deep Sets element). Opponent elements
# keep the raw compact-state width; only the ego branch grows.
CRITIC_EGO_STATE_DIM = AGENT_STATE_DIM + TRACK_PREVIEW_DIM


@dataclass(frozen=True)
class CriticShapes:
    """Omniscient masked agent-set critic; never serialized into deploy artifacts."""

    ego_state_dim: int
    element_dim: int
    condition_dim: int
    max_other_agents: int
    hidden_dim: int
    mlp_sizes: tuple[int, ...] = DEFAULT_CRITIC_MLP


def critic_shapes(
    cfg: ExperimentConfig,
    ego_state_dim: int = CRITIC_EGO_STATE_DIM,
    element_dim: int = CRITIC_ELEMENT_DIM,
) -> CriticShapes:
    return CriticShapes(
        ego_state_dim=ego_state_dim,
        element_dim=element_dim,
        condition_dim=cfg.agents.condition_dim,
        max_other_agents=cfg.worlds.max_agents_per_world - 1,
        hidden_dim=CRITIC_ELEMENT_HIDDEN,
        mlp_sizes=DEFAULT_CRITIC_MLP,
    )


@dataclass
class CriticOutput:
    values: Any  # [B]


@runtime_checkable
class CentralValueCritic(Protocol):
    def shapes(self) -> CriticShapes:
        ...

    def forward(
        self,
        ego_state: Any,
        other_agents: Any,
        other_mask: Any,
        private_condition: Any,
    ) -> CriticOutput:
        """
        ego_state: [B, D_ego] privileged ego features
        other_agents: [B, N, D_el] unordered opponent/track elements
        other_mask: [B, N] True for valid elements (inactive slots excluded)
        private_condition: [B, C] ego private style only
        """


class DeepSetsCentralCritic(nn.Module):
    """Masked channel-wise max-pool Deep Sets → [1024,1024,1024] scalar value."""

    def __init__(self, shapes: CriticShapes):
        super().__init__()
        if shapes.condition_dim != CONDITION_DIM:
            raise ValueError(
                f"condition_dim={shapes.condition_dim} != schema {CONDITION_DIM}"
            )
        self._shapes = shapes
        self.element_encoder = nn.Sequential(
            nn.Linear(shapes.element_dim, shapes.hidden_dim),
            nn.ReLU(),
            nn.Linear(shapes.hidden_dim, shapes.hidden_dim),
            nn.ReLU(),
        )
        self.condition_encoder = ConditionEncoder(
            shapes.condition_dim, CRITIC_CONDITION_EMBED
        )
        backbone_in = (
            shapes.ego_state_dim + shapes.hidden_dim + CRITIC_CONDITION_EMBED
        )
        self.backbone = mlp(
            [backbone_in] + list(shapes.mlp_sizes) + [1],
            nn.ReLU,
            nn.Identity,
        )
        orthogonal_init_(self)

    def shapes(self) -> CriticShapes:
        return self._shapes

    def encode_set(
        self, other_agents: torch.Tensor, other_mask: torch.Tensor
    ) -> torch.Tensor:
        """Permutation-invariant masked max pool over the agent/track set."""
        if other_agents.dim() != 3:
            raise ValueError(
                f"other_agents shape {tuple(other_agents.shape)}; expected [B,N,D]"
            )
        if other_mask.shape != other_agents.shape[:2]:
            raise ValueError(
                f"other_mask shape {tuple(other_mask.shape)}; "
                f"expected {tuple(other_agents.shape[:2])}"
            )
        emb = self.element_encoder(other_agents)
        # Inactive slots must not win the max; use a large negative fill.
        fill = torch.finfo(emb.dtype).min
        mask = other_mask.to(device=emb.device, dtype=torch.bool).unsqueeze(-1)
        emb = emb.masked_fill(~mask, fill)
        pooled = emb.max(dim=1).values
        # All-inactive rows → zeros (max over empty set).
        any_valid = mask.squeeze(-1).any(dim=-1)
        pooled = torch.where(any_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return pooled

    def forward(
        self,
        ego_state: torch.Tensor,
        other_agents: torch.Tensor,
        other_mask: torch.Tensor,
        private_condition: torch.Tensor,
    ) -> CriticOutput:
        if ego_state.shape[-1] != self._shapes.ego_state_dim:
            raise ValueError(
                f"ego_state dim {ego_state.shape[-1]} != {self._shapes.ego_state_dim}"
            )
        if private_condition.shape[-1] != self._shapes.condition_dim:
            raise ValueError(
                f"private_condition dim {private_condition.shape[-1]} != "
                f"{self._shapes.condition_dim}"
            )
        pooled = self.encode_set(other_agents, other_mask)
        cond = self.condition_encoder(private_condition)
        x = torch.cat([ego_state, pooled, cond], dim=-1)
        values = self.backbone(x).squeeze(-1)
        return CriticOutput(values=values)


def build_critic(
    cfg: ExperimentConfig,
    ego_state_dim: int = CRITIC_EGO_STATE_DIM,
    element_dim: int = CRITIC_ELEMENT_DIM,
) -> CentralValueCritic:
    shapes = critic_shapes(cfg, ego_state_dim=ego_state_dim, element_dim=element_dim)
    return DeepSetsCentralCritic(shapes)


def _masked_ego_track_preview(
    compact_state: torch.Tensor,
    *,
    track_id: torch.Tensor,
    atlas: PackedTrackAtlasView,
    active: torch.Tensor,
) -> torch.Tensor:
    """Ordered ego-relative centerline preview, zeroed for inactive slots.

    Static opponents (``trainable=0``, ``active=1``) hold a real track position,
    so their preview is meaningful and only truly inactive padding is zeroed —
    otherwise a padded slot's stale/garbage pose would pollute the value target.
    """
    vx = compact_state[..., STATE_INDEX["vx"]]
    vy = compact_state[..., STATE_INDEX["vy"]]
    speed = torch.sqrt(vx * vx + vy * vy)
    preview = sample_track_lookahead(
        atlas,
        track_id=track_id,
        s=compact_state[..., STATE_INDEX["frenet_s"]],
        x=compact_state[..., STATE_INDEX["x"]],
        y=compact_state[..., STATE_INDEX["y"]],
        yaw=compact_state[..., STATE_INDEX["yaw"]],
        speed=speed,
    )
    mask = active.to(device=preview.device, dtype=preview.dtype).unsqueeze(-1)
    return preview * mask


def pack_critic_features(
    compact_state: torch.Tensor,
    *,
    world_id: torch.Tensor,
    active: torch.Tensor,
    max_agents_per_world: int,
    track_id: torch.Tensor,
    atlas: PackedTrackAtlasView,
    centralized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack privileged critic tensors from compact [..., S, D] agent state.

    Returns:
      ego_state [..., S, D + TRACK_PREVIEW_DIM]: compact state concatenated
        with an ordered ego-relative track preview (never pooled — a Deep Sets
        max-pool would destroy arc order, so this always lands in the ego
        branch, for every ``centralized`` setting).
      other_agents [..., S, N, D]
      other_mask [..., S, N]
    """
    if compact_state.shape[-1] != AGENT_STATE_DIM:
        raise ValueError(
            f"compact_state last dim {compact_state.shape[-1]} != {AGENT_STATE_DIM}"
        )
    *prefix, num_slots, _ = compact_state.shape
    n_others = max(int(max_agents_per_world) - 1, 0)
    device = compact_state.device
    dtype = compact_state.dtype
    preview = _masked_ego_track_preview(
        compact_state, track_id=track_id, atlas=atlas, active=active
    ).to(device=device, dtype=dtype)
    ego = torch.cat([compact_state, preview], dim=-1)
    if n_others == 0 or not centralized:
        others = torch.zeros(
            *prefix, num_slots, max(n_others, 1), AGENT_STATE_DIM,
            device=device, dtype=dtype,
        )
        mask = torch.zeros(
            *prefix, num_slots, max(n_others, 1), device=device, dtype=torch.bool
        )
        if n_others == 0:
            others = others[..., :0, :]
            mask = mask[..., :0]
        return ego, others, mask

    # Device-resident pack: reshape to [B, W, A, D] and gather other slots
    # without Python/.item() host sync. Worlds are contiguous agent blocks.
    del world_id  # layout-implied by max_agents_per_world tiling
    flat = compact_state.reshape(-1, num_slots, AGENT_STATE_DIM)
    act = active.reshape(-1, num_slots).to(dtype=torch.bool)
    batch = flat.shape[0]
    n_agents = int(max_agents_per_world)
    if num_slots % n_agents != 0:
        raise ValueError("num_slots must be divisible by max_agents_per_world")
    n_worlds = num_slots // n_agents
    world = flat.view(batch, n_worlds, n_agents, AGENT_STATE_DIM)
    act_w = act.view(batch, n_worlds, n_agents)
    ego_ids = torch.arange(n_agents, device=device)
    offsets = torch.arange(1, n_agents, device=device)
    src = (ego_ids.unsqueeze(1) + offsets.unsqueeze(0)) % n_agents  # [A, N]
    gathered = world[:, :, src, :]  # [B, W, A, N, D]
    yaw = world[..., 2:3].unsqueeze(3)  # [B, W, A, 1, 1]
    dx = gathered[..., 0:1] - world[..., 0:1].unsqueeze(3)
    dy = gathered[..., 1:2] - world[..., 1:2].unsqueeze(3)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    rel_x = c * dx + s * dy
    rel_y = -s * dx + c * dy
    rel_yaw = gathered[..., 2:3] - yaw
    others = gathered.clone()
    others[..., 0:1] = rel_x
    others[..., 1:2] = rel_y
    others[..., 2:3] = rel_yaw
    oth_act = act_w[:, :, src]  # [B, W, A, N]
    mask = act_w.unsqueeze(-1) & oth_act
    others = others.reshape(batch, num_slots, n_others, AGENT_STATE_DIM)
    mask = mask.reshape(batch, num_slots, n_others)
    out_shape = (*prefix, num_slots, n_others, AGENT_STATE_DIM)
    mask_shape = (*prefix, num_slots, n_others)
    return ego, others.reshape(out_shape), mask.reshape(mask_shape)


def critic_values_over_time(
    critic: Any,
    ego_state: torch.Tensor,
    other_agents: torch.Tensor,
    other_mask: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """Score [T, S, ...] critic features into float32 values [T, S]."""
    t, s, d = ego_state.shape
    n = other_agents.shape[2]
    el = other_agents.shape[3]
    if condition.ndim == 2:
        flat_cond = condition.unsqueeze(0).expand(t, -1, -1).reshape(t * s, -1)
    else:
        flat_cond = condition.reshape(t * s, -1)
    out = critic.forward(
        ego_state.reshape(t * s, d),
        other_agents.reshape(t * s, n, el),
        other_mask.reshape(t * s, n),
        flat_cond,
    )
    values = out.values if hasattr(out, "values") else out
    # Value targets / GAE reductions stay FP32 even under BF16 autocast.
    return values.to(dtype=torch.float32).reshape(t, s)


def architecture_metadata(cfg: ExperimentConfig) -> dict[str, Any]:
    shapes = critic_shapes(cfg)
    return {
        "name": CRITIC_ARCHITECTURE_NAME,
        "training_only": True,
        "ego_state_dim": shapes.ego_state_dim,
        "ego_state_layout": {
            "compact_state_dim": AGENT_STATE_DIM,
            "track_preview_dim": TRACK_PREVIEW_DIM,
        },
        "element_dim": shapes.element_dim,
        "condition_dim": shapes.condition_dim,
        "max_other_agents": shapes.max_other_agents,
        "hidden_dim": shapes.hidden_dim,
        "mlp_sizes": list(shapes.mlp_sizes),
        "pooling": "masked_channel_max",
        "condition_embed_dim": CRITIC_CONDITION_EMBED,
    }

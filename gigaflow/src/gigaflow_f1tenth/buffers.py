"""Compact GPU rollout buffer (state-only primary design)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from gigaflow_f1tenth.config import AGENT_STATE_DIM, ExperimentConfig

# Privileged agent-state width shared with critic packing / simulator restore.
DEFAULT_AGENT_STATE_DIM = AGENT_STATE_DIM

# Compact state channel layout (float32 [S, AGENT_STATE_DIM]).
STATE_INDEX = {
    "x": 0,
    "y": 1,
    "yaw": 2,
    "vx": 3,
    "vy": 4,
    "yaw_rate": 5,
    "steer": 6,
    "applied_effort": 7,
    "ax": 8,
    "ay": 9,
    "frenet_s": 10,
    "frenet_ey": 11,
    "progress_s": 12,
    "boundary_distance": 13,
    "contact": 14,
    "wall_contact": 15,
    "contact_closing_speed": 16,
    "mass": 17,
    "mu": 18,
    "drive_scale": 19,
    "steer_scale": 20,
    "lidar_range_noise_std": 21,
    "lidar_dropout_prob": 22,
    "lidar_far_dropout_prob": 23,
    "lidar_angle_bias": 24,
    "lidar_extrinsic_x": 25,
    "lidar_extrinsic_y": 26,
    "lidar_extrinsic_yaw": 27,
    "lidar_sector_start": 28,
    "lidar_sector_width": 29,
    "active": 30,
    "stalled_steps": 31,
    "accel_scale": 32,
    "vmax_scale": 33,
    # Command history and wheel speeds feed the proprioception channels, so the
    # observation is reconstructible from state alone and need not be stored.
    "executed_long_0": 34,
    "executed_long_1": 35,
    "executed_steer_0": 36,
    "executed_steer_1": 37,
    "executed_steer_2": 38,
    "executed_steer_3": 39,
    "omega_0": 40,
    "omega_1": 41,
    "omega_2": 42,
    "omega_3": 43,
    "frenet_segment": 44,
}


@dataclass(frozen=True)
class BufferShapes:
    """Fixed-capacity shapes for flattened (world, agent_slot) storage."""

    num_worlds: int
    max_agents_per_world: int
    num_slots: int
    rollout_length: int
    action_dim: int
    gru_hidden_dim: int
    condition_dim: int
    state_dim: int


def buffer_shapes(
    cfg: ExperimentConfig,
    *,
    state_dim: int = DEFAULT_AGENT_STATE_DIM,
) -> BufferShapes:
    num_slots = cfg.worlds.num_worlds * cfg.worlds.max_agents_per_world
    return BufferShapes(
        num_worlds=cfg.worlds.num_worlds,
        max_agents_per_world=cfg.worlds.max_agents_per_world,
        num_slots=num_slots,
        rollout_length=cfg.ppo.rollout_length,
        action_dim=cfg.agents.action_dim,
        gru_hidden_dim=cfg.agents.gru_hidden_dim,
        condition_dim=cfg.agents.condition_dim,
        state_dim=int(state_dim),
    )


@dataclass
class CompactRolloutBatch:
    """Collection-time fields. Actor obs are reconstructed, never stored."""

    state: Tensor  # [T, S, state_dim]
    next_state: Tensor  # [T, S, state_dim] pre-respawn next state (GAE bootstrap)
    actions: Tensor  # [T, S, A]
    rewards: Tensor  # [T, S]
    valid: Tensor  # [T, S] trainable & active
    done: Tensor  # [T, S] true terminal
    timeout: Tensor  # [T, S] truncation for bootstrap
    reset_mask: Tensor  # [T, S] GRU clear rows
    track_id: Tensor  # [S]
    condition: Tensor  # [T, S, C] private style at action time (respawn-safe)
    sensor_noise_seed: Tensor  # [T, S]
    episode_id: Tensor  # [T, S] sensor-noise RNG stream, changes on respawn
    episode_step: Tensor  # [T, S] sensor-noise RNG stream
    gru_start: Tensor  # [S, H] rollout-start hidden state
    old_logp: Tensor  # [T, S] behavior log-prob (collection-time)
    obs_digest: Tensor  # [T, S] int64 digest of the action-time observation
    pre_tanh: Tensor  # [T, S, A] Gaussian sample before tanh squash


def observation_digest(sensor_obs: Tensor) -> Tensor:
    """Bit-pattern digest of float32 observations over the last dimension.

    Storing one int64 per agent-step instead of the 1097-wide observation keeps a
    reference the reconstruction must reproduce; summing the raw float32 words in
    int64 cannot overflow and changes whenever any word's bits change.

    Callers must always pass the *raw*, pre-normalization sensor observation.
    The actor's running sensor normalizer mutates over training, so digesting
    normalized values would make this gate depend on when it ran relative to a
    statistics update instead of on reconstruction correctness alone.
    """
    if sensor_obs.dtype != torch.float32:
        raise ValueError(f"observation digest needs float32, got {sensor_obs.dtype}")
    words = sensor_obs.contiguous().view(torch.int32)
    return words.sum(dim=-1, dtype=torch.int64)


@runtime_checkable
class RolloutBuffer(Protocol):
    def shapes(self) -> BufferShapes:
        ...

    def reset(self) -> None:
        ...

    def store_step(self, t: int, **fields: Any) -> None:
        ...

    def finalize(self) -> CompactRolloutBatch:
        ...


def pack_compact_agent_state(arrays: Any) -> Tensor:
    """Pack simulator SoA torch views into [S, AGENT_STATE_DIM] float32."""
    device = arrays.x.device
    s = int(arrays.x.shape[0])
    out = torch.zeros(s, DEFAULT_AGENT_STATE_DIM, device=device, dtype=torch.float32)
    out[:, 0] = arrays.x.to(dtype=torch.float32)
    out[:, 1] = arrays.y.to(dtype=torch.float32)
    out[:, 2] = arrays.yaw.to(dtype=torch.float32)
    out[:, 3] = arrays.vx.to(dtype=torch.float32)
    out[:, 4] = arrays.vy.to(dtype=torch.float32)
    out[:, 5] = arrays.yaw_rate.to(dtype=torch.float32)
    out[:, 6] = arrays.steer.to(dtype=torch.float32)
    out[:, 7] = arrays.applied_effort.to(dtype=torch.float32)
    out[:, 8] = arrays.ax.to(dtype=torch.float32)
    out[:, 9] = arrays.ay.to(dtype=torch.float32)
    out[:, 10] = arrays.frenet_s.to(dtype=torch.float32)
    out[:, 11] = arrays.frenet_ey.to(dtype=torch.float32)
    out[:, 12] = arrays.progress_s.to(dtype=torch.float32)
    out[:, 13] = arrays.boundary_distance.to(dtype=torch.float32)
    out[:, 14] = arrays.contact.to(dtype=torch.float32)
    out[:, 15] = arrays.wall_contact.to(dtype=torch.float32)
    out[:, 16] = arrays.contact_closing_speed.to(dtype=torch.float32)
    out[:, 17] = arrays.mass.to(dtype=torch.float32)
    out[:, 18] = arrays.mu.to(dtype=torch.float32)
    out[:, 19] = arrays.drive_scale.to(dtype=torch.float32)
    out[:, 20] = arrays.steer_scale.to(dtype=torch.float32)
    out[:, 21] = arrays.lidar_range_noise_std.to(dtype=torch.float32)
    out[:, 22] = arrays.lidar_dropout_prob.to(dtype=torch.float32)
    out[:, 23] = arrays.lidar_far_dropout_prob.to(dtype=torch.float32)
    out[:, 24] = arrays.lidar_angle_bias.to(dtype=torch.float32)
    out[:, 25] = arrays.lidar_extrinsic_x.to(dtype=torch.float32)
    out[:, 26] = arrays.lidar_extrinsic_y.to(dtype=torch.float32)
    out[:, 27] = arrays.lidar_extrinsic_yaw.to(dtype=torch.float32)
    out[:, 28] = arrays.lidar_sector_start.to(dtype=torch.float32)
    out[:, 29] = arrays.lidar_sector_width.to(dtype=torch.float32)
    out[:, 30] = arrays.active.to(dtype=torch.float32)
    out[:, 31] = arrays.stalled_steps.to(dtype=torch.float32)
    out[:, 32] = arrays.accel_scale.to(dtype=torch.float32)
    out[:, 33] = arrays.vmax_scale.to(dtype=torch.float32)
    out[:, 34] = arrays.executed_long_0.to(dtype=torch.float32)
    out[:, 35] = arrays.executed_long_1.to(dtype=torch.float32)
    out[:, 36] = arrays.executed_steer_0.to(dtype=torch.float32)
    out[:, 37] = arrays.executed_steer_1.to(dtype=torch.float32)
    out[:, 38] = arrays.executed_steer_2.to(dtype=torch.float32)
    out[:, 39] = arrays.executed_steer_3.to(dtype=torch.float32)
    out[:, 40:44] = arrays.omega.to(dtype=torch.float32)
    out[:, 44] = arrays.frenet_segment.to(dtype=torch.float32)
    return out


def unpack_compact_agent_state(state: Tensor, arrays: Any) -> None:
    """Write packed [S, AGENT_STATE_DIM] state back into simulator SoA torch views."""
    if state.ndim != 2 or state.shape[-1] != DEFAULT_AGENT_STATE_DIM:
        raise ValueError(
            f"state shape {tuple(state.shape)}; expected (S, {DEFAULT_AGENT_STATE_DIM})"
        )
    s = state.to(device=arrays.x.device, dtype=torch.float32)
    arrays.x.copy_(s[:, 0])
    arrays.y.copy_(s[:, 1])
    arrays.yaw.copy_(s[:, 2])
    arrays.vx.copy_(s[:, 3])
    arrays.vy.copy_(s[:, 4])
    arrays.yaw_rate.copy_(s[:, 5])
    arrays.steer.copy_(s[:, 6])
    arrays.applied_effort.copy_(s[:, 7])
    arrays.ax.copy_(s[:, 8])
    arrays.ay.copy_(s[:, 9])
    arrays.frenet_s.copy_(s[:, 10])
    arrays.frenet_ey.copy_(s[:, 11])
    arrays.progress_s.copy_(s[:, 12])
    arrays.boundary_distance.copy_(s[:, 13])
    arrays.contact.copy_(s[:, 14].to(dtype=arrays.contact.dtype))
    arrays.wall_contact.copy_(s[:, 15].to(dtype=arrays.wall_contact.dtype))
    arrays.contact_closing_speed.copy_(s[:, 16])
    arrays.mass.copy_(s[:, 17])
    arrays.mu.copy_(s[:, 18])
    arrays.drive_scale.copy_(s[:, 19])
    arrays.steer_scale.copy_(s[:, 20])
    arrays.lidar_range_noise_std.copy_(s[:, 21])
    arrays.lidar_dropout_prob.copy_(s[:, 22])
    arrays.lidar_far_dropout_prob.copy_(s[:, 23])
    arrays.lidar_angle_bias.copy_(s[:, 24])
    arrays.lidar_extrinsic_x.copy_(s[:, 25])
    arrays.lidar_extrinsic_y.copy_(s[:, 26])
    arrays.lidar_extrinsic_yaw.copy_(s[:, 27])
    arrays.lidar_sector_start.copy_(
        s[:, 28].round().to(dtype=arrays.lidar_sector_start.dtype)
    )
    arrays.lidar_sector_width.copy_(
        s[:, 29].round().to(dtype=arrays.lidar_sector_width.dtype)
    )
    arrays.active.copy_(s[:, 30].round().clamp(0, 1).to(dtype=arrays.active.dtype))
    arrays.stalled_steps.copy_(
        s[:, 31].round().to(dtype=arrays.stalled_steps.dtype)
    )
    arrays.accel_scale.copy_(s[:, 32])
    arrays.vmax_scale.copy_(s[:, 33])
    arrays.executed_long_0.copy_(s[:, 34])
    arrays.executed_long_1.copy_(s[:, 35])
    arrays.executed_steer_0.copy_(s[:, 36])
    arrays.executed_steer_1.copy_(s[:, 37])
    arrays.executed_steer_2.copy_(s[:, 38])
    arrays.executed_steer_3.copy_(s[:, 39])
    arrays.omega.copy_(s[:, 40:44])
    arrays.frenet_segment.copy_(
        s[:, 44].round().to(dtype=arrays.frenet_segment.dtype)
    )


class TensorRolloutBuffer:
    """Fixed-shape, preallocated compact rollout storage (CUDA-graph friendly)."""

    def __init__(
        self,
        shapes: BufferShapes,
        device: str | torch.device,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._shapes = shapes
        self.device = torch.device(device)
        self.dtype = dtype
        t = shapes.rollout_length
        s = shapes.num_slots
        a = shapes.action_dim
        h = shapes.gru_hidden_dim
        c = shapes.condition_dim
        sd = shapes.state_dim
        self.state = torch.zeros(t, s, sd, device=self.device, dtype=dtype)
        self.next_state = torch.zeros(t, s, sd, device=self.device, dtype=dtype)
        self.actions = torch.zeros(t, s, a, device=self.device, dtype=dtype)
        self.rewards = torch.zeros(t, s, device=self.device, dtype=dtype)
        self.valid = torch.zeros(t, s, device=self.device, dtype=torch.bool)
        self.done = torch.zeros(t, s, device=self.device, dtype=torch.bool)
        self.timeout = torch.zeros(t, s, device=self.device, dtype=torch.bool)
        self.reset_mask = torch.zeros(t, s, device=self.device, dtype=torch.bool)
        self.track_id = torch.zeros(s, device=self.device, dtype=torch.int32)
        self.condition = torch.zeros(t, s, c, device=self.device, dtype=dtype)
        self.sensor_noise_seed = torch.zeros(
            t, s, device=self.device, dtype=torch.int64
        )
        self.episode_id = torch.zeros(t, s, device=self.device, dtype=torch.int32)
        self.episode_step = torch.zeros(t, s, device=self.device, dtype=torch.int32)
        self.gru_start = torch.zeros(s, h, device=self.device, dtype=dtype)
        self.old_logp = torch.zeros(t, s, device=self.device, dtype=dtype)
        self.obs_digest = torch.zeros(t, s, device=self.device, dtype=torch.int64)
        self.pre_tanh = torch.zeros(t, s, a, device=self.device, dtype=dtype)
        self._filled = torch.zeros(t, device=self.device, dtype=torch.bool)

    def shapes(self) -> BufferShapes:
        return self._shapes

    def reset(self) -> None:
        self.state.zero_()
        self.next_state.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.valid.zero_()
        self.done.zero_()
        self.timeout.zero_()
        self.reset_mask.zero_()
        self.track_id.zero_()
        self.condition.zero_()
        self.sensor_noise_seed.zero_()
        self.episode_id.zero_()
        self.episode_step.zero_()
        self.gru_start.zero_()
        self.old_logp.zero_()
        self.obs_digest.zero_()
        self.pre_tanh.zero_()
        self._filled.zero_()

    def store_step(self, t: int, **fields: Any) -> None:
        t = int(t)
        if t < 0 or t >= self._shapes.rollout_length:
            raise IndexError(
                f"step t={t} out of range for rollout_length="
                f"{self._shapes.rollout_length}"
            )
        for key, value in fields.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown rollout field {key!r}")
            if key in ("track_id", "gru_start"):
                # Slot-level fields may be written once or refreshed each step.
                getattr(self, key).copy_(torch.as_tensor(value, device=self.device))
            else:
                slot = getattr(self, key)
                slot[t].copy_(torch.as_tensor(value, device=self.device))
        self._filled[t] = True

    def set_rollout_start_hidden(self, hidden: Tensor) -> None:
        self.gru_start.copy_(hidden.to(device=self.device, dtype=self.dtype))

    def finalize(self) -> CompactRolloutBatch:
        if not bool(self._filled.all()):
            missing = (~self._filled).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"rollout incomplete; missing steps {missing}")
        return CompactRolloutBatch(
            state=self.state,
            next_state=self.next_state,
            actions=self.actions,
            rewards=self.rewards,
            valid=self.valid,
            done=self.done,
            timeout=self.timeout,
            reset_mask=self.reset_mask,
            track_id=self.track_id,
            condition=self.condition,
            sensor_noise_seed=self.sensor_noise_seed,
            episode_id=self.episode_id,
            episode_step=self.episode_step,
            gru_start=self.gru_start,
            old_logp=self.old_logp,
            obs_digest=self.obs_digest,
            pre_tanh=self.pre_tanh,
        )


def allocate_rollout_buffer(
    cfg: ExperimentConfig,
    device: str,
    *,
    state_dim: int = DEFAULT_AGENT_STATE_DIM,
) -> TensorRolloutBuffer:
    shapes = buffer_shapes(cfg, state_dim=state_dim)
    return TensorRolloutBuffer(shapes, device=device)

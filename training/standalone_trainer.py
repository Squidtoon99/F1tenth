#!/usr/bin/env python3
"""Single-process QRSAC trainer: F1tenthEnv + trajectory replay, no Reverb/Redis/S3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F

from f1tenth_policy import (
    ObsNormalizer,
    actor_architecture_from_module,
    actor_from_architecture,
    architectures_match,
    build_sensor_artifact_payload,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    SENSOR_POLICY_FORMAT_VERSION,
    STEERING_ACTION_MODE,
)

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.sensors import ACTOR_LIDAR_DIM
from f1tenth_env.utils import episode_length_for_track
from evaluation import actor_is_recurrent, deterministic_rollout
from fixed_opponents import FixedChampionManager, load_opponent_pool
from run_layout import checkpoint_dir, config_snapshot_path, default_run_dir, run_log_path
from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from qrsac.replay import (
    REPLAY_BURN_IN,
    REPLAY_CHECKPOINT_INTERVAL,
    REPLAY_HIDDEN_DTYPE,
    REPLAY_OBS_DTYPE,
    REPLAY_TRAIN_LEN,
    TrajectoryReplayBuffer,
)
from qrsac.spinningup.core import ALLOWED_LIDAR_POOL_BINS, GRU_HIDDEN_DIM

# Compatibility alias used by recurrent actor / artifact tests.
reference_actor_from_architecture = actor_from_architecture
_REEXPORTS = (architectures_match, SENSOR_POLICY_FORMAT_VERSION)

LOGGER_NAME = "standalone_trainer"
# Privileged opponent block [384:392): rel_xy, rel_vxy, rel_axy, gap_norm, ey.
# gap_norm (index +6) is (s_other - s_self) / (0.5 * track_length).
OPP_OBS_BASE_IDX = 384
OPP_OBS_DIM = 8
OPP_TRACK_GAP_IDX = OPP_OBS_BASE_IDX + 6
OPP_OBS_END_IDX = OPP_OBS_BASE_IDX + OPP_OBS_DIM

# Dual-observation replay: only these capacities are chosen automatically.
REPLAY_CAPACITY_REQUESTED = 2_000_000
REPLAY_CAPACITY_FALLBACK = 1_000_000


def _tensor_shape(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record so lines appear promptly."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_trainer_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Dedicated trainer logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler = FlushingStreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def estimate_dual_replay_bytes(
    capacity: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    n_step: int,
    num_envs: int,
    *,
    obs_dtype: torch.dtype = REPLAY_OBS_DTYPE,
    burn_in: int = REPLAY_BURN_IN,
    train_len: int = REPLAY_TRAIN_LEN,
    checkpoint_interval: int = REPLAY_CHECKPOINT_INTERVAL,
    hidden_dim: int = GRU_HIDDEN_DIM,
) -> dict[str, int]:
    """Byte estimate for per-env trajectory replay (float16 obs + hidden ckpts)."""
    obs_item = torch.tensor([], dtype=obs_dtype).element_size()
    h_item = torch.tensor([], dtype=REPLAY_HIDDEN_DTYPE).element_size()
    f32 = torch.tensor([], dtype=torch.float32).element_size()
    i64 = torch.tensor([], dtype=torch.long).element_size()
    raw_steps = int(capacity) // int(num_envs)
    steps_per_env = (raw_steps // int(checkpoint_interval)) * int(checkpoint_interval)
    effective = steps_per_env * int(num_envs)
    num_ckpt = steps_per_env // int(checkpoint_interval) if steps_per_env else 0
    ring_obs = (
        int(num_envs) * steps_per_env * (actor_obs_dim + critic_obs_dim) * obs_item
    )
    ring_aux = int(num_envs) * steps_per_env * (
        act_dim * f32 + f32 + f32 + 1 + 1 + i64
    )  # action, reward, done, reset(bool~1), opponent_visible(bool~1), episode_id
    hidden_bytes = int(num_envs) * num_ckpt * int(hidden_dim) * h_item
    seq_len = int(burn_in) + int(train_len) + int(n_step)
    total = ring_obs + ring_aux + hidden_bytes
    return {
        "capacity": int(effective),
        "requested_capacity": int(capacity),
        "steps_per_env": int(steps_per_env),
        "num_checkpoints": int(num_ckpt),
        "seq_len": int(seq_len),
        "ring_obs_bytes": int(ring_obs),
        "ring_aux_bytes": int(ring_aux),
        "hidden_bytes": int(hidden_bytes),
        "total_bytes": int(total),
    }


def _update_headroom_ok(
    device: torch.device,
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    batch_size: int,
    *,
    num_sequences: int | None = None,
    seq_len: int = REPLAY_BURN_IN + REPLAY_TRAIN_LEN + 7,
    hidden_dim: int = GRU_HIDDEN_DIM,
) -> bool:
    """True when a float32 sequence batch still fits after replay alloc."""
    if device.type != "cuda":
        return True
    n_seq = int(num_sequences) if num_sequences is not None else max(
        1, int(batch_size) // REPLAY_TRAIN_LEN
    )
    try:
        torch.empty(
            n_seq, seq_len, actor_obs_dim, device=device, dtype=torch.float32
        )
        torch.empty(
            n_seq, seq_len, critic_obs_dim, device=device, dtype=torch.float32
        )
        torch.empty(n_seq, seq_len, act_dim, device=device, dtype=torch.float32)
        torch.empty(n_seq, hidden_dim, device=device, dtype=torch.float32)
        return True
    except torch.cuda.OutOfMemoryError:
        return False


def make_dual_replay_buffer(
    *,
    capacity: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    n_step: int,
    gamma: float,
    num_envs: int,
    device: torch.device,
    batch_size: int,
    log: logging.Logger | None = None,
    allow_fallback: bool = True,
    burn_in: int = REPLAY_BURN_IN,
    train_len: int = REPLAY_TRAIN_LEN,
    checkpoint_interval: int = REPLAY_CHECKPOINT_INTERVAL,
    hidden_dim: int = GRU_HIDDEN_DIM,
) -> tuple[TrajectoryReplayBuffer, int, dict[str, int]]:
    """Allocate trajectory replay at ``capacity``, with explicit 2M→1M CUDA fallback.

    Only ``REPLAY_CAPACITY_REQUESTED`` may fall back to ``REPLAY_CAPACITY_FALLBACK``.
    Any other requested capacity is used as-is (tests) or re-raised on failure.
    """
    logger = log or logging.getLogger(LOGGER_NAME)
    estimate = estimate_dual_replay_bytes(
        capacity,
        actor_obs_dim,
        critic_obs_dim,
        act_dim,
        n_step,
        num_envs,
        burn_in=burn_in,
        train_len=train_len,
        checkpoint_interval=checkpoint_interval,
        hidden_dim=hidden_dim,
    )
    logger.info(
        "Replay allocation estimate: capacity=%d steps_per_env=%d total_bytes=%d "
        "(%.2f GiB) ring_obs_bytes=%d hidden_bytes=%d actor_dim=%d critic_dim=%d "
        "dtype=%s seq_len=%d",
        estimate["capacity"],
        estimate["steps_per_env"],
        estimate["total_bytes"],
        estimate["total_bytes"] / (1024**3),
        estimate["ring_obs_bytes"],
        estimate["hidden_bytes"],
        actor_obs_dim,
        critic_obs_dim,
        str(REPLAY_OBS_DTYPE).replace("torch.", ""),
        estimate["seq_len"],
    )

    def _alloc(cap: int) -> TrajectoryReplayBuffer:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return TrajectoryReplayBuffer(
            capacity=cap,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            act_dim=act_dim,
            n_step=n_step,
            gamma=gamma,
            num_envs=num_envs,
            device=device,
            burn_in=burn_in,
            train_len=train_len,
            checkpoint_interval=checkpoint_interval,
            hidden_dim=hidden_dim,
            obs_dtype=REPLAY_OBS_DTYPE,
        )

    seq_len = int(burn_in) + int(train_len) + int(n_step)
    num_sequences = max(1, int(batch_size) // int(train_len))

    try:
        buffer = _alloc(capacity)
    except torch.cuda.OutOfMemoryError:
        if (
            not allow_fallback
            or capacity != REPLAY_CAPACITY_REQUESTED
            or device.type != "cuda"
        ):
            raise
        logger.warning(
            "CUDA OOM allocating replay capacity=%d; falling back to %d",
            capacity,
            REPLAY_CAPACITY_FALLBACK,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        capacity = REPLAY_CAPACITY_FALLBACK
        estimate = estimate_dual_replay_bytes(
            capacity,
            actor_obs_dim,
            critic_obs_dim,
            act_dim,
            n_step,
            num_envs,
            burn_in=burn_in,
            train_len=train_len,
            checkpoint_interval=checkpoint_interval,
            hidden_dim=hidden_dim,
        )
        logger.info(
            "Replay fallback estimate: capacity=%d total_bytes=%d (%.2f GiB)",
            estimate["capacity"],
            estimate["total_bytes"],
            estimate["total_bytes"] / (1024**3),
        )
        buffer = _alloc(capacity)
    else:
        if not _update_headroom_ok(
            device,
            actor_obs_dim,
            critic_obs_dim,
            act_dim,
            batch_size,
            num_sequences=num_sequences,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
        ):
            if (
                allow_fallback
                and capacity == REPLAY_CAPACITY_REQUESTED
                and device.type == "cuda"
            ):
                logger.warning(
                    "Insufficient CUDA headroom for a sequence update at "
                    "capacity=%d; falling back to %d",
                    capacity,
                    REPLAY_CAPACITY_FALLBACK,
                )
                del buffer
                torch.cuda.empty_cache()
                capacity = REPLAY_CAPACITY_FALLBACK
                estimate = estimate_dual_replay_bytes(
                    capacity,
                    actor_obs_dim,
                    critic_obs_dim,
                    act_dim,
                    n_step,
                    num_envs,
                    burn_in=burn_in,
                    train_len=train_len,
                    checkpoint_interval=checkpoint_interval,
                    hidden_dim=hidden_dim,
                )
                buffer = _alloc(capacity)
            else:
                raise RuntimeError(
                    f"Insufficient device headroom for QR-SAC update after "
                    f"allocating replay capacity={capacity}."
                )

    selected = int(buffer.capacity)
    logger.info("Replay capacity selected: %d", selected)
    return buffer, selected, estimate


def augment_actor_lidar_beam_shift(
    actor_obs: torch.Tensor,
    *,
    max_shift_beams: int,
    shifts: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one reflected-pad angular LiDAR shift per sequence window.

    ``actor_obs`` is ``(B, T, actor_dim)``. The same shift is used for every
    frame in a window so the GRU does not see artificial yaw jitter.
    """
    max_shift = int(max_shift_beams)
    if max_shift <= 0:
        return actor_obs
    if actor_obs.dim() != 3 or actor_obs.shape[-1] < ACTOR_LIDAR_DIM:
        raise ValueError(
            f"actor_obs shape={tuple(actor_obs.shape)}; expected (B, T, >= "
            f"{ACTOR_LIDAR_DIM})"
        )
    batch = actor_obs.shape[0]
    lidar = actor_obs[..., :ACTOR_LIDAR_DIM]
    proprio = actor_obs[..., ACTOR_LIDAR_DIM:]
    if shifts is None:
        shifts = torch.randint(
            -max_shift,
            max_shift + 1,
            (batch,),
            device=actor_obs.device,
            dtype=torch.long,
            generator=generator,
        )
    else:
        shifts = shifts.to(device=actor_obs.device, dtype=torch.long)
        if shifts.shape != (batch,):
            raise ValueError(
                f"shifts shape={tuple(shifts.shape)}; expected ({batch},)"
            )
    padded = F.pad(lidar, (max_shift, max_shift), mode="reflect")
    starts = max_shift + shifts
    index = starts.view(batch, 1, 1) + torch.arange(
        ACTOR_LIDAR_DIM, device=actor_obs.device, dtype=torch.long
    ).view(1, 1, ACTOR_LIDAR_DIM)
    index = index.expand(batch, actor_obs.shape[1], ACTOR_LIDAR_DIM)
    shifted = torch.gather(padded, dim=-1, index=index)
    return torch.cat([shifted, proprio], dim=-1)


def unpack_sensor_observations(
    obs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a ``with_sensors=True`` env return into actor/critic float32 tensors."""
    if not isinstance(obs, dict):
        raise TypeError(
            "Asymmetric trainer requires with_sensors=True dict observations "
            f"(got {type(obs)!r}). Legacy flat observations are not supported."
        )
    if "actor" not in obs or "frenet" not in obs:
        raise KeyError(
            "Sensor observation dict must contain 'actor' and 'frenet' "
            f"(keys={sorted(obs)})."
        )
    return obs["actor"].to(torch.float32), obs["frenet"].to(torch.float32)


def learner_updates_for_transitions(
    collected_transitions: int,
    batch_size: int,
    sampled_rows_per_transition: float,
    row_budget: float,
) -> tuple[int, float]:
    row_budget += collected_transitions * sampled_rows_per_transition
    updates = int(row_budget // batch_size)
    return updates, row_budget - updates * batch_size


def interval_crossed(previous: int, current: int, interval: int) -> bool:
    return interval > 0 and current // interval > previous // interval


def training_should_continue(
    env_transitions: int, total_transitions: int, continuous: bool
) -> bool:
    """Finite budget terminates by default; ``continuous`` ignores the cap."""
    return bool(continuous) or env_transitions < total_transitions


class RunningStats:
    """Accumulates scalar means / min / max / totals for named diagnostics.

    Values are kept as on-device tensors and only synced to Python floats at
    log time to avoid a host sync on every environment step.
    """

    def __init__(self):
        self._sum: dict[str, torch.Tensor] = {}
        self._count: dict[str, int | torch.Tensor] = {}
        self._min: dict[str, torch.Tensor] = {}
        self._max: dict[str, torch.Tensor] = {}

    def add_mean(
        self, key: str, value: torch.Tensor, *, track_range: bool = False
    ) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.mean()
        self._count[key] = self._count.get(key, 0) + 1
        if track_range:
            vmin, vmax = v.min(), v.max()
            self._min[key] = (
                vmin if key not in self._min else torch.minimum(self._min[key], vmin)
            )
            self._max[key] = (
                vmax if key not in self._max else torch.maximum(self._max[key], vmax)
            )

    def add_event_mean(
        self, key: str, value: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Accumulate a per-event mean over rows where ``mask`` is true."""
        v = value.detach().reshape(-1).float()
        m = mask.detach().reshape(-1).to(v.dtype)
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + (v * m).sum()
        self._count[key] = self._count.get(key, 0) + m.sum()

    def add_total(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.sum()

    def mean(self, key: str) -> float:
        count = float(self._count.get(key, 0))
        if count == 0:
            return float("nan")
        return float(self._sum[key]) / count

    def total(self, key: str) -> float:
        return float(self._sum[key]) if key in self._sum else 0.0

    def vmin(self, key: str) -> float:
        return float(self._min[key]) if key in self._min else float("nan")

    def vmax(self, key: str) -> float:
        return float(self._max[key]) if key in self._max else float("nan")

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()
        self._min.clear()
        self._max.clear()


def accumulate_step_diagnostics(
    diag: RunningStats,
    reward: torch.Tensor,
    actions: torch.Tensor,
    obs: torch.Tensor,
    extras: dict,
) -> None:
    """Fold one env step's reward terms, metrics and terminations into diag."""
    diag.add_mean("reward/step", reward, track_range=True)

    terms = extras.get("rewards", {}).get("terms", {})
    for name, value in terms.items():
        if isinstance(value, torch.Tensor):
            diag.add_mean(f"reward_term/{name}", value)

    metrics = extras.get("metrics", {})
    for name in (
        "speed_xy",
        "lateral_error",
        "oob_mask",
        "progress_ds",
        "lap_count",
        "laps_completed",
        "opp_speed",
        "nonfinite_obs_envs",
        "nonfinite_reward_envs",
        "nonfinite_state_envs",
    ):
        value = metrics.get(name)
        if isinstance(value, torch.Tensor):
            if name.startswith("nonfinite_") or name == "laps_completed":
                diag.add_total(f"metric/{name}", value)
            else:
                diag.add_mean(f"metric/{name}", value)

    oob_mask = metrics.get("oob_mask")
    oob_penalty = terms.get("oob_penalty")
    if isinstance(oob_mask, torch.Tensor) and isinstance(oob_penalty, torch.Tensor):
        diag.add_event_mean(
            "reward_term/oob_penalty_when_oob", oob_penalty, oob_mask > 0
        )
    oob_impact = terms.get("oob_impact")
    if isinstance(oob_impact, torch.Tensor):
        oob_impact_events = oob_impact != 0
        diag.add_total(
            "metric/oob_impact_events", oob_impact_events.to(oob_impact.dtype)
        )
        diag.add_event_mean(
            "reward_term/oob_impact_when_event", oob_impact, oob_impact_events
        )
    boundary = terms.get("boundary_contact")
    if isinstance(boundary, torch.Tensor):
        boundary_events = boundary != 0
        diag.add_total(
            "metric/boundary_contact_events",
            boundary_events.to(boundary.dtype),
        )
        diag.add_event_mean(
            "reward_term/boundary_contact_when_event",
            boundary,
            boundary_events,
        )

    for name, value in extras.get("termination", {}).items():
        if isinstance(value, torch.Tensor):
            diag.add_total(f"term/{name}", value)

    if actions.ndim == 2 and actions.shape[1] >= 2:
        diag.add_mean("action/throttle", actions[:, 0], track_range=True)
        diag.add_mean("action/steer", actions[:, 1], track_range=True)
    diag.add_mean("obs/abs", obs.abs(), track_range=True)


def accumulate_completed_episode_lifespans(
    diag: RunningStats,
    completed_episode_steps: torch.Tensor,
    done: torch.Tensor,
    control_dt: float,
) -> None:
    """Accumulate exact pre-reset lifespans for completed episodes."""
    lifespan_s = completed_episode_steps.to(torch.float32) * float(control_dt)
    diag.add_event_mean("episode/lifespan_s", lifespan_s, done)


TRAINING_SUMMARY_REWARD_TAIL = (
    "policy_loss=%.4f critic_loss=%.4f mean_ep_reward=%.4f "
    "episode_lifespan=%.3fs (n=%d)"
)


def training_summary_reward_tail_args(
    *,
    mean_policy_loss: float,
    mean_critic_loss: float,
    mean_ep_reward: float,
    diag: RunningStats,
    ep_count: int,
) -> tuple[float, float, float, float, int]:
    return (
        mean_policy_loss,
        mean_critic_loss,
        mean_ep_reward,
        diag.mean("episode/lifespan_s"),
        int(ep_count),
    )


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into ``base`` in place, returning ``base``.

    Nested mappings are merged key-by-key; every other value (including lists)
    replaces the base value wholesale (deep-copied so the patch is not aliased).
    """
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def initial_training_protocol_state() -> dict:
    """Process-local one-shot protocol flags (unset on every cold start)."""
    return {
        "replay_full_reinit_done": False,
        "replay_full_reinit_count": 0,
        "replay_full_reinit_transitions": None,
    }


def maybe_replay_full_reinit(
    *,
    enabled: bool,
    protocol_state: dict,
    buffer: TrajectoryReplayBuffer,
    trainer: QRSACTrainer,
    models: Models,
    actor_normalizer: ObsNormalizer,
    env_transitions: int,
    learner_hidden: torch.Tensor | None,
    champion_mgr: FixedChampionManager | None = None,
    env=None,
    log: logging.Logger | None = None,
    wandb_run=None,
) -> bool:
    """Trigger Lee et al. 2025 replay-full reinit exactly once when buffer fills.

    The fixed champion is immutable across reinitialization.
    """
    del actor_normalizer  # norms retained; signature kept for call-site stability
    if not enabled or protocol_state.get("replay_full_reinit_done"):
        return False
    if int(buffer.size) < int(buffer.capacity):
        return False
    trainer.reinitialize_networks()
    if learner_hidden is not None:
        learner_hidden.zero_()
    # Fixed champion must not change on replay-full reinit.
    if env is not None and champion_mgr is not None:
        champion_mgr.bootstrap_opponent(env, resample=True)
    protocol_state["replay_full_reinit_done"] = True
    protocol_state["replay_full_reinit_count"] = (
        int(protocol_state.get("replay_full_reinit_count", 0)) + 1
    )
    protocol_state["replay_full_reinit_transitions"] = int(env_transitions)
    logger = log or logging.getLogger(LOGGER_NAME)
    logger.info(
        "Replay-full network reinitialization (Lee et al. 2025): "
        "actor/critics/targets/Adam reset at transitions=%d buffer=%d/%d "
        "reinit_count=%d; replay + obs normalizers retained; live GRU cleared; "
        "fixed champion unchanged",
        env_transitions,
        int(buffer.size),
        int(buffer.capacity),
        protocol_state["replay_full_reinit_count"],
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/replay_full_reinit": 1,
                "train/replay_full_reinit_count": protocol_state[
                    "replay_full_reinit_count"
                ],
                "train/replay_full_reinit_transitions": env_transitions,
            },
            step=env_transitions,
        )
    return True


def validate_model_architecture(cfg: dict) -> None:
    """Reject unsupported actor types / pool widths before network construction."""
    model = cfg["model"]
    if "hidden_layers" in model:
        raise ValueError(
            "model.hidden_layers is no longer supported; set both "
            "model.actor_hidden_layers and model.critic_hidden_layers"
        )
    actor_type = model.get("actor_type")
    if actor_type != ACTOR_ARCHITECTURE_NAME:
        raise ValueError(
            f"Unsupported model.actor_type={actor_type!r}; "
            f"expected {ACTOR_ARCHITECTURE_NAME!r}"
        )
    pool_bins = model.get("lidar_pool_bins")
    if pool_bins not in ALLOWED_LIDAR_POOL_BINS:
        raise ValueError(
            f"model.lidar_pool_bins must be one of "
            f"{sorted(ALLOWED_LIDAR_POOL_BINS)}, got {pool_bins!r}"
        )
    for key in ("actor_hidden_layers", "critic_hidden_layers"):
        layers = model.get(key)
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError(f"model.{key} must be a non-empty list of ints")
        if any(int(width) <= 0 for width in layers):
            raise ValueError(f"model.{key} entries must be positive ints")
    for lr_key in ("actor_lr", "critic_lr"):
        lr = model.get(lr_key)
        if not isinstance(lr, (int, float)) or float(lr) <= 0.0:
            raise ValueError(f"model.{lr_key} must be a positive float, got {lr!r}")
    if not isinstance(model.get("replay_full_reinit"), bool):
        raise ValueError(
            "model.replay_full_reinit must be a bool, got "
            f"{model.get('replay_full_reinit')!r}"
        )
    if not isinstance(model.get("lidar_aug_enabled"), bool):
        raise ValueError(
            "model.lidar_aug_enabled must be a bool, got "
            f"{model.get('lidar_aug_enabled')!r}"
        )
    max_shift = model.get("lidar_aug_max_shift_beams")
    if not isinstance(max_shift, int) or max_shift < 0:
        raise ValueError(
            "model.lidar_aug_max_shift_beams must be a non-negative int, got "
            f"{max_shift!r}"
        )
    if max_shift >= ACTOR_LIDAR_DIM:
        raise ValueError(
            f"model.lidar_aug_max_shift_beams={max_shift} must be < lidar_dim="
            f"{ACTOR_LIDAR_DIM}"
        )


def make_policy_network(cfg: dict):
    validate_model_architecture(cfg)
    model = cfg["model"]
    return make_actor(
        actor_type=model["actor_type"],
        obs_dim=cfg["obs"]["num_actor_obs"],
        act_dim=cfg["env"]["num_actions"],
        hidden_sizes=model["actor_hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=model["lidar_pool_bins"],
    )


def make_q_network(cfg: dict) -> QuantileCritic:
    validate_model_architecture(cfg)
    return QuantileCritic(
        obs_dim=cfg["obs"]["num_obs"],
        act_dim=cfg["env"]["num_actions"],
        hidden_sizes=cfg["model"]["critic_hidden_layers"],
        num_quantiles=cfg["model"]["num_quantiles"],
    )


def make_target_q_network(cfg: dict) -> QuantileCritic:
    target_q = make_q_network(cfg)
    for param in target_q.parameters():
        param.requires_grad = False
    return target_q


def build_env_cfg(cfg: dict, **extra) -> dict:
    """Assemble ``env_cfg`` for ``F1tenthEnv``, including root-level ``sensor``.

    Sensor settings live at ``cfg["sensor"]`` (not under ``env``) so JSON patches
    to ``sensor.*`` merge through ``build_config``. Callers must use this helper
    (or equivalent) rather than passing ``cfg["env"]`` alone.

    Actor architecture keys are forwarded so ``PolicyOpponent`` is built with the
    same concrete actor as the learner (homogeneous self-play).
    """
    model = cfg["model"]
    return {
        **extra,
        **cfg["env"],
        "sensor": cfg["sensor"],
        "actor_type": model["actor_type"],
        "actor_hidden_layers": list(model["actor_hidden_layers"]),
        "lidar_pool_bins": int(model["lidar_pool_bins"]),
    }


# Gated reward-scale coefficients that compute_rewards enables purely by their
# presence in reward_scales, so they are intentionally absent from DEFAULT_CONFIG
# (adding them there would activate the term). build_config injects them for 1v1,


# and a --config patch may set them, so the validator accepts them under
# reward.reward_scales even though they are not in the reference shape.
_OPTIONAL_REWARD_SCALE_KEYS = frozenset({"passing", "collision", "rear_end"})


def validate_config_patch(
    patch: dict, reference: dict = DEFAULT_CONFIG, path: str = ""
) -> None:
    """Reject a JSON patch that strays from the ``DEFAULT_CONFIG`` shape.

    A key is rejected when it is absent from ``reference`` at the same nesting
    depth, or when it maps a mapping onto a scalar (or vice-versa). Both are
    raised early with the offending dotted path so a typo cannot silently create
    an ignored config key. Known optional gated reward-scale keys (enabled by
    presence, so absent from ``DEFAULT_CONFIG``) are accepted under
    ``reward.reward_scales`` as scalars. ``model.hidden_layers`` is rejected with
    an actionable split-key message rather than a generic unknown-key error.
    """
    if not isinstance(patch, dict):
        raise ValueError(
            f"config patch at '{path or '<root>'}' must be a JSON object, "
            f"got {type(patch).__name__}"
        )
    for key, value in patch.items():
        loc = f"{path}.{key}" if path else key
        if key not in reference:
            if path == "model" and key == "hidden_layers":
                raise ValueError(
                    "model.hidden_layers is no longer supported; set both "
                    "model.actor_hidden_layers and model.critic_hidden_layers"
                )
            if path == "reward.reward_scales" and key in _OPTIONAL_REWARD_SCALE_KEYS:
                if isinstance(value, dict):
                    raise ValueError(
                        f"type mismatch for config key '{loc}': expected a "
                        f"scalar, got a mapping"
                    )
                continue
            raise ValueError(f"unknown config key '{loc}' (not in DEFAULT_CONFIG)")
        ref_val = reference[key]
        ref_is_map = isinstance(ref_val, dict)
        val_is_map = isinstance(value, dict)
        if ref_is_map != val_is_map:
            raise ValueError(
                f"type mismatch for config key '{loc}': expected "
                f"{'a mapping' if ref_is_map else 'a scalar'}, got "
                f"{'a mapping' if val_is_map else 'a scalar'}"
            )
        if ref_is_map:
            validate_config_patch(value, ref_val, loc)


def load_config_patch(path: str) -> tuple[dict, dict]:
    """Read, parse and validate a JSON config patch.

    Returns the parsed patch and a provenance record (absolute path, sha256 of
    the raw file bytes, and the parsed contents) for the run snapshot / W&B.
    """
    raw = Path(path).read_bytes()
    patch = json.loads(raw)
    validate_config_patch(patch)
    meta = {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "contents": patch,
    }
    return patch, meta


def config_provenance(patch_meta: dict | None, explicit: set[str]) -> dict:
    """Assemble the config-resolution provenance recorded with every run."""
    return {
        "precedence": ["DEFAULT_CONFIG", "config_patch", "cli_args"],
        "patch": patch_meta,
        "explicit_cli_args": sorted(explicit),
    }


def build_run_snapshot(
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
    cfg: dict,
    provenance: dict,
) -> dict:
    """Serializable run snapshot: resolved config plus config provenance."""
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "config": cfg,
        "config_provenance": provenance,
    }


def build_config(
    args: argparse.Namespace,
    patch: dict | None = None,
    explicit: set[str] | None = None,
) -> dict:
    """Resolve the run config with precedence DEFAULT_CONFIG < patch < CLI.

    ``explicit`` is the set of arg dests the user actually passed; only those
    override the (already patch-merged) config, so argparse defaults never
    clobber patch values. Config-mapped scalars are also written back onto
    ``args`` so the training loop, which reads some of them directly, sees the
    resolved value.
    """
    explicit = set() if explicit is None else explicit
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if patch:
        validate_config_patch(patch)
        _deep_merge(cfg, patch)

    def cli_override(dest: str, section: str, key: str) -> None:
        if dest in explicit:
            cfg[section][key] = getattr(args, dest)

    cli_override("total_transitions", "schedule", "total_transitions")

    # Episode horizon: patch value or track-derived default (no CLI duplicate).
    if not (patch and "episode_length" in patch.get("env", {})):
        workspace_dir = str(Path(__file__).resolve().parent)
        cfg["env"]["episode_length"] = episode_length_for_track(
            track=cfg["env"]["track"],
            workspace_dir=workspace_dir,
            ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
            lap_multiplier=float(cfg["env"].get("episode_lap_multiplier", 3.0)),
        )

    cfg["env"]["domain_randomization"] = {
        **cfg["env"]["domain_randomization"],
        "enabled": True,
        "obs_latency_steps_range": [0, 0],
        "obs_noise_std_range": [0.0, 0.0],
    }

    actor_dim = int(cfg["obs"]["num_actor_obs"])
    critic_dim = int(cfg["obs"]["num_obs"])
    if actor_dim == critic_dim:
        raise ValueError(
            "Asymmetric trainer requires distinct actor/critic observation "
            f"dimensions; got num_actor_obs={actor_dim} num_obs={critic_dim}."
        )

    use_fixed = bool(getattr(args, "fixed_opponents", False)) or bool(
        cfg.get("fixed_opponents", {}).get("entries")
    )
    use_1v1 = use_fixed or args.opponent != "none"
    patch_env = (patch or {}).get("env", {})
    if use_1v1:
        if use_fixed:
            cfg["env"]["opponent_strategy"] = "mixed"
            if "opponent_mix" not in patch_env:
                mix = cfg["env"].setdefault("opponent_mix", {})
                mix["scripted_weight"] = 0.5
                mix["policy_weight"] = 0.5
                mix["policy_speed_cap_prob"] = 0.5
                mix["policy_speed_cap_range"] = [5.0, 7.0]
        else:
            cfg["env"]["opponent_strategy"] = args.opponent

    # Canonical Lee / ADR-0011 path only.
    cfg["env"]["steering_action_mode"] = "delta"
    if float(cfg["env"].get("steering_delta_max_rad", 0.0)) <= 0.0:
        raise ValueError("env.steering_delta_max_rad must be positive")
    cfg["env"]["term_oob_mode"] = "full_car_out"
    if "reset_stationary_probability" not in patch_env:
        cfg["env"]["reset_stationary_probability"] = 0.10
    scales = cfg["reward"]["reward_scales"]
    scales["oob_penalty"] = float(scales.get("oob_penalty", 0.02))
    scales["oob_impact"] = float(scales.get("oob_impact", scales["oob_penalty"]))
    for dead in (
        "wall_penalty",
        "wall_impact",
        "tyre_slip_penalty",
        "lateral",
        "smoothness",
        "overtake",
    ):
        scales.pop(dead, None)
    cfg["reward"]["oob_margin_m"] = 0.0
    cfg["reward"]["terminal_oob_skip_seconds"] = float(
        cfg["reward"].get("terminal_oob_skip_seconds", 10.0)
    )
    cfg["reward"]["rear_end_gate"] = "any_contact"
    validate_model_architecture(cfg)

    # Mirror schedule/model scalars onto args for the training loop.
    args.batch_size = int(cfg["model"]["batch_size"])
    args.buffer_capacity = int(cfg["model"]["replay_buffer_limit"])
    args.alpha = float(cfg["model"]["alpha"])
    args.min_train_transitions = int(cfg["model"]["minimum_train_transitions"])
    args.sampled_rows_per_transition = float(
        cfg["model"]["sampled_replay_rows_per_transition"]
    )
    args.log_interval_transitions = int(cfg["schedule"]["log_interval_transitions"])
    args.export_interval_transitions = int(
        cfg["schedule"]["export_interval_transitions"]
    )
    args.eval_interval_transitions = int(cfg["schedule"]["eval_interval_transitions"])
    args.total_transitions = int(cfg["schedule"]["total_transitions"])
    args.track = str(cfg["env"]["track"])
    return cfg


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_models(
    cfg: dict,
    device: torch.device,
    alpha: float = 0.01,
    compile: bool = False,
    compile_mode: str = "default",
) -> tuple[Models, QRSACTrainer]:
    # Networks/optimizers stay float32 even when the simulator uses float64.
    net_dtype = torch.float32
    models = Models(
        actor=make_policy_network(cfg).to(device=device, dtype=net_dtype),
        critic1=make_q_network(cfg).to(device=device, dtype=net_dtype),
        critic2=make_q_network(cfg).to(device=device, dtype=net_dtype),
        critic1_target=make_target_q_network(cfg).to(device=device, dtype=net_dtype),
        critic2_target=make_target_q_network(cfg).to(device=device, dtype=net_dtype),
    )
    models.critic1_target.load_state_dict(models.critic1.state_dict())
    models.critic2_target.load_state_dict(models.critic2.state_dict())

    trainer = QRSACTrainer(
        models,
        device=device,
        gamma=cfg["model"]["rew_gamma"],
        n_step=cfg["model"]["n_step"],
        alpha=alpha,
        smooth_factor=0.005,
        compile=compile,
        compile_mode=compile_mode,
        actor_lr=float(cfg["model"]["actor_lr"]),
        critic_lr=float(cfg["model"]["critic_lr"]),
    )
    return models, trainer


def save_policy_artifact(
    models: Models,
    env_transitions: int,
    artifact_dir: Path,
    normalizer: ObsNormalizer,
    cfg: dict,
    protocol_state: dict | None = None,
):
    """Save actor-only simulation artifact with sensor layout metadata."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"policy_{env_transitions}.pt"
    payload = build_sensor_artifact_payload(
        actor_state_dict=models.actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor_architecture_from_module(models.actor),
        env_transitions=env_transitions,
        actor_obs_dim=int(cfg["obs"]["num_actor_obs"]),
        critic_obs_dim=int(cfg["obs"]["num_obs"]),
        actor_layout_version=int(cfg["obs"]["actor_layout_version"]),
        action_dim=int(cfg["env"]["num_actions"]),
        action_scale=float(cfg["env"]["clip_actions"]),
        config_version=int(cfg["config_version"]),
        policy_format_version=int(cfg["policy_format_version"]),
        longitudinal_mode=str(cfg["env"].get("longitudinal_mode", "force")),
        steering_action_mode=str(cfg["env"].get("steering_action_mode", "delta")),
        steering_delta_max_rad=float(
            cfg["env"].get("steering_delta_max_rad", math.pi / 60.0)
        ),
        f_drive_max=float(cfg["env"].get("f_drive_max", 23.0)),
        f_brake_max=float(cfg["env"].get("f_brake_max", 5.2)),
        control_hz=float(
            1.0
            / (
                float(cfg["env"].get("sim_dt", 0.005))
                * float(cfg["env"].get("control_interval", 20))
            )
        ),
        simulator_id=str(cfg["simulator"]["id"]),
        simulator_version=int(cfg["simulator"]["version"]),
        training_protocol=(
            protocol_state
            if protocol_state is not None
            else initial_training_protocol_state()
        ),
    )
    torch.save(payload, path)
    logging.getLogger(LOGGER_NAME).info("Saved policy artifact to %s", path)
    return path


def load_init_ckpt(
    models: Models,
    normalizer: ObsNormalizer,
    path: str,
    device: torch.device,
    *,
    expected_layout_version: int,
    expected_critic_obs_dim: int | None = None,
    expected_steering_action_mode: str = "delta",
    expected_steering_delta_max_rad: float | None = None,
) -> int:
    """Warm-start the actor + actor-normalizer from a sensor policy artifact.

    Returns the artifact's env-transition count so self-play can seed its first
    snapshot at the policy's true maturity. Critics start fresh (not exported).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = actor_architecture_from_module(models.actor)
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=int(models.actor.obs_dim),
        expected_action_dim=int(models.actor.act_dim),
        expected_layout_version=expected_layout_version,
        expected_architecture=architecture,
        expected_critic_obs_dim=expected_critic_obs_dim,
        expected_steering_action_mode=expected_steering_action_mode,
        expected_steering_delta_max_rad=expected_steering_delta_max_rad,
    )
    models.actor.load_state_dict(payload["actor"], strict=True)
    normalizer.load_state_dict(payload["obs_norm"])
    init_transitions = int(payload.get("env_transitions", 0))
    logging.getLogger(LOGGER_NAME).info(
        "Warm-started from %s (env_transitions=%d)", path, init_transitions
    )
    return init_transitions


def run_eval_video(
    eval_state: dict,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    models: Models,
    normalizer: "ObsNormalizer",
    control_interval: int,
    clip_actions: float,
    run_dir: Path,
    step: int,
    num_steps: int,
    num_show: int,
    live: bool,
    wandb_run,
    log: logging.Logger,
    champion_mgr: "FixedChampionManager | None" = None,
) -> None:
    """Deterministic eval rollout rendered to an mp4 (and optionally live Rerun).

    Uses a cached env instance (``num_show`` parallel cars overlaid on the track)
    so the training env, replay buffer and RNG state are never touched. Best-effort:
    any failure is logged and swallowed so a long run is never taken down by viz.
    """
    from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz

    num_show = max(1, int(num_show))
    env = eval_state.get("env")
    if env is None:
        # Actor-only solo (1v0) eval with the learner's observation stream.
        env = F1tenthEnv(
            num_envs=num_show,
            env_cfg={
                **env_cfg,
                "opponent_strategy": "none",
                "launch_strategy_data": {"num_cars": num_show},
                "domain_randomization": {
                    **env_cfg["domain_randomization"],
                    "enabled": False,
                },
            },
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            show_viewer=False,
            enable_recording=False,
        )
        eval_state["env"] = env

    mp4_dir = run_dir / "eval"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = str(mp4_dir / f"eval_{step}.mp4")
    viz = RolloutVisualizer(
        centerline=env.track_state["centerline"],
        w_tr_left=env.track_state["w_tr_left"],
        w_tr_right=env.track_state["w_tr_right"],
        car_length=float(env_cfg.get("car_length", 0.568)),
        car_width=float(env_cfg.get("car_width", 0.296)),
        num_show=num_show,
        live=live,
        mp4_path=mp4_path,
        fps=10,
        has_opponent=False,
        rr_app_id="f1tenth_train_eval",
        rr_spawn=live,
    )

    was_training = models.actor.training
    models.actor.eval()

    def render_step(_step, rollout_env, _state_before, _reward, done, _extras):
        st = rollout_env.read_state()
        ego_xy = st["base_pos"][:, :2].cpu().numpy()
        ego_yaw = np.array(
            [yaw_from_quat_wxyz(q.tolist()) for q in st["base_quat"]]
        )
        speed = torch.linalg.norm(st["base_lin_vel"][:, :2], dim=-1).cpu().numpy()
        viz.render(
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            speed=speed,
            opp_xy=None,
            opp_yaw=0.0,
            done=done.cpu().numpy(),
        )

    deterministic_rollout(
        env,
        models.actor,
        normalizer.normalize,
        num_steps=num_steps,
        control_interval=control_interval,
        clip_actions=clip_actions,
        callback=render_step,
        with_sensors=True,
    )
    out = viz.close()
    models.actor.train(was_training)
    if wandb_run is not None and out is not None:
        import wandb

        wandb_run.log({"eval/rollout": wandb.Video(out, format="mp4")}, step=step)
    log.info("Eval rollout video written to %s", out)


def _build_parser() -> argparse.ArgumentParser:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Standalone QRSAC trainer (Lee/ADR-0011 sensor path)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config patch merged over DEFAULT_CONFIG. Precedence is "
        "DEFAULT_CONFIG < patch < explicit runtime CLI flags. Reward/model/"
        "schedule/opponent knobs belong in the patch, not the CLI.",
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument(
        "--total-transitions",
        type=int,
        default=cfg["schedule"]["total_transitions"],
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=False,
        help="Ignore --total-transitions termination (unbounded training).",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="none",
        choices=["none", "scripted", "policy"],
        help="Solo ('none'), scripted centerline, or frozen policy opponent. "
        "Fixed champion mix uses --fixed-opponents + config entries.",
    )
    parser.add_argument(
        "--fixed-opponents",
        action="store_true",
        default=False,
        help="Enable fixed champion pool from config fixed_opponents.entries.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile QR-SAC (default on; --no-compile for reference).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="Warm-start actor + obs-normalizer from a sensor policy artifact.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run artifact directory (default: outputs/runs/<run-id>/).",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log metrics to Weights & Biases (default on; --no-wandb disables).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.getenv("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    return parser


def _explicit_cli_dests(argv: list[str] | None = None) -> set[str]:
    """Dest names the user actually passed on the command line.

    Re-parses ``argv`` with every argparse default suppressed, so only options
    present on the command line land in the namespace. This is what lets an
    unpassed CLI flag avoid clobbering a --config patch value.
    """
    sentinel = _build_parser()
    for action in sentinel._actions:
        action.default = argparse.SUPPRESS
    return set(vars(sentinel.parse_args(argv)).keys())


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, set[str]]:
    args = _build_parser().parse_args(argv)
    return args, _explicit_cli_dests(argv)


def main():
    load_dotenv()
    args, explicit = parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.compile / Inductor may emit kernels without a deterministic impl; warn
    # instead of erroring on the opt-in fast path (numerics caveat is documented).
    torch.use_deterministic_algorithms(True, warn_only=args.compile)

    patch, patch_meta = (None, None)
    if args.config is not None:
        patch, patch_meta = load_config_patch(args.config)
    cfg = build_config(args, patch=patch, explicit=explicit)
    provenance = config_provenance(patch_meta, explicit)
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": args.num_envs},
    )
    clip_actions = cfg["env"]["clip_actions"]
    control_interval = cfg["env"]["control_interval"]
    n_step = model_cfg["n_step"]

    device = select_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    run_id = args.run_id or uuid.uuid4().hex[:8]
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = checkpoint_dir(run_dir)
    config_path = config_snapshot_path(run_dir)
    trainer_log_path = run_log_path(run_dir)

    log = setup_trainer_logging(log_file=trainer_log_path)
    log.info("Run id: %s  run_dir: %s  checkpoints: %s", run_id, run_dir, ckpt_dir)

    snapshot = build_run_snapshot(run_id, run_dir, args, cfg, provenance)
    config_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    log.info("Wrote config snapshot to %s", config_path)
    log.info("Using device: %s", device)
    scales = reward_cfg["reward_scales"]
    log.info(
        "Effective experiment: steering_action_mode=%s steering_delta_max_rad=%.12f "
        "control_hz=%.1f total_transitions=%d oob=-%.6f*dt*speed_kph^2 "
        "boundary_contact=-%.3f term_oob_mode=%s replay=%d compile=%s "
        "compile_mode=%s",
        env_cfg.get("steering_action_mode", "delta"),
        float(env_cfg.get("steering_delta_max_rad", math.pi / 60.0)),
        1.0 / (float(env_cfg["sim_dt"]) * int(env_cfg["control_interval"])),
        int(args.total_transitions),
        float(scales.get("oob_penalty", 0.0)),
        float(reward_cfg.get("boundary_contact_penalty", 0.0)),
        env_cfg.get("term_oob_mode", "full_car_out"),
        int(model_cfg["replay_buffer_limit"]),
        bool(args.compile),
        args.compile_mode,
    )

    env = F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        enable_recording=False,
    )
    log.info(
        "Episode horizon: %.1fs -> %d control steps (track=%s)",
        float(env_cfg["episode_length"]),
        int(env.max_episode_steps),
        args.track,
    )

    models, trainer = build_models(
        cfg,
        device,
        alpha=args.alpha,
        compile=args.compile,
        compile_mode=args.compile_mode,
    )
    actor_obs_dim = int(obs_cfg["num_actor_obs"])
    critic_obs_dim = int(obs_cfg["num_obs"])
    act_dim = cfg["env"]["num_actions"]
    buffer, selected_capacity, replay_estimate = make_dual_replay_buffer(
        capacity=args.buffer_capacity,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        n_step=n_step,
        gamma=model_cfg["rew_gamma"],
        num_envs=args.num_envs,
        device=device,
        batch_size=args.batch_size,
        log=log,
        allow_fallback=True,
    )
    args.buffer_capacity = selected_capacity
    actor_normalizer = ObsNormalizer(
        obs_dim=actor_obs_dim,
        device=device,
        eps=float(obs_cfg.get("norm_eps", 1e-8)),
        clip=float(obs_cfg.get("norm_clip", 10.0)),
    )
    critic_normalizer = ObsNormalizer(
        obs_dim=critic_obs_dim,
        device=device,
        eps=float(obs_cfg.get("norm_eps", 1e-8)),
        clip=float(obs_cfg.get("norm_clip", 10.0)),
    )

    if args.init_ckpt is not None:
        load_init_ckpt(
            models,
            actor_normalizer,
            args.init_ckpt,
            device,
            expected_layout_version=int(obs_cfg["actor_layout_version"]),
            expected_critic_obs_dim=int(obs_cfg["num_obs"]),
            expected_steering_action_mode=str(
                env_cfg.get("steering_action_mode", STEERING_ACTION_MODE)
            ),
            expected_steering_delta_max_rad=float(
                env_cfg.get("steering_delta_max_rad", math.pi / 60.0)
            ),
        )

    champion_mgr: FixedChampionManager | None = None
    fixed_entries = list(cfg.get("fixed_opponents", {}).get("entries") or [])
    if getattr(args, "fixed_opponents", False) or fixed_entries:
        if not fixed_entries:
            raise ValueError(
                "--fixed-opponents requires config fixed_opponents.entries"
            )
        selection = load_opponent_pool(
            fixed_entries,
            expected_architecture=actor_architecture_from_module(models.actor),
            expected_actor_obs_dim=actor_obs_dim,
            expected_action_dim=int(cfg["env"]["num_actions"]),
            expected_layout_version=int(obs_cfg["actor_layout_version"]),
            expected_critic_obs_dim=int(obs_cfg["num_obs"]),
            expected_steering_action_mode=str(
                env_cfg.get("steering_action_mode", STEERING_ACTION_MODE)
            ),
            expected_steering_delta_max_rad=float(
                env_cfg.get("steering_delta_max_rad", math.pi / 60.0)
            ),
            device=device,
            log=log,
        )
        entries, weights = selection
        champion_mgr = FixedChampionManager(
            entries,
            weights,
            seed=int(getattr(args, "seed", 0) or 0),
            log=log,
        )

    use_1v1 = champion_mgr is not None or args.opponent != "none"
    continuous = bool(getattr(args, "continuous", False))
    if continuous:
        log.info(
            "Continuous mode enabled: training runs until interrupted "
            "(finite total_transitions=%d is ignored for termination).",
            args.total_transitions,
        )
    wandb_run = None
    if args.wandb:
        import wandb

        tags = ["standalone", run_id, "asymmetric"]
        if continuous:
            tags.append("continuous")
        if platform.system() == "Darwin":
            tags.append("mac")
        init_kwargs = {
            "project": os.getenv("WANDB_PROJECT", "f1tenth-genesis"),
            "name": f"standalone_{run_id}",
            "id": run_id,
            "config": {
                **cfg,
                **vars(args),
                "config_provenance": provenance,
                "replay_capacity_selected": selected_capacity,
                "replay_estimate": replay_estimate,
            },
            "mode": os.getenv("WANDB_MODE", args.wandb_mode),
            "dir": str(run_dir),
            "tags": tags,
        }
        if entity := os.getenv("WANDB_ENTITY"):
            init_kwargs["entity"] = entity
        wandb_run = wandb.init(**init_kwargs)

    if champion_mgr is not None:
        champion_mgr.bootstrap_opponent(env)
    raw_obs, _ = env.reset(with_sensors=True)
    actor_obs, critic_obs = unpack_sensor_observations(raw_obs)
    actor_normalizer.update(actor_obs)
    critic_normalizer.update(critic_obs)
    vector_ticks = 0
    env_transitions = 0
    replay_inserts = torch.zeros((), device=device, dtype=torch.long)
    sampled_replay_rows = 0
    gradient_updates = 0
    learner_row_budget = 0.0
    episode_rewards = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    ep_return_sum = torch.zeros((), device=device, dtype=torch.float32)
    ep_return_count = torch.zeros((), device=device, dtype=torch.float32)
    policy_loss_accum = torch.zeros((), device=device)
    critic_loss_accum = torch.zeros((), device=device)
    loss_count = 0
    diag = RunningStats()
    t_start = time.perf_counter()
    last_log_time = t_start
    last_log_transitions = 0
    last_log_replay_inserts = 0
    last_log_gradient_updates = 0
    last_log_sampled_rows = 0
    eval_state: dict = {}
    num_sequences = max(1, int(args.batch_size) // REPLAY_TRAIN_LEN)
    # Cold-start only: one-shot flags begin unset; systemd restart => new process.
    protocol_state = initial_training_protocol_state()
    lidar_aug_enabled = bool(cfg["model"]["lidar_aug_enabled"])
    lidar_aug_max_shift = int(cfg["model"]["lidar_aug_max_shift_beams"])
    replay_full_reinit_enabled = bool(cfg["model"]["replay_full_reinit"])
    log.info(
        "Training protocol: actor_lr=%.3e critic_lr=%.3e "
        "replay_full_reinit=%s lidar_aug_enabled=%s lidar_aug_max_shift_beams=%d",
        trainer.actor_lr,
        trainer.critic_lr,
        replay_full_reinit_enabled,
        lidar_aug_enabled,
        lidar_aug_max_shift,
    )
    # Live GRU carry for the learner: one hidden per env. Checkpoints store the
    # pre-action state; done/reset rows are zeroed after the env step.
    recurrent_actor = actor_is_recurrent(models.actor)
    learner_hidden = (
        models.actor.initial_hidden(
            args.num_envs, device=device, dtype=torch.float32
        )
        if recurrent_actor
        else None
    )
    collection_actor_step = models.actor.step if recurrent_actor else None
    if recurrent_actor and args.compile:
        collection_actor_step = torch.compile(
            collection_actor_step, mode=args.compile_mode
        )

    try:
        while training_should_continue(
            env_transitions, args.total_transitions, continuous
        ):
            previous_transitions = env_transitions
            # Pre-action hidden is what trajectory replay checkpoints at boundaries.
            hidden_checkpoint = learner_hidden

            if env_transitions < args.min_train_transitions:
                actions = (
                    torch.rand(
                        args.num_envs, act_dim, device=device, dtype=torch.float32
                    )
                    * 2
                    * clip_actions
                    - clip_actions
                )
            elif recurrent_actor:
                with torch.no_grad():
                    if args.compile and args.compile_mode == "reduce-overhead":
                        torch.compiler.cudagraph_mark_step_begin()
                    actions, _, learner_hidden = collection_actor_step(
                        actor_normalizer.normalize(actor_obs),
                        learner_hidden,
                        reset_mask=None,
                        deterministic=False,
                        with_logprob=False,
                    )
                actions = actions.clamp(-clip_actions, clip_actions)
            else:
                with torch.no_grad():
                    actions, _ = models.actor(
                        actor_normalizer.normalize(actor_obs),
                        deterministic=False,
                        with_logprob=False,
                    )
                actions = actions.clamp(-clip_actions, clip_actions)

            next_raw_obs, reward, done, extras = env.step(
                actions.to(rt.tc_float),
                n_steps=control_interval,
                with_sensors=True,
            )
            vector_ticks += 1
            env_transitions += args.num_envs
            next_actor_obs, next_critic_obs = unpack_sensor_observations(next_raw_obs)
            reward = reward.to(torch.float32)

            episode_rewards += reward
            done_f = done.to(episode_rewards.dtype)
            ep_return_sum += (episode_rewards * done_f).sum()
            ep_return_count += done_f.sum()
            episode_rewards = episode_rewards * (1.0 - done_f)

            accumulate_step_diagnostics(diag, reward, actions, critic_obs, extras)
            accumulate_completed_episode_lifespans(
                diag,
                extras["metrics"]["completed_episode_steps"],
                done,
                env.control_dt,
            )
            diag.add_mean(
                "obs/norm_abs",
                actor_normalizer.normalize(actor_obs).abs(),
                track_range=True,
            )
            if use_1v1 and critic_obs.shape[-1] > OPP_OBS_BASE_IDX:
                opp_block = critic_obs[:, OPP_OBS_BASE_IDX:]
                in_range = (opp_block != 0).any(dim=-1).to(critic_obs.dtype)
                diag.add_mean("metric/opponent_presence", in_range)

            replay_inserts += buffer.add(
                actor_obs,
                critic_obs,
                actions,
                reward,
                done,
                hidden=hidden_checkpoint,
            )
            if learner_hidden is not None:
                done_b = done.bool()
                if done_b.any():
                    learner_hidden[done_b] = 0
            maybe_replay_full_reinit(
                enabled=replay_full_reinit_enabled,
                protocol_state=protocol_state,
                buffer=buffer,
                trainer=trainer,
                models=models,
                actor_normalizer=actor_normalizer,
                env_transitions=env_transitions,
                learner_hidden=learner_hidden,
                champion_mgr=champion_mgr,
                env=env,
                log=log,
                wandb_run=wandb_run,
            )
            actor_normalizer.update(next_actor_obs)
            critic_normalizer.update(next_critic_obs)
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs

            if env_transitions >= args.min_train_transitions:
                updates_due, learner_row_budget = learner_updates_for_transitions(
                    args.num_envs,
                    args.batch_size,
                    args.sampled_rows_per_transition,
                    learner_row_budget,
                )
                if not buffer.is_ready(num_sequences):
                    learner_row_budget += updates_due * args.batch_size
                    updates_due = 0
                boot_lo = buffer.burn_in + buffer.n_step
                boot_hi = buffer.burn_in + buffer.train_len + buffer.n_step
                for _ in range(updates_due):
                    batch = buffer.sample(num_sequences)
                    replay_stats = buffer.last_sample_metrics
                    _rs = torch.tensor(
                        [
                            replay_stats["available_visible_frac"],
                            replay_stats["available_not_visible_frac"],
                            replay_stats["sampled_visible_frac"],
                            replay_stats["sampled_not_visible_frac"],
                            replay_stats["fallback"],
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    diag.add_mean("replay/opp_visible_available", _rs[0])
                    diag.add_mean("replay/opp_not_visible_available", _rs[1])
                    diag.add_mean("replay/opp_visible_sampled", _rs[2])
                    diag.add_mean("replay/opp_not_visible_sampled", _rs[3])
                    diag.add_mean("replay/opp_visibility_fallback", _rs[4])
                    actor_raw = batch["actor_obs"]
                    if lidar_aug_enabled and lidar_aug_max_shift > 0:
                        actor_raw = augment_actor_lidar_beam_shift(
                            actor_raw, max_shift_beams=lidar_aug_max_shift
                        )
                    normalized_actor = actor_normalizer.normalize(actor_raw)
                    batch["actor_obs"] = normalized_actor
                    # Exact t+n train window from the same augmented+normalized
                    # tensor so burn-in / train / bootstrap stay aligned.
                    normalized_bootstrap_actor = normalized_actor[:, boot_lo:boot_hi]
                    batch["bootstrap_actor_obs"] = normalized_bootstrap_actor
                    normalized_critic = critic_normalizer.normalize(
                        batch["critic_obs"]
                    )
                    batch["critic_obs"] = normalized_critic
                    batch["bootstrap_critic_obs"] = normalized_critic[
                        :, boot_lo:boot_hi
                    ]
                    if recurrent_actor:
                        losses = trainer.update_from_sequences(batch)
                    else:
                        train_lo = buffer.burn_in
                        train_hi = train_lo + buffer.train_len
                        rows = num_sequences * buffer.train_len
                        losses = trainer.update(
                            {
                                "actor_obs": normalized_actor[
                                    :, train_lo:train_hi
                                ].reshape(rows, actor_obs_dim),
                                "critic_obs": normalized_critic[
                                    :, train_lo:train_hi
                                ].reshape(rows, critic_obs_dim),
                                "action": batch["action"][
                                    :, train_lo:train_hi
                                ].reshape(rows, act_dim),
                                "reward": batch["n_step_reward"].reshape(rows),
                                "next_actor_obs": normalized_bootstrap_actor.reshape(
                                    rows, actor_obs_dim
                                ),
                                "next_critic_obs": batch[
                                    "bootstrap_critic_obs"
                                ].reshape(rows, critic_obs_dim),
                                "done": batch["n_step_done"].reshape(rows),
                            }
                        )
                    gradient_updates += 1
                    sampled_replay_rows += num_sequences * REPLAY_TRAIN_LEN
                    policy_loss_accum += losses.policy_loss
                    critic_loss_accum += losses.critic_loss
                    loss_count += 1

            if interval_crossed(
                previous_transitions,
                env_transitions,
                args.log_interval_transitions,
            ):
                now = time.perf_counter()
                elapsed = now - last_log_time
                ri = int(replay_inserts)
                bsize = int(buffer.size)
                window_transitions = env_transitions - last_log_transitions
                vector_ticks_per_sec = (
                    window_transitions / args.num_envs / max(elapsed, 1e-6)
                )
                transitions_per_sec = window_transitions / max(elapsed, 1e-6)
                inserts_per_sec = (
                    ri - last_log_replay_inserts
                ) / max(elapsed, 1e-6)
                sampled_rows_per_sec = (
                    sampled_replay_rows - last_log_sampled_rows
                ) / max(elapsed, 1e-6)
                updates_per_sec = (
                    gradient_updates - last_log_gradient_updates
                ) / max(elapsed, 1e-6)
                ep_count = int(ep_return_count.item())
                mean_ep_reward = (
                    float(ep_return_sum / ep_return_count)
                    if ep_count > 0
                    else float("nan")
                )
                mean_policy_loss = (
                    float(policy_loss_accum / loss_count)
                    if loss_count
                    else float("nan")
                )
                mean_critic_loss = (
                    float(critic_loss_accum / loss_count)
                    if loss_count
                    else float("nan")
                )
                buffer_fill_pct = 100.0 * bsize / buffer.capacity
                log.info(
                    "ticks=%d transitions=%d replay_inserts=%d buffer=%d/%d (%.1f%%) "
                    "gradient_updates=%d ticks/s=%.1f transitions/s=%.1f "
                    "inserts/s=%.1f sampled_rows/s=%.1f updates/s=%.2f "
                    + TRAINING_SUMMARY_REWARD_TAIL,
                    vector_ticks,
                    env_transitions,
                    ri,
                    bsize,
                    buffer.capacity,
                    buffer_fill_pct,
                    gradient_updates,
                    vector_ticks_per_sec,
                    transitions_per_sec,
                    inserts_per_sec,
                    sampled_rows_per_sec,
                    updates_per_sec,
                    *training_summary_reward_tail_args(
                        mean_policy_loss=mean_policy_loss,
                        mean_critic_loss=mean_critic_loss,
                        mean_ep_reward=mean_ep_reward,
                        diag=diag,
                        ep_count=ep_count,
                    ),
                )
                window_env_steps = float(window_transitions)
                nf_obs_rate = diag.total("metric/nonfinite_obs_envs") / window_env_steps
                nf_reward_rate = (
                    diag.total("metric/nonfinite_reward_envs") / window_env_steps
                )
                nf_state_rate = (
                    diag.total("metric/nonfinite_state_envs") / window_env_steps
                )

                if use_1v1:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f passing=%.4f collision=%.4f "
                        "oob_penalty=%.4f steer_chg=%.4f steer_hist=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/passing"),
                        diag.mean("reward_term/collision"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/steering_change"),
                        diag.mean("reward_term/steering_history"),
                    )
                else:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f oob_penalty=%.4f "
                        "steer_chg=%.4f steer_hist=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/steering_change"),
                        diag.mean("reward_term/steering_history"),
                    )
                log.info(
                    "  reward_events: oob_when=%.4f oob_impact_when=%.4f "
                    "boundary_when=%.4f oob_impact_events=%d "
                    "boundary_events=%d",
                    diag.mean("reward_term/oob_penalty_when_oob"),
                    diag.mean("reward_term/oob_impact_when_event"),
                    diag.mean("reward_term/boundary_contact_when_event"),
                    int(diag.total("metric/oob_impact_events")),
                    int(diag.total("metric/boundary_contact_events")),
                )
                log.info(
                    "  env: speed=%.3f opp_speed=%.3f lat_err=%.3f oob_frac=%.3f "
                    "progress_ds=%.4f laps_completed=%d | "
                    "throttle[%.2f..%.2f] steer[%.2f..%.2f] obs_absmax=%.2f "
                    "norm_obs_absmax=%.2f",
                    diag.mean("metric/speed_xy"),
                    diag.mean("metric/opp_speed"),
                    diag.mean("metric/lateral_error"),
                    diag.mean("metric/oob_mask"),
                    diag.mean("metric/progress_ds"),
                    int(diag.total("metric/laps_completed")),
                    diag.vmin("action/throttle"),
                    diag.vmax("action/throttle"),
                    diag.vmin("action/steer"),
                    diag.vmax("action/steer"),
                    diag.vmax("obs/abs"),
                    diag.vmax("obs/norm_abs"),
                )
                log.info(
                    "  nonfinite: obs_rate=%.2e reward_rate=%.2e state_rate=%.2e",
                    nf_obs_rate,
                    nf_reward_rate,
                    nf_state_rate,
                )
                if use_1v1:
                    log.info(
                        "  terminations: time_out=%d oob=%d "
                        "collision=%d not_moving=%d invalid=%d | "
                        "opp_presence=%.3f",
                        int(diag.total("term/time_out")),
                        int(diag.total("term/out_of_bounds")),
                        int(diag.total("term/collision")),
                        int(diag.total("term/not_moving")),
                        int(diag.total("term/invalid_state")),
                        diag.mean("metric/opponent_presence"),
                    )
                    if champion_mgr is not None:
                        meta = champion_mgr.metadata()
                        if champion_mgr.pool_size > 1:
                            log.info(
                                "  fixed_opponent_pool: size=%d assignments=%s",
                                meta["pool_size"],
                                [
                                    (
                                        entry["checkpoint"].rsplit("/", 1)[-1],
                                        f"w={entry['weight']:.2f}",
                                        f"p={entry['sampled_fraction']:.3f}",
                                    )
                                    for entry in meta["entries"]
                                ],
                            )
                        else:
                            log.info(
                                "  fixed_champion: ckpt=%s transitions=%d",
                                meta["checkpoint"],
                                meta["transitions"],
                            )
                else:
                    log.info(
                        "  terminations: time_out=%d oob=%d "
                        "not_moving=%d invalid=%d",
                        int(diag.total("term/time_out")),
                        int(diag.total("term/out_of_bounds")),
                        int(diag.total("term/not_moving")),
                        int(diag.total("term/invalid_state")),
                    )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "env_transitions": env_transitions,
                            "vector_ticks": vector_ticks,
                            "replay_inserts": ri,
                            "sampled_replay_rows": sampled_replay_rows,
                            "gradient_updates": gradient_updates,
                            "buffer/size": bsize,
                            "train/policy_loss": mean_policy_loss,
                            "train/critic_loss": mean_critic_loss,
                            "train/mean_ep_reward": mean_ep_reward,
                            "episode/lifespan_mean_s": diag.mean(
                                "episode/lifespan_s"
                            ),
                            "perf/vector_ticks_per_sec": vector_ticks_per_sec,
                            "perf/env_transitions_per_sec": transitions_per_sec,
                            "perf/replay_inserts_per_sec": inserts_per_sec,
                            "perf/sampled_replay_rows_per_sec": sampled_rows_per_sec,
                            "perf/gradient_updates_per_sec": updates_per_sec,
                            "reward/total_mean": diag.mean("reward/step"),
                            "reward/total_min": diag.vmin("reward/step"),
                            "reward/total_max": diag.vmax("reward/step"),
                            "reward/progress": diag.mean("reward_term/progress"),
                            "reward/passing": diag.mean("reward_term/passing"),
                            "reward/rear_end": diag.mean("reward_term/rear_end"),
                            "reward/collision": diag.mean(
                                "reward_term/collision"
                            ),
                            "reward/oob_penalty": diag.mean(
                                "reward_term/oob_penalty"
                            ),
                            "reward/oob_penalty_when_oob": diag.mean(
                                "reward_term/oob_penalty_when_oob"
                            ),
                            "reward/oob_impact": diag.mean(
                                "reward_term/oob_impact"
                            ),
                            "reward/oob_impact_when_event": diag.mean(
                                "reward_term/oob_impact_when_event"
                            ),
                            "reward/steering_change": diag.mean(
                                "reward_term/steering_change"
                            ),
                            "reward/steering_history": diag.mean(
                                "reward_term/steering_history"
                            ),
                            "env/speed_xy": diag.mean("metric/speed_xy"),
                            "env/lateral_error": diag.mean("metric/lateral_error"),
                            "env/oob_frac": diag.mean("metric/oob_mask"),
                            "env/oob_impact_events": diag.total(
                                "metric/oob_impact_events"
                            ),
                            "env/progress_ds": diag.mean("metric/progress_ds"),
                            "env/lap_count": diag.mean("metric/lap_count"),
                            "env/laps_completed": diag.total("metric/laps_completed"),
                            "action/throttle_max": diag.vmax("action/throttle"),
                            "action/steer_max": diag.vmax("action/steer"),
                            "obs/absmax": diag.vmax("obs/abs"),
                            "term/time_out": diag.total("term/time_out"),
                            "term/out_of_bounds": diag.total("term/out_of_bounds"),
                            "term/not_moving": diag.total("term/not_moving"),
                            "term/invalid_state": diag.total("term/invalid_state"),
                            "nonfinite/obs_rate": nf_obs_rate,
                            "nonfinite/reward_rate": nf_reward_rate,
                            "nonfinite/state_rate": nf_state_rate,
                        },
                        step=env_transitions,
                    )
                policy_loss_accum.zero_()
                critic_loss_accum.zero_()
                ep_return_sum.zero_()
                ep_return_count.zero_()
                loss_count = 0
                diag.reset()
                last_log_time = now
                last_log_transitions = env_transitions
                last_log_replay_inserts = ri
                last_log_sampled_rows = sampled_replay_rows
                last_log_gradient_updates = gradient_updates

            if interval_crossed(
                previous_transitions,
                env_transitions,
                args.export_interval_transitions,
            ):
                save_policy_artifact(
                    models,
                    env_transitions,
                    ckpt_dir,
                    actor_normalizer,
                    cfg,
                    protocol_state=protocol_state,
                )

            if (
                args.eval_interval_transitions > 0
                and interval_crossed(
                    previous_transitions,
                    env_transitions,
                    args.eval_interval_transitions,
                )
            ):
                try:
                    run_eval_video(
                        eval_state=eval_state,
                        env_cfg=env_cfg,
                        obs_cfg=obs_cfg,
                        reward_cfg=reward_cfg,
                        models=models,
                        normalizer=actor_normalizer,
                        control_interval=control_interval,
                        clip_actions=clip_actions,
                        run_dir=run_dir,
                        step=env_transitions,
                        num_steps=int(
                            cfg["schedule"].get("eval_video_steps", 600)
                        ),
                        num_show=int(
                            cfg["schedule"].get("eval_video_num_envs", 1)
                        ),
                        live=bool(
                            cfg["schedule"].get("eval_video_live", False)
                        ),
                        wandb_run=wandb_run,
                        log=log,
                        champion_mgr=champion_mgr,
                    )
                except Exception as exc:
                    log.warning("Eval video rollout failed (continuing): %s", exc)

        save_policy_artifact(
            models,
            env_transitions,
            ckpt_dir,
            actor_normalizer,
            cfg,
            protocol_state=protocol_state,
        )
    finally:
        try:
            env.close()
        except Exception as exc:
            log.warning("env.close() failed during shutdown: %s", exc)
        eval_env = eval_state.get("env")
        if eval_env is not None:
            try:
                eval_env.close()
            except Exception as exc:
                log.warning("eval env.close() failed during shutdown: %s", exc)
        if wandb_run is not None:
            wandb_run.finish()

    log.info(
        "Training finished after %d vector ticks, %d transitions, and %d updates.",
        vector_ticks,
        env_transitions,
        gradient_updates,


    )


if __name__ == "__main__":
    main()

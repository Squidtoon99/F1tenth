#!/usr/bin/env python3
"""Single-process QR-SAC/PPO trainer for the F1TENTH sensor policy."""

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
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F

from f1tenth_policy import (
    TRAINING_I_BRAKE_MAX_A,
    TRAINING_I_DRIVE_MAX_A,
    TRAINING_I_SLEW_A_PER_S,
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
from ppo import PPOTrainer, pure_timeout_mask
from selfplay import SelfPlayManager
from run_layout import (
    checkpoint_dir,
    config_snapshot_path,
    default_run_dir,
    run_lock_path,
    run_log_path,
)
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
RESUME_STATE_FILENAME = "resume_state.pt"
RESUME_FORMAT_VERSION = 1
DEFAULT_POLICY_ARTIFACT_KEEP = 3
# Privileged opponent block [384:392): rel_xy, rel_vxy, rel_axy, gap_norm, ey.
# gap_norm (index +6) is (s_other - s_self) / (0.5 * track_length).
OPP_OBS_BASE_IDX = 384
OPP_OBS_DIM = 8
OPP_TRACK_GAP_IDX = OPP_OBS_BASE_IDX + 6
OPP_OBS_END_IDX = OPP_OBS_BASE_IDX + OPP_OBS_DIM

# Dual-observation replay: only these capacities are chosen automatically.
REPLAY_CAPACITY_REQUESTED = 2_000_000
REPLAY_CAPACITY_FALLBACK = 1_000_000


@dataclass
class ActorModels:
    actor: nn.Module


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
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)
    else:
        stdout_handler = FlushingStreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(fmt)
        handlers.append(stdout_handler)
    for name in (LOGGER_NAME, "qrsac"):
        target = logging.getLogger(name)
        target.setLevel(level)
        target.propagate = False
        for handler in target.handlers[:]:
            target.removeHandler(handler)
        for handler in handlers:
            target.addHandler(handler)
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


def actor_lr_schedule_active(
    actor_freeze_transitions: int,
    actor_lr_ramp_transitions: int,
) -> bool:
    return actor_freeze_transitions > 0 or actor_lr_ramp_transitions > 0


def effective_actor_learning_rate(
    env_transitions: int,
    actor_freeze_transitions: int,
    actor_lr_ramp_transitions: int,
    base_lr: float,
) -> float:
    if env_transitions < actor_freeze_transitions:
        return 0.0
    if actor_lr_ramp_transitions <= 0:
        return base_lr
    ramp_end = actor_freeze_transitions + actor_lr_ramp_transitions
    if env_transitions >= ramp_end:
        return base_lr
    progress = (env_transitions - actor_freeze_transitions) / actor_lr_ramp_transitions
    return base_lr * progress


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
        "wall_contact",
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

    wall_contact = terms.get("wall_contact")
    wall_contact_metric = metrics.get("wall_contact")
    if isinstance(wall_contact, torch.Tensor) and isinstance(
        wall_contact_metric, torch.Tensor
    ):
        wall_contact_events = wall_contact_metric > 0
        diag.add_total(
            "metric/wall_contact_events",
            wall_contact_events.to(wall_contact_metric.dtype),
        )
        diag.add_event_mean(
            "reward_term/wall_contact_when_event",
            wall_contact,
            wall_contact_events,
        )

    for term_name in ("boundary_contact", "oob_impact"):
        term_value = terms.get(term_name)
        if isinstance(term_value, torch.Tensor):
            fired = term_value != 0
            diag.add_total(
                f"metric/{term_name}_events",
                fired.to(term_value.dtype),
            )
            diag.add_event_mean(
                f"reward_term/{term_name}_when_event", term_value, fired
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


def initial_training_protocol_state(algorithm: str = "qrsac") -> dict:
    """Process-local one-shot protocol flags (unset on every cold start)."""
    return {
        "algorithm": algorithm,
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
    selfplay_mgr: SelfPlayManager | None = None,
    env=None,
    log: logging.Logger | None = None,
    wandb_run=None,
) -> bool:
    """Trigger Lee et al. 2025 replay-full reinit exactly once when buffer fills.

    Fixed champion is immutable across reinitialization; self-play re-seeds
    the opponent from the reinitialized learner.
    """
    if not enabled or protocol_state.get("replay_full_reinit_done"):
        return False
    if int(buffer.size) < int(buffer.capacity):
        return False
    trainer.reinitialize_networks()
    if learner_hidden is not None:
        learner_hidden.zero_()
    if env is not None and selfplay_mgr is not None:
        selfplay_mgr.reseed_after_reinit(
            models, actor_normalizer, env, int(env_transitions)
        )
    if env is not None and champion_mgr is not None:
        champion_mgr.bootstrap_opponent(env, resample=True)
    protocol_state["replay_full_reinit_done"] = True
    protocol_state["replay_full_reinit_count"] = (
        int(protocol_state.get("replay_full_reinit_count", 0)) + 1
    )
    protocol_state["replay_full_reinit_transitions"] = int(env_transitions)
    logger = log or logging.getLogger(LOGGER_NAME)
    if selfplay_mgr is not None:
        gru_detail = (
            "live learner/opponent GRU cleared; self-play pool reseeding"
        )
    elif champion_mgr is not None:
        gru_detail = "live GRU cleared; fixed champion unchanged"
    else:
        gru_detail = "live GRU cleared"
    logger.info(
        "Replay-full network reinitialization (Lee et al. 2025): "
        "actor/critics/targets/Adam reset at transitions=%d buffer=%d/%d "
        "reinit_count=%d; replay + obs normalizers retained; %s",
        env_transitions,
        int(buffer.size),
        int(buffer.capacity),
        protocol_state["replay_full_reinit_count"],
        gru_detail,
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
    projection_dim = model.get("lidar_projection_dim")
    if not isinstance(projection_dim, int) or projection_dim <= 0:
        raise ValueError(
            "model.lidar_projection_dim must be a positive int, got "
            f"{projection_dim!r}"
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
    freeze = model.get("actor_freeze_transitions", 0)
    if not isinstance(freeze, int) or freeze < 0:
        raise ValueError(
            "model.actor_freeze_transitions must be a non-negative int, got "
            f"{freeze!r}"
        )
    ramp = model.get("actor_lr_ramp_transitions", 0)
    if not isinstance(ramp, int) or ramp < 0:
        raise ValueError(
            "model.actor_lr_ramp_transitions must be a non-negative int, got "
            f"{ramp!r}"
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


def validate_algorithm_config(cfg: dict, num_envs: int) -> None:
    algorithm = cfg.get("algorithm")
    if algorithm not in ("qrsac", "ppo"):
        raise ValueError(
            f"algorithm must be 'qrsac' or 'ppo', got {algorithm!r}"
        )
    if algorithm == "qrsac":
        return
    ppo = cfg["ppo"]
    for key in ("rollout_steps", "epochs", "env_minibatches"):
        value = ppo.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"ppo.{key} must be a positive int, got {value!r}")
    env_minibatches = min(num_envs, int(ppo["env_minibatches"]))
    if num_envs % env_minibatches != 0:
        raise ValueError(
            f"num_envs={num_envs} must be divisible by "
            f"effective ppo.env_minibatches={env_minibatches}"
        )
    for key in (
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "value_clip",
        "actor_lr",
        "value_lr",
        "max_grad_norm",
        "entropy_coef",
    ):
        value = ppo.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"ppo.{key} must be a positive finite number, got {value!r}")
    for key in ("gamma", "gae_lambda"):
        if float(ppo[key]) > 1.0:
            raise ValueError(f"ppo.{key} must be <= 1.0, got {ppo[key]!r}")
    if not isinstance(ppo.get("advantage_filter_enabled"), bool):
        raise ValueError(
            "ppo.advantage_filter_enabled must be a bool, got "
            f"{ppo.get('advantage_filter_enabled')!r}"
        )
    discard_fraction = ppo.get("advantage_filter_discard_fraction")
    if (
        not isinstance(discard_fraction, (int, float))
        or isinstance(discard_fraction, bool)
        or not math.isfinite(float(discard_fraction))
        or float(discard_fraction) < 0.0
        or float(discard_fraction) >= 1.0
    ):
        raise ValueError(
            "ppo.advantage_filter_discard_fraction must be a finite number in "
            f"[0.0, 1.0), got {discard_fraction!r}"
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
        lidar_projection_dim=model["lidar_projection_dim"],
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
        "lidar_projection_dim": int(model["lidar_projection_dim"]),
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


def training_source_git_state(training_root: Path) -> dict:
    repo_root = training_root.parent
    overlay_marker = training_root / ".overlay-active"
    if overlay_marker.exists():
        raise RuntimeError(
            f"Training source overlay is active ({overlay_marker}). "
            "Stop the overlay launcher so tracked files are restored."
        )
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "training"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    head_proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    head = head_proc.stdout.strip() if head_proc.returncode == 0 else None
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "git_head": head,
        "git_branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "training_tree_clean": len(dirty) == 0,
        "training_tree_dirty_files": dirty or None,
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
        "source_git": provenance.get("source_git"),
        "effective_wall_contact": {
            "geometry": "first projected footprint edge intersects mapped wall",
            "car_length_m": float(cfg["env"]["car_length"]),
            "car_width_m": float(cfg["env"]["car_width"]),
            "coefficient_per_m": float(
                cfg["reward"]["wall_contact_coefficient"]
            ),
            "control_dt_s": (
                float(cfg["env"]["sim_dt"])
                * int(cfg["env"]["control_interval"])
            ),
            "formula": "-coefficient * speed_mps",
        },
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
    if "algorithm" in explicit:
        cfg["algorithm"] = args.algorithm
    cli_override(
        "selfplay_snapshot_interval", "selfplay", "snapshot_interval_transitions"
    )
    cli_override(
        "selfplay_refresh_interval", "selfplay", "refresh_interval_transitions"
    )
    cli_override("selfplay_pool_size", "selfplay", "pool_size")
    cli_override("selfplay_sample", "selfplay", "sample_mode")
    cli_override("selfplay_mixed_latest_prob", "selfplay", "mixed_latest_prob")
    cli_override("selfplay_anchor_ckpt", "selfplay", "anchor_ckpt")
    cli_override("selfplay_anchor_prob", "selfplay", "anchor_prob")

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

    self_play = getattr(args, "self_play", False) or getattr(
        args, "mixed_opponents", False
    )
    use_fixed = bool(getattr(args, "fixed_opponents", False)) or bool(
        cfg.get("fixed_opponents", {}).get("entries")
    )
    if self_play and use_fixed:
        raise ValueError("--self-play and --fixed-opponents are mutually exclusive")
    use_1v1 = use_fixed or self_play or args.opponent != "none"
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
        elif self_play:
            if getattr(args, "mixed_opponents", False):
                cfg["env"]["opponent_strategy"] = "mixed"
            else:
                cfg["env"]["opponent_strategy"] = "policy"
        else:
            cfg["env"]["opponent_strategy"] = args.opponent

    steering_mode = str(cfg["env"].get("steering_action_mode", "delta"))
    if steering_mode not in ("absolute", "delta"):
        raise ValueError(
            "env.steering_action_mode must be 'absolute' or 'delta', got "
            f"{steering_mode!r}"
        )
    cfg["env"]["steering_action_mode"] = steering_mode
    if steering_mode == "delta":
        if float(cfg["env"].get("steering_delta_max_rad", 0.0)) <= 0.0:
            raise ValueError("env.steering_delta_max_rad must be positive")
    if "reset_stationary_probability" not in patch_env:
        cfg["env"]["reset_stationary_probability"] = 0.10
    scales = cfg["reward"]["reward_scales"]
    for dead in (
        "wall_penalty",
        "wall_impact",
        "tyre_slip_penalty",
        "lateral",
        "smoothness",
        "overtake",
    ):
        scales.pop(dead, None)
    cfg["reward"]["rear_end_gate"] = "any_contact"
    validate_model_architecture(cfg)
    validate_algorithm_config(cfg, int(args.num_envs))
    args.algorithm = str(cfg["algorithm"])

    # Mirror schedule/model scalars onto args for the training loop.
    args.batch_size = int(cfg["model"]["batch_size"])
    args.buffer_capacity = int(cfg["model"]["replay_buffer_limit"])
    args.alpha = float(cfg["model"]["alpha"])
    args.actor_freeze_transitions = int(cfg["model"]["actor_freeze_transitions"])
    args.actor_lr_ramp_transitions = int(cfg["model"]["actor_lr_ramp_transitions"])
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
        actor_freeze_transitions=int(cfg["model"]["actor_freeze_transitions"]),
        actor_lr_ramp_transitions=int(cfg["model"]["actor_lr_ramp_transitions"]),
    )
    return models, trainer


def build_ppo_models(
    cfg: dict,
    device: torch.device,
    actor_normalizer: ObsNormalizer,
    critic_normalizer: ObsNormalizer,
    num_envs: int,
    compile: bool = False,
    compile_mode: str = "default",
) -> tuple[ActorModels, PPOTrainer]:
    ppo_cfg = cfg["ppo"]
    env_minibatches = min(int(num_envs), int(ppo_cfg["env_minibatches"]))
    actor = make_policy_network(cfg).to(device=device, dtype=torch.float32)
    models = ActorModels(actor=actor)
    trainer = PPOTrainer(
        actor,
        int(cfg["obs"]["num_obs"]),
        actor_normalizer,
        critic_normalizer,
        device,
        value_hidden_sizes=cfg["model"]["critic_hidden_layers"],
        rollout_steps=int(ppo_cfg["rollout_steps"]),
        num_epochs=int(ppo_cfg["epochs"]),
        env_minibatch_size=int(num_envs) // env_minibatches,
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_ratio=float(ppo_cfg["clip_ratio"]),
        value_clip=float(ppo_cfg["value_clip"]),
        actor_lr=float(ppo_cfg["actor_lr"]),
        value_lr=float(ppo_cfg["value_lr"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        action_clip=float(cfg["env"]["clip_actions"]),
        compile=compile,
        compile_mode=compile_mode,
        advantage_filter_enabled=bool(ppo_cfg["advantage_filter_enabled"]),
        advantage_filter_discard_fraction=float(
            ppo_cfg["advantage_filter_discard_fraction"]
        ),
        lidar_aug_enabled=bool(cfg["model"]["lidar_aug_enabled"]),
        lidar_aug_max_shift_beams=int(cfg["model"]["lidar_aug_max_shift_beams"]),
    )
    return models, trainer


def save_policy_artifact(
    models,
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
        f_drive_max=float(cfg["env"].get("f_drive_max", 26.5)),
        f_brake_max=float(cfg["env"].get("f_brake_max", 23.1)),
        i_drive_max_a=float(cfg["env"].get("i_drive_max_a", TRAINING_I_DRIVE_MAX_A)),
        i_brake_max_a=float(cfg["env"].get("i_brake_max_a", TRAINING_I_BRAKE_MAX_A)),
        i_slew_a_per_s=float(cfg["env"].get("i_slew_a_per_s", TRAINING_I_SLEW_A_PER_S)),
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
    models,
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


def resume_state_path(run_dir: Path) -> Path:
    return run_dir / RESUME_STATE_FILENAME


def build_rng_state(device: torch.device) -> dict:
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "name": np_state[0],
            "keys": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": np_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np_entry = state["numpy"]
    if isinstance(np_entry, dict):
        np.random.set_state(
            (
                np_entry["name"],
                np.array(np_entry["keys"], dtype=np.uint32),
                np_entry["pos"],
                np_entry["has_gauss"],
                np_entry["cached_gaussian"],
            )
        )
    else:
        np.random.set_state(np_entry)
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
    torch.set_rng_state(torch_state.cpu())
    if device.type == "cuda" and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        if isinstance(cuda_states, list):
            cuda_states = [
                s.cpu() if isinstance(s, torch.Tensor) else torch.as_tensor(s, dtype=torch.uint8)
                for s in cuda_states
            ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_resume_state(
    path: Path,
    *,
    algorithm: str,
    models,
    trainer,
    actor_normalizer: ObsNormalizer,
    critic_normalizer: ObsNormalizer | None,
    env_transitions: int,
    gradient_updates: int,
    vector_ticks: int,
    sampled_replay_rows: int,
    device: torch.device,
) -> None:
    payload = {
        "format": "training_resume",
        "format_version": RESUME_FORMAT_VERSION,
        "algorithm": algorithm,
        "env_transitions": int(env_transitions),
        "gradient_updates": int(gradient_updates),
        "vector_ticks": int(vector_ticks),
        "sampled_replay_rows": int(sampled_replay_rows),
        "actor": models.actor.state_dict(),
        "actor_obs_norm": actor_normalizer.state_dict(),
        "rng_state": build_rng_state(device),
    }
    if algorithm == "ppo":
        payload["value_critic"] = trainer.value_critic.state_dict()
        if critic_normalizer is None:
            raise ValueError("critic_normalizer is required for PPO resume")
        payload["critic_obs_norm"] = critic_normalizer.state_dict()
        payload["actor_optimizer"] = trainer.actor_optimizer.state_dict()
        payload["value_optimizer"] = trainer.value_optimizer.state_dict()
    else:
        raise ValueError(f"resume is not implemented for algorithm={algorithm!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    logging.getLogger(LOGGER_NAME).info("Saved resume state to %s", path)


def load_resume_state(
    path: str,
    *,
    algorithm: str,
    models,
    trainer,
    actor_normalizer: ObsNormalizer,
    critic_normalizer: ObsNormalizer | None,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "training_resume":
        raise ValueError(f"Not a training resume checkpoint: {path}")
    if payload.get("algorithm") != algorithm:
        raise ValueError(
            f"Resume checkpoint algorithm={payload.get('algorithm')!r} "
            f"does not match requested {algorithm!r}"
        )
    models.actor.load_state_dict(payload["actor"], strict=True)
    actor_normalizer.load_state_dict(payload["actor_obs_norm"])
    if algorithm == "ppo":
        trainer.value_critic.load_state_dict(payload["value_critic"], strict=True)
        if critic_normalizer is None:
            raise ValueError("critic_normalizer is required for PPO resume")
        critic_normalizer.load_state_dict(payload["critic_obs_norm"])
        trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        trainer.value_optimizer.load_state_dict(payload["value_optimizer"])
    restore_rng_state(payload["rng_state"], device)
    logging.getLogger(LOGGER_NAME).info(
        "Resumed from %s (env_transitions=%d gradient_updates=%d)",
        path,
        int(payload["env_transitions"]),
        int(payload["gradient_updates"]),
    )
    return {
        "env_transitions": int(payload["env_transitions"]),
        "gradient_updates": int(payload["gradient_updates"]),
        "vector_ticks": int(payload["vector_ticks"]),
        "sampled_replay_rows": int(payload.get("sampled_replay_rows", 0)),
    }


def prune_policy_artifacts(
    artifact_dir: Path, *, keep: int = DEFAULT_POLICY_ARTIFACT_KEEP
) -> None:
    paths = sorted(
        artifact_dir.glob("policy_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


def run_eval_video(
    eval_state: dict,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    models,
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
    selfplay_mgr: "SelfPlayManager | None" = None,
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
        description="Standalone QR-SAC/PPO trainer (Lee/ADR-0011 sensor path)"
    )
    parser.add_argument(
        "--algorithm",
        choices=["qrsac", "ppo"],
        default=cfg["algorithm"],
        help="Training algorithm; explicitly overrides config.algorithm.",
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
        "--self-play",
        action="store_true",
        default=False,
        help="Enable delayed self-play (implies --opponent policy): snapshot the "
        "learner into a pool and refresh the frozen policy opponent periodically.",
    )
    parser.add_argument(
        "--mixed-opponents",
        action="store_true",
        default=False,
        help="GT Sophy-style mixed opponent population on reset (implies --self-play).",
    )
    parser.add_argument(
        "--selfplay-snapshot-interval",
        type=int,
        default=cfg["selfplay"]["snapshot_interval_transitions"],
        help="Environment transitions between learner snapshots.",
    )
    parser.add_argument(
        "--selfplay-refresh-interval",
        type=int,
        default=cfg["selfplay"]["refresh_interval_transitions"],
        help="Environment transitions between opponent policy refreshes.",
    )
    parser.add_argument(
        "--selfplay-pool-size",
        type=int,
        default=cfg["selfplay"]["pool_size"],
        help="Maximum number of past learner snapshots kept in the opponent pool.",
    )
    parser.add_argument(
        "--selfplay-sample",
        type=str,
        default=cfg["selfplay"]["sample_mode"],
        choices=["latest", "uniform", "mixed"],
        help="How to sample an opponent snapshot from the pool.",
    )
    parser.add_argument(
        "--selfplay-mixed-latest-prob",
        type=float,
        default=cfg["selfplay"]["mixed_latest_prob"],
        help="When sample=mixed, probability of picking the latest snapshot.",
    )
    parser.add_argument(
        "--selfplay-anchor-ckpt",
        type=str,
        default=cfg["selfplay"]["anchor_ckpt"],
        help="Immutable incumbent policy artifact for the opponent population.",
    )
    parser.add_argument(
        "--selfplay-anchor-prob",
        type=float,
        default=cfg["selfplay"]["anchor_prob"],
        help="Probability of sampling the anchor instead of the rolling pool.",
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
        help="torch.compile learner updates (default on; --no-compile for reference).",
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
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default=None,
        help="Resume full optimizer/critic/normalizer state from a rolling checkpoint.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run artifact directory (default: outputs/runs/<run-id>/).",
    )
    parser.add_argument(
        "--reuse-run-dir",
        action="store_true",
        help=(
            "Allow attaching to a run directory whose config.json belongs to a "
            "different run (manual cleanup required; default is refuse)."
        ),
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


def run_identity_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "algorithm": str(args.algorithm),
        "config": str(Path(args.config).resolve()) if args.config else None,
        "total_transitions": int(args.total_transitions),
        "init_ckpt": (
            str(Path(args.init_ckpt).resolve())
            if getattr(args, "init_ckpt", None)
            else None
        ),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
    }


def fingerprint_from_snapshot(snapshot: dict) -> dict:
    args = snapshot.get("args", {})
    return {
        "algorithm": str(
            args.get("algorithm", snapshot.get("config", {}).get("algorithm", "qrsac"))
        ),
        "config": args.get("config"),
        "total_transitions": int(args.get("total_transitions", 0)),
        "init_ckpt": args.get("init_ckpt"),
        "seed": int(args.get("seed", 0)),
        "num_envs": int(args.get("num_envs", 0)),
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run_attach_error(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def assert_run_dir_exclusive(run_dir: Path, args: argparse.Namespace) -> None:
    """Refuse to attach when run_dir already belongs to another live or completed run."""
    fingerprint = run_identity_fingerprint(args)
    config_path = config_snapshot_path(run_dir)
    log_path = run_log_path(run_dir)
    lock_path = run_lock_path(run_dir)

    if log_path.is_file():
        log_text = log_path.read_text(errors="replace")
        if "Training finished" in log_text:
            _run_attach_error(
                f"{run_dir} already contains a completed run (Training finished in run.log). "
                "Pick a new --run-id or archive the directory before restarting."
            )

    if config_path.is_file():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _run_attach_error(
                f"{run_dir}/config.json exists but is unreadable ({exc}). "
                "Refusing to overwrite an ambiguous run directory."
            )
        existing_fp = fingerprint_from_snapshot(existing)
        if existing_fp != fingerprint and not getattr(args, "reuse_run_dir", False):
            _run_attach_error(
                f"{run_dir} belongs to a different run "
                f"(existing config={existing_fp.get('config')} "
                f"total_transitions={existing_fp.get('total_transitions')} "
                f"init_ckpt={existing_fp.get('init_ckpt')}); "
                f"refusing to attach "
                f"(config={fingerprint.get('config')} "
                f"total_transitions={fingerprint.get('total_transitions')} "
                f"init_ckpt={fingerprint.get('init_ckpt')})."
            )

    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _run_attach_error(
                f"{run_dir}/run.lock exists but is unreadable ({exc}). "
                "Another trainer may own this directory; resolve manually."
            )
        other_pid = lock.get("pid")
        lock_fp = lock.get("fingerprint")
        if isinstance(other_pid, int) and other_pid != os.getpid() and _pid_alive(other_pid):
            _run_attach_error(
                f"{run_dir} is locked by live trainer pid={other_pid} "
                f"(started {lock.get('started_at')}). Refusing second attach."
            )
        if lock_fp and lock_fp != fingerprint:
            _run_attach_error(
                f"{run_dir}/run.lock fingerprint does not match this launch "
                f"(locked={lock_fp}, requested={fingerprint})."
            )


def write_run_lock(run_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": run_identity_fingerprint(args),
    }
    run_lock_path(run_dir).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


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

    patch, patch_meta = (None, None)
    if args.config is not None:
        patch, patch_meta = load_config_patch(args.config)
    cfg = build_config(args, patch=patch, explicit=explicit)
    # Inductor and PPO's adaptive-pooling backward have no deterministic CUDA
    # implementation; retain deterministic checks but warn on those paths.
    torch.use_deterministic_algorithms(
        True, warn_only=args.compile or cfg["algorithm"] == "ppo"
    )
    training_root = Path(__file__).resolve().parent
    source_git = training_source_git_state(training_root)
    provenance = config_provenance(patch_meta, explicit)
    provenance["algorithm"] = cfg["algorithm"]
    provenance["source_git"] = source_git
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]
    algorithm = cfg["algorithm"]
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
    assert_run_dir_exclusive(run_dir, args)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = checkpoint_dir(run_dir)
    config_path = config_snapshot_path(run_dir)
    trainer_log_path = run_log_path(run_dir)
    write_run_lock(run_dir, args)

    log = setup_trainer_logging(log_file=trainer_log_path)
    log.info("Run id: %s  run_dir: %s  checkpoints: %s", run_id, run_dir, ckpt_dir)

    snapshot = build_run_snapshot(run_id, run_dir, args, cfg, provenance)
    config_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    log.info("Wrote config snapshot to %s", config_path)
    log.info("Using device: %s", device)
    if algorithm == "qrsac":
        log.info(
            "Effective experiment: steering_action_mode=%s steering_delta_max_rad=%.12f "
            "control_hz=%.1f total_transitions=%d wall_contact_geometry=%s "
            "footprint_m=%.3fx%.3f wall_contact_coefficient=%.1f/m "
            "wall_contact_formula='-coefficient*speed_mps' "
            "replay=%d compile=%s compile_mode=%s",
            env_cfg.get("steering_action_mode", "delta"),
            float(env_cfg.get("steering_delta_max_rad", math.pi / 60.0)),
            1.0 / (float(env_cfg["sim_dt"]) * int(env_cfg["control_interval"])),
            int(args.total_transitions),
            "first_projected_footprint_edge_intersects_mapped_wall",
            float(env_cfg["car_length"]),
            float(env_cfg["car_width"]),
            float(reward_cfg["wall_contact_coefficient"]),
            int(model_cfg["replay_buffer_limit"]),
            bool(args.compile),
            args.compile_mode,
        )
    else:
        log.info(
            "Effective experiment: algorithm=ppo steering_action_mode=%s "
            "steering_delta_max_rad=%.12f control_hz=%.1f total_transitions=%d "
            "wall_contact_geometry=%s footprint_m=%.3fx%.3f "
            "wall_contact_coefficient=%.1f/m "
            "wall_contact_formula='-coefficient*speed_mps' "
            "compile=%s compile_mode=%s",
            env_cfg.get("steering_action_mode", "delta"),
            float(env_cfg.get("steering_delta_max_rad", math.pi / 60.0)),
            1.0 / (float(env_cfg["sim_dt"]) * int(env_cfg["control_interval"])),
            int(args.total_transitions),
            "first_projected_footprint_edge_intersects_mapped_wall",
            float(env_cfg["car_length"]),
            float(env_cfg["car_width"]),
            float(reward_cfg["wall_contact_coefficient"]),
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

    actor_obs_dim = int(obs_cfg["num_actor_obs"])
    critic_obs_dim = int(obs_cfg["num_obs"])
    act_dim = cfg["env"]["num_actions"]
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
    if algorithm == "qrsac":
        models, trainer = build_models(
            cfg,
            device,
            alpha=args.alpha,
            compile=args.compile,
            compile_mode=args.compile_mode,
        )
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
    else:
        models, trainer = build_ppo_models(
            cfg,
            device,
            actor_normalizer,
            critic_normalizer,
            args.num_envs,
            compile=args.compile,
            compile_mode=args.compile_mode,
        )
        buffer = None
        selected_capacity = 0
        replay_estimate = None

    resume_counters = None
    if args.resume_ckpt is not None and args.init_ckpt is not None:
        raise ValueError("Cannot use both --init-ckpt and --resume-ckpt")
    if args.resume_ckpt is not None:
        if algorithm != "ppo":
            raise ValueError("--resume-ckpt is only supported for PPO")
        resume_counters = load_resume_state(
            args.resume_ckpt,
            algorithm=algorithm,
            models=models,
            trainer=trainer,
            actor_normalizer=actor_normalizer,
            critic_normalizer=critic_normalizer,
            device=device,
        )
        init_transitions = resume_counters["env_transitions"]
    elif args.init_ckpt is not None:
        init_transitions = load_init_ckpt(
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
    else:
        init_transitions = 0

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

    selfplay_mgr: SelfPlayManager | None = None
    if getattr(args, "self_play", False) or getattr(args, "mixed_opponents", False):
        sp_cfg = cfg["selfplay"]
        selfplay_mgr = SelfPlayManager(
            pool_size=int(sp_cfg["pool_size"]),
            snapshot_interval_transitions=int(
                sp_cfg["snapshot_interval_transitions"]
            ),
            refresh_interval_transitions=int(
                sp_cfg["refresh_interval_transitions"]
            ),
            sample_mode=str(sp_cfg["sample_mode"]),
            mixed_latest_prob=float(sp_cfg["mixed_latest_prob"]),
            anchor_prob=float(sp_cfg["anchor_prob"]),
            expected_architecture=actor_architecture_from_module(models.actor),
            log=log,
        )
        if sp_cfg.get("anchor_ckpt"):
            selfplay_mgr.load_anchor(
                str(sp_cfg["anchor_ckpt"]),
                device,
                actor_obs_dim,
                int(cfg["env"]["num_actions"]),
                expected_layout_version=int(obs_cfg["actor_layout_version"]),
                expected_architecture=actor_architecture_from_module(models.actor),
                expected_critic_obs_dim=int(obs_cfg["num_obs"]),
                expected_steering_action_mode=str(
                    env_cfg.get("steering_action_mode", STEERING_ACTION_MODE)
                ),
                expected_steering_delta_max_rad=float(
                    env_cfg.get("steering_delta_max_rad", math.pi / 60.0)
                ),
            )

    use_1v1 = (
        champion_mgr is not None
        or selfplay_mgr is not None
        or args.opponent != "none"
    )
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
        tags.append(algorithm)
        wandb_config = {
            **cfg,
            **vars(args),
            "config_provenance": provenance,
        }
        if algorithm == "qrsac":
            wandb_config.update(
                {
                    "replay_capacity_selected": selected_capacity,
                    "replay_estimate": replay_estimate,
                }
            )
        init_kwargs = {
            "project": os.getenv("WANDB_PROJECT", "f1tenth-genesis"),
            "name": f"standalone_{run_id}",
            "id": run_id,
            "config": wandb_config,
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
    if algorithm == "ppo":
        trainer.initialize(actor_obs, critic_obs)
    if selfplay_mgr is not None:
        selfplay_mgr.seed_snapshot(
            SelfPlayManager.make_snapshot(
                models, actor_normalizer, transitions=init_transitions
            )
        )
        selfplay_mgr.bootstrap_opponent(env)
    if resume_counters is not None:
        vector_ticks = resume_counters["vector_ticks"]
        env_transitions = resume_counters["env_transitions"]
        sampled_replay_rows = resume_counters["sampled_replay_rows"]
        gradient_updates = resume_counters["gradient_updates"]
    else:
        vector_ticks = 0
        env_transitions = 0
        sampled_replay_rows = 0
        gradient_updates = 0
    replay_inserts = (
        torch.zeros((), device=device, dtype=torch.long)
        if algorithm == "qrsac"
        else 0
    )
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
    last_log_transitions = env_transitions
    last_log_replay_inserts = 0
    last_log_gradient_updates = 0
    last_log_sampled_rows = 0
    eval_state: dict = {}
    num_sequences = max(1, int(args.batch_size) // REPLAY_TRAIN_LEN)
    # Cold-start only: one-shot flags begin unset; systemd restart => new process.
    protocol_state = initial_training_protocol_state(algorithm)
    lidar_aug_enabled = bool(cfg["model"]["lidar_aug_enabled"])
    lidar_aug_max_shift = int(cfg["model"]["lidar_aug_max_shift_beams"])
    replay_full_reinit_enabled = bool(cfg["model"]["replay_full_reinit"])
    actor_freeze_transitions = int(cfg["model"]["actor_freeze_transitions"])
    actor_lr_ramp_transitions = int(cfg["model"]["actor_lr_ramp_transitions"])
    actor_unfreeze_logged = False
    if algorithm == "qrsac":
        log.info(
            "Training protocol: actor_lr=%.3e critic_lr=%.3e alpha=%.4f "
            "actor_freeze_transitions=%d actor_lr_ramp_transitions=%d "
            "replay_full_reinit=%s lidar_aug_enabled=%s lidar_aug_max_shift_beams=%d",
            trainer.actor_lr,
            trainer.critic_lr,
            args.alpha,
            actor_freeze_transitions,
            actor_lr_ramp_transitions,
            replay_full_reinit_enabled,
            lidar_aug_enabled,
            lidar_aug_max_shift,
        )
    else:
        log.info(
            "Training protocol: algorithm=ppo rollout_steps=%d epochs=%d "
            "env_minibatches=%d env_minibatch_size=%d actor_lr=%.3e value_lr=%.3e "
            "compile=%s compile_mode=%s lidar_aug_enabled=%s "
            "lidar_aug_max_shift_beams=%d",
            trainer.rollout_steps,
            trainer.num_epochs,
            args.num_envs // trainer.env_minibatch_size,
            trainer.env_minibatch_size,
            float(cfg["ppo"]["actor_lr"]),
            float(cfg["ppo"]["value_lr"]),
            trainer.compile,
            trainer.compile_mode,
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
        if recurrent_actor and algorithm == "qrsac"
        else None
    )
    collection_actor_step = (
        models.actor.step if recurrent_actor and algorithm == "qrsac" else None
    )
    if recurrent_actor and algorithm == "qrsac" and args.compile:
        collection_actor_step = torch.compile(
            collection_actor_step, mode=args.compile_mode
        )
    ppo_metric_accum: dict[str, float] = {}

    try:
        while (
            training_should_continue(
                env_transitions, args.total_transitions, continuous
            )
            or (
                algorithm == "ppo"
                and trainer.rollout_position > 0
            )
        ):
            previous_transitions = env_transitions
            # Pre-action hidden is what trajectory replay checkpoints at boundaries.
            hidden_checkpoint = learner_hidden

            if algorithm == "ppo":
                actions = trainer.act(actor_obs, critic_obs)
            elif env_transitions < args.min_train_transitions:
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
            ppo_metrics = None
            if algorithm == "ppo":
                ppo_timed_out = pure_timeout_mask(extras["termination"])
                ppo_metrics = trainer.observe(
                    next_actor_obs,
                    next_critic_obs,
                    reward,
                    done,
                    timed_out=ppo_timed_out,
                    timeout_critic_obs=extras["observations"]["terminal_critic"],
                )
                if ppo_metrics is not None:
                    minibatches = int(ppo_metrics["minibatches"])
                    gradient_updates += int(ppo_metrics["actor_updates"])
                    gradient_updates += int(ppo_metrics["value_updates"])
                    sampled_replay_rows += (
                        minibatches
                        * trainer.rollout_steps
                        * trainer.env_minibatch_size
                    )
                    policy_loss_accum += float(ppo_metrics["policy_loss"])
                    critic_loss_accum += float(ppo_metrics["value_loss"])
                    loss_count += 1
                    for key, value in ppo_metrics.items():
                        ppo_metric_accum[key] = ppo_metric_accum.get(key, 0.0) + float(
                            value
                        )

            episode_rewards += reward
            done_f = done.to(episode_rewards.dtype)
            ep_return_sum += (episode_rewards * done_f).sum()
            ep_return_count += done_f.sum()
            if selfplay_mgr is not None and critic_obs.shape[-1] > OPP_TRACK_GAP_IDX:
                done_bool = done.bool()
                if done_bool.any():
                    ego_minus_opp = -critic_obs[done_bool, OPP_TRACK_GAP_IDX]
                    selfplay_mgr.record_episode_outcomes(ego_minus_opp)
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

            if algorithm == "qrsac":
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
                    selfplay_mgr=selfplay_mgr,
                    env=env,
                    log=log,
                    wandb_run=wandb_run,
                )
                actor_normalizer.update(next_actor_obs)
                critic_normalizer.update(next_critic_obs)
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs

            if algorithm == "qrsac" and env_transitions >= args.min_train_transitions:
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
                    trainer.actor_frozen = (
                        env_transitions < actor_freeze_transitions
                    )
                    if actor_lr_schedule_active(
                        actor_freeze_transitions, actor_lr_ramp_transitions
                    ) and not trainer.actor_frozen:
                        trainer.set_actor_learning_rate(
                            effective_actor_learning_rate(
                                env_transitions,
                                actor_freeze_transitions,
                                actor_lr_ramp_transitions,
                                trainer.actor_lr,
                            )
                        )
                    if (
                        not trainer.actor_frozen
                        and actor_freeze_transitions > 0
                        and not actor_unfreeze_logged
                    ):
                        actor_unfreeze_logged = True
                        log.info(
                            "Actor unfreeze at transitions=%d "
                            "(actor_freeze_transitions=%d, "
                            "actor_lr_ramp_transitions=%d, actor_lr=%.3e)",
                            env_transitions,
                            actor_freeze_transitions,
                            actor_lr_ramp_transitions,
                            effective_actor_learning_rate(
                                env_transitions,
                                actor_freeze_transitions,
                                actor_lr_ramp_transitions,
                                trainer.actor_lr,
                            ),
                        )
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

            if selfplay_mgr is not None and (
                algorithm == "qrsac" or ppo_metrics is not None
            ):
                selfplay_mgr.maybe_snapshot(
                    models, actor_normalizer, env_transitions
                )
                selfplay_mgr.maybe_refresh(env, env_transitions)

            if interval_crossed(
                previous_transitions,
                env_transitions,
                args.log_interval_transitions,
            ):
                now = time.perf_counter()
                elapsed = now - last_log_time
                ri = int(replay_inserts)
                bsize = int(buffer.size) if buffer is not None else 0
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
                if algorithm == "qrsac":
                    buffer_fill_pct = 100.0 * bsize / buffer.capacity
                    log.info(
                        "ticks=%d transitions=%d replay_inserts=%d "
                        "buffer=%d/%d (%.1f%%) gradient_updates=%d "
                        "ticks/s=%.1f transitions/s=%.1f inserts/s=%.1f "
                        "sampled_rows/s=%.1f updates/s=%.2f "
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
                else:
                    log.info(
                        "ticks=%d transitions=%d optimized_rows=%d "
                        "gradient_updates=%d ticks/s=%.1f transitions/s=%.1f "
                        "optimized_rows/s=%.1f updates/s=%.2f "
                        + TRAINING_SUMMARY_REWARD_TAIL,
                        vector_ticks,
                        env_transitions,
                        sampled_replay_rows,
                        gradient_updates,
                        vector_ticks_per_sec,
                        transitions_per_sec,
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
                    if loss_count:
                        log.info(
                            "  ppo: value_loss=%.5f neg_logp=%.5f approx_kl=%.5f "
                            "policy_entropy=%.3f policy_clip_frac=%.3f value_clip_frac=%.3f "
                            "actor_grad_norm=%.3f value_grad_norm=%.3f "
                            "epochs=%.2f minibatches=%.2f "
                            "advantage_filter_discard_rate=%.4f "
                            "advantage_filter_retained=%.1f",
                            ppo_metric_accum["value_loss"] / loss_count,
                            ppo_metric_accum["negative_log_prob"] / loss_count,
                            ppo_metric_accum["approx_kl"] / loss_count,
                            ppo_metric_accum["policy_entropy"] / loss_count,
                            ppo_metric_accum["policy_clip_fraction"] / loss_count,
                            ppo_metric_accum["value_clip_fraction"] / loss_count,
                            ppo_metric_accum["actor_grad_norm"] / loss_count,
                            ppo_metric_accum["value_grad_norm"] / loss_count,
                            ppo_metric_accum["epochs"] / loss_count,
                            ppo_metric_accum["minibatches"] / loss_count,
                            ppo_metric_accum["advantage_filter_discard_rate"] / loss_count,
                            ppo_metric_accum["advantage_filter_retained"] / loss_count,
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
                        "wall_contact=%.4f steer_chg=%.4f steer_hist=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/passing"),
                        diag.mean("reward_term/collision"),
                        diag.mean("reward_term/wall_contact"),
                        diag.mean("reward_term/steering_change"),
                        diag.mean("reward_term/steering_history"),
                    )
                else:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f wall_contact=%.4f "
                        "steer_chg=%.4f steer_hist=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/wall_contact"),
                        diag.mean("reward_term/steering_change"),
                        diag.mean("reward_term/steering_history"),
                    )
                log.info(
                    "  reward_events: wall_contact_when=%.4f wall_contact_events=%d "
                    "boundary_contact_when=%.4f boundary_contact_events=%d "
                    "oob_impact_when=%.4f oob_impact_events=%d",
                    diag.mean("reward_term/wall_contact_when_event"),
                    int(diag.total("metric/wall_contact_events")),
                    diag.mean("reward_term/boundary_contact_when_event"),
                    int(diag.total("metric/boundary_contact_events")),
                    diag.mean("reward_term/oob_impact_when_event"),
                    int(diag.total("metric/oob_impact_events")),
                )
                log.info(
                    "  env: speed=%.3f opp_speed=%.3f lat_err=%.3f "
                    "wall_contact_frac=%.3f "
                    "progress_ds=%.4f laps_completed=%d | "
                    "throttle[%.2f..%.2f] steer[%.2f..%.2f] obs_absmax=%.2f "
                    "norm_obs_absmax=%.2f",
                    diag.mean("metric/speed_xy"),
                    diag.mean("metric/opp_speed"),
                    diag.mean("metric/lateral_error"),
                    diag.mean("metric/wall_contact"),
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
                    elif selfplay_mgr is not None:
                        opp_age = (
                            env_transitions - selfplay_mgr.opponent_transitions
                            if selfplay_mgr.opponent_transitions is not None
                            else -1
                        )
                        log.info(
                            "  selfplay: pool_size=%d opp_transitions=%s "
                            "opp_age=%d win_rate=%.3f (n=%d)",
                            len(selfplay_mgr.pool),
                            selfplay_mgr.opponent_transitions,
                            opp_age,
                            selfplay_mgr.win_rate(),
                            selfplay_mgr._episode_total,
                        )
                        selfplay_mgr.reset_win_stats()
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
                    wandb_metrics = {
                            "env_transitions": env_transitions,
                            "vector_ticks": vector_ticks,
                            "gradient_updates": gradient_updates,
                            "train/policy_loss": mean_policy_loss,
                            "train/critic_loss": mean_critic_loss,
                            "train/mean_ep_reward": mean_ep_reward,
                            "episode/lifespan_mean_s": diag.mean(
                                "episode/lifespan_s"
                            ),
                            "perf/vector_ticks_per_sec": vector_ticks_per_sec,
                            "perf/env_transitions_per_sec": transitions_per_sec,
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
                            "reward/wall_contact": diag.mean(
                                "reward_term/wall_contact"
                            ),
                            "reward/wall_contact_when_event": diag.mean(
                                "reward_term/wall_contact_when_event"
                            ),
                            "reward/boundary_contact": diag.mean(
                                "reward_term/boundary_contact"
                            ),
                            "reward/boundary_contact_events": diag.total(
                                "metric/boundary_contact_events"
                            ),
                            "reward/oob_impact": diag.mean(
                                "reward_term/oob_impact"
                            ),
                            "reward/oob_impact_events": diag.total(
                                "metric/oob_impact_events"
                            ),
                            "reward/steering_change": diag.mean(
                                "reward_term/steering_change"
                            ),
                            "reward/steering_history": diag.mean(
                                "reward_term/steering_history"
                            ),
                            "env/speed_xy": diag.mean("metric/speed_xy"),
                            "env/lateral_error": diag.mean("metric/lateral_error"),
                            "env/wall_contact_frac": diag.mean(
                                "metric/wall_contact"
                            ),
                            "env/wall_contact_events": diag.total(
                                "metric/wall_contact_events"
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
                    }
                    if algorithm == "qrsac":
                        wandb_metrics.update(
                            {
                                "replay_inserts": ri,
                                "sampled_replay_rows": sampled_replay_rows,
                                "buffer/size": bsize,
                                "perf/replay_inserts_per_sec": inserts_per_sec,
                                "perf/sampled_replay_rows_per_sec": (
                                    sampled_rows_per_sec
                                ),
                            }
                        )
                    else:
                        wandb_metrics.update(
                            {
                                "optimized_rows": sampled_replay_rows,
                                "perf/optimized_rows_per_sec": sampled_rows_per_sec,
                            }
                        )
                        if loss_count:
                            wandb_metrics.update(
                                {
                                    f"ppo/{key}": value / loss_count
                                    for key, value in ppo_metric_accum.items()
                                }
                            )
                    wandb_run.log(
                        wandb_metrics,
                        step=env_transitions,
                    )
                policy_loss_accum.zero_()
                critic_loss_accum.zero_()
                ep_return_sum.zero_()
                ep_return_count.zero_()
                loss_count = 0
                ppo_metric_accum.clear()
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
                prune_policy_artifacts(ckpt_dir)
                if algorithm == "ppo":
                    save_resume_state(
                        resume_state_path(run_dir),
                        algorithm=algorithm,
                        models=models,
                        trainer=trainer,
                        actor_normalizer=actor_normalizer,
                        critic_normalizer=critic_normalizer,
                        env_transitions=env_transitions,
                        gradient_updates=gradient_updates,
                        vector_ticks=vector_ticks,
                        sampled_replay_rows=sampled_replay_rows,
                        device=device,
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
                        selfplay_mgr=selfplay_mgr,
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
        prune_policy_artifacts(ckpt_dir)
        if algorithm == "ppo":
            save_resume_state(
                resume_state_path(run_dir),
                algorithm=algorithm,
                models=models,
                trainer=trainer,
                actor_normalizer=actor_normalizer,
                critic_normalizer=critic_normalizer,
                env_transitions=env_transitions,
                gradient_updates=gradient_updates,
                vector_ticks=vector_ticks,
                sampled_replay_rows=sampled_replay_rows,
                device=device,
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

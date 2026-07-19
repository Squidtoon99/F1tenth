#!/usr/bin/env python3
"""Single-process QRSAC trainer: F1tenthEnv + in-memory n-step replay, no Reverb/Redis/S3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn

from f1tenth_contract import OBS_PREPROCESSING_VERSION, validate_policy_artifact
from f1tenth_contract.action import CONTROL_HZ

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.utils import episode_length_for_track
from evaluation import deterministic_rollout
from run_layout import checkpoint_dir, config_snapshot_path, default_run_dir, run_log_path
from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from qrsac.spinningup.core import ALLOWED_LIDAR_POOL_BINS

LOGGER_NAME = "standalone_trainer"
# Opponent-relative block starts right after the base observation (tyre_load ends
# the base vector); index 4 within it is the signed along-track gap s_other-s_self.
OPP_OBS_BASE_IDX = 384
OPP_TRACK_GAP_IDX = OPP_OBS_BASE_IDX + 4

# Dual-observation replay: only these capacities are chosen automatically.
REPLAY_CAPACITY_REQUESTED = 2_000_000
REPLAY_CAPACITY_FALLBACK = 1_000_000
REPLAY_OBS_DTYPE = torch.float16
ARTIFACT_SCOPE_SIM_TRAINING = "simulation_training_only"
SENSOR_POLICY_FORMAT_VERSION = 3


def _tensor_shape(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def normalize_actor_architecture(architecture: dict) -> dict:
    """JSON-round-trip so list/tuple and int/numpy scalar forms compare equal."""
    return json.loads(
        json.dumps(architecture, default=lambda v: int(v) if hasattr(v, "item") else v)
    )


def architectures_match(left, right) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return normalize_actor_architecture(left) == normalize_actor_architecture(right)


def actor_architecture_from_module(actor) -> dict:
    return normalize_actor_architecture(dict(actor.actor_architecture))


def reference_actor_from_architecture(architecture: dict):
    """Construct an eager actor whose state_dict shapes match ``architecture``."""
    arch = normalize_actor_architecture(architecture)
    name = arch.get("name")
    if name not in ("flat_mlp", "lidar_cnn"):
        raise ValueError(
            f"Unsupported actor_architecture.name={name!r}; "
            "expected 'flat_mlp' or 'lidar_cnn'"
        )
    return make_actor(
        actor_type=name,
        obs_dim=int(arch["obs_dim"]),
        act_dim=int(arch["action_dim"]),
        hidden_sizes=list(arch["hidden_layers"]),
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=int(arch.get("pool_bins", 32)),
        lidar_dim=int(arch.get("lidar_dim", 1081)),
        proprio_dim=int(arch.get("proprio_dim", 12)),
    )


def _first_state_dict_mismatch(
    actual: dict, expected: dict
) -> str | None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        return f"missing actor key {missing[0]!r}"
    unexpected = sorted(actual_keys - expected_keys)
    if unexpected:
        return f"unexpected actor key {unexpected[0]!r}"
    for key in sorted(expected_keys):
        want = _tensor_shape(expected[key])
        got = _tensor_shape(actual[key])
        if got != want:
            return f"actor[{key!r}] shape={got}; expected {want}"
    return None


class SelfPlaySnapshot(dict):
    """CPU snapshot: actor state_dict + obs-norm stats + transition count."""

    actor: dict[str, torch.Tensor]
    mean: torch.Tensor
    var: torch.Tensor
    transitions: int


class SelfPlayManager:
    """Delayed self-play: snapshot learner into a pool, refresh opponent periodically."""

    def __init__(
        self,
        pool_size: int = 5,
        snapshot_interval_transitions: int = 10_240_000,
        refresh_interval_transitions: int = 2_560_000,
        sample_mode: str = "mixed",
        mixed_latest_prob: float = 0.8,
        anchor_prob: float = 0.0,
        expected_architecture: dict | None = None,
        log: logging.Logger | None = None,
    ):
        self.pool_size = pool_size
        self.snapshot_interval_transitions = snapshot_interval_transitions
        self.refresh_interval_transitions = refresh_interval_transitions
        self.sample_mode = sample_mode
        self.mixed_latest_prob = mixed_latest_prob
        self.anchor_prob = anchor_prob
        self.expected_architecture = (
            normalize_actor_architecture(expected_architecture)
            if expected_architecture is not None
            else None
        )
        self.anchor: SelfPlaySnapshot | None = None
        self.log = log or logging.getLogger(LOGGER_NAME)
        self.pool: deque[SelfPlaySnapshot] = deque(maxlen=pool_size)
        self.opponent_transitions: int | None = None
        self._last_snapshot_transitions = 0
        self._last_refresh_transitions = 0
        self._episode_wins = 0
        self._episode_total = 0

    def _require_homogeneous_architecture(self, architecture: dict) -> dict:
        arch = normalize_actor_architecture(architecture)
        if self.expected_architecture is None:
            self.expected_architecture = arch
            return arch
        if not architectures_match(arch, self.expected_architecture):
            raise ValueError(
                "Self-play pool is architecture-homogeneous; got "
                f"{arch!r}, expected {self.expected_architecture!r}"
            )
        return arch

    @staticmethod
    def make_snapshot(
        models: Models, normalizer: ObsNormalizer, transitions: int
    ) -> SelfPlaySnapshot:
        return SelfPlaySnapshot(
            actor={
                k: v.detach().cpu().clone()
                for k, v in models.actor.state_dict().items()
            },
            mean=normalizer.mean.detach().cpu().clone(),
            var=normalizer.var.detach().cpu().clone(),
            transitions=transitions,
            actor_architecture=actor_architecture_from_module(models.actor),
        )

    def load_anchor(
        self,
        path: str,
        device: torch.device,
        obs_dim: int,
        action_dim: int,
        *,
        expected_layout_version: int | None = None,
        expected_architecture: dict | None = None,
        expected_critic_obs_dim: int | None = None,
    ) -> None:
        """Load an immutable incumbent anchor from a policy artifact.

        The anchor is kept separate from the rolling ``deque`` (never evicted)
        and sampled with ``anchor_prob`` on each refresh/selection.
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        architecture = expected_architecture
        if architecture is None and self.expected_architecture is not None:
            architecture = self.expected_architecture
        if architecture is None and isinstance(payload, dict):
            architecture = payload.get("actor_architecture")
        if expected_layout_version is not None or architecture is not None:
            validate_sensor_policy_artifact(
                payload,
                expected_actor_obs_dim=obs_dim,
                expected_action_dim=action_dim,
                expected_layout_version=(
                    expected_layout_version
                    if expected_layout_version is not None
                    else int(payload.get("actor_layout_version", 1))
                ),
                expected_architecture=architecture,
                expected_critic_obs_dim=expected_critic_obs_dim,
            )
            if architecture is None:
                architecture = payload.get("actor_architecture")
        else:
            validate_policy_artifact(
                payload, expected_obs_dim=obs_dim, expected_action_dim=action_dim
            )
            architecture = payload.get("actor_architecture")
            if architecture is None:
                architecture = {
                    "name": "flat_mlp",
                    "version": 1,
                    "obs_dim": int(obs_dim),
                    "hidden_layers": [],
                    "activation": "relu",
                    "action_dim": int(action_dim),
                }
        architecture = self._require_homogeneous_architecture(architecture)
        self.anchor = SelfPlaySnapshot(
            actor={
                k: v.detach().cpu().clone() for k, v in payload["actor"].items()
            },
            mean=payload["obs_norm"]["mean"].detach().cpu().clone(),
            var=payload["obs_norm"]["var"].detach().cpu().clone(),
            transitions=int(payload.get("env_transitions", 0)),
            actor_architecture=architecture,
        )
        self.log.info(
            "Self-play anchor loaded from %s (transitions=%d anchor_prob=%.3f)",
            path,
            self.anchor["transitions"],
            self.anchor_prob,
        )

    def seed_snapshot(self, snapshot: SelfPlaySnapshot) -> None:
        arch = snapshot.get("actor_architecture")
        if arch is None:
            raise ValueError("Self-play snapshot is missing actor_architecture")
        self._require_homogeneous_architecture(arch)
        self.pool.append(snapshot)
        if self.opponent_transitions is None:
            self.opponent_transitions = snapshot["transitions"]

    def maybe_snapshot(
        self, models: Models, normalizer: ObsNormalizer, transitions: int
    ) -> bool:
        if (
            transitions - self._last_snapshot_transitions
            < self.snapshot_interval_transitions
        ):
            return False
        snap = self.make_snapshot(models, normalizer, transitions)
        self._require_homogeneous_architecture(snap["actor_architecture"])
        self.pool.append(snap)
        self._last_snapshot_transitions = transitions
        self.log.info(
            "Self-play snapshot pushed at transitions=%d (pool_size=%d)",
            transitions,
            len(self.pool),
        )
        return True

    def _sample_snapshot(self) -> SelfPlaySnapshot | None:
        if self.anchor is not None and random.random() < self.anchor_prob:
            return self.anchor
        if not self.pool:
            return None
        if self.sample_mode == "latest":
            return self.pool[-1]
        if self.sample_mode == "uniform":
            return random.choice(list(self.pool))
        if random.random() < self.mixed_latest_prob:
            return self.pool[-1]
        return random.choice(list(self.pool))

    def maybe_refresh(self, env: F1tenthEnv, transitions: int) -> bool:
        if not self.pool:
            return False
        if (
            transitions - self._last_refresh_transitions
            < self.refresh_interval_transitions
        ):
            return False
        snap = self._sample_snapshot()
        if snap is None:
            return False
        env.refresh_opponent_policy(
            snap["actor"],
            snap["mean"],
            snap["var"],
            actor_architecture=snap.get("actor_architecture"),
        )
        self.opponent_transitions = snap["transitions"]
        self._last_refresh_transitions = transitions
        self.log.info(
            "Self-play opponent refreshed at transitions=%d from snapshot transitions=%d "
            "(pool_size=%d sample=%s)",
            transitions,
            snap["transitions"],
            len(self.pool),
            self.sample_mode,
        )
        return True

    def bootstrap_opponent(self, env: F1tenthEnv) -> None:
        """Load the initial learner snapshot into the environment opponent."""
        if not self.pool:
            return
        snap = self.pool[-1]
        env.refresh_opponent_policy(
            snap["actor"],
            snap["mean"],
            snap["var"],
            actor_architecture=snap.get("actor_architecture"),
        )
        self.opponent_transitions = snap["transitions"]
        self.log.info(
            "Self-play opponent bootstrapped from snapshot transitions=%d (pool_size=%d)",
            snap["transitions"],
            len(self.pool),
        )

    def refresh_eval_opponent(self, env: F1tenthEnv) -> bool:
        """Inject a pool snapshot into a separate eval env (same sampling as
        ``maybe_refresh`` but ungated), so eval videos race the ego against the
        self-play pool instead of an untrained default opponent."""
        snap = self._sample_snapshot()
        if snap is None:
            return False
        env.refresh_opponent_policy(
            snap["actor"],
            snap["mean"],
            snap["var"],
            actor_architecture=snap.get("actor_architecture"),
        )
        return True

    def record_episode_outcomes(self, ego_minus_opp_gap: torch.Tensor) -> None:
        """Win proxy: ego ahead on track when ``s_self - s_other > 0``."""
        wins = (ego_minus_opp_gap > 0).sum().item()
        self._episode_wins += int(wins)
        self._episode_total += int(ego_minus_opp_gap.numel())

    def win_rate(self) -> float:
        if self._episode_total == 0:
            return float("nan")
        return self._episode_wins / self._episode_total

    def reset_win_stats(self) -> None:
        self._episode_wins = 0
        self._episode_total = 0


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


def validate_model_architecture(cfg: dict) -> None:
    """Reject unsupported actor types / pool widths before network construction."""
    model = cfg["model"]
    if "hidden_layers" in model:
        raise ValueError(
            "model.hidden_layers is no longer supported; set both "
            "model.actor_hidden_layers and model.critic_hidden_layers"
        )
    actor_type = model.get("actor_type")
    if actor_type not in ("flat_mlp", "lidar_cnn"):
        raise ValueError(
            f"Unsupported model.actor_type={actor_type!r}; "
            "expected 'flat_mlp' or 'lidar_cnn'"
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


def migrate_legacy_model_config(cfg: dict) -> dict:
    """In-memory migration for persisted run configs that only have hidden_layers.

    Used by evaluators opening asymmetric format-2 artifacts. Does not write the
    migrated config back to disk. New user patches must set the split keys.
    """
    model = cfg.get("model")
    if not isinstance(model, dict) or "hidden_layers" not in model:
        return cfg
    if "actor_hidden_layers" in model or "critic_hidden_layers" in model:
        raise ValueError(
            "Persisted config mixes model.hidden_layers with "
            "model.actor_hidden_layers / model.critic_hidden_layers; "
            "keep only one representation"
        )
    hidden = list(model.pop("hidden_layers"))
    model["actor_type"] = "flat_mlp"
    model["actor_hidden_layers"] = hidden
    model["critic_hidden_layers"] = list(hidden)
    model.setdefault("lidar_pool_bins", 32)
    return cfg


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
    obs_dim = cfg["obs"]["num_obs"]
    action_dim = cfg["env"]["num_actions"]
    return QuantileCritic(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg["model"]["critic_hidden_layers"],
        num_quantiles=cfg["model"]["num_quantiles"],
    )


def make_target_q_network(cfg: dict) -> QuantileCritic:
    target_q = make_q_network(cfg)
    for param in target_q.parameters():
        param.requires_grad = False
    return target_q


class NStepReplayBuffer:
    """Per-env n-step deques feeding a dual-observation float16 ring buffer."""

    def __init__(
        self,
        capacity: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        n_step: int,
        gamma: float,
        num_envs: int,
        device: torch.device,
        obs_dtype: torch.dtype = REPLAY_OBS_DTYPE,
    ):
        if actor_obs_dim == critic_obs_dim:
            raise ValueError(
                "Dual replay requires distinct actor/critic observation dimensions; "
                f"got actor_obs_dim={actor_obs_dim} critic_obs_dim={critic_obs_dim}."
            )
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.num_envs = num_envs
        self.device = device
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.obs_dtype = obs_dtype
        # ``ptr``/``size`` live on device so the write cursor advances without a
        # host synchronization every step.
        self.size = torch.zeros((), device=device, dtype=torch.long)
        self.ptr = torch.zeros((), device=device, dtype=torch.long)
        self._ready = False

        # Each step emits at most ``num_envs * (n_step + 1)`` candidate rows.
        # Masked-out candidates are scattered into a private, per-candidate
        # scratch region past ``capacity`` so every write index is unique (no
        # duplicate-index scatter, which is nondeterministic / disallowed under
        # ``use_deterministic_algorithms``) and no valid row is clobbered.
        self._max_emit = num_envs * (n_step + 1)
        rows = capacity + self._max_emit
        self.actor_obs = torch.zeros(
            rows, actor_obs_dim, device=device, dtype=obs_dtype
        )
        self.critic_obs = torch.zeros(
            rows, critic_obs_dim, device=device, dtype=obs_dtype
        )
        self.action = torch.zeros(rows, act_dim, device=device, dtype=torch.float32)
        self.reward = torch.zeros(rows, device=device, dtype=torch.float32)
        self.next_actor_obs = torch.zeros(
            rows, actor_obs_dim, device=device, dtype=obs_dtype
        )
        self.next_critic_obs = torch.zeros(
            rows, critic_obs_dim, device=device, dtype=obs_dtype
        )
        self.done = torch.zeros(rows, device=device, dtype=torch.float32)
        self._scratch = torch.arange(
            capacity, rows, device=device, dtype=torch.long
        )

        self._gamma_powers = torch.tensor(
            [gamma**k for k in range(n_step)], device=device, dtype=torch.float32
        )

        # Vectorized per-env n-step windows kept on device as circular buffers.
        # All envs advance in lockstep, so a single write column index ``w_pos``
        # is shared. ``w_len`` counts valid entries per env (reset to 0 on done).
        # Windows stay float32; ring storage is float16.
        self.w_actor_obs = torch.zeros(
            num_envs, n_step, actor_obs_dim, device=device, dtype=torch.float32
        )
        self.w_critic_obs = torch.zeros(
            num_envs, n_step, critic_obs_dim, device=device, dtype=torch.float32
        )
        self.w_act = torch.zeros(
            num_envs, n_step, act_dim, device=device, dtype=torch.float32
        )
        self.w_rew = torch.zeros(num_envs, n_step, device=device, dtype=torch.float32)
        self.w_len = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.w_pos = 0
        self._arange_n = torch.arange(n_step, device=device)

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized n-step accumulation with **no host synchronization**.

        Writes the current dual-observation transition into each env's circular
        window, then emits completed n-step samples (block A) and truncated
        terminal tails (block B) in two ring passes — same candidate order as a
        single A||B pack, without building a ``num_envs*(n_step+1)`` cat or an
        ``(num_envs, n, n)`` reward gather. Returns inserted row count."""
        num_envs, n = self.num_envs, self.n_step
        col = self.w_pos
        self.w_actor_obs[:, col] = actor_obs.detach()
        self.w_critic_obs[:, col] = critic_obs.detach()
        self.w_act[:, col] = actions.detach()
        self.w_rew[:, col] = rewards.detach()
        self.w_len = torch.clamp(self.w_len + 1, max=n)
        self.w_pos = (col + 1) % n
        dones_b = dones.bool()

        # Block A: completed n-step samples from full, non-terminal windows.
        # ``w_pos`` now points at the oldest entry of a full window.
        oldest = self.w_pos
        order = (oldest + self._arange_n) % n
        rew_a = (self.w_rew[:, order] * self._gamma_powers).sum(dim=1)
        valid_a = (self.w_len == n) & ~dones_b
        nxt_actor = next_actor_obs.detach()
        nxt_critic = next_critic_obs.detach()

        slot_a = torch.cumsum(valid_a.long(), dim=0) - 1
        n_emit_a = valid_a.long().sum()
        pos_a = torch.where(
            valid_a,
            torch.remainder(self.ptr + slot_a, self.capacity),
            self._scratch[:num_envs],
        )
        self.actor_obs[pos_a] = self.w_actor_obs[:, oldest].to(self.obs_dtype)
        self.critic_obs[pos_a] = self.w_critic_obs[:, oldest].to(self.obs_dtype)
        self.action[pos_a] = self.w_act[:, oldest]
        self.reward[pos_a] = rew_a
        self.next_actor_obs[pos_a] = nxt_actor.to(self.obs_dtype)
        self.next_critic_obs[pos_a] = nxt_critic.to(self.obs_dtype)
        self.done[pos_a] = 0.0

        # Block B: truncated tails for done envs. Chronological window index
        # ``start[:,k]`` is the k-th oldest entry; discounted returns use an O(n)
        # backward recurrence instead of an (n, n) gather.
        start = (
            self.w_pos - self.w_len.unsqueeze(1) + self._arange_n.unsqueeze(0)
        ) % n
        horizon = self.w_len.unsqueeze(1) - self._arange_n.unsqueeze(0)
        chron_rew = torch.gather(self.w_rew, 1, start)
        # S_k = r_k + gamma * S_{k+1} for k < w_len (oldest-first chronology).
        rew_b = torch.zeros(num_envs, n, device=self.device, dtype=torch.float32)
        next_s = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        for k in range(n - 1, -1, -1):
            in_window = self.w_len > k
            s_k = chron_rew[:, k] + self.gamma * next_s
            rew_b[:, k] = torch.where(in_window, s_k, rew_b[:, k])
            next_s = torch.where(in_window, s_k, torch.zeros_like(next_s))

        valid_b = dones_b.unsqueeze(1) & (horizon > 0)
        valid_b_flat = valid_b.reshape(num_envs * n)
        slot_b = torch.cumsum(valid_b_flat.long(), dim=0) - 1
        pos_b = torch.where(
            valid_b_flat,
            torch.remainder(self.ptr + n_emit_a + slot_b, self.capacity),
            self._scratch[num_envs:],
        )
        start_exp_actor = start.unsqueeze(-1).expand(num_envs, n, self.actor_obs_dim)
        start_exp_critic = start.unsqueeze(-1).expand(num_envs, n, self.critic_obs_dim)
        start_exp_act = start.unsqueeze(-1).expand(num_envs, n, self.act_dim)
        env_idx = (
            torch.arange(num_envs, device=self.device)
            .unsqueeze(1)
            .expand(num_envs, n)
            .reshape(num_envs * n)
        )
        self.actor_obs[pos_b] = torch.gather(
            self.w_actor_obs, 1, start_exp_actor
        ).reshape(num_envs * n, self.actor_obs_dim).to(self.obs_dtype)
        self.critic_obs[pos_b] = torch.gather(
            self.w_critic_obs, 1, start_exp_critic
        ).reshape(num_envs * n, self.critic_obs_dim).to(self.obs_dtype)
        self.action[pos_b] = torch.gather(
            self.w_act, 1, start_exp_act
        ).reshape(num_envs * n, self.act_dim)
        self.reward[pos_b] = rew_b.reshape(num_envs * n)
        self.next_actor_obs[pos_b] = nxt_actor[env_idx].to(self.obs_dtype)
        self.next_critic_obs[pos_b] = nxt_critic[env_idx].to(self.obs_dtype)
        self.done[pos_b] = 1.0

        n_emit = n_emit_a + valid_b_flat.long().sum()
        self.ptr = torch.remainder(self.ptr + n_emit, self.capacity)
        self.size = torch.clamp(self.size + n_emit, max=self.capacity)
        self.w_len = torch.where(dones_b, torch.zeros_like(self.w_len), self.w_len)
        return n_emit

    def is_ready(self, batch_size: int) -> bool:
        """Whether the buffer can serve a ``batch_size`` draw. Reads the device
        size once per step only until it first fills, then never again."""
        if not self._ready and int(self.size) >= batch_size:
            self._ready = True
        return self._ready

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        # Caller gates on ``is_ready``; keep this path sync-free (no ``randint``
        # high-bound read) by sampling floats scaled to the current size.
        # Observations are cast to float32 before normalization / network use.
        idx = (torch.rand(batch_size, device=self.device) * self.size).long()
        idx = torch.minimum(idx, self.size - 1)
        return {
            "actor_obs": self.actor_obs[idx].to(torch.float32),
            "critic_obs": self.critic_obs[idx].to(torch.float32),
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_actor_obs": self.next_actor_obs[idx].to(torch.float32),
            "next_critic_obs": self.next_critic_obs[idx].to(torch.float32),
            "done": self.done[idx],
        }


class ObsNormalizer:
    """Running mean/variance observation normalizer (Welford parallel update).

    Estimates per-feature mean and variance from observations actually experienced
    during training, then normalizes obs at network-input time. The replay buffer
    keeps RAW observations; normalization is applied with the current statistics
    wherever an observation enters a network, so there is no stale-normalization
    drift across the buffer. Stats are kept on-device in float32.
    """

    def __init__(
        self,
        obs_dim: int,
        device: torch.device,
        eps: float = 1e-8,
        clip: float = 10.0,
    ):
        self.device = device
        self.eps = float(eps)
        self.clip = float(clip)
        self.mean = torch.zeros(obs_dim, device=device, dtype=torch.float32)
        self.var = torch.ones(obs_dim, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Chan et al. parallel variance update from a (batch, obs_dim) tensor."""
        x = x.to(torch.float32)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * (self.count * batch_count / tot_count)
        self.var = m2 / tot_count
        self.count = tot_count

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(torch.float32) - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].to(self.device, dtype=torch.float32)
        self.var = state["var"].to(self.device, dtype=torch.float32)
        self.count = float(state["count"])


def estimate_dual_replay_bytes(
    capacity: int,
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    n_step: int,
    num_envs: int,
    *,
    obs_dtype: torch.dtype = REPLAY_OBS_DTYPE,
) -> dict[str, int]:
    """Byte estimate for dual float16 ring storage plus float32 windows/aux."""
    obs_item = torch.tensor([], dtype=obs_dtype).element_size()
    f32 = torch.tensor([], dtype=torch.float32).element_size()
    i64 = torch.tensor([], dtype=torch.long).element_size()
    rows = capacity + num_envs * (n_step + 1)
    ring_obs = 2 * rows * (actor_obs_dim + critic_obs_dim) * obs_item
    ring_aux = rows * (act_dim + 1 + 1) * f32  # action + reward + done
    window = num_envs * n_step * (
        (actor_obs_dim + critic_obs_dim + act_dim + 1) * f32
    )
    misc = num_envs * i64 + (n_step + num_envs * (n_step + 1)) * i64
    total = ring_obs + ring_aux + window + misc
    return {
        "capacity": int(capacity),
        "rows": int(rows),
        "ring_obs_bytes": int(ring_obs),
        "ring_aux_bytes": int(ring_aux),
        "window_bytes": int(window),
        "total_bytes": int(total),
    }


def _update_headroom_ok(
    device: torch.device,
    actor_obs_dim: int,
    critic_obs_dim: int,
    act_dim: int,
    batch_size: int,
) -> bool:
    """True when a float32 asymmetric update batch still fits after replay alloc."""
    if device.type != "cuda":
        return True
    try:
        torch.empty(batch_size, actor_obs_dim, device=device, dtype=torch.float32)
        torch.empty(batch_size, critic_obs_dim, device=device, dtype=torch.float32)
        torch.empty(batch_size, actor_obs_dim, device=device, dtype=torch.float32)
        torch.empty(batch_size, critic_obs_dim, device=device, dtype=torch.float32)
        torch.empty(batch_size, act_dim, device=device, dtype=torch.float32)
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
) -> tuple[NStepReplayBuffer, int, dict[str, int]]:
    """Allocate dual replay at ``capacity``, with explicit 2M→1M CUDA fallback.

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
    )
    logger.info(
        "Replay allocation estimate: capacity=%d rows=%d total_bytes=%d (%.2f GiB) "
        "ring_obs_bytes=%d actor_dim=%d critic_dim=%d dtype=%s",
        estimate["capacity"],
        estimate["rows"],
        estimate["total_bytes"],
        estimate["total_bytes"] / (1024**3),
        estimate["ring_obs_bytes"],
        actor_obs_dim,
        critic_obs_dim,
        str(REPLAY_OBS_DTYPE).replace("torch.", ""),
    )

    def _alloc(cap: int) -> NStepReplayBuffer:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return NStepReplayBuffer(
            capacity=cap,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            act_dim=act_dim,
            n_step=n_step,
            gamma=gamma,
            num_envs=num_envs,
            device=device,
            obs_dtype=REPLAY_OBS_DTYPE,
        )

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
            device, actor_obs_dim, critic_obs_dim, act_dim, batch_size
        ):
            if (
                allow_fallback
                and capacity == REPLAY_CAPACITY_REQUESTED
                and device.type == "cuda"
            ):
                logger.warning(
                    "Insufficient CUDA headroom for an asymmetric update at "
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
                )
                buffer = _alloc(capacity)
            else:
                raise RuntimeError(
                    f"Insufficient device headroom for QR-SAC update after "
                    f"allocating replay capacity={capacity}."
                )

    logger.info("Replay capacity selected: %d", capacity)
    return buffer, capacity, estimate


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
        "wall_contact",
        "progress_ds",
        "lap_count",
        "laps_completed",
        "opp_speed",
        "nonfinite_obs_envs",
        "nonfinite_reward_envs",
        "nonfinite_state_envs",
        "dr/tire_friction",
        "dr/vehicle_mass",
        "dr/drive_scale",
        "dr/steer_bias",
        "dr/action_latency_steps",
        "dr/obs_latency_steps",
        "dr/obs_noise_std",
    ):
        value = metrics.get(name)
        if isinstance(value, torch.Tensor):
            if name.startswith("nonfinite_") or name == "laps_completed":
                diag.add_total(f"metric/{name}", value)
            else:
                diag.add_mean(
                    f"metric/{name}",
                    value,
                    track_range=name.startswith("dr/"),
                )

    oob_mask = metrics.get("oob_mask")
    oob_penalty = terms.get("oob_penalty")
    if isinstance(oob_mask, torch.Tensor) and isinstance(oob_penalty, torch.Tensor):
        diag.add_event_mean(
            "reward_term/oob_penalty_when_oob", oob_penalty, oob_mask > 0
        )
    wall_mask = metrics.get("wall_contact")
    wall_penalty = terms.get("wall_penalty")
    if isinstance(wall_mask, torch.Tensor) and isinstance(wall_penalty, torch.Tensor):
        diag.add_total("metric/wall_contact_count", wall_mask)
        diag.add_event_mean(
            "reward_term/wall_penalty_when_contact", wall_penalty, wall_mask > 0
        )
    wall_impact = terms.get("wall_impact")
    if isinstance(wall_impact, torch.Tensor):
        impact_events = wall_impact != 0
        diag.add_total(
            "metric/wall_impact_events", impact_events.to(wall_impact.dtype)
        )
        diag.add_event_mean(
            "reward_term/wall_impact_when_event", wall_impact, impact_events
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
_OPTIONAL_REWARD_SCALE_KEYS = frozenset(
    {"overtake", "passing", "collision", "rear_end"}
)


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
        elif key in cfg[section]:
            setattr(args, dest, cfg[section][key])

    cli_override("track", "env", "track")
    cli_override("batch_size", "model", "batch_size")
    cli_override("min_train_transitions", "model", "minimum_train_transitions")
    cli_override(
        "sampled_rows_per_transition", "model", "sampled_replay_rows_per_transition"
    )
    cli_override("buffer_capacity", "model", "replay_buffer_limit")
    cli_override("alpha", "model", "alpha")
    cli_override("total_transitions", "schedule", "total_transitions")
    cli_override("log_interval_transitions", "schedule", "log_interval_transitions")
    cli_override(
        "export_interval_transitions", "schedule", "export_interval_transitions"
    )
    cli_override("eval_interval_transitions", "schedule", "eval_interval_transitions")

    # Episode horizon precedence: explicit CLI > patch > track-derived default.
    if "episode_length" in explicit:
        cfg["env"]["episode_length"] = float(args.episode_length)
    elif patch and "episode_length" in patch.get("env", {}):
        cfg["env"]["episode_length"] = float(cfg["env"]["episode_length"])
    else:
        lap_multiplier = (
            float(args.episode_lap_multiplier)
            if "episode_lap_multiplier" in explicit
            else float(cfg["env"].get("episode_lap_multiplier", 3.0))
        )
        workspace_dir = str(Path(__file__).resolve().parent)
        cfg["env"]["episode_length"] = episode_length_for_track(
            track=cfg["env"]["track"],
            workspace_dir=workspace_dir,
            ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
            lap_multiplier=lap_multiplier,
        )

    if args.n_step is not None:
        cfg["model"]["n_step"] = args.n_step

    cfg["env"]["domain_randomization"] = {
        **cfg["env"]["domain_randomization"],
        "enabled": True,
        # Asymmetric experiment: critic sees current privileged Frenet state.
        # Sensor latency/noise belongs only on the actor stream.
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

    if getattr(args, "zero_tyre_slip_obs", False):
        cfg["obs"]["zero_tyre_slip_obs"] = True

    cli_override(
        "selfplay_snapshot_interval", "selfplay", "snapshot_interval_transitions"
    )
    cli_override(
        "selfplay_refresh_interval", "selfplay", "refresh_interval_transitions"
    )
    cli_override("selfplay_pool_size", "selfplay", "pool_size")
    cli_override("selfplay_sample", "selfplay", "sample_mode")
    cli_override("selfplay_anchor_ckpt", "selfplay", "anchor_ckpt")
    cli_override("selfplay_anchor_prob", "selfplay", "anchor_prob")

    # 1v1: enable the opponent and passing reward. The policy input remains the
    # fixed 390-d layout in both solo and opponent modes.
    # --mixed-opponents implies self-play (the policy half of the mix is refreshed
    # from the learner snapshot pool just like pure self-play).
    self_play = args.self_play or args.mixed_opponents
    use_1v1 = self_play or args.opponent != "none"
    if use_1v1:
        if args.mixed_opponents:
            cfg["env"]["opponent_strategy"] = "mixed"
        elif args.self_play:
            cfg["env"]["opponent_strategy"] = "policy"
        else:
            cfg["env"]["opponent_strategy"] = args.opponent
        cli_override("opponent_target_speed", "env", "opponent_target_speed")
        cli_override("opponent_spawn_gap_min", "env", "opponent_spawn_gap_min_m")
        cli_override("opponent_spawn_gap_max", "env", "opponent_spawn_gap_max_m")
        cli_override("opponent_spawn_behind_prob", "env", "opponent_spawn_behind_prob")
        cli_override(
            "opponent_spawn_lateral_independent",
            "env",
            "opponent_spawn_lateral_independent",
        )
        cli_override("opponent_reset_speed_min", "env", "opponent_reset_speed_min_mps")
        cli_override("opponent_reset_speed_max", "env", "opponent_reset_speed_max_mps")
        if args.opponent_ckpt is not None:
            cfg["env"]["opponent_ckpt"] = args.opponent_ckpt
        # reward_scales resolve from the config/patch (the single source of truth
        # for the Maggiore coefficients); a CLI flag only wins when explicitly
        # passed, so a --config patch is never clobbered by an argparse default.
        if "passing_scale" in explicit:
            cfg["reward"]["reward_scales"]["passing"] = args.passing_scale
        if "collision_scale" in explicit:
            cfg["reward"]["reward_scales"]["collision"] = args.collision_scale
        if "rear_end_scale" in explicit:
            cfg["reward"]["reward_scales"]["rear_end"] = args.rear_end_scale
        if "overtake_scale" in explicit:
            cfg["reward"]["reward_scales"]["overtake"] = args.overtake_scale
        # Closing-speed threshold for collision termination (0.0 = terminate on any
        # overlap). Below it, contacts only apply penalties/physics and the episode
        # continues.
        if args.collision_term_speed is not None:
            cfg["env"]["collision_term_speed_mps"] = float(args.collision_term_speed)
    validate_model_architecture(cfg)
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
    )
    return models, trainer


def _validate_sensor_artifact_common(
    payload,
    *,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    expected_critic_obs_dim: int | None,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Sensor policy artifact must be a mapping.")
    obs_dim = payload.get("obs_dim")
    actor_obs_dim = payload.get("actor_obs_dim", obs_dim)
    if (
        obs_dim is None
        or actor_obs_dim is None
        or int(obs_dim) != int(expected_actor_obs_dim)
        or int(actor_obs_dim) != int(expected_actor_obs_dim)
    ):
        raise ValueError(
            f"Sensor policy artifact obs_dim={obs_dim!r} "
            f"actor_obs_dim={actor_obs_dim!r}; expected actor dim "
            f"{expected_actor_obs_dim}. Privileged/symmetric schemas "
            f"(e.g. 390-D Frenet actors) are not supported."
        )
    layout = payload.get("actor_layout_version")
    if layout is None or int(layout) != int(expected_layout_version):
        raise ValueError(
            f"Sensor policy artifact actor_layout_version={layout!r}; "
            f"expected {expected_layout_version}."
        )
    if "critic_norm" in payload or "critic_obs_norm" in payload:
        raise ValueError(
            "Sensor policy artifact must not include critic normalization."
        )
    for critic_key in ("critic1", "critic2", "critic1_target", "critic2_target"):
        if critic_key in payload:
            raise ValueError(
                f"Sensor policy artifact must not include critic weights ({critic_key})."
            )
    action_dim = payload.get("action_dim")
    if action_dim is None or int(action_dim) != int(expected_action_dim):
        raise ValueError(
            f"Sensor policy artifact action_dim={action_dim!r}; "
            f"expected {expected_action_dim}."
        )
    mode = payload.get("longitudinal_mode", payload.get("throttle_mode"))
    if mode != "force":
        raise ValueError(
            f"Sensor policy artifact longitudinal_mode={mode!r}; expected 'force'."
        )
    control_hz = payload.get("control_hz")
    if control_hz is None or abs(float(control_hz) - CONTROL_HZ) > 1.0e-6:
        raise ValueError(
            f"Sensor policy artifact control_hz={control_hz!r}; expected {CONTROL_HZ}."
        )
    if expected_critic_obs_dim is not None:
        critic_obs_dim = payload.get("critic_obs_dim")
        if critic_obs_dim is None or int(critic_obs_dim) != int(expected_critic_obs_dim):
            raise ValueError(
                f"Sensor policy artifact critic_obs_dim={critic_obs_dim!r}; "
                f"expected provenance dim {expected_critic_obs_dim}."
            )
    stats = payload.get("obs_norm")
    if not isinstance(stats, dict):
        raise ValueError("Sensor policy artifact is missing obs_norm statistics.")
    for name in ("mean", "var"):
        shape = _tensor_shape(stats.get(name))
        if shape != (int(expected_actor_obs_dim),):
            raise ValueError(
                f"Sensor policy artifact obs_norm.{name} shape={shape}; "
                f"expected ({expected_actor_obs_dim},)."
            )


def _validate_format3_actor_state(
    payload: dict,
    *,
    expected_architecture: dict,
    expected_action_dim: int,
) -> None:
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, dict):
        raise ValueError(
            "Format-3 sensor policy artifact is missing actor_architecture metadata."
        )
    if not architectures_match(architecture, expected_architecture):
        raise ValueError(
            "Sensor policy artifact actor_architecture mismatch: "
            f"got {normalize_actor_architecture(architecture)!r}, "
            f"expected {normalize_actor_architecture(expected_architecture)!r}"
        )
    actor = payload.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Sensor policy artifact is missing the actor state_dict.")
    reference = reference_actor_from_architecture(expected_architecture)
    mismatch = _first_state_dict_mismatch(actor, reference.state_dict())
    if mismatch is not None:
        raise ValueError(f"Sensor policy artifact {mismatch}")
    mu_shape = _tensor_shape(actor.get("mu_layer.weight"))
    log_std_shape = _tensor_shape(actor.get("log_std_layer.weight"))
    if len(mu_shape) != 2 or mu_shape[0] != int(expected_action_dim):
        raise ValueError(
            f"Sensor policy artifact mu_layer.weight shape={mu_shape}; "
            f"expected ({expected_action_dim}, *)."
        )
    if len(log_std_shape) != 2 or log_std_shape[0] != int(expected_action_dim):
        raise ValueError(
            f"Sensor policy artifact log_std_layer.weight shape={log_std_shape}; "
            f"expected ({expected_action_dim}, *)."
        )
    reference.load_state_dict(actor, strict=True)


def validate_sensor_policy_artifact(
    payload,
    *,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    expected_architecture: dict | None = None,
    expected_critic_obs_dim: int | None = None,
) -> None:
    """Validate simulation-training sensor policy artifacts."""
    if not isinstance(payload, dict):
        raise ValueError("Sensor policy artifact must be a mapping.")
    version = payload.get("policy_format_version")
    if version is None:
        raise ValueError("Sensor policy artifact is missing policy_format_version.")
    version = int(version)

    if version >= SENSOR_POLICY_FORMAT_VERSION:
        if expected_architecture is None:
            raise ValueError(
                "Format-3 sensor policy artifacts require expected_architecture."
            )
        scope = payload.get("artifact_scope")
        if scope != ARTIFACT_SCOPE_SIM_TRAINING:
            raise ValueError(
                f"Sensor policy artifact artifact_scope={scope!r}; "
                f"expected {ARTIFACT_SCOPE_SIM_TRAINING!r}."
            )
        _validate_sensor_artifact_common(
            payload,
            expected_actor_obs_dim=expected_actor_obs_dim,
            expected_action_dim=expected_action_dim,
            expected_layout_version=expected_layout_version,
            expected_critic_obs_dim=expected_critic_obs_dim,
        )
        _validate_format3_actor_state(
            payload,
            expected_architecture=expected_architecture,
            expected_action_dim=expected_action_dim,
        )
        return

    if version < 2:
        raise ValueError(
            f"Sensor policy artifact policy_format_version={version!r} is unsupported; "
            "need >=2."
        )
    if expected_architecture is None:
        raise ValueError(
            "Format-2 sensor policy artifacts require expected_architecture "
            "so flat-only compatibility can be enforced."
        )
    expected_name = normalize_actor_architecture(expected_architecture).get("name")
    if expected_name != "flat_mlp":
        raise ValueError(
            "Format-2 flat-MLP artifacts cannot warm-start or seed a "
            f"{expected_name!r} actor; there is no MLP-to-CNN weight migration."
        )
    _validate_sensor_artifact_common(
        payload,
        expected_actor_obs_dim=expected_actor_obs_dim,
        expected_action_dim=expected_action_dim,
        expected_layout_version=expected_layout_version,
        expected_critic_obs_dim=expected_critic_obs_dim,
    )
    validate_policy_artifact(
        payload,
        expected_obs_dim=expected_actor_obs_dim,
        expected_action_dim=expected_action_dim,
    )
    reference = reference_actor_from_architecture(expected_architecture)
    mismatch = _first_state_dict_mismatch(payload["actor"], reference.state_dict())
    if mismatch is not None:
        raise ValueError(f"Sensor policy artifact {mismatch}")
    reference.load_state_dict(payload["actor"], strict=True)


def save_policy_artifact(
    models: Models,
    env_transitions: int,
    artifact_dir: Path,
    normalizer: ObsNormalizer,
    cfg: dict,
):
    """Save actor-only simulation artifact with sensor layout metadata."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"policy_{env_transitions}.pt"
    actor_obs_dim = int(cfg["obs"]["num_actor_obs"])
    architecture = actor_architecture_from_module(models.actor)
    payload = {
        "actor": models.actor.state_dict(),
        "obs_norm": normalizer.state_dict(),
        "env_transitions": env_transitions,
        "obs_dim": actor_obs_dim,
        "actor_obs_dim": actor_obs_dim,
        "critic_obs_dim": int(cfg["obs"]["num_obs"]),
        "actor_layout_version": int(cfg["obs"]["actor_layout_version"]),
        "action_dim": int(cfg["env"]["num_actions"]),
        "action_scale": float(cfg["env"]["clip_actions"]),
        "config_version": int(cfg["config_version"]),
        "policy_format_version": int(cfg["policy_format_version"]),
        "artifact_scope": ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": architecture,
        "longitudinal_mode": str(cfg["env"].get("longitudinal_mode", "force")),
        "f_drive_max": float(cfg["env"].get("f_drive_max", 23.0)),
        "f_brake_max": float(cfg["env"].get("f_brake_max", 23.0)),
        "control_hz": float(
            1.0
            / (
                float(cfg["env"].get("sim_dt", 0.005))
                * float(cfg["env"].get("control_interval", 10))
            )
        ),
        "simulator_id": str(cfg["simulator"]["id"]),
        "simulator_version": int(cfg["simulator"]["version"]),
        "observation_preprocessing_version": OBS_PREPROCESSING_VERSION,
    }
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
        # Actor-only solo (1v0) eval: sensor observations, no opponent stream.
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
    parser = argparse.ArgumentParser(description="Standalone QRSAC trainer (1v0, single process)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config patch merged over DEFAULT_CONFIG (nested "
        "env/reward/model/schedule/selfplay/obs shape only). Precedence is "
        "DEFAULT_CONFIG < patch < explicitly-passed config CLI flags; runtime "
        "options (num-envs, seed, device, run/wandb/eval, init-ckpt, compile) "
        "stay CLI-only.",
    )
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--total-transitions",
        type=int,
        default=cfg["schedule"]["total_transitions"],
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=False,
        help="Opt-in unbounded training: ignore --total-transitions termination "
        "while keeping the finite budget as the default. Periodic artifacts, "
        "logs, and eval still run.",
    )
    parser.add_argument("--batch-size", type=int, default=cfg["model"]["batch_size"])
    parser.add_argument(
        "--sampled-rows-per-transition",
        type=float,
        default=cfg["model"]["sampled_replay_rows_per_transition"],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=cfg["model"]["alpha"],
        help="SAC entropy coefficient (fixed). Default 0.01 matches GT Sophy; "
        "lower temperature lets the policy commit to a fast racing line rather "
        "than staying overly stochastic.",
    )
    parser.add_argument(
        "--min-train-transitions",
        type=int,
        default=cfg["model"]["minimum_train_transitions"],
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help=f"N-step horizon (default: {cfg['model']['n_step']} from config)",
    )
    parser.add_argument("--track", type=str, default=cfg["env"]["track"])
    parser.add_argument(
        "--episode-length",
        type=float,
        default=None,
        help="Episode horizon in seconds. Default: derived from the track centerline "
        "length (episode_lap_multiplier laps at expected_lap_speed_mps).",
    )
    parser.add_argument(
        "--episode-lap-multiplier",
        type=float,
        default=None,
        help="Number of laps the auto-derived episode horizon should cover "
        "(overrides config env.episode_lap_multiplier). Ignored if --episode-length "
        "is set.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="none",
        choices=["none", "scripted", "policy"],
        help="1v1 opponent: 'none' (solo/1v0), 'scripted' (centerline follower), "
        "or 'policy' (frozen-policy self-play opponent).",
    )
    parser.add_argument(
        "--opponent-target-speed",
        type=float,
        default=cfg["env"]["opponent_target_speed"],
        help="Scripted opponent target speed (m/s); keep below ego pace so an "
        "overtake is feasible.",
    )
    parser.add_argument(
        "--opponent-spawn-gap-min",
        type=float,
        default=cfg["env"]["opponent_spawn_gap_min_m"],
        help="Minimum opponent spawn gap magnitude (m) along the centerline.",
    )
    parser.add_argument(
        "--opponent-spawn-gap-max",
        type=float,
        default=cfg["env"]["opponent_spawn_gap_max_m"],
        help="Maximum opponent spawn gap magnitude (m) along the centerline.",
    )
    parser.add_argument(
        "--opponent-spawn-behind-prob",
        type=float,
        default=cfg["env"]["opponent_spawn_behind_prob"],
        help="Probability the opponent spawns behind the ego (else ahead).",
    )
    parser.add_argument(
        "--opponent-spawn-lateral-independent",
        action=argparse.BooleanOptionalAction,
        default=cfg["env"]["opponent_spawn_lateral_independent"],
        help="Sample an independent lateral offset for the opponent at reset.",
    )
    parser.add_argument(
        "--opponent-reset-speed-min",
        type=float,
        default=cfg["env"]["opponent_reset_speed_min_mps"],
        help="Minimum opponent launch speed at reset (m/s).",
    )
    parser.add_argument(
        "--opponent-reset-speed-max",
        type=float,
        default=cfg["env"]["opponent_reset_speed_max_mps"],
        help="Maximum opponent launch speed at reset (m/s).",
    )
    parser.add_argument(
        "--opponent-ckpt",
        type=str,
        default=None,
        help="Checkpoint for the 'policy' opponent (deferred self-play path).",
    )
    parser.add_argument(
        "--passing-scale",
        type=float,
        default=cfg["reward"]["reward_scales"]["passing"],
        help="Reward scale (Maggiore Rps coefficient) for the 1v1 passing term "
        "(track position gained on the opponent). Defaults to config "
        "reward.reward_scales.passing; only used when --opponent is not 'none'.",
    )
    parser.add_argument(
        "--collision-scale",
        type=float,
        default=cfg["reward"]["reward_scales"]["collision"],
        help="Reward scale (Maggiore Rc coefficient) for the GT Sophy any-collision "
        "penalty on car-car overlap. Defaults to config "
        "reward.reward_scales.collision; only used when --opponent is not 'none'.",
    )
    parser.add_argument(
        "--rear-end-scale",
        type=float,
        default=cfg["reward"]["reward_scales"]["rear_end"],
        help="Reward scale (Maggiore Rr coefficient) for the GT Sophy rear-end "
        "penalty (closing-speed^2 when colliding with an opponent ahead). Defaults "
        "to config reward.reward_scales.rear_end; only used when --opponent is not "
        "'none'.",
    )
    parser.add_argument(
        "--overtake-scale",
        type=float,
        default=0.0,
        help="Reward scale for the one-time overtake-completed bonus (opponent goes "
        "from ahead to behind within overtake_gap_m). 0.0 (default) disables it -- the "
        "continuous passing reward already drives overtaking, so the discrete bonus "
        "fired at noise level and is off. Only used when --opponent is not 'none'.",
    )
    parser.add_argument(
        "--collision-term-speed",
        type=float,
        default=None,
        help="Closing-speed threshold (m/s) above which a car-car collision ends "
        "the episode. Below it, low-speed contacts still incur penalties and contact "
        "physics but the agent keeps driving. 0.0 terminates on any overlap. "
        "Defaults to config env.collision_term_speed_mps.",
    )
    parser.add_argument(
        "--zero-tyre-slip-obs",
        action="store_true",
        default=False,
        help="Zero obs[372:380] in training to match deploy/gym (no slip sensing).",
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
        help="Use the GT Sophy-style mixed opponent population: each env row is "
        "randomly assigned scripted or self-play policy on reset (implies --self-play "
        "so the policy snapshots refresh). Mix weights come from config opponent_mix.",
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
        help="How to sample an opponent snapshot from the pool (mixed: 80%% latest).",
    )
    parser.add_argument(
        "--selfplay-anchor-ckpt",
        type=str,
        default=cfg["selfplay"]["anchor_ckpt"],
        help="Immutable incumbent policy artifact added to the opponent population "
        "and never evicted from the rolling pool. Default: no anchor.",
    )
    parser.add_argument(
        "--selfplay-anchor-prob",
        type=float,
        default=cfg["selfplay"]["anchor_prob"],
        help="Probability of sampling the immutable anchor instead of the rolling "
        "pool on each opponent refresh/selection (0.0 disables it).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32"],
        help="Warp simulator precision (float32 only).",
    )
    parser.add_argument(
        "--export-interval-transitions",
        type=int,
        default=cfg["schedule"]["export_interval_transitions"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "torch.compile the QR-SAC networks + quantile loss, plus fused Adam "
            "and foreach polyak (default: on). Use --no-compile for the strict "
            "deterministic reference path (tests, debugging, reproducibility)."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode when --compile is set. 'reduce-overhead' captures "
            "CUDA graphs to collapse per-kernel launch overhead."
        ),
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=cfg["model"]["replay_buffer_limit"],
    )
    parser.add_argument(
        "--log-interval-transitions",
        type=int,
        default=cfg["schedule"]["log_interval_transitions"],
    )
    parser.add_argument(
        "--eval-interval-transitions",
        type=int,
        default=cfg["schedule"]["eval_interval_transitions"],
        help="Render a deterministic eval rollout every N transitions (0=off). Uses a "
        "separate 1-env instance so training data/state is never touched. Logs an "
        "mp4 to W&B (and/or streams live to Rerun with --eval-video-live).",
    )
    parser.add_argument(
        "--eval-video-steps",
        type=int,
        default=600,
        help="Length (control steps) of each eval rollout video.",
    )
    parser.add_argument(
        "--eval-video-num-envs",
        type=int,
        default=1,
        help="How many env instances to draw overlaid in the eval video (swarm "
        "view showing the spread of policy behaviour).",
    )
    parser.add_argument(
        "--eval-video-live",
        action="store_true",
        help="Also stream the eval rollout live to a Rerun viewer as it renders.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log metrics to Weights & Biases (default: on; use --no-wandb to disable).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.getenv("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="Warm-start the actor + obs-normalizer from a saved policy artifact.",
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for this run's artifacts (checkpoints/, run.log, config.json). "
        "Default: outputs/runs/<run-id>/. Explicit paths support legacy layouts.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B run group for comparing related experiments",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default=None,
        help="Human-readable hypothesis description for W&B metadata",
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

    init_transitions = 0
    if args.init_ckpt is not None:
        init_transitions = load_init_ckpt(
            models,
            actor_normalizer,
            args.init_ckpt,
            device,
            expected_layout_version=int(obs_cfg["actor_layout_version"]),
            expected_critic_obs_dim=int(obs_cfg["num_obs"]),
        )

    selfplay_mgr: SelfPlayManager | None = None
    if args.self_play or args.mixed_opponents:
        sp_cfg = cfg["selfplay"]
        selfplay_mgr = SelfPlayManager(
            pool_size=sp_cfg["pool_size"],
            snapshot_interval_transitions=sp_cfg["snapshot_interval_transitions"],
            refresh_interval_transitions=sp_cfg["refresh_interval_transitions"],
            sample_mode=sp_cfg["sample_mode"],
            mixed_latest_prob=sp_cfg["mixed_latest_prob"],
            anchor_prob=sp_cfg["anchor_prob"],
            expected_architecture=actor_architecture_from_module(models.actor),
            log=log,
        )
        if sp_cfg["anchor_ckpt"]:
            selfplay_mgr.load_anchor(
                sp_cfg["anchor_ckpt"],
                device,
                actor_obs_dim,
                cfg["env"]["num_actions"],
                expected_layout_version=int(obs_cfg["actor_layout_version"]),
                expected_architecture=actor_architecture_from_module(models.actor),
                expected_critic_obs_dim=int(obs_cfg["num_obs"]),
            )
        # Seed/bootstrap after the first sensor batch initializes the actor normalizer.

    use_1v1 = args.self_play or args.mixed_opponents or args.opponent != "none"
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
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        if args.hypothesis:
            init_kwargs["notes"] = args.hypothesis
        wandb_run = wandb.init(**init_kwargs)

    raw_obs, _ = env.reset(with_sensors=True)
    actor_obs, critic_obs = unpack_sensor_observations(raw_obs)
    actor_normalizer.update(actor_obs)
    critic_normalizer.update(critic_obs)
    if selfplay_mgr is not None:
        # First-batch actor normalizer stats must exist before the initial snapshot.
        selfplay_mgr.seed_snapshot(
            SelfPlayManager.make_snapshot(
                models, actor_normalizer, transitions=init_transitions
            )
        )
        selfplay_mgr.bootstrap_opponent(env)
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

    try:
        while training_should_continue(
            env_transitions, args.total_transitions, continuous
        ):
            previous_transitions = env_transitions

            if env_transitions < args.min_train_transitions:
                actions = (
                    torch.rand(
                        args.num_envs, act_dim, device=device, dtype=torch.float32
                    )
                    * 2
                    * clip_actions
                    - clip_actions
                )
            else:
                with torch.no_grad():
                    actions, _ = models.actor(
                        actor_normalizer.normalize(actor_obs),
                        deterministic=False,
                        with_logprob=False,
                    )
                actions = actions.clamp(-clip_actions, clip_actions)

            next_raw_obs, reward, done, extras = env.step(
                actions.to(rt.tc_float), n_steps=control_interval, with_sensors=True
            )
            vector_ticks += 1
            env_transitions += args.num_envs
            next_actor_obs, next_critic_obs = unpack_sensor_observations(next_raw_obs)
            reward = reward.to(torch.float32)

            episode_rewards += reward
            done_f = done.to(episode_rewards.dtype)
            ep_return_sum += (episode_rewards * done_f).sum()
            ep_return_count += done_f.sum()
            if selfplay_mgr is not None and critic_obs.shape[-1] > OPP_TRACK_GAP_IDX:
                done_bool = done.bool()
                if done_bool.any():
                    # Opponent block index 4 is ``s_other - s_self``; negate.
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

            replay_inserts += buffer.add(
                actor_obs,
                critic_obs,
                actions,
                reward,
                next_actor_obs,
                next_critic_obs,
                done,
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
                if not buffer.is_ready(args.batch_size):
                    learner_row_budget += updates_due * args.batch_size
                    updates_due = 0
                for _ in range(updates_due):
                    batch = buffer.sample(args.batch_size)
                    # Buffer stores RAW obs; normalize with current stats.
                    batch["actor_obs"] = actor_normalizer.normalize(
                        batch["actor_obs"]
                    )
                    batch["next_actor_obs"] = actor_normalizer.normalize(
                        batch["next_actor_obs"]
                    )
                    batch["critic_obs"] = critic_normalizer.normalize(
                        batch["critic_obs"]
                    )
                    batch["next_critic_obs"] = critic_normalizer.normalize(
                        batch["next_critic_obs"]
                    )
                    losses = trainer.update(batch)
                    gradient_updates += 1
                    sampled_replay_rows += args.batch_size
                    policy_loss_accum += losses.policy_loss
                    critic_loss_accum += losses.critic_loss
                    loss_count += 1

            if selfplay_mgr is not None:
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
                        "progress=%.4f passing=%.4f collision=%.4f oob_penalty=%.4f "
                        "wall=%.4f impact=%.4f tyre_slip=%.4f smooth=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/passing"),
                        diag.mean("reward_term/collision"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/wall_penalty"),
                        diag.mean("reward_term/wall_impact"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                    )
                else:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f oob_penalty=%.4f wall=%.4f impact=%.4f "
                        "tyre_slip=%.4f smooth=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/wall_penalty"),
                        diag.mean("reward_term/wall_impact"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                    )
                log.info(
                    "  reward_events: oob_when=%.4f wall_when=%.4f "
                    "impact_when=%.4f wall_contacts=%d impact_events=%d",
                    diag.mean("reward_term/oob_penalty_when_oob"),
                    diag.mean("reward_term/wall_penalty_when_contact"),
                    diag.mean("reward_term/wall_impact_when_event"),
                    int(diag.total("metric/wall_contact_count")),
                    int(diag.total("metric/wall_impact_events")),
                )
                log.info(
                    "  env: speed=%.3f opp_speed=%.3f lat_err=%.3f oob_frac=%.3f "
                    "wall_frac=%.3f progress_ds=%.4f laps_completed=%d | "
                    "throttle[%.2f..%.2f] steer[%.2f..%.2f] obs_absmax=%.2f "
                    "norm_obs_absmax=%.2f",
                    diag.mean("metric/speed_xy"),
                    diag.mean("metric/opp_speed"),
                    diag.mean("metric/lateral_error"),
                    diag.mean("metric/oob_mask"),
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
                    "  dr: tire_mu[%.3f..%.3f] mass[%.3f..%.3f] "
                    "act_latency[%.0f..%.0f] obs_latency[%.0f..%.0f] "
                    "obs_noise[%.4f..%.4f]",
                    diag.vmin("metric/dr/tire_friction"),
                    diag.vmax("metric/dr/tire_friction"),
                    diag.vmin("metric/dr/vehicle_mass"),
                    diag.vmax("metric/dr/vehicle_mass"),
                    diag.vmin("metric/dr/action_latency_steps"),
                    diag.vmax("metric/dr/action_latency_steps"),
                    diag.vmin("metric/dr/obs_latency_steps"),
                    diag.vmax("metric/dr/obs_latency_steps"),
                    diag.vmin("metric/dr/obs_noise_std"),
                    diag.vmax("metric/dr/obs_noise_std"),
                )
                log.info(
                    "  nonfinite: obs_rate=%.2e reward_rate=%.2e state_rate=%.2e",
                    nf_obs_rate,
                    nf_reward_rate,
                    nf_state_rate,
                )
                if use_1v1:
                    log.info(
                        "  terminations: time_out=%d oob=%d wall_impact=%d "
                        "collision=%d not_moving=%d invalid=%d | "
                        "opp_presence=%.3f",
                        int(diag.total("term/time_out")),
                        int(diag.total("term/out_of_bounds")),
                        int(diag.total("term/wall_impact")),
                        int(diag.total("term/collision")),
                        int(diag.total("term/not_moving")),
                        int(diag.total("term/invalid_state")),
                        diag.mean("metric/opponent_presence"),
                    )
                    if selfplay_mgr is not None:
                        opp_age = (
                            env_transitions - selfplay_mgr.opponent_transitions
                            if selfplay_mgr.opponent_transitions is not None
                            else -1
                        )
                        log.info(
                            "  selfplay: pool_size=%d opp_transitions=%s opp_age=%d "
                            "win_rate=%.3f (n=%d)",
                            len(selfplay_mgr.pool),
                            selfplay_mgr.opponent_transitions,
                            opp_age,
                            selfplay_mgr.win_rate(),
                            selfplay_mgr._episode_total,
                        )
                        selfplay_mgr.reset_win_stats()
                else:
                    log.info(
                        "  terminations: time_out=%d oob=%d wall_impact=%d "
                        "not_moving=%d invalid=%d",
                        int(diag.total("term/time_out")),
                        int(diag.total("term/out_of_bounds")),
                        int(diag.total("term/wall_impact")),
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
                            "reward/overtake": diag.mean("reward_term/overtake"),
                            "reward/rear_end": diag.mean("reward_term/rear_end"),
                            "reward/collision": diag.mean(
                                "reward_term/collision"
                            ),
                            "reward/oob_penalty": diag.mean(
                                "reward_term/oob_penalty"
                            ),
                            "reward/wall_penalty": diag.mean(
                                "reward_term/wall_penalty"
                            ),
                            "reward/wall_impact": diag.mean(
                                "reward_term/wall_impact"
                            ),
                            "reward/oob_penalty_when_oob": diag.mean(
                                "reward_term/oob_penalty_when_oob"
                            ),
                            "reward/wall_penalty_when_contact": diag.mean(
                                "reward_term/wall_penalty_when_contact"
                            ),
                            "reward/wall_impact_when_event": diag.mean(
                                "reward_term/wall_impact_when_event"
                            ),
                            "reward/tyre_slip_penalty": diag.mean(
                                "reward_term/tyre_slip_penalty"
                            ),
                            "reward/smoothness": diag.mean(
                                "reward_term/smoothness"
                            ),
                            "env/speed_xy": diag.mean("metric/speed_xy"),
                            "env/lateral_error": diag.mean("metric/lateral_error"),
                            "env/oob_frac": diag.mean("metric/oob_mask"),
                            "env/wall_frac": diag.mean("metric/wall_contact"),
                            "env/wall_contact_count": diag.total(
                                "metric/wall_contact_count"
                            ),
                            "env/wall_impact_events": diag.total(
                                "metric/wall_impact_events"
                            ),
                            "env/progress_ds": diag.mean("metric/progress_ds"),
                            "env/lap_count": diag.mean("metric/lap_count"),
                            "env/laps_completed": diag.total("metric/laps_completed"),
                            "dr/tire_friction_min": diag.vmin(
                                "metric/dr/tire_friction"
                            ),
                            "dr/tire_friction_max": diag.vmax(
                                "metric/dr/tire_friction"
                            ),
                            "dr/vehicle_mass_min": diag.vmin(
                                "metric/dr/vehicle_mass"
                            ),
                            "dr/vehicle_mass_max": diag.vmax(
                                "metric/dr/vehicle_mass"
                            ),
                            "action/throttle_max": diag.vmax("action/throttle"),
                            "action/steer_max": diag.vmax("action/steer"),
                            "obs/absmax": diag.vmax("obs/abs"),
                            "term/time_out": diag.total("term/time_out"),
                            "term/out_of_bounds": diag.total("term/out_of_bounds"),
                            "term/wall_impact": diag.total("term/wall_impact"),
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
                    models, env_transitions, ckpt_dir, actor_normalizer, cfg
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
                        num_steps=args.eval_video_steps,
                        num_show=args.eval_video_num_envs,
                        live=args.eval_video_live,
                        wandb_run=wandb_run,
                        log=log,
                        selfplay_mgr=selfplay_mgr,
                    )
                except Exception as exc:
                    log.warning("Eval video rollout failed (continuing): %s", exc)

        save_policy_artifact(
            models, env_transitions, ckpt_dir, actor_normalizer, cfg
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

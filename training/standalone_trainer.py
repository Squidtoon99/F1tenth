#!/usr/bin/env python3
"""Single-process QRSAC trainer: F1tenthEnv + in-memory n-step replay, no Reverb/Redis/S3."""

from __future__ import annotations

import argparse
import copy
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

from dotenv import load_dotenv

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.utils import episode_length_for_track
from evaluation import deterministic_rollout
from run_layout import checkpoint_dir, config_snapshot_path, default_run_dir, run_log_path
from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor

LOGGER_NAME = "standalone_trainer"
RECENT_EPISODES_MAX = 50
# Opponent-relative block starts right after the base observation (tyre_load ends
# the base vector); index 4 within it is the signed along-track gap s_other-s_self.
OPP_OBS_BASE_IDX = 384
OPP_TRACK_GAP_IDX = OPP_OBS_BASE_IDX + 4


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
        log: logging.Logger | None = None,
    ):
        self.pool_size = pool_size
        self.snapshot_interval_transitions = snapshot_interval_transitions
        self.refresh_interval_transitions = refresh_interval_transitions
        self.sample_mode = sample_mode
        self.mixed_latest_prob = mixed_latest_prob
        self.log = log or logging.getLogger(LOGGER_NAME)
        self.pool: deque[SelfPlaySnapshot] = deque(maxlen=pool_size)
        self.opponent_transitions: int | None = None
        self._last_snapshot_transitions = 0
        self._last_refresh_transitions = 0
        self._episode_wins = 0
        self._episode_total = 0

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
        )

    def seed_snapshot(self, snapshot: SelfPlaySnapshot) -> None:
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
        self.pool.append(snap)
        self._last_snapshot_transitions = transitions
        self.log.info(
            "Self-play snapshot pushed at transitions=%d (pool_size=%d)",
            transitions,
            len(self.pool),
        )
        return True

    def _sample_snapshot(self) -> SelfPlaySnapshot | None:
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
        env.refresh_opponent_policy(snap["actor"], snap["mean"], snap["var"])
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
        env.refresh_opponent_policy(snap["actor"], snap["mean"], snap["var"])
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
        env.refresh_opponent_policy(snap["actor"], snap["mean"], snap["var"])
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


def make_policy_network(cfg: dict) -> SquashedGaussianMLPActor:
    obs_dim = cfg["obs"]["num_obs"]
    action_dim = cfg["env"]["num_actions"]
    return SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg["model"]["hidden_layers"],
        activation=nn.ReLU,
        act_limit=1.0,
    )


def make_q_network(cfg: dict) -> QuantileCritic:
    obs_dim = cfg["obs"]["num_obs"]
    action_dim = cfg["env"]["num_actions"]
    return QuantileCritic(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden_sizes=cfg["model"]["hidden_layers"],
        num_quantiles=cfg["model"]["num_quantiles"],
    )


def make_target_q_network(cfg: dict) -> QuantileCritic:
    target_q = make_q_network(cfg)
    for param in target_q.parameters():
        param.requires_grad = False
    return target_q


class NStepReplayBuffer:
    """Per-env n-step deques feeding a preallocated tensor ring buffer on device."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        n_step: int,
        gamma: float,
        num_envs: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.num_envs = num_envs
        self.device = device
        self.size = 0
        self.ptr = 0

        self.obs = torch.zeros(capacity, obs_dim, device=device, dtype=torch.float32)
        self.action = torch.zeros(
            capacity, act_dim, device=device, dtype=torch.float32
        )
        self.reward = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_obs = torch.zeros(
            capacity, obs_dim, device=device, dtype=torch.float32
        )
        self.done = torch.zeros(capacity, device=device, dtype=torch.float32)

        self._gamma_powers = torch.tensor(
            [gamma**k for k in range(n_step)], device=device, dtype=torch.float32
        )

        # Vectorized per-env n-step windows kept on device as circular buffers.
        # All envs advance in lockstep, so a single write column index ``w_pos``
        # is shared. ``w_len`` counts valid entries per env (reset to 0 on done).
        self.w_obs = torch.zeros(num_envs, n_step, obs_dim, device=device, dtype=torch.float32)
        self.w_act = torch.zeros(num_envs, n_step, act_dim, device=device, dtype=torch.float32)
        self.w_rew = torch.zeros(num_envs, n_step, device=device, dtype=torch.float32)
        self.w_len = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.w_pos = 0
        self._arange_n = torch.arange(n_step, device=device)

    def _store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> int:
        count = int(obs.shape[0])
        if count == 0:
            return 0
        positions = (
            self.ptr + torch.arange(count, device=self.device)
        ) % self.capacity
        self.obs[positions] = obs
        self.action[positions] = action
        self.reward[positions] = reward
        self.next_obs[positions] = next_obs
        self.done[positions] = done
        self.ptr = int((self.ptr + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)
        return count

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> int:
        """Vectorized n-step accumulation. Writes the current transition into each
        env's circular window, emits completed n-step samples for all full windows
        in a single batched scatter, then clears windows for done envs. The only
        host sync is one ``nonzero`` per step (independent of ``num_envs``)."""
        col = self.w_pos
        self.w_obs[:, col] = obs.detach()
        self.w_act[:, col] = actions.detach()
        self.w_rew[:, col] = rewards.detach()
        self.w_len = torch.clamp(self.w_len + 1, max=self.n_step)
        self.w_pos = (col + 1) % self.n_step

        # After advancing, column ``w_pos`` is the oldest entry of a full window;
        # ``order`` lists columns oldest -> newest for the discounted sum.
        oldest = self.w_pos
        order = (oldest + self._arange_n) % self.n_step
        n_step_reward = (self.w_rew[:, order] * self._gamma_powers).sum(dim=1)
        obs0 = self.w_obs[:, oldest]
        act0 = self.w_act[:, oldest]
        done_f = dones.detach().to(torch.float32)

        emit_mask = (self.w_len == self.n_step) & ~dones.bool()
        idx = torch.nonzero(emit_mask, as_tuple=False).squeeze(-1)
        n_emit = self._store(
            obs0[idx],
            act0[idx],
            n_step_reward[idx],
            next_obs[idx].detach(),
            done_f[idx],
        )

        done_mask = dones.bool()
        env_arange = torch.arange(self.num_envs, device=self.device)
        for offset in range(self.n_step):
            terminal_mask = done_mask & (self.w_len > offset)
            terminal_idx = torch.nonzero(
                terminal_mask, as_tuple=False
            ).squeeze(-1)
            if terminal_idx.numel() == 0:
                continue
            start = (self.w_pos - self.w_len + offset) % self.n_step
            horizon = self.w_len - offset
            columns = (
                start.unsqueeze(1) + self._arange_n.unsqueeze(0)
            ) % self.n_step
            rewards = self.w_rew[env_arange.unsqueeze(1), columns]
            valid = self._arange_n.unsqueeze(0) < horizon.unsqueeze(1)
            returns = (
                rewards * self._gamma_powers.unsqueeze(0) * valid
            ).sum(dim=1)
            n_emit += self._store(
                self.w_obs[terminal_idx, start[terminal_idx]],
                self.w_act[terminal_idx, start[terminal_idx]],
                returns[terminal_idx],
                next_obs[terminal_idx].detach(),
                torch.ones_like(done_f[terminal_idx]),
            )

        # Clear windows for done envs (sync-free masked write).
        self.w_len = torch.where(
            dones.bool(), torch.zeros_like(self.w_len), self.w_len
        )
        return n_emit

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(
                f"Buffer has {self.size} samples, need at least {batch_size} to sample."
            )
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_obs": self.next_obs[idx],
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


class RunningStats:
    """Accumulates scalar means / min / max / totals for named diagnostics.

    Values are kept as on-device tensors and only synced to Python floats at
    log time to avoid a host sync on every environment step.
    """

    def __init__(self):
        self._sum: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}
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

    def add_total(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.sum()

    def mean(self, key: str) -> float:
        if self._count.get(key, 0) == 0:
            return float("nan")
        return float(self._sum[key]) / self._count[key]

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

    for name, value in extras.get("rewards", {}).get("terms", {}).items():
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
        "dr/tire_friction",
        "dr/vehicle_mass",
        "dr/mass_scale",
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

    for name, value in extras.get("termination", {}).items():
        if isinstance(value, torch.Tensor):
            diag.add_total(f"term/{name}", value)

    if actions.ndim == 2 and actions.shape[1] >= 2:
        diag.add_mean("action/throttle", actions[:, 0], track_range=True)
        diag.add_mean("action/steer", actions[:, 1], track_range=True)
    diag.add_mean("obs/abs", obs.abs(), track_range=True)


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    cfg["model"]["batch_size"] = args.batch_size
    cfg["model"]["minimum_train_transitions"] = args.min_train_transitions
    cfg["model"]["sampled_replay_rows_per_transition"] = (
        args.sampled_rows_per_transition
    )
    cfg["model"]["replay_buffer_limit"] = args.buffer_capacity
    cfg["model"]["alpha"] = args.alpha
    cfg["schedule"]["total_transitions"] = args.total_transitions
    cfg["schedule"]["log_interval_transitions"] = args.log_interval_transitions
    cfg["schedule"]["export_interval_transitions"] = (
        args.export_interval_transitions
    )
    cfg["schedule"]["eval_interval_transitions"] = args.eval_interval_transitions

    # Episode horizon: explicit override, else derive from the track centerline
    # length so each track gets ~episode_lap_multiplier laps of racing time.
    if getattr(args, "episode_length", None) is not None:
        cfg["env"]["episode_length"] = float(args.episode_length)
    else:
        lap_multiplier = (
            float(args.episode_lap_multiplier)
            if getattr(args, "episode_lap_multiplier", None) is not None
            else float(cfg["env"].get("episode_lap_multiplier", 3.0))
        )
        workspace_dir = str(Path(__file__).resolve().parent)
        cfg["env"]["episode_length"] = episode_length_for_track(
            track=args.track,
            workspace_dir=workspace_dir,
            ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
            lap_multiplier=lap_multiplier,
        )

    if args.n_step is not None:
        cfg["model"]["n_step"] = args.n_step

    cfg["env"]["domain_randomization"] = {
        **DEFAULT_CONFIG["env"]["domain_randomization"],
        "enabled": True,
    }

    if getattr(args, "zero_tyre_slip_obs", False):
        cfg["obs"]["zero_tyre_slip_obs"] = True

    sp_defaults = DEFAULT_CONFIG["selfplay"]
    cfg["selfplay"] = {
        "snapshot_interval_transitions": args.selfplay_snapshot_interval,
        "refresh_interval_transitions": args.selfplay_refresh_interval,
        "pool_size": args.selfplay_pool_size,
        "sample_mode": args.selfplay_sample,
        "mixed_latest_prob": sp_defaults["mixed_latest_prob"],
    }

    # 1v1: enable the opponent + opponent observation block + passing reward.
    # Trained from scratch, so we just size the networks/normalizer at the larger
    # num_obs - no checkpoint surgery. 1v0 (opponent "none") leaves everything as
    # the unchanged solo config.
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
        cfg["env"]["opponent_target_speed"] = args.opponent_target_speed
        cfg["env"]["opponent_spawn_gap_min_m"] = args.opponent_spawn_gap_min
        cfg["env"]["opponent_spawn_gap_max_m"] = args.opponent_spawn_gap_max
        cfg["env"]["opponent_spawn_behind_prob"] = args.opponent_spawn_behind_prob
        cfg["env"]["opponent_spawn_lateral_independent"] = (
            args.opponent_spawn_lateral_independent
        )
        cfg["env"]["opponent_reset_speed_min_mps"] = args.opponent_reset_speed_min
        cfg["env"]["opponent_reset_speed_max_mps"] = args.opponent_reset_speed_max
        if args.opponent_ckpt is not None:
            cfg["env"]["opponent_ckpt"] = args.opponent_ckpt
        cfg["obs"]["enable_opponent_obs"] = True
        cfg["obs"]["num_obs"] = int(cfg["obs"]["num_obs"]) + int(
            cfg["obs"]["opponent_obs_dim"]
        )
        # Activate the passing reward term (gated by presence of this scale).
        cfg["reward"]["reward_scales"]["passing"] = args.passing_scale
        # Activate the GT Sophy any-collision penalty (gated by this scale).
        cfg["reward"]["reward_scales"]["collision"] = args.collision_scale
        # Activate the GT Sophy rear-end penalty Rr (gated by this scale). 0.0
        # leaves it out of the reward breakdown entirely.
        if float(args.rear_end_scale) != 0.0:
            cfg["reward"]["reward_scales"]["rear_end"] = args.rear_end_scale
        # Activate the overtake-completed bonus (gated by this scale). 0.0 leaves it
        # out of the reward breakdown entirely.
        if float(args.overtake_scale) != 0.0:
            cfg["reward"]["reward_scales"]["overtake"] = args.overtake_scale
        # Closing-speed threshold for collision termination (0.0 = terminate on any
        # overlap). Below it, contacts only apply penalties/physics and the episode
        # continues.
        if args.collision_term_speed is not None:
            cfg["env"]["collision_term_speed_mps"] = float(args.collision_term_speed)
    return cfg


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_models(
    cfg: dict, device: torch.device, alpha: float = 0.01
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
    )
    return models, trainer


def save_policy_artifact(
    models: Models,
    env_transitions: int,
    artifact_dir: Path,
    normalizer: ObsNormalizer,
    cfg: dict,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"policy_{env_transitions}.pt"
    payload = {
        "actor": models.actor.state_dict(),
        "obs_norm": normalizer.state_dict(),
        "env_transitions": env_transitions,
        "obs_dim": int(cfg["obs"]["num_obs"]),
        "action_dim": int(cfg["env"]["num_actions"]),
        "action_scale": float(cfg["env"]["clip_actions"]),
        "config_version": int(cfg["config_version"]),
        "policy_format_version": int(cfg["policy_format_version"]),
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
    }
    torch.save(payload, path)
    logging.getLogger(LOGGER_NAME).info("Saved policy artifact to %s", path)
    return path


def load_init_ckpt(
    models: Models,
    normalizer: ObsNormalizer,
    path: str,
    device: torch.device,
) -> int:
    """Warm-start the actor + obs-normalizer from a saved policy artifact.

    Returns the artifact's env-transition count so self-play can seed its first
    snapshot at the policy's true maturity. Critics start fresh (not exported).
    """
    payload = torch.load(path, map_location=device)
    models.actor.load_state_dict(payload["actor"])
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
    import numpy as np

    from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz

    num_show = max(1, int(num_show))
    env = eval_state.get("env")
    if env is None:
        env = F1tenthEnv(
            num_envs=num_show,
            env_cfg={
                **env_cfg,
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
        has_opponent=env.has_opponent,
        rr_app_id="f1tenth_train_eval",
        rr_spawn=live,
    )

    was_training = models.actor.training
    models.actor.eval()
    if selfplay_mgr is not None and getattr(env, "has_opponent", False):
        selfplay_mgr.refresh_eval_opponent(env)

    def render_step(_step, rollout_env, _state_before, _reward, done, _extras):
        st = rollout_env.backend.read_state()
        ego_xy = st["base_pos"][:, :2].cpu().numpy()
        ego_yaw = np.array(
            [yaw_from_quat_wxyz(q.tolist()) for q in st["base_quat"]]
        )
        speed = torch.linalg.norm(st["base_lin_vel"][:, :2], dim=-1).cpu().numpy()
        opp_xy = opp_yaw = None
        if rollout_env.has_opponent and "opp_base_pos" in st:
            opp_xy = st["opp_base_pos"][:, :2].cpu().numpy()
            opp_yaw = np.array(
                [yaw_from_quat_wxyz(q.tolist()) for q in st["opp_base_quat"]]
            )
        viz.render(
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            speed=speed,
            opp_xy=opp_xy,
            opp_yaw=opp_yaw if opp_yaw is not None else 0.0,
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
    )
    out = viz.close()
    models.actor.train(was_training)
    if wandb_run is not None and out is not None:
        import wandb

        wandb_run.log({"eval/rollout": wandb.Video(out, format="mp4")}, step=step)
    log.info("Eval rollout video written to %s", out)


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Standalone QRSAC trainer (1v0, single process)")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--total-transitions",
        type=int,
        default=cfg["schedule"]["total_transitions"],
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
        default=2.0,
        help="Reward scale for the 1v1 passing term (track position gained on the "
        "opponent). Only used when --opponent is not 'none'.",
    )
    parser.add_argument(
        "--collision-scale",
        type=float,
        default=1.0,
        help="Reward scale for the GT Sophy any-collision penalty (-collision_k on "
        "car-car overlap). Only used when --opponent is not 'none'.",
    )
    parser.add_argument(
        "--rear-end-scale",
        type=float,
        default=0.5,
        help="Reward scale for the GT Sophy rear-end penalty Rr (-rear_end_k * "
        "closing-speed^2 when colliding with an opponent ahead). 0.0 disables it. "
        "Only used when --opponent is not 'none'.",
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="64",
        choices=["32", "64"],
        help="Torch simulator precision.",
    )
    parser.add_argument(
        "--export-interval-transitions",
        type=int,
        default=cfg["schedule"]["export_interval_transitions"],
    )
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = build_config(args)
    obs_cfg = cfg["obs"]
    reward_cfg = cfg["reward"]
    model_cfg = cfg["model"]
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": args.num_envs},
        **cfg["env"],
    }
    clip_actions = cfg["env"]["clip_actions"]
    control_interval = cfg["env"]["control_interval"]
    n_step = model_cfg["n_step"]

    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float64 if args.precision == "64" else torch.float32,
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

    snapshot = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "args": {k: v for k, v in vars(args).items() if v is not None},
        "config": cfg,
    }
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

    models, trainer = build_models(cfg, device, alpha=args.alpha)
    buffer = NStepReplayBuffer(
        capacity=args.buffer_capacity,
        obs_dim=obs_cfg["num_obs"],
        act_dim=cfg["env"]["num_actions"],
        n_step=n_step,
        gamma=model_cfg["rew_gamma"],
        num_envs=args.num_envs,
        device=device,
    )
    normalizer = ObsNormalizer(
        obs_dim=obs_cfg["num_obs"],
        device=device,
        eps=float(obs_cfg.get("norm_eps", 1e-8)),
        clip=float(obs_cfg.get("norm_clip", 10.0)),
    )

    init_transitions = 0
    if args.init_ckpt is not None:
        init_transitions = load_init_ckpt(models, normalizer, args.init_ckpt, device)

    selfplay_mgr: SelfPlayManager | None = None
    if args.self_play or args.mixed_opponents:
        sp_cfg = cfg["selfplay"]
        selfplay_mgr = SelfPlayManager(
            pool_size=sp_cfg["pool_size"],
            snapshot_interval_transitions=sp_cfg["snapshot_interval_transitions"],
            refresh_interval_transitions=sp_cfg["refresh_interval_transitions"],
            sample_mode=sp_cfg["sample_mode"],
            mixed_latest_prob=sp_cfg["mixed_latest_prob"],
            log=log,
        )
        selfplay_mgr.seed_snapshot(
            SelfPlayManager.make_snapshot(
                models, normalizer, transitions=init_transitions
            )
        )
        selfplay_mgr.bootstrap_opponent(env)

    use_1v1 = args.self_play or args.mixed_opponents or args.opponent != "none"
    wandb_run = None
    if args.wandb:
        import wandb

        tags = ["standalone", run_id]
        if platform.system() == "Darwin":
            tags.append("mac")
        init_kwargs = {
            "project": os.getenv("WANDB_PROJECT", "f1tenth-genesis"),
            "name": f"standalone_{run_id}",
            "id": run_id,
            "config": {**cfg, **vars(args)},
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

    obs, _ = env.reset()
    obs = obs.to(torch.float32)
    normalizer.update(obs)
    act_dim = cfg["env"]["num_actions"]
    vector_ticks = 0
    env_transitions = 0
    replay_inserts = 0
    sampled_replay_rows = 0
    gradient_updates = 0
    learner_row_budget = 0.0
    episode_rewards = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    recent_episode_rewards: deque[float] = deque(maxlen=RECENT_EPISODES_MAX)
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
        while env_transitions < args.total_transitions:
            previous_transitions = env_transitions
            bad_obs_mask = (~torch.isfinite(obs)).any(dim=1)
            if bad_obs_mask.any():
                diag.add_total(
                    "nonfinite/pre_step_obs_resets", bad_obs_mask.to(torch.float32)
                )
                reset_obs, _ = env.reset(envs_idx=bad_obs_mask)
                obs = obs.clone()
                obs[bad_obs_mask] = reset_obs[bad_obs_mask].to(torch.float32)

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
                        normalizer.normalize(obs),
                        deterministic=False,
                        with_logprob=False,
                    )
                actions = actions.clamp(-clip_actions, clip_actions)

            next_obs, reward, done, extras = env.step(
                actions.to(rt.tc_float), n_steps=control_interval
            )
            vector_ticks += 1
            env_transitions += args.num_envs
            next_obs = next_obs.to(torch.float32)
            reward = reward.to(torch.float32)
            episode_rewards += reward

            done_bool = done.bool()
            completed_returns = episode_rewards[done_bool]
            if completed_returns.numel() > 0:
                recent_episode_rewards.extend(completed_returns.tolist())
            if (
                selfplay_mgr is not None
                and done_bool.any()
                and obs.shape[-1] > OPP_TRACK_GAP_IDX
            ):
                # Opponent block index 4 is ``s_other - s_self``; negate for ego lead.
                ego_minus_opp = -obs[done_bool, OPP_TRACK_GAP_IDX]
                selfplay_mgr.record_episode_outcomes(ego_minus_opp)
            episode_rewards = torch.where(
                done_bool, torch.zeros_like(episode_rewards), episode_rewards
            )

            accumulate_step_diagnostics(diag, reward, actions, obs, extras)
            diag.add_mean(
                "obs/norm_abs",
                normalizer.normalize(obs).abs(),
                track_range=True,
            )

            if use_1v1 and obs.shape[-1] > OPP_OBS_BASE_IDX:
                opp_block = obs[:, OPP_OBS_BASE_IDX:]
                in_range = (opp_block != 0).any(dim=-1).to(obs.dtype)
                diag.add_mean("metric/opponent_presence", in_range)

            bad_obs_mask = (~torch.isfinite(next_obs)).any(dim=1)
            finite_ok = bool(
                torch.isfinite(reward).all() and not bad_obs_mask.any()
            )
            if finite_ok:
                replay_inserts += buffer.add(obs, actions, reward, next_obs, done)
                # Update running stats only from finite observations so a NaN/Inf
                # spin transient can never corrupt the normalizer.
                normalizer.update(next_obs)
            else:
                n_bad_reward = int((~torch.isfinite(reward)).sum().item())
                n_bad_obs_envs = int(bad_obs_mask.sum().item())
                diag.add_total(
                    "nonfinite/post_step_obs_bad",
                    bad_obs_mask.to(torch.float32),
                )
                diag.add_total(
                    "nonfinite/post_step_reward_bad",
                    (~torch.isfinite(reward)).to(torch.float32),
                )
                log.warning(
                    "Non-finite transition batch at %d transitions "
                    "(reward_bad=%d obs_bad_envs=%d); "
                    "skipping buffer add.",
                    env_transitions,
                    n_bad_reward,
                    n_bad_obs_envs,
                )
                if n_bad_obs_envs > 0:
                    reset_obs, _ = env.reset(envs_idx=bad_obs_mask)
                    next_obs = next_obs.clone()
                    next_obs[bad_obs_mask] = reset_obs[bad_obs_mask].to(torch.float32)
            obs = next_obs

            if env_transitions >= args.min_train_transitions:
                updates_due, learner_row_budget = learner_updates_for_transitions(
                    args.num_envs,
                    args.batch_size,
                    args.sampled_rows_per_transition,
                    learner_row_budget,
                )
                if buffer.size < args.batch_size:
                    learner_row_budget += updates_due * args.batch_size
                    updates_due = 0
                for _ in range(updates_due):
                    batch = buffer.sample(args.batch_size)
                    # Buffer stores RAW obs; normalize with current stats at input.
                    batch["obs"] = normalizer.normalize(batch["obs"])
                    batch["next_obs"] = normalizer.normalize(batch["next_obs"])
                    losses = trainer.update(batch)
                    gradient_updates += 1
                    sampled_replay_rows += args.batch_size
                    policy_loss_accum += losses.policy_loss
                    critic_loss_accum += losses.critic_loss
                    loss_count += 1

            if selfplay_mgr is not None:
                selfplay_mgr.maybe_snapshot(models, normalizer, env_transitions)
                selfplay_mgr.maybe_refresh(env, env_transitions)

            if interval_crossed(
                previous_transitions,
                env_transitions,
                args.log_interval_transitions,
            ):
                now = time.perf_counter()
                elapsed = now - last_log_time
                window_transitions = env_transitions - last_log_transitions
                vector_ticks_per_sec = (
                    window_transitions / args.num_envs / max(elapsed, 1e-6)
                )
                transitions_per_sec = window_transitions / max(elapsed, 1e-6)
                inserts_per_sec = (
                    replay_inserts - last_log_replay_inserts
                ) / max(elapsed, 1e-6)
                sampled_rows_per_sec = (
                    sampled_replay_rows - last_log_sampled_rows
                ) / max(elapsed, 1e-6)
                updates_per_sec = (
                    gradient_updates - last_log_gradient_updates
                ) / max(elapsed, 1e-6)
                mean_ep_reward = (
                    sum(recent_episode_rewards) / len(recent_episode_rewards)
                    if recent_episode_rewards
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
                buffer_fill_pct = 100.0 * buffer.size / buffer.capacity
                log.info(
                    "ticks=%d transitions=%d replay_inserts=%d buffer=%d/%d (%.1f%%) "
                    "gradient_updates=%d ticks/s=%.1f transitions/s=%.1f "
                    "inserts/s=%.1f sampled_rows/s=%.1f updates/s=%.2f "
                    "policy_loss=%.4f critic_loss=%.4f mean_ep_reward=%.4f (n=%d)",
                    vector_ticks,
                    env_transitions,
                    replay_inserts,
                    buffer.size,
                    buffer.capacity,
                    buffer_fill_pct,
                    gradient_updates,
                    vector_ticks_per_sec,
                    transitions_per_sec,
                    inserts_per_sec,
                    sampled_rows_per_sec,
                    updates_per_sec,
                    mean_policy_loss,
                    mean_critic_loss,
                    mean_ep_reward,
                    len(recent_episode_rewards),
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
                        "tyre_slip=%.4f smooth=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/passing"),
                        diag.mean("reward_term/collision"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                    )
                else:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f oob_penalty=%.4f tyre_slip=%.4f "
                        "smooth=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                    )
                log.info(
                    "  env: speed=%.3f opp_speed=%.3f lat_err=%.3f oob_frac=%.3f "
                    "progress_ds=%.4f laps_completed=%d | throttle[%.2f..%.2f] "
                    "steer[%.2f..%.2f] obs_absmax=%.2f norm_obs_absmax=%.2f",
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
                    "  nonfinite: obs_rate=%.2e reward_rate=%.2e state_rate=%.2e "
                    "pre_obs_reset=%d post_obs_bad=%d post_reward_bad=%d",
                    nf_obs_rate,
                    nf_reward_rate,
                    nf_state_rate,
                    int(diag.total("nonfinite/pre_step_obs_resets")),
                    int(diag.total("nonfinite/post_step_obs_bad")),
                    int(diag.total("nonfinite/post_step_reward_bad")),
                )
                if use_1v1:
                    log.info(
                        "  terminations: time_out=%d oob=%d collision=%d "
                        "not_moving=%d invalid=%d | opp_presence=%.3f",
                        int(diag.total("term/time_out")),
                        int(diag.total("term/out_of_bounds")),
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
                        "  terminations: time_out=%d oob=%d not_moving=%d invalid=%d",
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
                            "replay_inserts": replay_inserts,
                            "sampled_replay_rows": sampled_replay_rows,
                            "gradient_updates": gradient_updates,
                            "buffer/size": buffer.size,
                            "train/policy_loss": mean_policy_loss,
                            "train/critic_loss": mean_critic_loss,
                            "train/mean_ep_reward": mean_ep_reward,
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
                            "reward/tyre_slip_penalty": diag.mean(
                                "reward_term/tyre_slip_penalty"
                            ),
                            "reward/smoothness": diag.mean(
                                "reward_term/smoothness"
                            ),
                            "env/speed_xy": diag.mean("metric/speed_xy"),
                            "env/lateral_error": diag.mean("metric/lateral_error"),
                            "env/oob_frac": diag.mean("metric/oob_mask"),
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
                            "term/not_moving": diag.total("term/not_moving"),
                            "term/invalid_state": diag.total("term/invalid_state"),
                            "nonfinite/obs_rate": nf_obs_rate,
                            "nonfinite/reward_rate": nf_reward_rate,
                            "nonfinite/state_rate": nf_state_rate,
                            "nonfinite/pre_step_obs_resets": diag.total(
                                "nonfinite/pre_step_obs_resets"
                            ),
                            "nonfinite/post_step_obs_bad": diag.total(
                                "nonfinite/post_step_obs_bad"
                            ),
                            "nonfinite/post_step_reward_bad": diag.total(
                                "nonfinite/post_step_reward_bad"
                            ),
                        },
                        step=vector_ticks,
                    )
                policy_loss_accum.zero_()
                critic_loss_accum.zero_()
                loss_count = 0
                diag.reset()
                last_log_time = now
                last_log_transitions = env_transitions
                last_log_replay_inserts = replay_inserts
                last_log_sampled_rows = sampled_replay_rows
                last_log_gradient_updates = gradient_updates

            if interval_crossed(
                previous_transitions,
                env_transitions,
                args.export_interval_transitions,
            ):
                save_policy_artifact(
                    models, env_transitions, ckpt_dir, normalizer, cfg
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
                        normalizer=normalizer,
                        control_interval=control_interval,
                        clip_actions=clip_actions,
                        run_dir=run_dir,
                        step=vector_ticks,
                        num_steps=args.eval_video_steps,
                        num_show=args.eval_video_num_envs,
                        live=args.eval_video_live,
                        wandb_run=wandb_run,
                        log=log,
                        selfplay_mgr=selfplay_mgr,
                    )
                except Exception as exc:
                    log.warning("Eval video rollout failed (continuing): %s", exc)

        save_policy_artifact(models, env_transitions, ckpt_dir, normalizer, cfg)
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

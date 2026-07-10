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

import genesis as gs
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.utils import episode_length_for_track
from run_layout import checkpoint_dir, config_snapshot_path, default_run_dir, run_log_path
from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor

LOGGER_NAME = "standalone_trainer"
RECENT_EPISODES_MAX = 50
# Opponent-relative block starts right after the base observation (tyre_load ends
# the base vector); index 4 within it is the signed along-track gap s_other-s_self.
OPP_OBS_BASE_IDX = 384
OPP_TRACK_GAP_IDX = OPP_OBS_BASE_IDX + 4


class SelfPlaySnapshot(dict):
    """CPU snapshot: actor state_dict + obs-norm stats + learner step."""

    actor: dict[str, torch.Tensor]
    mean: torch.Tensor
    var: torch.Tensor
    step: int


class SelfPlayManager:
    """Delayed self-play: snapshot learner into a pool, refresh opponent periodically."""

    def __init__(
        self,
        pool_size: int = 5,
        snapshot_interval: int = 20_000,
        refresh_interval: int = 5_000,
        sample_mode: str = "mixed",
        mixed_latest_prob: float = 0.8,
        log: logging.Logger | None = None,
    ):
        self.pool_size = pool_size
        self.snapshot_interval = snapshot_interval
        self.refresh_interval = refresh_interval
        self.sample_mode = sample_mode
        self.mixed_latest_prob = mixed_latest_prob
        self.log = log or logging.getLogger(LOGGER_NAME)
        self.pool: deque[SelfPlaySnapshot] = deque(maxlen=pool_size)
        self.opponent_step: int | None = None
        self._last_snapshot_step = -1
        self._last_refresh_step = -1
        self._episode_wins = 0
        self._episode_total = 0

    @staticmethod
    def make_snapshot(
        models: Models, normalizer: ObsNormalizer, step: int
    ) -> SelfPlaySnapshot:
        return SelfPlaySnapshot(
            actor={
                k: v.detach().cpu().clone()
                for k, v in models.actor.state_dict().items()
            },
            mean=normalizer.mean.detach().cpu().clone(),
            var=normalizer.var.detach().cpu().clone(),
            step=step,
        )

    def seed_snapshot(self, snapshot: SelfPlaySnapshot) -> None:
        self.pool.append(snapshot)
        if self.opponent_step is None:
            self.opponent_step = snapshot["step"]

    def maybe_snapshot(
        self, models: Models, normalizer: ObsNormalizer, step: int
    ) -> bool:
        if step <= 0 or step % self.snapshot_interval != 0:
            return False
        if step == self._last_snapshot_step:
            return False
        snap = self.make_snapshot(models, normalizer, step)
        self.pool.append(snap)
        self._last_snapshot_step = step
        self.log.info(
            "Self-play snapshot pushed at step=%d (pool_size=%d)",
            step,
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

    def maybe_refresh(self, env: F1tenthEnv, step: int) -> bool:
        if not self.pool:
            return False
        if step <= 0 or step % self.refresh_interval != 0:
            return False
        if step == self._last_refresh_step:
            return False
        snap = self._sample_snapshot()
        if snap is None:
            return False
        env.refresh_opponent_policy(snap["actor"], snap["mean"], snap["var"])
        self.opponent_step = snap["step"]
        self._last_refresh_step = step
        self.log.info(
            "Self-play opponent refreshed at step=%d from snapshot step=%d "
            "(pool_size=%d sample=%s)",
            step,
            snap["step"],
            len(self.pool),
            self.sample_mode,
        )
        return True

    def bootstrap_opponent(self, env: F1tenthEnv) -> None:
        """Load the newest pool snapshot into the env opponent (step-0 warm start)."""
        if not self.pool:
            return
        snap = self.pool[-1]
        env.refresh_opponent_policy(snap["actor"], snap["mean"], snap["var"])
        self.opponent_step = snap["step"]
        self.log.info(
            "Self-play opponent bootstrapped from snapshot step=%d (pool_size=%d)",
            snap["step"],
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


def load_init_checkpoint(
    ckpt_path: str | Path,
    models: Models,
    normalizer: ObsNormalizer,
    device: torch.device,
    log: logging.Logger,
) -> None:
    """Warm-start learner actor + obs normalizer from a standalone checkpoint."""
    path = Path(ckpt_path)
    payload = torch.load(path, map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    log.info("Loaded init checkpoint from %s", path)


def _maybe_patch_headless_rasterizer() -> None:
    """Skip pyglet offscreen init when no GUI display is available (CI / agents).

    Matches scripts/physics_check.headless_gs_init; only applied when
    try_get_display_size fails so interactive Mac runs stay unchanged.
    """
    try:
        gs.utils.try_get_display_size()
    except Exception:
        import pyglet
        from genesis.vis.rasterizer import Rasterizer

        pyglet.options["headless"] = True

        def _headless_build(self):
            if self._context is None:
                return
            self.visualizer = self._context.visualizer

        Rasterizer.build = _headless_build


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record so lines appear promptly."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_trainer_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Dedicated logger isolated from Genesis root-logger / FPS timer output."""
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

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ):
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

        emit_mask = self.w_len == self.n_step
        idx = torch.nonzero(emit_mask, as_tuple=False).squeeze(-1)
        n_emit = int(idx.numel())
        if n_emit > 0:
            positions = (
                self.ptr + torch.arange(n_emit, device=self.device)
            ) % self.capacity
            self.obs[positions] = obs0[idx]
            self.action[positions] = act0[idx]
            self.reward[positions] = n_step_reward[idx]
            self.next_obs[positions] = next_obs[idx].detach()
            self.done[positions] = done_f[idx]
            self.ptr = int((self.ptr + n_emit) % self.capacity)
            self.size = min(self.size + n_emit, self.capacity)

        # Clear windows for done envs (sync-free masked write).
        self.w_len = torch.where(
            dones.bool(), torch.zeros_like(self.w_len), self.w_len
        )

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

    def add_mean(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().float()
        self._sum[key] = self._sum.get(key, v.new_zeros(())) + v.mean()
        self._count[key] = self._count.get(key, 0) + 1
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
    diag.add_mean("reward/step", reward)

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
    ):
        value = metrics.get(name)
        if isinstance(value, torch.Tensor):
            if name.startswith("nonfinite_") or name == "laps_completed":
                diag.add_total(f"metric/{name}", value)
            else:
                diag.add_mean(f"metric/{name}", value)

    for name, value in extras.get("termination", {}).items():
        if isinstance(value, torch.Tensor):
            diag.add_total(f"term/{name}", value)

    if actions.ndim == 2 and actions.shape[1] >= 2:
        diag.add_mean("action/throttle", actions[:, 0])
        diag.add_mean("action/steer", actions[:, 1])
    diag.add_mean("obs/abs", obs.abs())


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    if getattr(args, "physics", None) is not None:
        cfg["env"]["physics_backend"] = args.physics
    if getattr(args, "throttle_mode", None) is not None:
        cfg["env"]["throttle_mode"] = args.throttle_mode

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

    if getattr(args, "domain_randomization", False):
        cfg["env"]["domain_randomization"] = {
            **DEFAULT_CONFIG["env"]["domain_randomization"],
            "enabled": True,
        }

    if getattr(args, "zero_tyre_slip_obs", False):
        cfg["obs"]["zero_tyre_slip_obs"] = True

    sp_defaults = DEFAULT_CONFIG["selfplay"]
    cfg["selfplay"] = {
        "snapshot_interval": args.selfplay_snapshot_interval,
        "refresh_interval": args.selfplay_refresh_interval,
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
        cfg["env"]["opponent_spawn_gap_m"] = args.opponent_spawn_gap
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
        # Closing-speed threshold for collision termination (0.0 = terminate on any
        # overlap). Below it, contacts only apply penalties/physics and the episode
        # continues.
        if args.collision_term_speed is not None:
            cfg["env"]["collision_term_speed_mps"] = float(args.collision_term_speed)
        # Car-car contacts need a slightly softer / better-resolved constraint solve.
        cfg["env"]["solver_iterations"] = max(
            int(cfg["env"].get("solver_iterations", 50)), 80
        )
        cfg["env"]["solver_ls_iterations"] = max(
            int(cfg["env"].get("solver_ls_iterations", 50)), 80
        )
        cfg["env"]["constraint_timeconst"] = max(
            float(cfg["env"].get("constraint_timeconst", 0.02)), 0.04
        )
    return cfg


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_genesis_backend(name: str):
    """Resolve a Genesis compute backend. 'gpu' uses CUDA on NVIDIA and Metal on
    Apple Silicon. 'auto' only auto-selects a discrete CUDA GPU and otherwise
    stays on CPU (on Apple Silicon the CPU backend is faster for this workload,
    so Metal must be requested explicitly via --backend gpu/metal)."""
    if name == "cpu":
        return gs.cpu
    if name == "gpu":
        return gs.gpu
    if name == "metal":
        return getattr(gs, "metal", gs.gpu)
    if name == "cuda":
        return getattr(gs, "cuda", gs.gpu)
    return gs.gpu if torch.cuda.is_available() else gs.cpu


def build_models(
    cfg: dict, device: torch.device, alpha: float = 0.01
) -> tuple[Models, QRSACTrainer]:
    # Networks/optimizers stay float32 even when Genesis runs in precision="64"
    # (which flips torch's default dtype to float64); env outputs are bridged to
    # float32 at the boundary. Pin dtype explicitly so module creation under a
    # float64 default still yields float32 weights.
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


def save_checkpoint(
    models: Models,
    step: int,
    ckpt_dir: Path,
    normalizer: "ObsNormalizer | None" = None,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step}.pt"
    payload = {
        "step": step,
        "actor": models.actor.state_dict(),
        "critic1": models.critic1.state_dict(),
        "critic2": models.critic2.state_dict(),
    }
    if normalizer is not None:
        # Eval/deploy MUST apply these same obs stats (e.g. ros2_deploy) since the
        # policy was trained on normalized observations.
        payload["obs_norm"] = normalizer.state_dict()
    torch.save(payload, path)
    logging.getLogger(LOGGER_NAME).info("Saved checkpoint to %s", path)
    return path


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
            env_cfg={**env_cfg, "launch_strategy_data": {"num_cars": num_show}},
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
        car_length=float(env_cfg.get("car_length", 0.46)),
        car_width=float(env_cfg.get("car_width", 0.30)),
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
    obs, _ = env.reset()
    obs = obs.to(torch.float32)
    with torch.no_grad():
        for _ in range(num_steps):
            actions, _ = models.actor(
                normalizer.normalize(obs), deterministic=True, with_logprob=False
            )
            actions = actions.clamp(-clip_actions, clip_actions)
            obs, _, done, _ = env.step(
                actions.to(rt.tc_float), n_steps=control_interval
            )
            obs = obs.to(torch.float32)
            st = env.backend.read_state()
            ego_xy = st["base_pos"][:, :2].cpu().numpy()
            ego_yaw = np.array(
                [yaw_from_quat_wxyz(q.tolist()) for q in st["base_quat"]]
            )
            spd = torch.linalg.norm(st["base_lin_vel"][:, :2], dim=-1).cpu().numpy()
            opp_xy = opp_yaw = None
            if env.has_opponent and "opp_base_pos" in st:
                opp_xy = st["opp_base_pos"][:, :2].cpu().numpy()
                opp_yaw = np.array(
                    [yaw_from_quat_wxyz(q.tolist()) for q in st["opp_base_quat"]]
                )
            viz.render(
                ego_xy=ego_xy,
                ego_yaw=ego_yaw,
                speed=spd,
                opp_xy=opp_xy,
                opp_yaw=opp_yaw if opp_yaw is not None else 0.0,
                done=done.cpu().numpy(),
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
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=cfg["model"]["batch_size"])
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="SAC entropy coefficient (fixed). Default 0.01 matches GT Sophy; "
        "lower temperature lets the policy commit to a fast racing line rather "
        "than staying overly stochastic.",
    )
    parser.add_argument("--min-train-samples", type=int, default=5000)
    parser.add_argument(
        "--n-step",
        type=int,
        default=None,
        help=f"N-step horizon (default: {cfg['model']['n_step']} from config)",
    )
    parser.add_argument("--track", type=str, default=cfg["env"]["track"])
    parser.add_argument(
        "--physics",
        type=str,
        default=cfg["env"].get("physics_backend", "genesis"),
        choices=["genesis", "torch"],
        help="Physics backend: 'genesis' (rigid-body, default) or 'torch' "
        "(pure-Torch f1tenth_sim vehicle model). Same obs/action contract.",
    )
    parser.add_argument(
        "--throttle-mode",
        type=str,
        default=None,
        choices=["force", "speed"],
        help="Throttle semantics for the torch backend: 'force' (drive-force "
        "envelope) or 'speed' (VESC-style speed command). Default from config.",
    )
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
        "--opponent-spawn-gap",
        type=float,
        default=cfg["env"]["opponent_spawn_gap_m"],
        help="Meters the opponent spawns ahead of the ego on the centerline.",
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
        default=0.5,
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
        default=1.0,
        help="Reward scale for the GT Sophy rear-end penalty Rr (-rear_end_k * "
        "closing-speed^2 when colliding with an opponent ahead). 0.0 disables it. "
        "Only used when --opponent is not 'none'.",
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
        "--domain-randomization",
        action="store_true",
        default=False,
        help="Enable per-episode domain randomization (friction, mass, latency, obs noise).",
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
        default=cfg["selfplay"]["snapshot_interval"],
        help="Environment steps between learner snapshots added to the opponent pool.",
    )
    parser.add_argument(
        "--selfplay-refresh-interval",
        type=int,
        default=cfg["selfplay"]["refresh_interval"],
        help="Environment steps between opponent policy refreshes from the pool.",
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
        "--init-ckpt",
        type=str,
        default=None,
        help="Warm-start the learner actor (+ obs_norm) and seed the self-play pool "
        "from this standalone checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "metal"],
        help="Genesis sim backend. 'gpu' uses CUDA (NVIDIA) or Metal (Apple "
        "Silicon); 'auto' prefers any available GPU. Note: the Metal backend "
        "only supports precision=32.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="64",
        choices=["32", "64"],
        help="Genesis float precision. 64 is the strongest NaN-stability lever "
        "(slower / 2x memory); use 32 for fast iteration or GPU throughput.",
    )
    parser.add_argument("--ckpt-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--eval-video-interval",
        type=int,
        default=0,
        help="Render a deterministic eval rollout every N steps (0=off). Uses a "
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

    physics_backend = str(cfg["env"].get("physics_backend", "genesis"))
    if physics_backend == "torch":
        # The pure-Torch vehicle sim is Genesis-free: skip gs.init() entirely and
        # configure the env's own dtype/device runtime. No Genesis runtime / no GL.
        device = select_device(args.device)
        rt.configure(
            float_dtype=torch.float64 if args.precision == "64" else torch.float32,
            int_dtype=torch.int32,
            dev=device,
            eps=1e-12,
        )
    else:
        _maybe_patch_headless_rasterizer()
        backend = select_genesis_backend(args.backend)
        gs.init(
            backend=backend,
            precision=args.precision,
            performance_mode=True,
        )
        # Keep the RL pipeline (normalizer, networks, replay buffer) on the same
        # device as the Genesis sim so env outputs don't straddle two devices. On a
        # GPU backend gs.device is the accelerator (CUDA / Apple MPS); on CPU it is
        # cpu and we honour the explicit --device choice.
        if backend == gs.cpu:
            device = select_device(args.device)
        else:
            device = gs.device
        rt.configure(
            float_dtype=gs.tc_float, int_dtype=gs.tc_int, dev=device, eps=gs.EPS,
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

    if args.init_ckpt is not None:
        load_init_checkpoint(args.init_ckpt, models, normalizer, device, log)

    selfplay_mgr: SelfPlayManager | None = None
    if args.self_play or args.mixed_opponents:
        sp_cfg = cfg["selfplay"]
        selfplay_mgr = SelfPlayManager(
            pool_size=sp_cfg["pool_size"],
            snapshot_interval=sp_cfg["snapshot_interval"],
            refresh_interval=sp_cfg["refresh_interval"],
            sample_mode=sp_cfg["sample_mode"],
            mixed_latest_prob=sp_cfg["mixed_latest_prob"],
            log=log,
        )
        selfplay_mgr.seed_snapshot(
            SelfPlayManager.make_snapshot(models, normalizer, step=0)
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
            "resume": "allow",
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
    global_step = 0
    train_updates = 0
    episode_rewards = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    recent_episode_rewards: deque[float] = deque(maxlen=RECENT_EPISODES_MAX)
    policy_loss_accum = 0.0
    critic_loss_accum = 0.0
    loss_count = 0
    diag = RunningStats()
    last_batch: dict[str, torch.Tensor] | None = None
    t_start = time.perf_counter()
    consecutive_nan_steps = 0
    total_nan_resets = 0
    max_consecutive_nan = 20
    eval_state: dict = {}

    try:
        while global_step < args.total_steps:
            bad_obs_mask = (~torch.isfinite(obs)).any(dim=1)
            if bad_obs_mask.any():
                diag.add_total(
                    "nonfinite/pre_step_obs_resets", bad_obs_mask.to(torch.float32)
                )
                reset_obs, _ = env.reset(envs_idx=bad_obs_mask)
                obs = obs.clone()
                obs[bad_obs_mask] = reset_obs[bad_obs_mask].to(torch.float32)

            if buffer.size < args.min_train_samples:
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

            try:
                next_obs, reward, done, extras = env.step(
                    actions.to(rt.tc_float), n_steps=control_interval
                )
            except gs.GenesisException as exc:
                consecutive_nan_steps += 1
                total_nan_resets += 1
                diag.add_total("nonfinite/genesis_exceptions", torch.ones((), device=device))
                log.warning(
                    "Genesis raised at step %d (NaN constraint forces): %s. "
                    "Resetting envs (consecutive=%d total=%d).",
                    global_step,
                    exc,
                    consecutive_nan_steps,
                    total_nan_resets,
                )
                obs, _ = env.reset()
                obs = obs.to(torch.float32)
                episode_rewards.zero_()
                global_step += 1
                if global_step % args.ckpt_interval == 0:
                    save_checkpoint(models, global_step, ckpt_dir, normalizer)
                if consecutive_nan_steps >= max_consecutive_nan:
                    raise RuntimeError(
                        f"Physics NaN persisted for {consecutive_nan_steps} "
                        f"consecutive steps ({total_nan_resets} total NaN resets). "
                        "Decrease sim_dt or abort corrupted run."
                    ) from exc
                continue
            consecutive_nan_steps = 0
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
            diag.add_mean("obs/norm_abs", normalizer.normalize(obs).abs())

            if use_1v1 and obs.shape[-1] > OPP_OBS_BASE_IDX:
                opp_block = obs[:, OPP_OBS_BASE_IDX:]
                in_range = (opp_block != 0).any(dim=-1).to(obs.dtype)
                diag.add_mean("metric/opponent_presence", in_range)

            bad_obs_mask = (~torch.isfinite(next_obs)).any(dim=1)
            finite_ok = bool(
                torch.isfinite(reward).all() and not bad_obs_mask.any()
            )
            if finite_ok:
                buffer.add(obs, actions, reward, next_obs, done)
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
                    "Non-finite step at %d (reward_bad=%d obs_bad_envs=%d); "
                    "skipping buffer add.",
                    global_step,
                    n_bad_reward,
                    n_bad_obs_envs,
                )
                if n_bad_obs_envs > 0:
                    reset_obs, _ = env.reset(envs_idx=bad_obs_mask)
                    next_obs = next_obs.clone()
                    next_obs[bad_obs_mask] = reset_obs[bad_obs_mask].to(torch.float32)
            obs = next_obs
            global_step += 1

            if buffer.size >= args.min_train_samples:
                for _ in range(args.updates_per_step):
                    batch = buffer.sample(args.batch_size)
                    # Buffer stores RAW obs; normalize with current stats at input.
                    batch["obs"] = normalizer.normalize(batch["obs"])
                    batch["next_obs"] = normalizer.normalize(batch["next_obs"])
                    losses = trainer.update(batch)
                    train_updates += 1
                    policy_loss_accum += losses.policy_loss
                    critic_loss_accum += losses.critic_loss
                    loss_count += 1
                last_batch = batch

            if selfplay_mgr is not None:
                selfplay_mgr.maybe_snapshot(models, normalizer, global_step)
                selfplay_mgr.maybe_refresh(env, global_step)

            if global_step % args.log_interval == 0:
                elapsed = time.perf_counter() - t_start
                steps_per_sec = global_step / max(elapsed, 1e-6)
                mean_ep_reward = (
                    sum(recent_episode_rewards) / len(recent_episode_rewards)
                    if recent_episode_rewards
                    else float("nan")
                )
                mean_policy_loss = (
                    policy_loss_accum / loss_count if loss_count else float("nan")
                )
                mean_critic_loss = (
                    critic_loss_accum / loss_count if loss_count else float("nan")
                )
                buffer_fill_pct = 100.0 * buffer.size / buffer.capacity
                log.info(
                    "step=%d buffer=%d/%d (%.1f%%) train_updates=%d steps/s=%.1f "
                    "policy_loss=%.4f critic_loss=%.4f mean_ep_reward=%.4f (n=%d)",
                    global_step,
                    buffer.size,
                    buffer.capacity,
                    buffer_fill_pct,
                    train_updates,
                    steps_per_sec,
                    mean_policy_loss,
                    mean_critic_loss,
                    mean_ep_reward,
                    len(recent_episode_rewards),
                )
                mean_q = float("nan")
                if last_batch is not None:
                    with torch.no_grad():
                        mean_q = (
                            models.critic1(last_batch["obs"], last_batch["action"])
                            .mean()
                            .item()
                        )
                nstep_buf_reward_mean = (
                    float(buffer.reward[: buffer.size].mean())
                    if buffer.size > 0
                    else float("nan")
                )
                window_env_steps = float(args.log_interval * args.num_envs)
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
                        "tyre_slip=%.4f smooth=%.4f | nstep_buf_reward=%.4f mean_Q=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/passing"),
                        diag.mean("reward_term/collision"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                        nstep_buf_reward_mean,
                        mean_q,
                    )
                else:
                    log.info(
                        "  rewards: total[mean=%.4f min=%.4f max=%.4f] "
                        "progress=%.4f oob_penalty=%.4f tyre_slip=%.4f "
                        "smooth=%.4f | nstep_buf_reward=%.4f mean_Q=%.4f",
                        diag.mean("reward/step"),
                        diag.vmin("reward/step"),
                        diag.vmax("reward/step"),
                        diag.mean("reward_term/progress"),
                        diag.mean("reward_term/oob_penalty"),
                        diag.mean("reward_term/tyre_slip_penalty"),
                        diag.mean("reward_term/smoothness"),
                        nstep_buf_reward_mean,
                        mean_q,
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
                    "  nonfinite: obs_rate=%.2e reward_rate=%.2e state_rate=%.2e "
                    "genesis_exc=%d pre_obs_reset=%d post_obs_bad=%d post_reward_bad=%d",
                    nf_obs_rate,
                    nf_reward_rate,
                    nf_state_rate,
                    int(diag.total("nonfinite/genesis_exceptions")),
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
                            global_step - selfplay_mgr.opponent_step
                            if selfplay_mgr.opponent_step is not None
                            else -1
                        )
                        log.info(
                            "  selfplay: pool_size=%d opp_step=%s opp_age=%d "
                            "win_rate=%.3f (n=%d)",
                            len(selfplay_mgr.pool),
                            selfplay_mgr.opponent_step,
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
                            "step": global_step,
                            "buffer/size": buffer.size,
                            "train/updates": train_updates,
                            "train/policy_loss": mean_policy_loss,
                            "train/critic_loss": mean_critic_loss,
                            "train/mean_ep_reward": mean_ep_reward,
                            "train/mean_Q": mean_q,
                            "train/nstep_buf_reward": nstep_buf_reward_mean,
                            "perf/steps_per_sec": steps_per_sec,
                            "reward/total_mean": diag.mean("reward/step"),
                            "reward/total_min": diag.vmin("reward/step"),
                            "reward/total_max": diag.vmax("reward/step"),
                            "reward/progress": diag.mean("reward_term/progress"),
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
                            "nonfinite/genesis_exceptions": diag.total(
                                "nonfinite/genesis_exceptions"
                            ),
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
                        step=global_step,
                    )
                policy_loss_accum = 0.0
                critic_loss_accum = 0.0
                loss_count = 0
                diag.reset()

            if global_step % args.ckpt_interval == 0:
                save_checkpoint(models, global_step, ckpt_dir, normalizer)

            if (
                args.eval_video_interval > 0
                and global_step % args.eval_video_interval == 0
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
                        step=global_step,
                        num_steps=args.eval_video_steps,
                        num_show=args.eval_video_num_envs,
                        live=args.eval_video_live,
                        wandb_run=wandb_run,
                        log=log,
                        selfplay_mgr=selfplay_mgr,
                    )
                except Exception as exc:
                    log.warning("Eval video rollout failed (continuing): %s", exc)

        save_checkpoint(models, global_step, ckpt_dir, normalizer)
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

    log.info("Training finished after %d steps (%d updates).", global_step, train_updates)


if __name__ == "__main__":
    main()

"""Delayed self-play: snapshot learner into a pool, refresh opponent periodically."""

from __future__ import annotations

import logging
import random
from collections import deque

import torch

from f1tenth_policy import (
    ObsNormalizer,
    actor_architecture_from_module,
    architectures_match,
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)
from f1tenth_env import F1tenthEnv
from qrsac import Models


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
            dict(expected_architecture)
            if expected_architecture is not None
            else None
        )
        self.anchor: SelfPlaySnapshot | None = None
        self.log = log or logging.getLogger(__name__)
        self.pool: deque[SelfPlaySnapshot] = deque(maxlen=pool_size)
        self.opponent_transitions: int | None = None
        self._last_snapshot_transitions = 0
        self._last_refresh_transitions = 0
        self._episode_wins = 0
        self._episode_total = 0

    def _require_homogeneous_architecture(self, architecture: dict) -> dict:
        arch = dict(architecture)
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
        expected_layout_version: int,
        expected_architecture: dict | None = None,
        expected_critic_obs_dim: int | None = None,
        expected_steering_action_mode: str = "delta",
        expected_steering_delta_max_rad: float | None = None,
    ) -> None:
        payload = load_sensor_artifact(path, map_location=device)
        architecture = expected_architecture or payload.get("actor_architecture")
        validate_sensor_policy_artifact(
            payload,
            expected_actor_obs_dim=obs_dim,
            expected_action_dim=action_dim,
            expected_layout_version=expected_layout_version,
            expected_architecture=architecture,
            expected_critic_obs_dim=expected_critic_obs_dim,
            expected_steering_action_mode=expected_steering_action_mode,
            expected_steering_delta_max_rad=expected_steering_delta_max_rad,
        )
        architecture = self._require_homogeneous_architecture(
            architecture or payload["actor_architecture"]
        )
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
            "Self-play opponent refreshed at transitions=%d from snapshot "
            "transitions=%d (pool_size=%d sample=%s)",
            transitions,
            snap["transitions"],
            len(self.pool),
            self.sample_mode,
        )
        return True

    def bootstrap_opponent(self, env: F1tenthEnv) -> None:
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
            "Self-play opponent bootstrapped from snapshot transitions=%d "
            "(pool_size=%d)",
            snap["transitions"],
            len(self.pool),
        )

    def refresh_eval_opponent(self, env: F1tenthEnv) -> bool:
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

    def reseed_after_reinit(
        self,
        models: Models,
        normalizer: ObsNormalizer,
        env: F1tenthEnv,
        transitions: int,
    ) -> None:
        self.pool.clear()
        self.opponent_transitions = None
        self._last_snapshot_transitions = int(transitions)
        self._last_refresh_transitions = int(transitions)
        snap = self.make_snapshot(models, normalizer, transitions)
        self.seed_snapshot(snap)
        self.bootstrap_opponent(env)
        self.log.info(
            "Self-play pool reset and reseeding after replay-full reinit "
            "(transitions=%d pool_size=%d)",
            transitions,
            len(self.pool),
        )

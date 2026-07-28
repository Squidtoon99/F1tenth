"""Fixed opponent pool with per-episode weighted sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from f1tenth_policy import (
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)


@dataclass(frozen=True)
class ChampionEntry:
    checkpoint: str
    weight: float
    transitions: int
    actor: dict[str, torch.Tensor]
    mean: torch.Tensor
    var: torch.Tensor
    actor_architecture: dict


@dataclass(frozen=True)
class ChampionSelection:
    checkpoint: str
    weight: float
    transitions: int
    actor: dict[str, torch.Tensor]
    mean: torch.Tensor
    var: torch.Tensor
    actor_architecture: dict


def _normalize_entries(entries: list[dict[str, Any]]) -> list[tuple[str, float]]:
    if not entries:
        raise ValueError("fixed_opponents.entries must be a non-empty list")
    parsed: list[tuple[str, float]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"fixed_opponents.entries[{i}] must be a mapping")
        path = entry.get("checkpoint")
        weight = float(entry.get("weight", 1.0))
        if not path:
            raise ValueError(f"fixed_opponents.entries[{i}] missing checkpoint")
        if weight <= 0.0:
            raise ValueError(
                f"fixed_opponents.entries[{i}] weight must be positive, got {weight}"
            )
        parsed.append((str(path), weight))
    return parsed


def _load_entry(
    path: str,
    weight: float,
    *,
    expected_architecture: dict,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    expected_critic_obs_dim: int | None,
    expected_steering_action_mode: str,
    expected_steering_delta_max_rad: float,
    device: torch.device,
    log: logging.Logger,
) -> ChampionEntry:
    payload = load_sensor_artifact(path, map_location=device)
    champion_delta = payload.get("steering_delta_max_rad")
    if champion_delta is None:
        artifact_delta = expected_steering_delta_max_rad
    else:
        artifact_delta = float(champion_delta)
    validate_sensor_policy_artifact(
        payload,
        expected_architecture=expected_architecture,
        expected_actor_obs_dim=expected_actor_obs_dim,
        expected_action_dim=expected_action_dim,
        expected_layout_version=expected_layout_version,
        expected_critic_obs_dim=expected_critic_obs_dim,
        expected_steering_action_mode=expected_steering_action_mode,
        expected_steering_delta_max_rad=artifact_delta,
    )
    if (
        champion_delta is not None
        and abs(artifact_delta - float(expected_steering_delta_max_rad)) > 1.0e-9
    ):
        log.warning(
            "Fixed opponent %s steering_delta_max_rad=%.12f differs from learner "
            "env %.12f; opponent normalized actions use the learner env scale.",
            path,
            artifact_delta,
            float(expected_steering_delta_max_rad),
        )
    return ChampionEntry(
        checkpoint=path,
        weight=weight,
        transitions=int(payload.get("env_transitions", 0)),
        actor={k: v.detach().cpu().clone() for k, v in payload["actor"].items()},
        mean=payload["obs_norm"]["mean"].detach().cpu().clone(),
        var=payload["obs_norm"]["var"].detach().cpu().clone(),
        actor_architecture=dict(payload["actor_architecture"]),
    )


def load_opponent_pool(
    entries: list[dict[str, Any]],
    *,
    expected_architecture: dict,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    expected_critic_obs_dim: int | None,
    expected_steering_action_mode: str,
    expected_steering_delta_max_rad: float,
    device: torch.device = torch.device("cpu"),
    log: logging.Logger | None = None,
) -> tuple[list[ChampionEntry], torch.Tensor]:
    """Load every weighted pool entry once at startup."""
    parsed = _normalize_entries(entries)
    logger = log or logging.getLogger("fixed_opponents")
    loaded: list[ChampionEntry] = []
    for path, weight in parsed:
        loaded.append(
            _load_entry(
                path,
                weight,
                expected_architecture=expected_architecture,
                expected_actor_obs_dim=expected_actor_obs_dim,
                expected_action_dim=expected_action_dim,
                expected_layout_version=expected_layout_version,
                expected_critic_obs_dim=expected_critic_obs_dim,
                expected_steering_action_mode=expected_steering_action_mode,
                expected_steering_delta_max_rad=expected_steering_delta_max_rad,
                device=device,
                log=logger,
            )
        )
    weights = torch.tensor([entry.weight for entry in loaded], dtype=torch.float64)
    logger.info(
        "Fixed opponent pool loaded size=%d total_weight=%.4f",
        len(loaded),
        float(weights.sum().item()),
    )
    for i, entry in enumerate(loaded):
        logger.info(
            "  pool[%d] checkpoint=%s weight=%.4f transitions=%d",
            i,
            entry.checkpoint,
            entry.weight,
            entry.transitions,
        )
    return loaded, weights


def select_champion(
    entries: list[dict[str, Any]],
    *,
    seed: int,
    expected_architecture: dict,
    expected_actor_obs_dim: int,
    expected_action_dim: int,
    expected_layout_version: int,
    expected_critic_obs_dim: int | None,
    expected_steering_action_mode: str,
    expected_steering_delta_max_rad: float,
    device: torch.device = torch.device("cpu"),
    log: logging.Logger | None = None,
) -> ChampionSelection:
    """Deterministically pick one weighted champion and load it once."""
    parsed = _normalize_entries(entries)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    weights = torch.tensor([w for _, w in parsed], dtype=torch.float64)
    idx = int(torch.multinomial(weights, 1, generator=rng).item())
    path, weight = parsed[idx]
    logger = log or logging.getLogger("fixed_opponents")
    entry = _load_entry(
        path,
        weight,
        expected_architecture=expected_architecture,
        expected_actor_obs_dim=expected_actor_obs_dim,
        expected_action_dim=expected_action_dim,
        expected_layout_version=expected_layout_version,
        expected_critic_obs_dim=expected_critic_obs_dim,
        expected_steering_action_mode=expected_steering_action_mode,
        expected_steering_delta_max_rad=expected_steering_delta_max_rad,
        device=device,
        log=logger,
    )
    selection = ChampionSelection(
        checkpoint=entry.checkpoint,
        weight=entry.weight,
        transitions=entry.transitions,
        actor=entry.actor,
        mean=entry.mean,
        var=entry.var,
        actor_architecture=entry.actor_architecture,
    )
    logger.info(
        "Fixed champion selected index=%d/%d checkpoint=%s weight=%.4f "
        "transitions=%d seed=%d",
        idx,
        len(parsed),
        path,
        weight,
        selection.transitions,
        seed,
    )
    return selection


class FixedChampionManager:
    """Immutable opponent pool with per-reset weighted sampling."""

    def __init__(
        self,
        entries: list[ChampionEntry],
        weights: torch.Tensor,
        *,
        seed: int,
        log: logging.Logger | None = None,
    ):
        if not entries:
            raise ValueError("FixedChampionManager requires at least one entry")
        self.entries = entries
        self.weights = weights.to(dtype=torch.float64, device="cpu")
        self.seed = int(seed)
        self.log = log or logging.getLogger("fixed_opponents")
        self._sample_counter = 0
        self._assignment_counts = torch.zeros(len(entries), dtype=torch.int64)

    @property
    def pool_size(self) -> int:
        return len(self.entries)

    def _draw_indices(self, count: int) -> torch.Tensor:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + self._sample_counter)
        self._sample_counter += 1
        drawn = torch.multinomial(
            self.weights, int(count), replacement=True, generator=rng
        )
        for idx in drawn.tolist():
            self._assignment_counts[int(idx)] += 1
        return drawn.to(dtype=torch.long)

    def bootstrap_opponent(self, env, *, resample: bool = False) -> None:
        env.refresh_opponent_pool(self.entries)
        env.set_opponent_resample_callback(self.resample_opponents)
        if resample:
            self.sample_all(env)

    def sample_all(self, env) -> None:
        indices = self._draw_indices(env.num_envs)
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env.assign_opponent_policies(mask, indices.to(device=env.device))

    def resample_opponents(self, env, reset_mask: torch.Tensor) -> None:
        mask_b = reset_mask.to(dtype=torch.bool)
        count = int(mask_b.sum().item())
        if count == 0:
            return
        indices = self._draw_indices(count)
        env.assign_opponent_policies(mask_b, indices.to(device=env.device))

    def metadata(self) -> dict[str, Any]:
        total = int(self._assignment_counts.sum().item())
        proportions = [
            float(self._assignment_counts[i].item()) / total if total else 0.0
            for i in range(len(self.entries))
        ]
        return {
            "pool_size": len(self.entries),
            "entries": [
                {
                    "checkpoint": entry.checkpoint,
                    "weight": entry.weight,
                    "transitions": entry.transitions,
                    "sampled_fraction": proportions[i],
                }
                for i, entry in enumerate(self.entries)
            ],
            "checkpoint": self.entries[0].checkpoint,
            "transitions": self.entries[0].transitions,
            "actor_architecture": self.entries[0].actor_architecture,
        }

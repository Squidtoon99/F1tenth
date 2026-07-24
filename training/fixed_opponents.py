"""Fixed champion opponent pool."""

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
    logger = log or logging.getLogger("fixed_opponents")
    if (
        champion_delta is not None
        and abs(artifact_delta - float(expected_steering_delta_max_rad)) > 1.0e-9
    ):
        logger.warning(
            "Fixed champion steering_delta_max_rad=%.12f differs from learner "
            "env %.12f; opponent normalized actions use the learner env scale.",
            artifact_delta,
            float(expected_steering_delta_max_rad),
        )
    selection = ChampionSelection(
        checkpoint=path,
        weight=weight,
        transitions=int(payload.get("env_transitions", 0)),
        actor={k: v.detach().cpu().clone() for k, v in payload["actor"].items()},
        mean=payload["obs_norm"]["mean"].detach().cpu().clone(),
        var=payload["obs_norm"]["var"].detach().cpu().clone(),
        actor_architecture=dict(payload["actor_architecture"]),
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
    """Holds one immutable champion loaded at startup."""

    def __init__(self, selection: ChampionSelection, log: logging.Logger | None = None):
        self.selection = selection
        self.log = log or logging.getLogger("fixed_opponents")

    def bootstrap_opponent(self, env) -> None:
        env.refresh_opponent_policy(
            self.selection.actor,
            self.selection.mean,
            self.selection.var,
            actor_architecture=self.selection.actor_architecture,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.selection.checkpoint,
            "weight": self.selection.weight,
            "transitions": self.selection.transitions,
            "actor_architecture": self.selection.actor_architecture,
        }

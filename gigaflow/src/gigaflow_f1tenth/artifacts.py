"""Deployable actor artifact format (critic never included)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from gigaflow_f1tenth.config import ExperimentConfig
from gigaflow_f1tenth.model import (
    actor_architecture_from_module,
    architecture_metadata as actor_architecture_metadata,
)
from gigaflow_f1tenth.rewards import (
    CONDITION_DIM,
    CONDITION_FIELD_NAMES,
    DEPLOYMENT_STYLES,
    condition_schema_fields,
    deployment_style,
    normalize_condition_vector,
)

ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_SCOPE_SIM_TRAINING = "simulation_training_only"


@dataclass(frozen=True)
class ArtifactManifest:
    format_version: int
    scope: str
    actor_architecture: Mapping[str, Any]
    condition_schema: Mapping[str, Any]
    deployment_style: str
    track_manifest_hash: str | None


def condition_schema(cfg: ExperimentConfig) -> dict[str, Any]:
    rc = cfg.reward_conditioning
    style_name = cfg.evaluation.conservative_deployment_style
    style = deployment_style(style_name)
    return {
        "condition_dim": cfg.agents.condition_dim,
        "schema_dim": CONDITION_DIM,
        "field_names": list(CONDITION_FIELD_NAMES),
        "fields": condition_schema_fields(),
        "enabled": rc.enabled,
        "x_drive_a": rc.x_drive_a,
        "x_accel_a": rc.x_accel_a,
        "omit_comfort": True,
        "omit_goal_stop_line": True,
        "deployment_styles": sorted(DEPLOYMENT_STYLES),
        "conservative_deployment_style": style_name,
        "conservative_raw": style.as_dict(),
        "notes": (
            "Private normalized side channel; not part of the 1097-D sensor contract. "
            "Only estimable dynamics scales and private reward weights are included."
        ),
    }


def build_manifest(
    cfg: ExperimentConfig,
    deployment_style_name: str | None = None,
    track_manifest_hash: str | None = None,
) -> ArtifactManifest:
    style = deployment_style_name or cfg.evaluation.conservative_deployment_style
    return ArtifactManifest(
        format_version=ARTIFACT_FORMAT_VERSION,
        scope=ARTIFACT_SCOPE_SIM_TRAINING,
        actor_architecture=actor_architecture_metadata(cfg),
        condition_schema=condition_schema(cfg),
        deployment_style=style,
        track_manifest_hash=track_manifest_hash,
    )


def actor_state_dict_for_artifact(actor: torch.nn.Module) -> dict[str, Any]:
    """Weights suitable for deploy packages (caller must not pass a critic)."""
    return {k: v.detach().cpu() for k, v in actor.state_dict().items()}


def _sensor_normalizer_payload(
    actor: torch.nn.Module, override: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Prefer an explicit override; else read the actor's own running stats.

    The actor owns its normalizer (applied inside ``forward``), so the common
    path needs no caller wiring: exporting any built actor yields well-formed
    stats, even a freshly built one whose normalizer has not seen data yet.
    """
    if override is not None:
        return dict(override)
    normalizer = getattr(actor, "sensor_normalizer", None)
    if normalizer is not None and hasattr(normalizer, "to_dict"):
        return normalizer.to_dict()
    return {}


def build_actor_artifact_payload(
    cfg: ExperimentConfig,
    actor: torch.nn.Module,
    *,
    deployment_style_name: str | None = None,
    track_manifest_hash: str | None = None,
    sensor_normalizer: Mapping[str, Any] | None = None,
    condition_normalizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a deployable actor payload; critic fields are never included."""
    style = deployment_style_name or cfg.evaluation.conservative_deployment_style
    style_vec = normalize_condition_vector(deployment_style(style).raw_vector())
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "scope": ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": actor_architecture_from_module(actor),
        "condition_schema": condition_schema(cfg),
        "deployment_style": style,
        "deployment_condition_normalized": style_vec.tolist(),
        "track_manifest_hash": track_manifest_hash,
        "actor_state_dict": actor_state_dict_for_artifact(actor),
        "sensor_normalizer": _sensor_normalizer_payload(actor, sensor_normalizer),
        "condition_normalizer": dict(condition_normalizer or {"mode": "schema_ranges"}),
    }


@runtime_checkable
class ArtifactStore(Protocol):
    def write_actor_artifact(self, path: str, payload: Mapping[str, Any]) -> None:
        ...

    def read_actor_artifact(self, path: str) -> Mapping[str, Any]:
        ...


class FileArtifactStore:
    def write_actor_artifact(self, path: str, payload: Mapping[str, Any]) -> None:
        validate_actor_artifact(payload)
        torch.save(dict(payload), path)

    def read_actor_artifact(self, path: str) -> Mapping[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError("artifact payload must be a mapping")
        validate_actor_artifact(payload)
        return payload


def export_actor_artifact(
    cfg: ExperimentConfig, actor: Any, path: str, **kwargs: Any
) -> None:
    payload = build_actor_artifact_payload(cfg, actor, **kwargs)
    FileArtifactStore().write_actor_artifact(path, payload)


def validate_actor_artifact(payload: Mapping[str, Any]) -> None:
    if int(payload.get("format_version", -1)) != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported artifact format_version")
    if "critic" in payload or "central_critic" in payload:
        raise ValueError("deployable artifacts must not contain critic weights")
    arch = payload.get("actor_architecture")
    if not isinstance(arch, Mapping):
        raise ValueError("actor_architecture missing")
    if int(arch.get("sensor_obs_dim", -1)) != 1097:
        raise ValueError("artifact sensor_obs_dim must be 1097")
    if int(arch.get("action_dim", -1)) != 2:
        raise ValueError("artifact action_dim must be 2")
    schema = payload.get("condition_schema")
    if schema is not None:
        if not isinstance(schema, Mapping):
            raise ValueError("condition_schema must be a mapping")
        if int(schema.get("condition_dim", -1)) != CONDITION_DIM:
            raise ValueError(f"condition_schema.condition_dim must be {CONDITION_DIM}")
        if list(schema.get("field_names", ())) != list(CONDITION_FIELD_NAMES):
            raise ValueError("condition_schema.field_names must match the live layout")
    _validate_sensor_normalizer(payload.get("sensor_normalizer"), arch)


def _validate_sensor_normalizer(normalizer: Any, arch: Mapping[str, Any]) -> None:
    """The policy normalizes sensor obs internally; deploy inference needs
    these stats to match training, so an artifact without them is unusable."""
    if not isinstance(normalizer, Mapping) or not normalizer:
        raise ValueError("sensor_normalizer stats are required and must be non-empty")
    required = {"dim", "mean", "m2", "count"}
    missing = required - set(normalizer)
    if missing:
        raise ValueError(f"sensor_normalizer missing required keys: {sorted(missing)}")
    sensor_dim = int(arch.get("sensor_obs_dim", -1))
    if int(normalizer["dim"]) != sensor_dim:
        raise ValueError(
            f"sensor_normalizer.dim={normalizer['dim']} != sensor_obs_dim={sensor_dim}"
        )
    mean = torch.as_tensor(normalizer["mean"])
    m2 = torch.as_tensor(normalizer["m2"])
    if mean.shape[-1] != sensor_dim or m2.shape[-1] != sensor_dim:
        raise ValueError("sensor_normalizer mean/m2 must have length sensor_obs_dim")

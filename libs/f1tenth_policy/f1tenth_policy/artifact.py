
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from f1tenth_policy.actor import (
    actor_from_architecture,
    architectures_match,
    normalize_actor_architecture,
)
from f1tenth_policy.current import (
    TRAINING_I_BRAKE_MAX_A,
    TRAINING_I_DRIVE_MAX_A,
    TRAINING_I_SLEW_A_PER_S,
)
from f1tenth_policy.layout import (
    ACTOR_LAYOUT_VERSION,
    ACTOR_OBS_DIM,
    ARTIFACT_SCOPE_SIM_TRAINING,
    CONTROL_HZ,
    CRITIC_OBS_DIM,
    NUM_ACTIONS,
    OBS_PREPROCESSING_VERSION,
    SENSOR_POLICY_FORMAT_VERSION,
    STEERING_ACTION_MODE,
    STEERING_DELTA_MAX_RAD,
)


def _tensor_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _first_state_dict_mismatch(actual: Mapping, expected: Mapping) -> str | None:
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


def validate_sensor_policy_artifact(
    payload: Any,
    *,
    expected_architecture: Mapping[str, Any],
    expected_actor_obs_dim: int = ACTOR_OBS_DIM,
    expected_action_dim: int = NUM_ACTIONS,
    expected_layout_version: int = ACTOR_LAYOUT_VERSION,
    expected_critic_obs_dim: int | None = CRITIC_OBS_DIM,
    expected_steering_action_mode: str = STEERING_ACTION_MODE,
    expected_steering_delta_max_rad: float | None = STEERING_DELTA_MAX_RAD,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Sensor policy artifact must be a mapping.")
    version = payload.get("policy_format_version")
    if version is None:
        raise ValueError("Sensor policy artifact is missing policy_format_version.")
    version = int(version)
    if version < SENSOR_POLICY_FORMAT_VERSION:
        raise ValueError(
            f"Sensor policy artifact policy_format_version={version!r} is unsupported; "
            f"need >={SENSOR_POLICY_FORMAT_VERSION}."
        )
    scope = payload.get("artifact_scope")
    if scope != ARTIFACT_SCOPE_SIM_TRAINING:
        raise ValueError(
            f"Sensor policy artifact artifact_scope={scope!r}; "
            f"expected {ARTIFACT_SCOPE_SIM_TRAINING!r}."
        )

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
            f"{expected_actor_obs_dim}."
        )
    layout = payload.get("actor_layout_version")
    if layout is None or int(layout) != int(expected_layout_version):
        raise ValueError(
            f"Sensor policy artifact actor_layout_version={layout!r}; "
            f"expected {expected_layout_version}."
        )
    preprocessing = payload.get("observation_preprocessing_version")
    if preprocessing is None or int(preprocessing) != OBS_PREPROCESSING_VERSION:
        raise ValueError(
            "Sensor policy artifact observation_preprocessing_version="
            f"{preprocessing!r}; expected {OBS_PREPROCESSING_VERSION}."
        )
    for key in ("i_drive_max_a", "i_brake_max_a", "i_slew_a_per_s"):
        value = payload.get(key)
        if value is None:
            raise ValueError(
                f"Sensor policy artifact is missing {key} (physical current "
                "scale for normalized effort=1)."
            )
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"Sensor policy artifact {key}={value!r} must be finite and > 0."
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
    action_mode = payload.get("steering_action_mode", "absolute")
    if action_mode != expected_steering_action_mode:
        raise ValueError(
            f"Sensor policy artifact steering_action_mode={action_mode!r}; "
            f"expected {expected_steering_action_mode!r}."
        )
    if expected_steering_action_mode == "delta":
        delta_max = payload.get("steering_delta_max_rad")
        if delta_max is None or expected_steering_delta_max_rad is None:
            raise ValueError(
                "Delta sensor policy artifact is missing steering_delta_max_rad."
            )
        if abs(float(delta_max) - float(expected_steering_delta_max_rad)) > 1.0e-9:
            raise ValueError(
                "Sensor policy artifact steering_delta_max_rad="
                f"{delta_max!r}; expected {expected_steering_delta_max_rad!r}."
            )
    if expected_critic_obs_dim is not None:
        critic_obs_dim = payload.get("critic_obs_dim")
        if critic_obs_dim is None or int(critic_obs_dim) != int(expected_critic_obs_dim):
            raise ValueError(
                f"Sensor policy artifact critic_obs_dim={critic_obs_dim!r}; "
                f"expected provenance dim {expected_critic_obs_dim}."
            )
    stats = payload.get("obs_norm")
    if not isinstance(stats, Mapping):
        raise ValueError("Sensor policy artifact is missing obs_norm statistics.")
    for name in ("mean", "var"):
        shape = _tensor_shape(stats.get(name))
        if shape != (int(expected_actor_obs_dim),):
            raise ValueError(
                f"Sensor policy artifact obs_norm.{name} shape={shape}; "
                f"expected ({expected_actor_obs_dim},)."
            )

    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(
            "Sensor policy artifact is missing actor_architecture metadata."
        )
    if not architectures_match(architecture, expected_architecture):
        raise ValueError(
            "Sensor policy artifact actor_architecture mismatch: "
            f"got {normalize_actor_architecture(architecture)!r}, "
            f"expected {normalize_actor_architecture(expected_architecture)!r}"
        )
    actor = payload.get("actor")
    if not isinstance(actor, Mapping):
        raise ValueError("Sensor policy artifact is missing the actor state_dict.")
    reference = actor_from_architecture(expected_architecture)
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


def build_sensor_artifact_payload(
    *,
    actor_state_dict: Mapping[str, Any],
    obs_norm: Mapping[str, Any],
    actor_architecture: Mapping[str, Any],
    env_transitions: int,
    actor_obs_dim: int = ACTOR_OBS_DIM,
    critic_obs_dim: int = CRITIC_OBS_DIM,
    action_dim: int = NUM_ACTIONS,
    actor_layout_version: int = ACTOR_LAYOUT_VERSION,
    action_scale: float = 1.0,
    config_version: int = 4,
    policy_format_version: int = SENSOR_POLICY_FORMAT_VERSION,
    longitudinal_mode: str = "force",
    steering_action_mode: str = STEERING_ACTION_MODE,
    steering_delta_max_rad: float = STEERING_DELTA_MAX_RAD,
    f_drive_max: float = 23.0,
    f_brake_max: float = 5.2,
    i_drive_max_a: float = TRAINING_I_DRIVE_MAX_A,
    i_brake_max_a: float = TRAINING_I_BRAKE_MAX_A,
    i_slew_a_per_s: float = TRAINING_I_SLEW_A_PER_S,
    control_hz: float = CONTROL_HZ,
    simulator_id: str = "f1tenth-torch",
    simulator_version: int = 1,
    training_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "actor": dict(actor_state_dict),
        "obs_norm": dict(obs_norm),
        "env_transitions": int(env_transitions),
        "obs_dim": int(actor_obs_dim),
        "actor_obs_dim": int(actor_obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "actor_layout_version": int(actor_layout_version),
        "action_dim": int(action_dim),
        "action_scale": float(action_scale),
        "config_version": int(config_version),
        "policy_format_version": int(policy_format_version),
        "artifact_scope": ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": normalize_actor_architecture(actor_architecture),
        "longitudinal_mode": str(longitudinal_mode),
        "steering_action_mode": str(steering_action_mode),
        "steering_delta_max_rad": float(steering_delta_max_rad),
        "f_drive_max": float(f_drive_max),
        "f_brake_max": float(f_brake_max),
        "i_drive_max_a": float(i_drive_max_a),
        "i_brake_max_a": float(i_brake_max_a),
        "i_slew_a_per_s": float(i_slew_a_per_s),
        "control_hz": float(control_hz),
        "simulator_id": str(simulator_id),
        "simulator_version": int(simulator_version),
        "observation_preprocessing_version": OBS_PREPROCESSING_VERSION,
        "training_protocol": dict(training_protocol or {}),
    }


def load_sensor_artifact(path: str, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Sensor policy artifact payload must be a dict")
    return payload

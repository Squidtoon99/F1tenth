from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from f1tenth_contract.action import CONTROL_HZ
from f1tenth_contract.observation import NUM_ACTIONS, NUM_OBS

POLICY_FORMAT_VERSION = 2
OBS_PREPROCESSING_VERSION = 1


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def validate_policy_artifact(
    payload: Any,
    *,
    expected_obs_dim: int = NUM_OBS,
    expected_action_dim: int = NUM_ACTIONS,
    expected_control_hz: float = CONTROL_HZ,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must be a policy artifact mapping.")

    version = payload.get("policy_format_version")
    if version is None or int(version) < POLICY_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint policy_format_version={version!r} is unsupported; "
            f"need >={POLICY_FORMAT_VERSION}."
        )

    mode = payload.get("longitudinal_mode", payload.get("throttle_mode"))
    if mode != "force":
        raise ValueError(
            f"Checkpoint longitudinal_mode={mode!r} is unsupported; need 'force'."
        )

    obs_dim = payload.get("obs_dim")
    if obs_dim is None or int(obs_dim) != expected_obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim={obs_dim!r}; expected fixed {expected_obs_dim}."
        )

    action_dim = payload.get("action_dim")
    if action_dim is None or int(action_dim) != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={action_dim!r}; expected {expected_action_dim}."
        )

    control_hz = payload.get("control_hz")
    if control_hz is None or abs(float(control_hz) - expected_control_hz) > 1.0e-6:
        raise ValueError(
            f"Checkpoint control_hz={control_hz!r}; expected {expected_control_hz}."
        )

    stats = payload.get("obs_norm")
    if not isinstance(stats, Mapping):
        raise ValueError("Checkpoint is missing obs_norm statistics.")
    for name in ("mean", "var"):
        shape = _shape(stats.get(name))
        if shape != (expected_obs_dim,):
            raise ValueError(
                f"Checkpoint obs_norm.{name} shape={shape}; "
                f"expected ({expected_obs_dim},)."
            )

    actor = payload.get("actor")
    if not isinstance(actor, Mapping):
        raise ValueError("Checkpoint is missing the actor state_dict.")
    actor_input_shape = _shape(actor.get("net.0.weight"))
    if len(actor_input_shape) != 2 or actor_input_shape[1] != expected_obs_dim:
        raise ValueError(
            f"Checkpoint actor input shape={actor_input_shape}; "
            f"expected width {expected_obs_dim}."
        )
    actor_output_shape = _shape(actor.get("mu_layer.weight"))
    if len(actor_output_shape) != 2 or actor_output_shape[0] != expected_action_dim:
        raise ValueError(
            f"Checkpoint actor output shape={actor_output_shape}; "
            f"expected {expected_action_dim} actions."
        )

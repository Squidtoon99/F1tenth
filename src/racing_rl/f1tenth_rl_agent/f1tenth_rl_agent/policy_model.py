"""Deploy policy adapters: classical 392-D MLP + shared sensor GRU actor."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn

from f1tenth_contract import validate_policy_artifact
from f1tenth_policy import (
    CONTROL_HZ,
    ObsNormalizer as SharedObsNormalizer,
    SquashedGaussianLidarGRUActor,
    actor_from_architecture,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.actor import _squashed_gaussian_forward, mlp, normalize_actor_architecture

# Classical symmetric actor (392-D path). Sensor path uses f1tenth_policy.


class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        return _squashed_gaussian_forward(
            self.net,
            self.mu_layer,
            self.log_std_layer,
            self.act_limit,
            obs,
            deterministic,
            with_logprob,
        )


# Sensor aliases over the shared package.
make_sensor_actor = actor_from_architecture
SquashedGaussianLidarGRUActor = SquashedGaussianLidarGRUActor


class ObsNormalizer(SharedObsNormalizer):
    """Deploy normalizer; accepts precomputed mean/var like the prior API."""

    def __init__(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        eps: float,
        clip: float,
        device: torch.device,
    ):
        super().__init__(
            obs_dim=int(torch.as_tensor(mean).numel()),
            device=device,
            eps=eps,
            clip=clip,
        )
        self.mean = torch.as_tensor(mean, device=device, dtype=torch.float32)
        self.var = torch.as_tensor(var, device=device, dtype=torch.float32)


def load_actor(
    checkpoint_path: str,
    obs_dim: int,
    act_dim: int,
    hidden_sizes,
    act_limit: float,
    state_dict_key: str,
    device: torch.device,
) -> SquashedGaussianMLPActor:
    actor = SquashedGaussianMLPActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=list(hidden_sizes),
        activation=nn.ReLU,
        act_limit=act_limit,
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_policy_artifact(
        payload, expected_obs_dim=obs_dim, expected_action_dim=act_dim
    )
    if isinstance(payload, dict) and state_dict_key in payload:
        state_dict = payload[state_dict_key]
    else:
        raise ValueError(
            f"Checkpoint missing '{state_dict_key}' state_dict under the "
            "current/force policy artifact format."
        )
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor


def load_obs_norm(
    checkpoint_path: str,
    obs_dim: int,
    device: torch.device,
    eps: float,
    clip: float,
) -> ObsNormalizer | None:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_policy_artifact(payload, expected_obs_dim=obs_dim)
    stats = payload.get("obs_norm")
    if not isinstance(stats, dict) or "mean" not in stats or "var" not in stats:
        return None
    return ObsNormalizer(
        mean=torch.as_tensor(stats["mean"]),
        var=torch.as_tensor(stats["var"]),
        eps=eps,
        clip=clip,
        device=device,
    )


def _sensor_artifact_for_loader(payload: Mapping) -> dict:
    relaxed = dict(payload)
    if relaxed.get("control_hz") is not None:
        relaxed["control_hz"] = CONTROL_HZ
    return relaxed


def checkpoint_steering_delta_max_rad(payload: Mapping) -> float | None:
    contract = payload.get("action_contract")
    if isinstance(contract, Mapping):
        nested = contract.get("steering_delta_max_rad")
        if nested is not None:
            return float(nested)
    top = payload.get("steering_delta_max_rad")
    if top is not None:
        return float(top)
    return None


def load_sensor_actor(
    checkpoint_path: str,
    state_dict_key: str,
    device: torch.device,
    *,
    expected_steering_action_mode: str = "delta",
    expected_steering_delta_max_rad: float | None = math.pi / 60.0,
) -> SquashedGaussianLidarGRUActor:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(
            "Sensor policy artifact is missing actor_architecture metadata."
        )
    validate_sensor_policy_artifact(
        _sensor_artifact_for_loader(payload),
        expected_architecture=architecture,
        expected_steering_action_mode=expected_steering_action_mode,
        expected_steering_delta_max_rad=expected_steering_delta_max_rad,
    )
    if not isinstance(payload, Mapping) or state_dict_key not in payload:
        raise ValueError(
            f"Checkpoint missing '{state_dict_key}' state_dict under the "
            "sensor policy artifact."
        )
    actor = make_sensor_actor(architecture)
    actor.load_state_dict(payload[state_dict_key])
    actor.to(device)
    actor.eval()
    return actor


def load_sensor_obs_norm(
    checkpoint_path: str,
    device: torch.device,
    eps: float,
    clip: float,
    *,
    expected_steering_action_mode: str = "delta",
    expected_steering_delta_max_rad: float | None = math.pi / 60.0,
) -> ObsNormalizer:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError(
            "Sensor policy artifact is missing actor_architecture metadata."
        )
    validate_sensor_policy_artifact(
        _sensor_artifact_for_loader(payload),
        expected_architecture=architecture,
        expected_steering_action_mode=expected_steering_action_mode,
        expected_steering_delta_max_rad=expected_steering_delta_max_rad,
    )
    stats = payload.get("obs_norm")
    if not isinstance(stats, Mapping) or "mean" not in stats or "var" not in stats:
        raise ValueError("Sensor policy artifact is missing obs_norm statistics.")
    return ObsNormalizer(
        mean=torch.as_tensor(stats["mean"]),
        var=torch.as_tensor(stats["var"]),
        eps=eps,
        clip=clip,
        device=device,
    )


def architectures_match(left, right) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    return normalize_actor_architecture(left) == normalize_actor_architecture(right)


GIGAFLOW_ARCHITECTURE_NAME = "lidar_cnn_gru_conditioned"


def peek_actor_architecture_name(payload: Mapping) -> str:
    arch = payload.get("actor_architecture")
    if isinstance(arch, Mapping):
        return str(arch.get("name", ""))
    return ""


def load_checkpoint_payload(checkpoint_path: str, device: torch.device) -> Mapping:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def _obs_norm_from_welford(
    stats: Mapping,
    device: torch.device,
    eps: float,
    clip: float,
) -> ObsNormalizer:
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    if "var" in stats:
        var = torch.as_tensor(stats["var"], dtype=torch.float32)
    else:
        m2 = torch.as_tensor(stats["m2"], dtype=torch.float32)
        count = float(torch.as_tensor(stats["count"]).reshape(-1)[0])
        var = m2 / max(count, 1.0)
    return ObsNormalizer(mean=mean, var=var, eps=eps, clip=clip, device=device)


def load_gigaflow_actor(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, Mapping]:
    """Load a 1,097-D conditioned Gigaflow actor and frozen IRL condition.

    Accepts ``actor_state_dict`` with or without a ``baked_condition`` buffer.
    If top-level ``deployment_condition_normalized`` is absent, that buffer is
    used as the frozen condition.
    """
    from gigaflow_f1tenth.artifacts import load_irl_actor

    return load_irl_actor(checkpoint_path, device=device)


def load_folded_sensor_actor(
    checkpoint_path: str,
    device: torch.device,
    *,
    eps: float = 1e-8,
    clip: float = 10.0,
) -> tuple[SquashedGaussianLidarGRUActor, ObsNormalizer | None, Mapping]:
    """Load a 272-D GRU-in ``lidar_cnn_gru`` artifact (folded Gigaflow or Sophy).

    Skips this branch's 10 Hz format-4 control_hz pin so a 10 Hz or 20 Hz
    folded checkpoint still loads on a 20 Hz car node.
    """
    payload = load_checkpoint_payload(checkpoint_path, device)
    name = peek_actor_architecture_name(payload)
    if name and name != "lidar_cnn_gru":
        raise ValueError(
            f"folded sensor loader expected lidar_cnn_gru, got {name!r}"
        )
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("folded sensor artifact missing actor_architecture")
    state = payload.get("actor")
    if not isinstance(state, Mapping):
        state = payload.get("actor_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("folded sensor artifact missing actor weights")
    actor = make_sensor_actor(architecture)
    actor.load_state_dict(state)
    actor.to(device)
    actor.eval()
    obs_norm = None
    stats = payload.get("obs_norm")
    if isinstance(stats, Mapping) and "mean" in stats:
        obs_norm = _obs_norm_from_welford(stats, device, eps, clip)
    else:
        welford = payload.get("sensor_normalizer")
        if isinstance(welford, Mapping) and "mean" in welford:
            obs_norm = _obs_norm_from_welford(welford, device, eps, clip)
    return actor, obs_norm, payload

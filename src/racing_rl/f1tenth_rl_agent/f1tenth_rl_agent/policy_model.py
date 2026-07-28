"""Deploy policy adapters: classical 392-D MLP + shared sensor GRU actor."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn

from f1tenth_contract import validate_policy_artifact
from f1tenth_policy import (
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
        payload,
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
        payload,
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

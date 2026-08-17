"""Shared policy layout and GRU actor smoke tests (real torch modules)."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from f1tenth_policy import (
    ACTOR_OBS_DIM,
    LIDAR_DIM,
    PROPRIO_DIM,
    make_actor,
    steer_history_angles_and_deltas,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.artifact import build_sensor_artifact_payload
from f1tenth_policy.normalizer import ObsNormalizer


def test_layout_dimensions():
    assert ACTOR_OBS_DIM == LIDAR_DIM + PROPRIO_DIM == 1097


def test_steer_history_deltas():
    angles, deltas = steer_history_angles_and_deltas((0.30, 0.20, 0.10, 0.0))
    assert angles == (0.30, 0.20, 0.10)
    assert all(abs(d - 0.10) < 1e-12 for d in deltas)


def test_gru_actor_step_and_sequence():
    torch.manual_seed(0)
    actor = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    obs = torch.randn(4, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(4, device=obs.device, dtype=obs.dtype)
    action, logp, h1 = actor.step(obs, hidden, deterministic=True)
    assert action.shape == (4, 2)
    assert logp.shape == (4,)
    assert h1.shape == (4, actor.gru_hidden_dim)

    seq = obs.unsqueeze(1).expand(4, 3, ACTOR_OBS_DIM).contiguous()
    actions, logps, h2 = actor.forward_sequence(seq, hidden, deterministic=True)
    assert actions.shape == (4, 3, 2)
    assert logps.shape == (4, 3)
    assert h2.shape == (4, actor.gru_hidden_dim)


def test_artifact_roundtrip_validation():
    torch.manual_seed(1)
    actor = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(torch.randn(32, ACTOR_OBS_DIM))
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=1000,
        f_brake_max=5.2,
    )
    validate_sensor_policy_artifact(
        payload,
        expected_architecture=actor.actor_architecture,
    )
    assert payload["i_drive_max_a"] == 80.0
    assert payload["i_brake_max_a"] == 20.0
    assert payload["i_slew_a_per_s"] == 200.0


def test_artifact_validation_rejects_missing_or_invalid_current_slew():
    torch.manual_seed(2)
    actor = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=1000,
    )

    missing = dict(payload)
    missing.pop("i_slew_a_per_s")
    with pytest.raises(ValueError, match="i_slew_a_per_s"):
        validate_sensor_policy_artifact(
            missing,
            expected_architecture=actor.actor_architecture,
        )

    invalid = dict(payload)
    invalid["i_slew_a_per_s"] = 0.0
    with pytest.raises(ValueError, match="i_slew_a_per_s"):
        validate_sensor_policy_artifact(
            invalid,
            expected_architecture=actor.actor_architecture,
        )

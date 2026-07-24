"""Deterministic fixed-champion selection (real artifact validation)."""

from __future__ import annotations

import torch
import torch.nn as nn

from f1tenth_policy import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    STEERING_ACTION_MODE,
    STEERING_DELTA_MAX_RAD,
    build_sensor_artifact_payload,
    make_actor,
)
from f1tenth_policy.normalizer import ObsNormalizer
from fixed_opponents import FixedChampionManager, select_champion


def _write_artifact(path, seed: int = 0):
    torch.manual_seed(seed)
    actor = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(torch.randn(16, ACTOR_OBS_DIM))
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=12345 + seed,
    )
    torch.save(payload, path)
    return actor.actor_architecture


def test_select_champion_is_deterministic(tmp_path):
    arch = None
    entries = []
    for i in range(3):
        p = tmp_path / f"champ_{i}.pt"
        arch = _write_artifact(p, seed=i)
        entries.append({"checkpoint": str(p), "weight": float(i + 1)})

    a = select_champion(
        entries,
        seed=42,
        expected_architecture=arch,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=2,
        expected_critic_obs_dim=CRITIC_OBS_DIM,
        expected_steering_action_mode=STEERING_ACTION_MODE,
        expected_steering_delta_max_rad=STEERING_DELTA_MAX_RAD,
    )
    b = select_champion(
        entries,
        seed=42,
        expected_architecture=arch,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=2,
        expected_critic_obs_dim=CRITIC_OBS_DIM,
        expected_steering_action_mode=STEERING_ACTION_MODE,
        expected_steering_delta_max_rad=STEERING_DELTA_MAX_RAD,
    )
    assert a.checkpoint == b.checkpoint
    assert a.transitions == b.transitions

    c = select_champion(
        entries,
        seed=99,
        expected_architecture=arch,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=2,
        expected_critic_obs_dim=CRITIC_OBS_DIM,
        expected_steering_action_mode=STEERING_ACTION_MODE,
        expected_steering_delta_max_rad=STEERING_DELTA_MAX_RAD,
    )
    # Different seed may pick a different entry; metadata stays immutable.
    mgr = FixedChampionManager(a)
    meta = mgr.metadata()
    assert meta["checkpoint"] == a.checkpoint
    assert meta["transitions"] == a.transitions
    assert c.checkpoint in {e["checkpoint"] for e in entries}


def test_select_champion_allows_learner_steering_delta_mismatch(tmp_path, caplog):
    path = tmp_path / "champ.pt"
    arch = _write_artifact(path, seed=0)
    import logging

    caplog.set_level(logging.WARNING, logger="fixed_opponents")
    selection = select_champion(
        [{"checkpoint": str(path), "weight": 1.0}],
        seed=42,
        expected_architecture=arch,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=2,
        expected_critic_obs_dim=CRITIC_OBS_DIM,
        expected_steering_action_mode=STEERING_ACTION_MODE,
        expected_steering_delta_max_rad=STEERING_DELTA_MAX_RAD * 2.0,
    )
    assert selection.checkpoint == str(path)
    assert any("differs from learner" in r.message for r in caplog.records)

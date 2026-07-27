"""Grouped opponent-pool inference keeps per-env GRU state aligned."""

from __future__ import annotations

import torch
import torch.nn as nn

from f1tenth_env.opponents import PolicyOpponent
from f1tenth_policy import (
    ACTOR_OBS_DIM,
    build_sensor_artifact_payload,
    make_actor,
)
from f1tenth_policy.normalizer import ObsNormalizer
from fixed_opponents import ChampionEntry


def _artifact(entry_seed: int):
    torch.manual_seed(entry_seed)
    actor = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    actor.mu_layer.bias.data.fill_(float(entry_seed))
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(torch.randn(16, ACTOR_OBS_DIM))
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=1000 + entry_seed,
    )
    return ChampionEntry(
        checkpoint=f"/tmp/champ_{entry_seed}.pt",
        weight=1.0,
        transitions=1000 + entry_seed,
        actor={k: v.detach().cpu().clone() for k, v in payload["actor"].items()},
        mean=payload["obs_norm"]["mean"].detach().cpu().clone(),
        var=payload["obs_norm"]["var"].detach().cpu().clone(),
        actor_architecture=dict(payload["actor_architecture"]),
    )


def _make_pool_opponent(entries: list[ChampionEntry]) -> PolicyOpponent:
    template = make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU)
    opp = PolicyOpponent(template, torch.device("cpu"))
    opp.load_opponent_pool(entries)
    return opp


def test_pooled_inference_matches_single_policy_per_env():
    entries = [_artifact(0), _artifact(1)]
    pooled = _make_pool_opponent(entries)
    singles = [
        PolicyOpponent(
            make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU),
            torch.device("cpu"),
        )
        for _ in entries
    ]
    for single, entry in zip(singles, entries):
        single.load_snapshot(entry.actor, entry.mean, entry.var, entry.actor_architecture)

    obs = torch.randn(4, ACTOR_OBS_DIM)
    mask = torch.tensor([True, True, True, True])
    pooled.assign_policies(mask, torch.tensor([0, 1, 0, 1]))

    pooled_actions = pooled.act_observation(obs)
    expected = torch.empty_like(pooled_actions)
    for env_id, policy_idx in enumerate([0, 1, 0, 1]):
        expected[env_id] = singles[policy_idx].act_observation(obs[env_id : env_id + 1])[0]
        singles[policy_idx]._hidden.zero_()

    pooled._hidden.zero_()
    pooled.assign_policies(mask, torch.tensor([0, 1, 0, 1]))
    pooled_actions = pooled.act_observation(obs)
    assert torch.allclose(pooled_actions, expected, atol=1.0e-5)


def test_reset_clears_hidden_and_reassignment_changes_output():
    entries = [_artifact(0), _artifact(1)]
    pooled = _make_pool_opponent(entries)
    obs = torch.randn(2, ACTOR_OBS_DIM)
    mask = torch.ones(2, dtype=torch.bool)
    pooled.assign_policies(mask, torch.tensor([0, 0]))
    pooled.act_observation(obs)

    reset_mask = torch.tensor([True, False])
    pooled.reset(reset_mask)
    pooled.assign_policies(reset_mask, torch.tensor([1]))
    assert float(pooled._hidden[0].abs().sum()) == 0.0
    assert float(pooled._hidden[1].abs().sum()) > 0.0

    cold_policy1 = PolicyOpponent(
        make_actor(ACTOR_OBS_DIM, 2, [64, 64], activation=nn.ReLU),
        torch.device("cpu"),
    )
    cold_policy1.load_snapshot(
        entries[1].actor,
        entries[1].mean,
        entries[1].var,
        entries[1].actor_architecture,
    )
    expected_env0 = cold_policy1.act_observation(obs[0:1])[0]
    after_reset = pooled.act_observation(obs)
    assert torch.allclose(after_reset[0], expected_env0, atol=1.0e-5)

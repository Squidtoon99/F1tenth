from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from qrsac import Models, QuantileCritic, SquashedGaussianMLPActor
from standalone_trainer import (
    NStepReplayBuffer,
    ObsNormalizer,
    interval_crossed,
    learner_updates_for_transitions,
    save_policy_artifact,
)


def _models() -> Models:
    actor = SquashedGaussianMLPActor(4, 2, [8], nn.ReLU, 1.0)
    critic = QuantileCritic(4, 2, [8], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def test_replay_flushes_short_terminal_episode():
    replay = NStepReplayBuffer(
        capacity=16,
        obs_dim=2,
        act_dim=1,
        n_step=3,
        gamma=0.5,
        num_envs=1,
        device=torch.device("cpu"),
    )
    assert replay.add(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([1.0]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([False]),
    ) == 0
    assert replay.add(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.5]]),
        torch.tensor([2.0]),
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([True]),
    ) == 2
    assert replay.size == 2
    assert torch.allclose(replay.reward[:2], torch.tensor([2.0, 2.0]))
    assert torch.equal(replay.done[:2], torch.ones(2))
    assert torch.equal(replay.next_obs[:2], torch.tensor([[2.0, 0.0], [2.0, 0.0]]))


def test_observation_normalizer_matches_population_statistics():
    normalizer = ObsNormalizer(2, torch.device("cpu"), eps=0.0)
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    normalizer.update(values[:2])
    normalizer.update(values[2:])
    assert torch.allclose(normalizer.mean, values.mean(dim=0))
    assert torch.allclose(normalizer.var, values.var(dim=0, unbiased=False))
    assert torch.isfinite(normalizer.normalize(values)).all()


def test_update_budget_is_independent_of_vector_width():
    def collect(width: int) -> tuple[int, float]:
        updates = 0
        budget = 0.0
        for _ in range(4096 // width):
            due, budget = learner_updates_for_transitions(width, 1024, 2.0, budget)
            updates += due
        return updates, budget

    assert collect(64) == collect(512) == (8, 0.0)
    assert not interval_crossed(499, 500, 1000)
    assert interval_crossed(999, 1001, 1000)


def test_transition_cadences_are_independent_of_vector_width():
    def cadence(width: int) -> tuple[int, list[int]]:
        transitions = 0
        warmup_at = 0
        events = []
        while transitions < 4096:
            previous = transitions
            transitions += width
            if warmup_at == 0 and transitions >= 2048:
                warmup_at = transitions
            if interval_crossed(previous, transitions, 1024):
                events.append(transitions)
        return warmup_at, events

    expected = (2048, [1024, 2048, 3072, 4096])
    assert cadence(64) == cadence(512) == expected


def test_compact_policy_artifact_round_trip(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["obs"]["num_obs"] = 4
    cfg["env"]["num_actions"] = 2
    models = _models()
    normalizer = ObsNormalizer(4, torch.device("cpu"))
    normalizer.update(torch.arange(12, dtype=torch.float32).reshape(3, 4))

    path = save_policy_artifact(models, 1234, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "actor" in payload
    assert "obs_norm" in payload
    assert "critic1" not in payload
    assert payload["env_transitions"] == 1234
    assert payload["obs_dim"] == 4
    assert payload["action_dim"] == 2
    assert payload["config_version"] == cfg["config_version"]
    restored = _models()
    restored.actor.load_state_dict(payload["actor"])
    obs = torch.randn(3, 4)
    with torch.no_grad():
        expected, _ = models.actor(obs, deterministic=True, with_logprob=False)
        actual, _ = restored.actor(obs, deterministic=True, with_logprob=False)
    assert torch.equal(expected, actual)

    deploy_module = (
        Path(__file__).resolve().parents[2]
        / "src/racing_rl/f1tenth_rl_agent/f1tenth_rl_agent"
    )
    sys.path.insert(0, str(deploy_module))
    try:
        from policy_model import load_actor, load_obs_norm

        deployed = load_actor(
            str(path),
            obs_dim=4,
            act_dim=2,
            hidden_sizes=[8],
            act_limit=1.0,
            state_dict_key="actor",
            device=torch.device("cpu"),
        )
        deployed_norm = load_obs_norm(
            str(path), obs_dim=4, device=torch.device("cpu"), eps=1e-8, clip=10.0
        )
    finally:
        sys.path.remove(str(deploy_module))
    assert deployed_norm is not None
    with torch.no_grad():
        deployed_action, _ = deployed(
            deployed_norm.normalize(obs),
            deterministic=True,
            with_logprob=False,
        )
        expected_action, _ = models.actor(
            deployed_norm.normalize(obs),
            deterministic=True,
            with_logprob=False,
        )
    assert torch.equal(deployed_action, expected_action)

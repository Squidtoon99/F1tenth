from __future__ import annotations

import copy
import logging

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from qrsac import Models, QuantileCritic, SquashedGaussianMLPActor
from standalone_trainer import (
    REPLAY_CAPACITY_FALLBACK,
    REPLAY_CAPACITY_REQUESTED,
    REPLAY_OBS_DTYPE,
    NStepReplayBuffer,
    ObsNormalizer,
    estimate_dual_replay_bytes,
    interval_crossed,
    learner_updates_for_transitions,
    make_dual_replay_buffer,
    save_policy_artifact,
    training_should_continue,
    unpack_sensor_observations,
    validate_sensor_policy_artifact,
)

ACTOR_DIM = 4
CRITIC_DIM = 6


def _models(actor_dim: int = ACTOR_DIM, critic_dim: int = CRITIC_DIM) -> Models:
    actor = SquashedGaussianMLPActor(actor_dim, 2, [8], nn.ReLU, 1.0)
    critic = QuantileCritic(critic_dim, 2, [8], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def test_dual_replay_flushes_short_terminal_episode():
    replay = NStepReplayBuffer(
        capacity=16,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=1,
        n_step=3,
        gamma=0.5,
        num_envs=1,
        device=torch.device("cpu"),
    )
    assert replay.add(
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([1.0]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
        torch.tensor([False]),
    ) == 0
    assert replay.add(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
        torch.tensor([[0.5]]),
        torch.tensor([2.0]),
        torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0]]),
        torch.tensor([True]),
    ) == 2
    assert replay.size == 2
    assert torch.allclose(replay.reward[:2], torch.tensor([2.0, 2.0]))
    assert torch.equal(replay.done[:2], torch.ones(2))
    assert torch.equal(
        replay.next_actor_obs[:2].to(torch.float32),
        torch.tensor([[2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        replay.next_critic_obs[:2].to(torch.float32),
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]]
        ),
    )


def _reference_n_step_emit(
    steps: list[tuple[torch.Tensor, ...]],
    *,
    n_step: int,
    gamma: float,
    num_envs: int,
) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    """Independent per-env deque reference for n-step emit parity tests.

    Candidate order matches the vectorized buffer: all block-A rows (env-major),
    then all block-B truncated tails (env-major, offset-major).
    """
    from collections import deque

    windows = [deque(maxlen=n_step) for _ in range(num_envs)]
    emitted: list[dict[str, torch.Tensor]] = []
    counts: list[int] = []
    for actor, critic, act, rew, nxt_a, nxt_c, done in steps:
        block_a: list[dict[str, torch.Tensor]] = []
        block_b: list[dict[str, torch.Tensor]] = []
        for e in range(num_envs):
            windows[e].append(
                (
                    actor[e].clone(),
                    critic[e].clone(),
                    act[e].clone(),
                    float(rew[e]),
                )
            )
        for e in range(num_envs):
            done_e = bool(done[e])
            if (not done_e) and len(windows[e]) == n_step:
                ret = 0.0
                for k, (_, _, _, r) in enumerate(windows[e]):
                    ret += (gamma**k) * r
                a0, c0, u0, _ = windows[e][0]
                block_a.append(
                    {
                        "actor_obs": a0,
                        "critic_obs": c0,
                        "action": u0,
                        "reward": torch.tensor(ret),
                        "next_actor_obs": nxt_a[e].clone(),
                        "next_critic_obs": nxt_c[e].clone(),
                        "done": torch.tensor(0.0),
                    }
                )
        for e in range(num_envs):
            if not bool(done[e]):
                continue
            L = len(windows[e])
            for k in range(L):
                ret = 0.0
                for j in range(L - k):
                    ret += (gamma**j) * windows[e][k + j][3]
                a_k, c_k, u_k, _ = windows[e][k]
                block_b.append(
                    {
                        "actor_obs": a_k,
                        "critic_obs": c_k,
                        "action": u_k,
                        "reward": torch.tensor(ret),
                        "next_actor_obs": nxt_a[e].clone(),
                        "next_critic_obs": nxt_c[e].clone(),
                        "done": torch.tensor(1.0),
                    }
                )
            windows[e].clear()
        step_rows = block_a + block_b
        emitted.extend(step_rows)
        counts.append(len(step_rows))
    return emitted, counts


def test_dual_replay_emit_matches_reference_counts_and_alignment():
    """Slim two-pass add must match classic n-step emit order, rewards, dones."""
    torch.manual_seed(0)
    n_step = 3
    gamma = 0.5
    num_envs = 3
    replay = NStepReplayBuffer(
        capacity=64,
        actor_obs_dim=2,
        critic_obs_dim=3,
        act_dim=1,
        n_step=n_step,
        gamma=gamma,
        num_envs=num_envs,
        device=torch.device("cpu"),
        obs_dtype=torch.float32,
    )
    steps: list[tuple[torch.Tensor, ...]] = []
    for t in range(6):
        actor = torch.tensor(
            [[float(t), 10.0 + e] for e in range(num_envs)], dtype=torch.float32
        )
        critic = torch.tensor(
            [[float(t), 100.0 + e, 1.0] for e in range(num_envs)],
            dtype=torch.float32,
        )
        act = torch.tensor([[0.1 * t] for _ in range(num_envs)], dtype=torch.float32)
        rew = torch.tensor(
            [1.0 + 0.25 * e + 0.1 * t for e in range(num_envs)], dtype=torch.float32
        )
        nxt_a = actor + 1.0
        nxt_c = critic + 1.0
        done = torch.tensor(
            [t == 2 and e == 0 or t == 4 and e == 2 for e in range(num_envs)]
        )
        steps.append((actor, critic, act, rew, nxt_a, nxt_c, done))
        n_emit = replay.add(actor, critic, act, rew, nxt_a, nxt_c, done)
        assert int(n_emit) >= 0

    ref_rows, ref_counts = _reference_n_step_emit(
        steps, n_step=n_step, gamma=gamma, num_envs=num_envs
    )
    assert int(replay.size) == len(ref_rows) == sum(ref_counts)
    for i, row in enumerate(ref_rows):
        assert torch.allclose(replay.actor_obs[i], row["actor_obs"])
        assert torch.allclose(replay.critic_obs[i], row["critic_obs"])
        assert torch.allclose(replay.action[i], row["action"])
        assert torch.allclose(replay.reward[i], row["reward"])
        assert torch.allclose(replay.next_actor_obs[i], row["next_actor_obs"])
        assert torch.allclose(replay.next_critic_obs[i], row["next_critic_obs"])
        assert torch.allclose(replay.done[i], row["done"])
        # Actor/critic streams share the same transition marker (first feature).
        assert float(replay.actor_obs[i, 0]) == float(replay.critic_obs[i, 0])


def test_dual_replay_full_window_nonterminal_emit_count():
    replay = NStepReplayBuffer(
        capacity=32,
        actor_obs_dim=2,
        critic_obs_dim=3,
        act_dim=1,
        n_step=3,
        gamma=1.0,
        num_envs=2,
        device=torch.device("cpu"),
        obs_dtype=torch.float32,
    )
    total = 0
    for t in range(5):
        n_emit = replay.add(
            torch.full((2, 2), float(t)),
            torch.full((2, 3), float(t)),
            torch.zeros(2, 1),
            torch.ones(2),
            torch.full((2, 2), float(t + 1)),
            torch.full((2, 3), float(t + 1)),
            torch.zeros(2, dtype=torch.bool),
        )
        total += int(n_emit)
        if t < 2:
            assert int(n_emit) == 0
        else:
            # Both envs emit one completed n-step row once the window is full.
            assert int(n_emit) == 2
    assert total == 6
    assert int(replay.size) == 6
    assert torch.equal(replay.done[:6], torch.zeros(6))
    # gamma=1, three unit rewards → n-step return 3.
    assert torch.allclose(replay.reward[:6], torch.full((6,), 3.0))


def test_dual_replay_stores_float16_and_samples_float32():
    replay = NStepReplayBuffer(
        capacity=8,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=2,
        n_step=1,
        gamma=0.99,
        num_envs=1,
        device=torch.device("cpu"),
    )
    assert replay.actor_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.critic_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.next_actor_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.next_critic_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.action.dtype == torch.float32

    actor = torch.randn(1, ACTOR_DIM)
    critic = torch.randn(1, CRITIC_DIM)
    next_actor = torch.randn(1, ACTOR_DIM)
    next_critic = torch.randn(1, CRITIC_DIM)
    replay.add(
        actor,
        critic,
        torch.zeros(1, 2),
        torch.tensor([1.0]),
        next_actor,
        next_critic,
        torch.tensor([True]),
    )
    batch = replay.sample(1)
    assert batch["actor_obs"].dtype == torch.float32
    assert batch["critic_obs"].dtype == torch.float32
    assert batch["next_actor_obs"].dtype == torch.float32
    assert batch["next_critic_obs"].dtype == torch.float32
    assert batch["actor_obs"].shape[-1] == ACTOR_DIM
    assert batch["critic_obs"].shape[-1] == CRITIC_DIM


def test_dual_replay_keeps_actor_critic_streams_aligned():
    replay = NStepReplayBuffer(
        capacity=32,
        actor_obs_dim=2,
        critic_obs_dim=3,
        act_dim=1,
        n_step=2,
        gamma=1.0,
        num_envs=2,
        device=torch.device("cpu"),
    )
    for step in range(3):
        actor = torch.tensor([[float(step), 10.0], [float(step), 20.0]])
        critic = torch.tensor(
            [[float(step), 100.0, 1.0], [float(step), 200.0, 2.0]]
        )
        next_actor = actor + 1.0
        next_critic = critic + 1.0
        done = torch.tensor([False, step == 2])
        replay.add(
            actor,
            critic,
            torch.zeros(2, 1),
            torch.ones(2),
            next_actor,
            next_critic,
            done,
        )
    assert int(replay.size) >= 1
    batch = replay.sample(min(4, int(replay.size)))
    # Same transition index: actor marker equals critic marker.
    assert torch.equal(batch["actor_obs"][:, 0], batch["critic_obs"][:, 0])
    assert torch.equal(
        batch["next_actor_obs"][:, 0], batch["next_critic_obs"][:, 0]
    )


def test_equal_dimension_dual_replay_is_rejected():
    with pytest.raises(ValueError, match="distinct actor/critic"):
        NStepReplayBuffer(
            capacity=4,
            actor_obs_dim=5,
            critic_obs_dim=5,
            act_dim=2,
            n_step=1,
            gamma=0.99,
            num_envs=1,
            device=torch.device("cpu"),
        )


def test_separate_actor_critic_normalizers():
    actor_norm = ObsNormalizer(ACTOR_DIM, torch.device("cpu"), eps=0.0)
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"), eps=0.0)
    actor_vals = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    critic_vals = torch.arange(12, dtype=torch.float32).reshape(2, CRITIC_DIM)
    actor_norm.update(actor_vals)
    critic_norm.update(critic_vals)
    assert torch.allclose(actor_norm.mean, actor_vals.mean(dim=0))
    assert torch.allclose(critic_norm.mean, critic_vals.mean(dim=0))
    assert actor_norm.mean.numel() != critic_norm.mean.numel()
    assert torch.isfinite(actor_norm.normalize(actor_vals)).all()
    assert torch.isfinite(critic_norm.normalize(critic_vals)).all()


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


def test_sensor_policy_artifact_round_trip(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["obs"]["num_actor_obs"] = ACTOR_DIM
    cfg["obs"]["num_obs"] = CRITIC_DIM
    cfg["obs"]["actor_layout_version"] = 1
    cfg["env"]["num_actions"] = 2
    cfg["model"]["hidden_layers"] = [8]
    models = _models()
    normalizer = ObsNormalizer(ACTOR_DIM, torch.device("cpu"))
    normalizer.update(torch.arange(12, dtype=torch.float32).reshape(3, ACTOR_DIM))

    path = save_policy_artifact(models, 1234, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "actor" in payload
    assert "obs_norm" in payload
    assert "critic1" not in payload
    assert "critic_norm" not in payload
    assert payload["env_transitions"] == 1234
    assert payload["obs_dim"] == ACTOR_DIM
    assert payload["actor_obs_dim"] == ACTOR_DIM
    assert payload["critic_obs_dim"] == CRITIC_DIM
    assert payload["actor_layout_version"] == 1
    assert payload["action_dim"] == 2
    assert payload["config_version"] == cfg["config_version"]
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=ACTOR_DIM,
        expected_action_dim=2,
        expected_layout_version=1,
        expected_architecture=payload["actor_architecture"],
    )
    restored = _models()
    restored.actor.load_state_dict(payload["actor"])
    obs = torch.randn(3, ACTOR_DIM)
    with torch.no_grad():
        expected, _ = models.actor(obs, deterministic=True, with_logprob=False)
        actual, _ = restored.actor(obs, deterministic=True, with_logprob=False)
    assert torch.equal(expected, actual)


def test_stale_privileged_artifact_schema_is_rejected(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["obs"]["num_actor_obs"] = ACTOR_DIM
    cfg["obs"]["num_obs"] = CRITIC_DIM
    cfg["env"]["num_actions"] = 2
    cfg["model"]["hidden_layers"] = [8]
    models = _models()
    normalizer = ObsNormalizer(ACTOR_DIM, torch.device("cpu"))
    path = save_policy_artifact(models, 1, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["obs_dim"] = 390
    payload["actor_obs_dim"] = 390
    del payload["actor_layout_version"]
    with pytest.raises(ValueError, match="Privileged/symmetric"):
        validate_sensor_policy_artifact(
            payload,
            expected_actor_obs_dim=ACTOR_DIM,
            expected_action_dim=2,
            expected_layout_version=1,
            expected_architecture=payload["actor_architecture"],
        )


def test_replay_memory_estimate_and_fallback_constants():
    est_2m = estimate_dual_replay_bytes(
        REPLAY_CAPACITY_REQUESTED,
        actor_obs_dim=1093,
        critic_obs_dim=390,
        act_dim=2,
        n_step=7,
        num_envs=512,
    )
    est_1m = estimate_dual_replay_bytes(
        REPLAY_CAPACITY_FALLBACK,
        actor_obs_dim=1093,
        critic_obs_dim=390,
        act_dim=2,
        n_step=7,
        num_envs=512,
    )
    assert est_2m["capacity"] == REPLAY_CAPACITY_REQUESTED == 2_000_000
    assert est_1m["capacity"] == REPLAY_CAPACITY_FALLBACK == 1_000_000
    assert est_2m["total_bytes"] > est_1m["total_bytes"] > 0
    assert est_2m["ring_obs_bytes"] == 2 * est_2m["rows"] * (1093 + 390) * 2

    log = logging.getLogger("test_replay_alloc")
    buffer, selected, estimate = make_dual_replay_buffer(
        capacity=64,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=2,
        n_step=2,
        gamma=0.99,
        num_envs=2,
        device=torch.device("cpu"),
        batch_size=4,
        log=log,
        allow_fallback=True,
    )
    assert selected == 64
    assert estimate["capacity"] == 64
    assert buffer.capacity == 64
    assert buffer.actor_obs.dtype == REPLAY_OBS_DTYPE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_2m_float16_dual_replay_and_compiled_asymmetric_update():
    """Real 2M allocation + compiled QR-SAC update; 1M only on OOM/headroom fail."""
    from qrsac import QRSACTrainer
    from standalone_trainer import build_models

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    log = logging.getLogger("test_cuda_2m_replay")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    actor_dim = int(cfg["obs"]["num_actor_obs"])
    critic_dim = int(cfg["obs"]["num_obs"])
    act_dim = int(cfg["env"]["num_actions"])
    batch_size = int(cfg["model"]["batch_size"])
    n_step = int(cfg["model"]["n_step"])
    num_envs = 512

    buffer, selected, estimate = make_dual_replay_buffer(
        capacity=REPLAY_CAPACITY_REQUESTED,
        actor_obs_dim=actor_dim,
        critic_obs_dim=critic_dim,
        act_dim=act_dim,
        n_step=n_step,
        gamma=float(cfg["model"]["rew_gamma"]),
        num_envs=num_envs,
        device=device,
        batch_size=batch_size,
        log=log,
        allow_fallback=True,
    )
    assert selected in (REPLAY_CAPACITY_REQUESTED, REPLAY_CAPACITY_FALLBACK)
    assert buffer.actor_obs.dtype == REPLAY_OBS_DTYPE
    assert buffer.critic_obs.dtype == REPLAY_OBS_DTYPE
    assert buffer.actor_obs.shape[-1] == actor_dim
    assert buffer.critic_obs.shape[-1] == critic_dim
    assert estimate["capacity"] == selected

    # Fill enough rows for a real sample, then run a compiled asymmetric update.
    for _ in range(n_step + 2):
        buffer.add(
            torch.randn(num_envs, actor_dim, device=device),
            torch.randn(num_envs, critic_dim, device=device),
            torch.randn(num_envs, act_dim, device=device),
            torch.randn(num_envs, device=device),
            torch.randn(num_envs, actor_dim, device=device),
            torch.randn(num_envs, critic_dim, device=device),
            torch.zeros(num_envs, dtype=torch.bool, device=device),
        )
    assert buffer.is_ready(batch_size)
    batch = buffer.sample(batch_size)
    models, trainer = build_models(cfg, device, compile=True, compile_mode="default")
    assert isinstance(trainer, QRSACTrainer)
    assert trainer.actor_obs_dim == actor_dim
    assert trainer.critic_obs_dim == critic_dim
    losses = trainer.update(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    # Keep the selected capacity visible in failure messages / CI logs.
    assert selected == REPLAY_CAPACITY_REQUESTED or selected == REPLAY_CAPACITY_FALLBACK
    del buffer, models, trainer, batch
    torch.cuda.empty_cache()


def test_finite_versus_continuous_termination():
    assert training_should_continue(0, 100, continuous=False)
    assert training_should_continue(99, 100, continuous=False)
    assert not training_should_continue(100, 100, continuous=False)
    assert not training_should_continue(101, 100, continuous=False)
    assert training_should_continue(0, 100, continuous=True)
    assert training_should_continue(100, 100, continuous=True)
    assert training_should_continue(10**12, 100, continuous=True)


def test_unpack_sensor_observations_requires_dict():
    actor = torch.randn(2, ACTOR_DIM)
    critic = torch.randn(2, CRITIC_DIM)
    a, c = unpack_sensor_observations({"actor": actor, "frenet": critic})
    assert torch.equal(a, actor.to(torch.float32))
    assert torch.equal(c, critic.to(torch.float32))
    with pytest.raises(TypeError, match="with_sensors=True"):
        unpack_sensor_observations(actor)
    with pytest.raises(KeyError, match="actor"):
        unpack_sensor_observations({"frenet": critic})


def test_collection_to_asymmetric_update_pipeline():
    """Sensor dict → dual replay → separate norms → QR-SAC update."""
    from qrsac import QRSACTrainer

    torch.manual_seed(0)
    models = _models()
    trainer = QRSACTrainer(models, torch.device("cpu"), n_step=1)
    replay = NStepReplayBuffer(
        capacity=16,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=2,
        n_step=1,
        gamma=0.99,
        num_envs=2,
        device=torch.device("cpu"),
    )
    actor_norm = ObsNormalizer(ACTOR_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))

    for _ in range(3):
        actor_obs, critic_obs = unpack_sensor_observations(
            {
                "actor": torch.randn(2, ACTOR_DIM),
                "frenet": torch.randn(2, CRITIC_DIM),
            }
        )
        next_actor, next_critic = unpack_sensor_observations(
            {
                "actor": torch.randn(2, ACTOR_DIM),
                "frenet": torch.randn(2, CRITIC_DIM),
            }
        )
        actor_norm.update(actor_obs)
        critic_norm.update(critic_obs)
        replay.add(
            actor_obs,
            critic_obs,
            torch.zeros(2, 2),
            torch.ones(2),
            next_actor,
            next_critic,
            torch.tensor([False, True]),
        )
        actor_norm.update(next_actor)
        critic_norm.update(next_critic)

    assert replay.is_ready(4)
    batch = replay.sample(4)
    batch["actor_obs"] = actor_norm.normalize(batch["actor_obs"])
    batch["next_actor_obs"] = actor_norm.normalize(batch["next_actor_obs"])
    batch["critic_obs"] = critic_norm.normalize(batch["critic_obs"])
    batch["next_critic_obs"] = critic_norm.normalize(batch["next_critic_obs"])
    losses = trainer.update(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)

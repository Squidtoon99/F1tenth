from __future__ import annotations

import copy
import logging
import math

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_policy.layout import (
    ACTOR_LAYOUT_VERSION,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
)
from qrsac import Models, QuantileCritic, make_actor
from qrsac.replay import TrajectoryReplayBuffer
from qrsac.spinningup.core import GRU_HIDDEN_DIM, mlp, _squashed_gaussian_forward
from standalone_trainer import (
    OPP_OBS_BASE_IDX,
    OPP_OBS_END_IDX,
    REPLAY_BURN_IN,
    REPLAY_CAPACITY_FALLBACK,
    REPLAY_CAPACITY_REQUESTED,
    REPLAY_CHECKPOINT_INTERVAL,
    REPLAY_HIDDEN_DTYPE,
    REPLAY_OBS_DTYPE,
    REPLAY_TRAIN_LEN,
    ObsNormalizer,
    estimate_dual_replay_bytes,
    effective_actor_learning_rate,
    interval_crossed,
    learner_updates_for_transitions,
    make_dual_replay_buffer,
    save_policy_artifact,
    training_should_continue,
    unpack_sensor_observations,
    validate_sensor_policy_artifact,
)


class SquashedGaussianMLPActor(nn.Module):
    """Tiny feed-forward actor for non-artifact trainer unit tests only."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.obs_dim = obs_dim
        self.act_dim = act_dim

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


def _sensor_models() -> Models:
    actor = make_actor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=2,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=32,
    )
    critic = QuantileCritic(CRITIC_OBS_DIM, 2, [32, 32], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def _small_trajectory_buffer(
    *,
    num_envs: int = 2,
    capacity: int = 128,
    n_step: int = 3,
    gamma: float = 0.5,
    burn_in: int = 4,
    train_len: int = 4,
    checkpoint_interval: int = 4,
    hidden_dim: int = 8,
    obs_dtype: torch.dtype = REPLAY_OBS_DTYPE,
    actor_obs_dim: int = ACTOR_DIM,
    critic_obs_dim: int = CRITIC_DIM,
    act_dim: int = 1,
) -> TrajectoryReplayBuffer:
    return TrajectoryReplayBuffer(
        capacity=capacity,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=torch.device("cpu"),
        n_step=n_step,
        gamma=gamma,
        burn_in=burn_in,
        train_len=train_len,
        checkpoint_interval=checkpoint_interval,
        hidden_dim=hidden_dim,
        obs_dtype=obs_dtype,
    )


def _fill_lockstep(
    replay: TrajectoryReplayBuffer,
    steps: int,
    *,
    terminal_at: set[int] | None = None,
    hidden_fn=None,
) -> None:
    terminal_at = terminal_at or set()
    for t in range(steps):
        actor = torch.zeros(replay.num_envs, replay.actor_obs_dim)
        critic = torch.zeros(replay.num_envs, replay.critic_obs_dim)
        actor[:, 0] = float(t)
        critic[:, 0] = float(t)
        if replay.actor_obs_dim > 1:
            actor[:, 1] = torch.arange(replay.num_envs, dtype=torch.float32) + 10.0
        if replay.critic_obs_dim > 1:
            critic[:, 1] = torch.arange(replay.num_envs, dtype=torch.float32) + 100.0
        if replay.critic_obs_dim > 2:
            critic[:, 2] = 1.0
        act = torch.full(
            (replay.num_envs, replay.act_dim), 0.1 * t, dtype=torch.float32
        )
        rew = torch.tensor(
            [1.0 + 0.25 * e + 0.1 * t for e in range(replay.num_envs)],
            dtype=torch.float32,
        )
        done = torch.tensor(
            [t in terminal_at for _ in range(replay.num_envs)], dtype=torch.bool
        )
        hidden = None if hidden_fn is None else hidden_fn(t, replay)
        n_emit = replay.add(actor, critic, act, rew, done, hidden=hidden)
        assert int(n_emit) == replay.num_envs


def test_trajectory_replay_stores_float16_and_samples_float32():
    replay = _small_trajectory_buffer(num_envs=1, capacity=64, hidden_dim=8)
    assert replay.actor_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.critic_obs.dtype == REPLAY_OBS_DTYPE
    assert replay.hidden.dtype == REPLAY_HIDDEN_DTYPE
    assert replay.action.dtype == torch.float32
    assert replay.seq_len == 4 + 4 + 3

    def hidden_fn(t, buf):
        return torch.full((1, buf.hidden_dim), float(t + 1), dtype=torch.float32)

    _fill_lockstep(replay, replay.seq_len, hidden_fn=hidden_fn)
    assert replay.is_ready(1)
    batch = replay.sample(1)
    assert batch["actor_obs"].dtype == torch.float32
    assert batch["critic_obs"].dtype == torch.float32
    assert batch["hidden"].dtype == torch.float32
    assert batch["actor_obs"].shape == (1, replay.seq_len, ACTOR_DIM)
    assert batch["critic_obs"].shape == (1, replay.seq_len, CRITIC_DIM)
    assert batch["hidden"].shape == (1, 8)
    assert batch["n_step_reward"].shape == (1, replay.train_len)
    assert batch["bootstrap_actor_obs"].shape == (1, replay.train_len, ACTOR_DIM)
    # Checkpoint at start column 0 stores hidden from t=0 → ones.
    assert torch.allclose(batch["hidden"], torch.ones(1, 8))


def test_trajectory_replay_checkpoint_alignment_and_contiguity():
    torch.manual_seed(0)
    replay = _small_trajectory_buffer(
        num_envs=3,
        capacity=192,
        actor_obs_dim=2,
        critic_obs_dim=3,
        obs_dtype=torch.float32,
    )
    steps = replay.seq_len + replay.checkpoint_interval
    _fill_lockstep(replay, steps)
    assert replay.is_ready(6)
    batch = replay.sample(6)
    assert torch.equal(
        batch["start_index"] % replay.checkpoint_interval,
        torch.zeros(6, dtype=torch.long),
    )
    # Contiguous time markers along the sequence axis.
    markers = batch["actor_obs"][:, :, 0]
    assert torch.allclose(markers[:, 1:] - markers[:, :-1], torch.ones(6, replay.seq_len - 1))
    # Actor/critic streams share the same transition marker.
    assert torch.equal(batch["actor_obs"][:, :, 0], batch["critic_obs"][:, :, 0])
    # Per-env rows stay on one env index.
    for i in range(6):
        env = int(batch["env_index"][i])
        start = int(batch["start_index"][i])
        for k in range(replay.seq_len):
            col = (start + k) % replay.steps_per_env
            assert float(batch["actor_obs"][i, k, 0]) == float(
                replay.actor_obs[env, col, 0]
            )


def test_trajectory_replay_episode_reset_masks_and_ids():
    replay = _small_trajectory_buffer(num_envs=1, capacity=64)
    # Terminal at t=5 → step 6 is a new episode.
    _fill_lockstep(replay, replay.seq_len, terminal_at={5})
    assert bool(replay.reset[0, 0])
    assert not bool(replay.reset[0, 5])
    assert bool(replay.reset[0, 6])
    assert int(replay.episode_id[0, 5]) == 0
    assert int(replay.episode_id[0, 6]) == 1
    batch = replay.sample(1)
    assert batch["reset"].shape == (1, replay.seq_len)
    assert float(batch["reset"][0, 0]) == 1.0
    assert float(batch["reset"][0, 6]) == 1.0


def test_trajectory_replay_terminal_safe_n_step_targets():
    replay = _small_trajectory_buffer(
        num_envs=1, capacity=64, n_step=3, gamma=0.5, burn_in=4, train_len=4
    )
    # Unit rewards; terminal at first optimized step (t=burn_in=4).
    for t in range(replay.seq_len):
        replay.add(
            torch.full((1, ACTOR_DIM), float(t)),
            torch.full((1, CRITIC_DIM), float(t)),
            torch.zeros(1, 1),
            torch.ones(1),
            torch.tensor([t == 4]),
            hidden=torch.zeros(1, 8),
        )
    batch = replay.sample(1)
    # Optimized step 0 at t=4 is terminal: return is just r_4, done flag set.
    assert torch.allclose(batch["n_step_reward"][0, 0], torch.tensor(1.0))
    assert float(batch["n_step_done"][0, 0]) == 1.0
    # Optimized step 1 at t=5 is post-reset; no terminal in its n-step window.
    # rewards 1 + 0.5 + 0.25 = 1.75
    assert torch.allclose(batch["n_step_reward"][0, 1], torch.tensor(1.75))
    assert float(batch["n_step_done"][0, 1]) == 0.0


def test_trajectory_replay_seven_step_targets_match_manual():
    """Production horizon constants: 16 burn-in + 32 train + 7 lookahead."""
    n_step = 7
    gamma = 0.9896
    replay = TrajectoryReplayBuffer(
        capacity=256,
        actor_obs_dim=2,
        critic_obs_dim=3,
        act_dim=1,
        num_envs=1,
        device=torch.device("cpu"),
        n_step=n_step,
        gamma=gamma,
        burn_in=REPLAY_BURN_IN,
        train_len=REPLAY_TRAIN_LEN,
        checkpoint_interval=REPLAY_CHECKPOINT_INTERVAL,
        hidden_dim=8,
        obs_dtype=torch.float32,
    )
    assert replay.seq_len == 55
    rewards = []
    for t in range(replay.seq_len):
        r = 0.1 * (t + 1)
        rewards.append(r)
        replay.add(
            torch.tensor([[float(t), 1.0]]),
            torch.tensor([[float(t), 2.0, 3.0]]),
            torch.zeros(1, 1),
            torch.tensor([r]),
            torch.tensor([False]),
            hidden=torch.full((1, 8), float(t), dtype=torch.float32),
        )
    batch = replay.sample(1)
    assert batch["actor_obs"].shape == (1, 55, 2)
    assert batch["n_step_reward"].shape == (1, 32)
    assert batch["bootstrap_actor_obs"].shape == (1, 32, 2)
    for i in range(32):
        t = REPLAY_BURN_IN + i
        expected = 0.0
        for k in range(n_step):
            expected += (gamma**k) * rewards[t + k]
        assert torch.allclose(
            batch["n_step_reward"][0, i],
            torch.tensor(expected),
            atol=1e-5,
        )
        assert float(batch["bootstrap_actor_obs"][0, i, 0]) == float(t + n_step)


def test_trajectory_replay_wraparound_keeps_valid_windows():
    replay = _small_trajectory_buffer(num_envs=2, capacity=64, hidden_dim=8)
    # Fill past steps_per_env to force a circular overwrite.
    total_steps = replay.steps_per_env + replay.seq_len + 4
    _fill_lockstep(replay, total_steps)
    assert replay._size == replay.steps_per_env
    assert int(replay.size) == replay.capacity
    assert replay.is_ready(4)
    batch = replay.sample(4)
    assert batch["actor_obs"].shape == (4, replay.seq_len, ACTOR_DIM)
    # Starts remain checkpoint-aligned after wrap.
    assert torch.equal(
        batch["start_index"] % replay.checkpoint_interval,
        torch.zeros(4, dtype=torch.long),
    )
    # Contiguity across the modular index wrap.
    for i in range(4):
        env = int(batch["env_index"][i])
        start = int(batch["start_index"][i])
        for k in range(replay.seq_len):
            col = (start + k) % replay.steps_per_env
            assert torch.allclose(
                batch["actor_obs"][i, k],
                replay.actor_obs[env, col].to(torch.float32),
            )


def _reference_trajectory_sample(
    replay: TrajectoryReplayBuffer,
    env_idx: torch.Tensor,
    start: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Straightforward gather/n-step reference for sample() parity tests."""
    n_seq = int(env_idx.shape[0])
    time_idx = (start.unsqueeze(1) + replay._arange_seq) % replay.steps_per_env
    env_exp = env_idx.unsqueeze(1).expand(n_seq, replay.seq_len)
    actor = replay.actor_obs[env_exp, time_idx].to(torch.float32)
    critic = replay.critic_obs[env_exp, time_idx].to(torch.float32)
    action = replay.action[env_exp, time_idx]
    reward = replay.reward[env_exp, time_idx]
    done = replay.done[env_exp, time_idx]
    reset = replay.reset[env_exp, time_idx].to(torch.float32)
    episode_id = replay.episode_id[env_exp, time_idx]
    ckpt_idx = start // replay.checkpoint_interval
    hidden = replay.hidden[env_idx, ckpt_idx].to(torch.float32)
    rew_g = reward[:, replay._n_step_idx]
    done_g = done[:, replay._n_step_idx]
    prior_done = torch.cumsum(done_g, dim=-1) - done_g
    alive = (prior_done == 0).to(torch.float32)
    n_step_reward = (rew_g * replay._gamma_powers * alive).sum(dim=-1)
    n_step_done = ((done_g * alive).sum(dim=-1) > 0).to(torch.float32)
    boot_lo = replay.burn_in + replay.n_step
    boot_hi = replay.burn_in + replay.train_len + replay.n_step
    return {
        "actor_obs": actor,
        "critic_obs": critic,
        "action": action,
        "reward": reward,
        "done": done,
        "reset": reset,
        "episode_id": episode_id,
        "hidden": hidden,
        "n_step_reward": n_step_reward,
        "n_step_done": n_step_done,
        "bootstrap_actor_obs": actor[:, boot_lo:boot_hi],
        "bootstrap_critic_obs": critic[:, boot_lo:boot_hi],
        "env_index": env_idx,
        "start_index": start,
    }


def test_trajectory_sample_matches_reference_and_cached_starts():
    torch.manual_seed(0)
    # Large enough steps_per_env for production 55-step windows + several starts.
    replay = _small_trajectory_buffer(
        num_envs=4,
        capacity=1024,
        actor_obs_dim=2,
        critic_obs_dim=3,
        obs_dtype=torch.float16,
        n_step=7,
        burn_in=REPLAY_BURN_IN,
        train_len=REPLAY_TRAIN_LEN,
        checkpoint_interval=REPLAY_CHECKPOINT_INTERVAL,
        hidden_dim=8,
    )
    assert replay.seq_len == 55
    assert replay.steps_per_env >= 128
    _fill_lockstep(replay, replay.steps_per_env + replay.seq_len)
    assert replay.is_ready(8)
    starts_a = replay._checkpoint_starts().clone()
    starts_b = replay._checkpoint_starts()
    assert torch.equal(starts_a, starts_b)
    assert replay._cached_starts_key == (replay.ptr, replay._size)

    torch.manual_seed(123)
    batch = replay.sample(8)
    ref = _reference_trajectory_sample(
        replay, batch["env_index"], batch["start_index"]
    )
    for key in (
        "actor_obs",
        "critic_obs",
        "action",
        "reward",
        "done",
        "reset",
        "episode_id",
        "hidden",
        "n_step_reward",
        "n_step_done",
        "bootstrap_actor_obs",
        "bootstrap_critic_obs",
    ):
        assert torch.allclose(
            batch[key].float(), ref[key].float(), atol=1e-5, rtol=1e-5
        ), key
    assert torch.equal(
        batch["start_index"] % replay.checkpoint_interval,
        torch.zeros(8, dtype=torch.long),
    )
    assert torch.equal(
        batch["bootstrap_actor_obs"],
        batch["actor_obs"][
            :,
            replay.burn_in
            + replay.n_step : replay.burn_in
            + replay.train_len
            + replay.n_step,
        ],
    )


def test_equal_dimension_dual_replay_is_rejected():
    with pytest.raises(ValueError, match="distinct actor/critic"):
        TrajectoryReplayBuffer(
            capacity=64,
            actor_obs_dim=5,
            critic_obs_dim=5,
            act_dim=2,
            num_envs=1,
            device=torch.device("cpu"),
            n_step=3,
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


def test_bootstrap_slice_matches_separate_normalize():
    """Hot-path: one window normalize + slice == normalize(bootstrap) alone."""
    torch.manual_seed(0)
    burn_in, train_len, n_step = REPLAY_BURN_IN, REPLAY_TRAIN_LEN, 7
    seq_len = burn_in + train_len + n_step
    actor_norm = ObsNormalizer(ACTOR_DIM, torch.device("cpu"), eps=1e-8)
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"), eps=1e-8)
    actor_norm.update(torch.randn(256, ACTOR_DIM))
    critic_norm.update(torch.randn(256, CRITIC_DIM))
    actor_raw = torch.randn(4, seq_len, ACTOR_DIM)
    critic_raw = torch.randn(4, seq_len, CRITIC_DIM)
    boot_lo = burn_in + n_step
    boot_hi = burn_in + train_len + n_step
    boot_actor_raw = actor_raw[:, boot_lo:boot_hi]
    boot_critic_raw = critic_raw[:, boot_lo:boot_hi]

    normalized_actor = actor_norm.normalize(actor_raw)
    normalized_critic = critic_norm.normalize(critic_raw)
    assert torch.allclose(
        normalized_actor[:, boot_lo:boot_hi],
        actor_norm.normalize(boot_actor_raw),
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        normalized_critic[:, boot_lo:boot_hi],
        critic_norm.normalize(boot_critic_raw),
        atol=1e-6,
        rtol=1e-5,
    )


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


def test_effective_actor_learning_rate_ramps_after_freeze():
    base = 2.5e-5
    freeze = 20_000_000
    ramp = 10_000_000
    assert effective_actor_learning_rate(0, freeze, ramp, base) == 0.0
    assert effective_actor_learning_rate(freeze - 1, freeze, ramp, base) == 0.0
    assert effective_actor_learning_rate(freeze, freeze, ramp, base) == 0.0
    mid = freeze + ramp // 2
    assert effective_actor_learning_rate(mid, freeze, ramp, base) == pytest.approx(
        base * 0.5
    )
    assert effective_actor_learning_rate(freeze + ramp, freeze, ramp, base) == base
    assert effective_actor_learning_rate(freeze + ramp + 1, freeze, ramp, base) == base
    assert effective_actor_learning_rate(freeze + 1, freeze, 0, base) == base


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
    models = _sensor_models()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(torch.randn(3, ACTOR_OBS_DIM))

    path = save_policy_artifact(models, 1234, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "actor" in payload
    assert "obs_norm" in payload
    assert "critic1" not in payload
    assert "critic_norm" not in payload
    assert payload["env_transitions"] == 1234
    assert payload["obs_dim"] == ACTOR_OBS_DIM
    assert payload["actor_obs_dim"] == ACTOR_OBS_DIM
    assert payload["critic_obs_dim"] == CRITIC_OBS_DIM
    assert payload["actor_layout_version"] == ACTOR_LAYOUT_VERSION
    assert payload["action_dim"] == 2
    assert payload["steering_action_mode"] == "delta"
    assert payload["steering_delta_max_rad"] == pytest.approx(math.pi / 60.0)
    assert payload["config_version"] == cfg["config_version"]
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=ACTOR_LAYOUT_VERSION,
        expected_architecture=payload["actor_architecture"],
        expected_critic_obs_dim=CRITIC_OBS_DIM,
    )
    restored = _sensor_models()
    restored.actor.load_state_dict(payload["actor"])
    obs = torch.randn(3, ACTOR_OBS_DIM)
    with torch.no_grad():
        expected, _ = models.actor(obs, deterministic=True, with_logprob=False)
        actual, _ = restored.actor(obs, deterministic=True, with_logprob=False)
    assert torch.equal(expected, actual)


def test_delta_sensor_artifact_metadata_and_legacy_rejection(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    models = _sensor_models()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    path = save_policy_artifact(models, 2, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_sensor_policy_artifact(
        payload,
        expected_actor_obs_dim=ACTOR_OBS_DIM,
        expected_action_dim=2,
        expected_layout_version=ACTOR_LAYOUT_VERSION,
        expected_architecture=payload["actor_architecture"],
        expected_steering_action_mode="delta",
        expected_steering_delta_max_rad=math.pi / 60.0,
        expected_critic_obs_dim=CRITIC_OBS_DIM,
    )
    legacy = copy.deepcopy(payload)
    del legacy["steering_action_mode"]
    del legacy["steering_delta_max_rad"]
    with pytest.raises(ValueError, match="steering_action_mode"):
        validate_sensor_policy_artifact(
            legacy,
            expected_actor_obs_dim=ACTOR_OBS_DIM,
            expected_action_dim=2,
            expected_layout_version=ACTOR_LAYOUT_VERSION,
            expected_architecture=payload["actor_architecture"],
            expected_steering_action_mode="delta",
            expected_steering_delta_max_rad=math.pi / 60.0,
            expected_critic_obs_dim=CRITIC_OBS_DIM,
        )


def test_stale_privileged_artifact_schema_is_rejected(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    models = _sensor_models()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    path = save_policy_artifact(models, 1, tmp_path, normalizer, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["obs_dim"] = 392
    payload["actor_obs_dim"] = 392
    del payload["actor_layout_version"]
    with pytest.raises(ValueError, match="actor_layout_version|expected actor dim"):
        validate_sensor_policy_artifact(
            payload,
            expected_actor_obs_dim=ACTOR_OBS_DIM,
            expected_action_dim=2,
            expected_layout_version=ACTOR_LAYOUT_VERSION,
            expected_architecture=payload["actor_architecture"],
            expected_critic_obs_dim=CRITIC_OBS_DIM,
        )


def test_replay_memory_estimate_and_fallback_constants():
    est_2m = estimate_dual_replay_bytes(
        REPLAY_CAPACITY_REQUESTED,
        actor_obs_dim=1097,
        critic_obs_dim=392,
        act_dim=2,
        n_step=7,
        num_envs=512,
    )
    est_1m = estimate_dual_replay_bytes(
        REPLAY_CAPACITY_FALLBACK,
        actor_obs_dim=1097,
        critic_obs_dim=392,
        act_dim=2,
        n_step=7,
        num_envs=512,
    )
    # Effective capacity is checkpoint-aligned per env (may be slightly under request).
    assert est_2m["requested_capacity"] == REPLAY_CAPACITY_REQUESTED == 2_000_000
    assert est_1m["requested_capacity"] == REPLAY_CAPACITY_FALLBACK == 1_000_000
    assert est_2m["capacity"] <= REPLAY_CAPACITY_REQUESTED
    assert est_1m["capacity"] <= REPLAY_CAPACITY_FALLBACK
    assert est_2m["capacity"] > est_1m["capacity"] > 0
    assert est_2m["total_bytes"] > est_1m["total_bytes"] > 0
    assert est_2m["seq_len"] == 55
    assert est_2m["ring_obs_bytes"] == (
        est_2m["steps_per_env"] * 512 * (1097 + 392) * 2
    )
    assert est_2m["hidden_bytes"] == (
        512 * est_2m["num_checkpoints"] * GRU_HIDDEN_DIM * 2
    )
    # Trajectory storage is far smaller than the old dual next-obs IID ring.
    assert est_2m["total_bytes"] < 12 * (1024**3)

    log = logging.getLogger("test_replay_alloc")
    buffer, selected, estimate = make_dual_replay_buffer(
        capacity=128,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=2,
        n_step=3,
        gamma=0.99,
        num_envs=2,
        device=torch.device("cpu"),
        batch_size=8,
        log=log,
        allow_fallback=True,
        burn_in=4,
        train_len=4,
        checkpoint_interval=4,
        hidden_dim=8,
    )
    assert selected == buffer.capacity
    assert estimate["capacity"] == buffer.capacity
    assert buffer.actor_obs.dtype == REPLAY_OBS_DTYPE
    assert buffer.hidden.dtype == REPLAY_HIDDEN_DTYPE
    assert isinstance(buffer, TrajectoryReplayBuffer)


def test_cpu_trajectory_replay_schema_at_production_horizons():
    """CPU allocation/sample gate for production 55-step schema (no GPU/e2e)."""
    num_envs = 4
    buffer, selected, estimate = make_dual_replay_buffer(
        capacity=512,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=2,
        n_step=7,
        gamma=0.9896,
        num_envs=num_envs,
        device=torch.device("cpu"),
        batch_size=64,
        allow_fallback=True,
        hidden_dim=16,
    )
    assert selected == buffer.capacity
    assert buffer.seq_len == 55
    assert buffer.burn_in == REPLAY_BURN_IN
    assert buffer.train_len == REPLAY_TRAIN_LEN
    assert estimate["seq_len"] == 55
    hidden = torch.randn(num_envs, 16)
    for t in range(buffer.seq_len):
        buffer.add(
            torch.randn(num_envs, ACTOR_DIM),
            torch.randn(num_envs, CRITIC_DIM),
            torch.randn(num_envs, 2),
            torch.randn(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            hidden=hidden,
        )
    assert buffer.is_ready(2)
    batch = buffer.sample(2)
    assert batch["actor_obs"].shape == (2, 55, ACTOR_DIM)
    assert batch["critic_obs"].shape == (2, 55, CRITIC_DIM)
    assert batch["n_step_reward"].shape == (2, 32)
    assert batch["n_step_done"].shape == (2, 32)
    assert batch["bootstrap_actor_obs"].shape == (2, 32, ACTOR_DIM)
    assert batch["bootstrap_critic_obs"].shape == (2, 32, CRITIC_DIM)
    assert batch["hidden"].shape == (2, 16)
    assert batch["reset"].shape == (2, 55)


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


def test_collection_to_trajectory_replay_alignment():
    """Sensor dict → trajectory replay → separate norms preserve dual alignment."""
    torch.manual_seed(0)
    replay = _small_trajectory_buffer(num_envs=2, capacity=128, act_dim=2, hidden_dim=8)
    actor_norm = ObsNormalizer(ACTOR_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))

    for t in range(replay.seq_len):
        actor_obs, critic_obs = unpack_sensor_observations(
            {
                "actor": torch.randn(2, ACTOR_DIM),
                "frenet": torch.randn(2, CRITIC_DIM),
            }
        )
        # Stamp a shared marker so alignment is checkable after sampling.
        actor_obs[:, 0] = float(t)
        critic_obs[:, 0] = float(t)
        actor_norm.update(actor_obs)
        critic_norm.update(critic_obs)
        replay.add(
            actor_obs,
            critic_obs,
            torch.zeros(2, 2),
            torch.ones(2),
            torch.tensor([False, t == replay.seq_len - 1]),
            hidden=torch.zeros(2, 8),
        )

    assert replay.is_ready(2)
    batch = replay.sample(2)
    batch["actor_obs"] = actor_norm.normalize(batch["actor_obs"])
    batch["critic_obs"] = critic_norm.normalize(batch["critic_obs"])
    assert batch["actor_obs"].shape == (2, replay.seq_len, ACTOR_DIM)
    assert batch["critic_obs"].shape == (2, replay.seq_len, CRITIC_DIM)
    assert torch.isfinite(batch["actor_obs"]).all()
    assert torch.isfinite(batch["critic_obs"]).all()
    # Raw (pre-norm) markers remain aligned in storage.
    raw = replay.sample(2)
    assert torch.equal(raw["actor_obs"][:, :, 0], raw["critic_obs"][:, :, 0])


PROD_CRITIC_DIM = 392
BATCH_SCHEMA_KEYS = (
    "actor_obs",
    "critic_obs",
    "action",
    "reward",
    "done",
    "reset",
    "episode_id",
    "hidden",
    "n_step_reward",
    "n_step_done",
    "bootstrap_actor_obs",
    "bootstrap_critic_obs",
    "env_index",
    "start_index",
)


def _prod_trajectory_buffer(
    *,
    num_envs: int = 4,
    capacity: int = 1024,
    hidden_dim: int = 8,
) -> TrajectoryReplayBuffer:
    return TrajectoryReplayBuffer(
        capacity=capacity,
        actor_obs_dim=ACTOR_DIM,
        critic_obs_dim=PROD_CRITIC_DIM,
        act_dim=1,
        num_envs=num_envs,
        device=torch.device("cpu"),
        n_step=7,
        gamma=0.9896,
        burn_in=REPLAY_BURN_IN,
        train_len=REPLAY_TRAIN_LEN,
        checkpoint_interval=REPLAY_CHECKPOINT_INTERVAL,
        hidden_dim=hidden_dim,
        obs_dtype=torch.float32,
    )


def _fill_visibility_pattern(
    replay: TrajectoryReplayBuffer,
    steps: int,
    *,
    visible_at: dict[int, set[int]] | None = None,
    terminal_at: set[int] | None = None,
    hidden_fn=None,
) -> None:
    """Fill lockstep rows; ``visible_at[t]`` lists envs with a non-zero opp block."""
    visible_at = visible_at or {}
    terminal_at = terminal_at or set()
    for t in range(steps):
        actor = torch.zeros(replay.num_envs, replay.actor_obs_dim)
        critic = torch.zeros(replay.num_envs, replay.critic_obs_dim)
        actor[:, 0] = float(t)
        critic[:, 0] = float(t)
        for env in visible_at.get(t, set()):
            critic[env, OPP_OBS_BASE_IDX:OPP_OBS_END_IDX] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0]
            )
        act = torch.zeros(replay.num_envs, replay.act_dim)
        rew = torch.ones(replay.num_envs)
        done = torch.tensor(
            [t in terminal_at for _ in range(replay.num_envs)], dtype=torch.bool
        )
        hidden = None if hidden_fn is None else hidden_fn(t, replay)
        replay.add(actor, critic, act, rew, done, hidden=hidden)


def _window_is_visible(replay: TrajectoryReplayBuffer, env: int, start: int) -> bool:
    cols = [(start + k) % replay.steps_per_env for k in range(replay.seq_len)]
    return bool(replay.opponent_visible[env, cols].any().item())


def test_visibility_metadata_from_critic_opponent_block():
    replay = _prod_trajectory_buffer(num_envs=2, capacity=256)
    # Env 0 visible only at t=0; env 1 never visible.
    _fill_visibility_pattern(replay, 2, visible_at={0: {0}})
    assert bool(replay.opponent_visible[0, 0])
    assert not bool(replay.opponent_visible[0, 1])
    assert not bool(replay.opponent_visible[1, 0])
    assert not bool(replay.opponent_visible[1, 1])


def test_visibility_classification_across_burn_train_bootstrap():
    """Visible if opponent block is non-zero in ANY section of the 55-step window."""
    replay = _prod_trajectory_buffer(num_envs=3, capacity=1024)
    assert replay.seq_len == 55
    # Place a single visible spike in burn-in / train / bootstrap on envs 0/1/2.
    visible_at = {
        3: {0},  # burn-in [0, 16)
        20: {1},  # train [16, 48)
        50: {2},  # bootstrap tail [48, 55)
    }
    steps = replay.steps_per_env
    _fill_visibility_pattern(replay, steps, visible_at=visible_at)
    assert replay.is_ready(6)
    for env in (0, 1, 2):
        assert _window_is_visible(replay, env, start=0)
    # All-absent buffer: every complete window is opponent_not_visible.
    replay2 = _prod_trajectory_buffer(num_envs=1, capacity=512)
    _fill_visibility_pattern(replay2, replay2.steps_per_env, visible_at={})
    assert not _window_is_visible(replay2, 0, 0)
    vis_e, _, not_e, _ = replay2._visibility_buckets()
    assert int(vis_e.numel()) == 0
    assert int(not_e.numel()) > 0


def test_even_visibility_sampling_and_odd_extra_alternation():
    torch.manual_seed(0)
    replay = _prod_trajectory_buffer(num_envs=4, capacity=1024)
    # Env 0/1 always visible; env 2/3 never.
    visible_at = {t: {0, 1} for t in range(replay.steps_per_env)}
    _fill_visibility_pattern(replay, replay.steps_per_env, visible_at=visible_at)
    assert replay.is_ready(8)

    batch = replay.sample(8)
    n_vis = sum(
        _window_is_visible(replay, int(batch["env_index"][i]), int(batch["start_index"][i]))
        for i in range(8)
    )
    assert n_vis == 4
    assert replay.last_sample_metrics["sampled_visible_frac"] == 0.5
    assert replay.last_sample_metrics["fallback"] == 0.0

    # Odd batch: alternate which bucket gets the extra slot.
    replay._odd_extra_to_visible = True
    b1 = replay.sample(5)
    n1 = sum(
        _window_is_visible(replay, int(b1["env_index"][i]), int(b1["start_index"][i]))
        for i in range(5)
    )
    assert n1 == 3  # 3 visible + 2 not
    b2 = replay.sample(5)
    n2 = sum(
        _window_is_visible(replay, int(b2["env_index"][i]), int(b2["start_index"][i]))
        for i in range(5)
    )
    assert n2 == 2  # 2 visible + 3 not


def test_visibility_sampling_fallback_when_bucket_empty():
    replay = _prod_trajectory_buffer(num_envs=2, capacity=512)
    _fill_visibility_pattern(replay, replay.steps_per_env, visible_at={})
    assert replay.is_ready(4)
    batch = replay.sample(4)
    assert batch["actor_obs"].shape == (4, 55, ACTOR_DIM)
    assert replay.last_sample_metrics["fallback"] == 1.0
    assert replay.last_sample_metrics["sampled_visible_frac"] == 0.0
    assert replay.last_sample_metrics["available_visible_frac"] == 0.0

    # Sparse: only one visible window among many absent.
    replay2 = _prod_trajectory_buffer(num_envs=2, capacity=512)
    visible_at = {t: {0} for t in range(replay2.seq_len)}
    _fill_visibility_pattern(replay2, replay2.steps_per_env, visible_at=visible_at)
    batch2 = replay2.sample(6)
    assert replay2.last_sample_metrics["fallback"] == 0.0
    assert abs(replay2.last_sample_metrics["sampled_visible_frac"] - 0.5) < 1e-9
    assert set(batch2.keys()) == set(BATCH_SCHEMA_KEYS)


def test_visibility_sampling_preserves_schema_alignment_wrap_and_hidden():
    torch.manual_seed(1)
    replay = _prod_trajectory_buffer(num_envs=3, capacity=512, hidden_dim=8)

    def hidden_fn(t, buf):
        return torch.full(
            (buf.num_envs, buf.hidden_dim), float(t + 7), dtype=torch.float32
        )

    # Mix visibility; wrap the ring; include a reset mid-window.
    visible_at = {t: {0} for t in range(10)}
    visible_at.update({t: {1} for t in range(30, 40)})
    total = replay.steps_per_env + replay.seq_len + 8
    _fill_visibility_pattern(
        replay,
        total,
        visible_at=visible_at,
        terminal_at={12, replay.steps_per_env + 5},
        hidden_fn=hidden_fn,
    )
    assert replay.is_ready(4)
    batch = replay.sample(4)
    assert set(batch.keys()) == set(BATCH_SCHEMA_KEYS)
    assert batch["actor_obs"].shape == (4, 55, ACTOR_DIM)
    assert batch["critic_obs"].shape == (4, 55, PROD_CRITIC_DIM)
    assert batch["hidden"].shape == (4, 8)
    assert batch["reset"].shape == (4, 55)
    assert torch.equal(
        batch["start_index"] % replay.checkpoint_interval,
        torch.zeros(4, dtype=torch.long),
    )
    for i in range(4):
        env = int(batch["env_index"][i])
        start = int(batch["start_index"][i])
        ckpt = start // replay.checkpoint_interval
        assert torch.allclose(
            batch["hidden"][i],
            replay.hidden[env, ckpt].float(),
        )
        for k in range(replay.seq_len):
            col = (start + k) % replay.steps_per_env
            assert torch.allclose(
                batch["actor_obs"][i, k],
                replay.actor_obs[env, col].float(),
            )
            assert torch.allclose(
                batch["critic_obs"][i, k],
                replay.critic_obs[env, col].float(),
            )
    ref = _reference_trajectory_sample(
        replay, batch["env_index"], batch["start_index"]
    )
    for key in BATCH_SCHEMA_KEYS:
        if key in ("env_index", "start_index"):
            assert torch.equal(batch[key], ref[key])
        else:
            assert torch.allclose(
                batch[key].float(), ref[key].float(), atol=1e-5, rtol=1e-5
            ), key

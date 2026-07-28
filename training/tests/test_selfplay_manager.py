"""Genesis-free tests for SelfPlayManager snapshot pool and cadence."""

from __future__ import annotations

import copy
import random
from collections import Counter

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_policy import ObsNormalizer
from f1tenth_policy.layout import ACTOR_OBS_DIM
from qrsac import Models, QuantileCritic, make_actor
from selfplay import SelfPlayManager
from standalone_trainer import save_policy_artifact

DEVICE = torch.device("cpu")
OBS_DIM = ACTOR_OBS_DIM
ACT_DIM = 2
ANCHOR_TRANSITIONS = 3_409_920_000


def _make_models() -> Models:
    hidden = [16, 16]
    actor = make_actor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=hidden,
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
    )
    critic = QuantileCritic(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=hidden,
        num_quantiles=4,
    )
    critic_t = QuantileCritic(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=hidden,
        num_quantiles=4,
    )
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic_t),
        critic2_target=copy.deepcopy(critic_t),
    )


class _RecordingEnv:
    """Minimal real stand-in for F1tenthEnv that records opponent refreshes.

    SelfPlayManager.maybe_refresh only touches ``refresh_opponent_policy`` on the
    env, so a tiny real object (not a mock) is enough to exercise the cadence and
    sampling logic without pulling in Genesis.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_args = None

    def refresh_opponent_policy(self, actor, mean, var, actor_architecture=None) -> None:
        self.call_count += 1
        self.last_args = (actor, mean, var, actor_architecture)


def _recording_env() -> _RecordingEnv:
    return _RecordingEnv()


def _write_anchor_ckpt(tmp_path) -> str:
    """Save a real policy artifact usable as an immutable anchor."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["obs"]["num_actor_obs"] = OBS_DIM
    cfg["obs"]["num_obs"] = OBS_DIM + 1
    cfg["obs"]["actor_layout_version"] = 2
    cfg["env"]["num_actions"] = ACT_DIM
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    normalizer.update(torch.randn(8, OBS_DIM))
    path = save_policy_artifact(
        _make_models(), ANCHOR_TRANSITIONS, tmp_path, normalizer, cfg
    )
    return str(path)


def _load_test_anchor(mgr: SelfPlayManager, path: str) -> None:
    mgr.load_anchor(
        path,
        DEVICE,
        OBS_DIM,
        ACT_DIM,
        expected_layout_version=int(DEFAULT_CONFIG["obs"]["actor_layout_version"]),
        expected_steering_action_mode=str(
            DEFAULT_CONFIG["env"]["steering_action_mode"]
        ),
        expected_steering_delta_max_rad=float(
            DEFAULT_CONFIG["env"]["steering_delta_max_rad"]
        ),
    )


def test_pool_push_and_maxlen_eviction():
    mgr = SelfPlayManager(
        pool_size=3,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=10_000,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)

    for step in (1000, 2000, 3000, 4000):
        mgr.maybe_snapshot(models, normalizer, step)

    assert len(mgr.pool) == 3
    assert [s["transitions"] for s in mgr.pool] == [2000, 3000, 4000]


def test_snapshot_cadence_gating():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=100,
        refresh_interval_transitions=10_000,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)

    assert not mgr.maybe_snapshot(models, normalizer, 50)
    assert not mgr.maybe_snapshot(models, normalizer, 99)
    assert mgr.maybe_snapshot(models, normalizer, 100)
    assert len(mgr.pool) == 1
    assert not mgr.maybe_snapshot(models, normalizer, 100)


def test_refresh_cadence_gating():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=50,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    mgr.seed_snapshot(
        SelfPlayManager.make_snapshot(models, normalizer, transitions=0)
    )

    assert not mgr.maybe_refresh(env, 25)
    assert mgr.maybe_refresh(env, 50)
    assert env.call_count == 1
    assert not mgr.maybe_refresh(env, 50)
    assert mgr.maybe_refresh(env, 100)
    assert env.call_count == 2


def test_sample_latest():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="latest",
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)
        mgr.maybe_refresh(env, step)

    assert mgr.opponent_transitions == 30


def test_sample_uniform_covers_pool():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="uniform",
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)

    counts: Counter[int] = Counter()
    for refresh_step in range(100, 1100, 1):
        mgr.maybe_refresh(env, refresh_step)
        counts[mgr.opponent_transitions] += 1

    assert counts[10] > 0
    assert counts[20] > 0
    assert counts[30] > 0


def test_sample_mixed_favors_latest():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="mixed",
        mixed_latest_prob=0.8,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)

    latest_hits = 0
    for refresh_step in range(200, 1200, 1):
        mgr.maybe_refresh(env, refresh_step)
        if mgr.opponent_transitions == 30:
            latest_hits += 1

    assert latest_hits > 700


def test_win_rate_proxy():
    mgr = SelfPlayManager()
    mgr.record_episode_outcomes(torch.tensor([1.0, -1.0, 0.5, -0.1]))
    assert mgr.win_rate() == 0.5
    mgr.reset_win_stats()
    assert mgr._episode_total == 0
    assert mgr.win_rate() != mgr.win_rate()  # nan


def test_anchor_survives_pool_eviction(tmp_path):
    mgr = SelfPlayManager(
        pool_size=3,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=10_000,
        anchor_prob=0.5,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    _load_test_anchor(mgr, _write_anchor_ckpt(tmp_path))

    for step in range(1000, 21000, 1000):
        mgr.maybe_snapshot(models, normalizer, step)

    assert len(mgr.pool) == 3
    assert [s["transitions"] for s in mgr.pool] == [18000, 19000, 20000]
    assert mgr.anchor is not None
    assert mgr.anchor["transitions"] == ANCHOR_TRANSITIONS
    assert all(s["transitions"] != ANCHOR_TRANSITIONS for s in mgr.pool)
    assert mgr.anchor["mean"].shape == (OBS_DIM,)
    assert "net.0.weight" in mgr.anchor["actor"]


def test_anchor_sampling_probability_wiring(tmp_path):
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="latest",
        anchor_prob=0.5,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()
    _load_test_anchor(mgr, _write_anchor_ckpt(tmp_path))
    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)

    random.seed(0)
    trials = 2000
    anchor_hits = 0
    for refresh_step in range(1, trials + 1):
        mgr.maybe_refresh(env, refresh_step)
        if mgr.opponent_transitions == ANCHOR_TRANSITIONS:
            anchor_hits += 1

    assert 800 < anchor_hits < 1200


def test_no_anchor_is_backward_compatible():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="latest",
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    assert mgr.anchor is None
    assert mgr.anchor_prob == 0.0
    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)
        mgr.maybe_refresh(env, step)
    assert mgr.opponent_transitions == 30


def test_anchor_prob_zero_never_samples_anchor(tmp_path):
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval_transitions=1,
        refresh_interval_transitions=1,
        sample_mode="latest",
        anchor_prob=0.0,
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()
    _load_test_anchor(mgr, _write_anchor_ckpt(tmp_path))
    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)

    random.seed(0)
    for refresh_step in range(1, 501):
        mgr.maybe_refresh(env, refresh_step)
        assert mgr.opponent_transitions == 30

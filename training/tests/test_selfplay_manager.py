"""Genesis-free tests for SelfPlayManager snapshot pool and cadence."""

from __future__ import annotations

import copy
from collections import Counter

import torch
import torch.nn as nn

from qrsac import Models, QuantileCritic, SquashedGaussianMLPActor
from standalone_trainer import ObsNormalizer, SelfPlayManager

DEVICE = torch.device("cpu")
OBS_DIM = 390
ACT_DIM = 2


def _make_models() -> Models:
    hidden = [16, 16]
    actor = SquashedGaussianMLPActor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=hidden,
        activation=nn.ReLU,
        act_limit=1.0,
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

    def refresh_opponent_policy(self, actor, mean, var) -> None:
        self.call_count += 1
        self.last_args = (actor, mean, var)


def _recording_env() -> _RecordingEnv:
    return _RecordingEnv()


def test_pool_push_and_maxlen_eviction():
    mgr = SelfPlayManager(pool_size=3, snapshot_interval=1, refresh_interval=10_000)
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)

    for step in (1000, 2000, 3000, 4000):
        mgr.maybe_snapshot(models, normalizer, step)

    assert len(mgr.pool) == 3
    assert [s["step"] for s in mgr.pool] == [2000, 3000, 4000]


def test_snapshot_cadence_gating():
    mgr = SelfPlayManager(pool_size=5, snapshot_interval=100, refresh_interval=10_000)
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)

    assert not mgr.maybe_snapshot(models, normalizer, 50)
    assert not mgr.maybe_snapshot(models, normalizer, 150)
    assert mgr.maybe_snapshot(models, normalizer, 100)
    assert len(mgr.pool) == 1
    assert not mgr.maybe_snapshot(models, normalizer, 100)


def test_refresh_cadence_gating():
    mgr = SelfPlayManager(pool_size=5, snapshot_interval=1, refresh_interval=50)
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    mgr.seed_snapshot(SelfPlayManager.make_snapshot(models, normalizer, 0))

    assert not mgr.maybe_refresh(env, 25)
    assert mgr.maybe_refresh(env, 50)
    assert env.call_count == 1
    assert not mgr.maybe_refresh(env, 50)
    assert mgr.maybe_refresh(env, 100)
    assert env.call_count == 2


def test_sample_latest():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval=1,
        refresh_interval=1,
        sample_mode="latest",
    )
    models = _make_models()
    normalizer = ObsNormalizer(OBS_DIM, DEVICE)
    env = _recording_env()

    for step in (10, 20, 30):
        mgr.maybe_snapshot(models, normalizer, step)
        mgr.maybe_refresh(env, step)

    assert mgr.opponent_step == 30


def test_sample_uniform_covers_pool():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval=1,
        refresh_interval=1,
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
        counts[mgr.opponent_step] += 1

    assert counts[10] > 0
    assert counts[20] > 0
    assert counts[30] > 0


def test_sample_mixed_favors_latest():
    mgr = SelfPlayManager(
        pool_size=5,
        snapshot_interval=1,
        refresh_interval=1,
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
        if mgr.opponent_step == 30:
            latest_hits += 1

    assert latest_hits > 700


def test_win_rate_proxy():
    mgr = SelfPlayManager()
    mgr.record_episode_outcomes(torch.tensor([1.0, -1.0, 0.5, -0.1]))
    assert mgr.win_rate() == 0.5
    mgr.reset_win_stats()
    assert mgr._episode_total == 0
    assert mgr.win_rate() != mgr.win_rate()  # nan

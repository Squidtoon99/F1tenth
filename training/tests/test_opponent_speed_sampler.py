"""Per-episode opponent speed sampler used by eval and release videos."""
from __future__ import annotations

import numpy as np
import pytest

from eval_full_lap import (
    DEFAULT_OPPONENT_SPEED_RANGE,
    NOTAL_EVAL_OPPONENT_SPEED_RANGE,
    resolve_eval_opponent_speed_range,
    resolve_opponent_speed_range,
    sample_opponent_speeds,
)


def test_resolve_uses_training_target_speed_range():
    lo, hi = resolve_opponent_speed_range(
        {
            "opponent_target_speed": 3.5,
            "opponent_target_speed_range": [2.0, 5.0],
        }
    )
    assert (lo, hi) == (2.0, 5.0)


def test_resolve_swaps_inverted_range():
    assert resolve_opponent_speed_range(
        {"opponent_target_speed_range": [5.0, 2.0]}
    ) == (2.0, 5.0)


def test_resolve_falls_back_to_scalar_target():
    assert resolve_opponent_speed_range(
        {"opponent_target_speed": 3.5}
    ) == (3.5, 3.5)


def test_resolve_documented_range_when_config_omits_speeds():
    assert resolve_opponent_speed_range({}) == DEFAULT_OPPONENT_SPEED_RANGE
    assert DEFAULT_OPPONENT_SPEED_RANGE == (2.0, 5.0)


def test_notal_eval_uses_faster_range():
    lo, hi = resolve_eval_opponent_speed_range(
        {"opponent_target_speed_range": [2.0, 5.0]},
        {"reward_scales": {"tal": 0.0}},
    )
    assert (lo, hi) == NOTAL_EVAL_OPPONENT_SPEED_RANGE
    assert NOTAL_EVAL_OPPONENT_SPEED_RANGE == (6.5, 7.6)


def test_tal_eval_keeps_training_range():
    lo, hi = resolve_eval_opponent_speed_range(
        {"opponent_target_speed_range": [2.0, 5.0]},
        {"reward_scales": {"tal": 0.2}},
    )
    assert (lo, hi) == (2.0, 5.0)


def test_sample_is_per_opponent_reproducible_and_in_range():
    rng = np.random.default_rng(0)
    speeds = sample_opponent_speeds(8, 2.0, 5.0, rng)
    assert speeds.shape == (8,)
    assert speeds.dtype == np.float32
    assert np.all(speeds >= 2.0)
    assert np.all(speeds <= 5.0)
    assert len(set(np.round(speeds, 6).tolist())) == 8
    again = sample_opponent_speeds(8, 2.0, 5.0, np.random.default_rng(0))
    np.testing.assert_array_equal(speeds, again)
    other = sample_opponent_speeds(8, 2.0, 5.0, np.random.default_rng(1))
    assert not np.array_equal(speeds, other)


def test_sample_degenerate_range_is_constant():
    speeds = sample_opponent_speeds(4, 3.5, 3.5, np.random.default_rng(0))
    assert speeds.tolist() == [3.5, 3.5, 3.5, 3.5]


def test_sample_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        sample_opponent_speeds(0, 2.0, 5.0, np.random.default_rng(0))

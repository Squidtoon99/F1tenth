"""Unit tests for shared applied-current observation helper."""

from __future__ import annotations

import math

import pytest

from f1tenth_policy import (
    OBS_PREPROCESSING_VERSION,
    TRAINING_I_BRAKE_MAX_A,
    TRAINING_I_DRIVE_MAX_A,
    TRAINING_I_SLEW_A_PER_S,
    applied_current_fraction,
    assert_artifact_current_limits_match,
)


def test_obs_preprocessing_version_is_normalized_current():
    assert OBS_PREPROCESSING_VERSION == 3


def test_training_current_scale_is_80a_drive_20a_brake_envelope():
    assert TRAINING_I_DRIVE_MAX_A == pytest.approx(80.0)
    assert TRAINING_I_BRAKE_MAX_A == pytest.approx(20.0)
    assert TRAINING_I_SLEW_A_PER_S == pytest.approx(200.0)
    assert TRAINING_I_SLEW_A_PER_S / TRAINING_I_DRIVE_MAX_A == pytest.approx(2.5)


def test_applied_current_fraction_drive_and_brake():
    assert applied_current_fraction(40.0, 80.0, 20.0) == pytest.approx(0.5)
    assert applied_current_fraction(-10.0, 80.0, 20.0) == pytest.approx(-0.5)
    assert applied_current_fraction(0.0, 80.0, 20.0) == pytest.approx(0.0)


def test_applied_current_fraction_fail_closed_on_invalid_limits():
    with pytest.raises(ValueError, match="directional current limit"):
        applied_current_fraction(1.0, 0.0, 10.0)
    with pytest.raises(ValueError, match="directional current limit"):
        applied_current_fraction(-1.0, 10.0, -1.0)
    with pytest.raises(ValueError, match="directional current limit"):
        applied_current_fraction(1.0, math.nan, 10.0)


def test_applied_current_fraction_nonfinite_amps_map_to_zero():
    assert applied_current_fraction(math.nan, 100.0, 10.0) == 0.0
    assert applied_current_fraction(math.inf, 100.0, 10.0) == 0.0


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_artifact_current_match_rejects_invalid_runtime_drive_limit(invalid):
    payload = {"i_drive_max_a": 80.0, "i_brake_max_a": 20.0}

    with pytest.raises(ValueError, match="Deploy i_drive_max_a"):
        assert_artifact_current_limits_match(payload, invalid, 20.0)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_artifact_current_match_rejects_invalid_runtime_brake_limit(invalid):
    payload = {"i_drive_max_a": 80.0, "i_brake_max_a": 20.0}

    with pytest.raises(ValueError, match="Deploy i_brake_max_a"):
        assert_artifact_current_limits_match(payload, 80.0, invalid)

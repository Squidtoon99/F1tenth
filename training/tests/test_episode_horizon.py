from __future__ import annotations

import inspect
import os
import sys

import pytest

from standalone_trainer import validate_config_patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env.kernel import RewardParams  # noqa: E402
from f1tenth_env.utils import (  # noqa: E402
    TRACK_EPISODE_HORIZON_S,
    TRACK_FLYING_SPAWN_MPS,
    TRACK_RESET_SPEED_MPS,
    episode_length_for_track,
    track_flying_spawn_mps,
    track_reset_speed_mps,
)


def test_expected_lap_speed_key_is_gone():
    assert "expected_lap_speed_mps" not in DEFAULT_CONFIG["env"]
    assert "episode_lap_multiplier" not in DEFAULT_CONFIG["env"]
    with pytest.raises(ValueError, match="unknown config key"):
        validate_config_patch({"env": {"expected_lap_speed_mps": 3.5}})
    with pytest.raises(ValueError, match="unknown config key"):
        validate_config_patch({"env": {"tal_v_ref": 3.5}})


def test_reward_params_have_no_speed_target():
    assert not hasattr(RewardParams, "tal_v_ref")
    assert not hasattr(RewardParams, "tal_scale")
    params = inspect.signature(episode_length_for_track).parameters
    assert "ref_lap_speed_mps" not in params


def test_episode_horizon_is_frozen_seconds_not_a_pace():
    assert episode_length_for_track("Austin") == pytest.approx(
        TRACK_EPISODE_HORIZON_S["Austin"]
    )
    assert episode_length_for_track("IV_2026_SIM") == pytest.approx(
        TRACK_EPISODE_HORIZON_S["IV_2026_SIM"]
    )
    assert episode_length_for_track("courtyard_2") == pytest.approx(60.0)
    assert episode_length_for_track("Galaxy_1") == pytest.approx(60.0)
    assert episode_length_for_track("ECSS_Floor") == pytest.approx(
        TRACK_EPISODE_HORIZON_S["ECSS_Floor"]
    )
    with pytest.raises(ValueError, match="no frozen episode horizon"):
        episode_length_for_track("not_a_track")


def test_track_start_speeds_are_track_appropriate():
    assert TRACK_FLYING_SPAWN_MPS["courtyard_2"] == pytest.approx(0.0)
    assert TRACK_FLYING_SPAWN_MPS["Galaxy_1"] == pytest.approx(0.5)
    assert TRACK_FLYING_SPAWN_MPS["ECSS_Floor"] == pytest.approx(1.0)
    assert TRACK_FLYING_SPAWN_MPS["IV_2026_SIM"] == pytest.approx(1.5)
    assert TRACK_FLYING_SPAWN_MPS["Austin"] == pytest.approx(2.0)
    assert TRACK_RESET_SPEED_MPS["courtyard_2"] == (0.0, 1.0)
    assert TRACK_RESET_SPEED_MPS["Galaxy_1"] == (0.0, 1.5)
    assert TRACK_RESET_SPEED_MPS["ECSS_Floor"] == (0.0, 2.5)
    assert TRACK_RESET_SPEED_MPS["IV_2026_SIM"] == (1.0, 3.0)
    assert TRACK_RESET_SPEED_MPS["Austin"] == (0.0, 5.0)
    assert track_flying_spawn_mps("not_a_track") == pytest.approx(0.0)
    assert track_reset_speed_mps(None) == (0.0, 1.0)
    assert DEFAULT_CONFIG["env"]["reset_speed_min_mps"] == pytest.approx(1.0)
    assert DEFAULT_CONFIG["env"]["reset_speed_max_mps"] == pytest.approx(3.0)
    assert DEFAULT_CONFIG["env"]["max_speed"] == pytest.approx(15.0)
    for name, speed in TRACK_FLYING_SPAWN_MPS.items():
        assert speed < 7.0, name
    for name, (_lo, hi) in TRACK_RESET_SPEED_MPS.items():
        assert hi < 7.0, name

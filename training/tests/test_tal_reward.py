from __future__ import annotations

import copy
import inspect
import os
import sys

import pytest
import torch

from standalone_trainer import validate_config_patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env.kernel import RewardParams  # noqa: E402
from f1tenth_env.utils import (  # noqa: E402
    TRACK_EPISODE_HORIZON_S,
    TRACK_FLYING_SPAWN_MPS,
    TRACK_RESET_SPEED_MPS,
    episode_length_for_track,
    track_flying_spawn_mps,
    track_reset_speed_mps,
)


def _build_cfg() -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["reset_stationary_probability"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 2.0
    cfg["env"]["reset_speed_max_mps"] = 2.0
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["obs"]["enable_opponent_obs"] = False
    scales = cfg["reward"]["reward_scales"]
    for key in list(scales):
        scales[key] = 0.0
    cfg["reward"]["global_reward_scale"] = 1.0
    cfg["reward"]["wall_contact_coefficient"] = 0.0
    return cfg


def _make_env(cfg: dict) -> F1tenthEnv:
    return F1tenthEnv(
        num_envs=2,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def test_default_tal_scale_is_off():
    assert DEFAULT_CONFIG["reward"]["reward_scales"]["tal"] == 0.0
    validate_config_patch({"reward": {"reward_scales": {"tal": 0.0}}})


def test_nonzero_tal_is_rejected():
    with pytest.raises(ValueError, match="tal must be 0"):
        validate_config_patch({"reward": {"reward_scales": {"tal": 0.2}}})


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


def test_tal_scale_zero_emits_zero_term(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, 2, device=env.device)
        _, reward, _, extras = env.step(actions, n_steps=env.control_interval)
        tal = extras["rewards"]["terms"]["tal"]
        assert float(tal.abs().max()) == 0.0
        assert float(reward.abs().max()) == 0.0
    finally:
        env.close()


def test_nonzero_tal_cannot_build_env(warp_runtime):
    del warp_runtime
    cfg = _build_cfg()
    cfg["reward"]["reward_scales"]["tal"] = 0.2
    with pytest.raises(ValueError, match="tal must be 0"):
        _make_env(cfg)

from __future__ import annotations

import hashlib
import json

import pytest

from config import DEFAULT_CONFIG
from standalone_trainer import (
    _deep_merge,
    build_config,
    build_env_cfg,
    build_run_snapshot,
    config_provenance,
    load_config_patch,
    parse_args,
    validate_config_patch,
)

# Hermetic episode length via patch (CLI no longer duplicates JSON knobs).
_BASE_PATCH = {"env": {"episode_length": 60}}


def _resolve(argv, patch=None):
    merged = dict(_BASE_PATCH)
    if patch:
        _deep_merge(merged, patch)
    args, explicit = parse_args(argv)
    return build_config(args, patch=merged, explicit=explicit)


def test_deep_merge_recurses_and_replaces():
    base = {"a": {"x": 1, "y": 2}, "b": 3, "c": [1, 2]}
    out = _deep_merge(base, {"a": {"y": 20, "z": 30}, "c": [9]})
    assert out is base
    assert base["a"] == {"x": 1, "y": 20, "z": 30}
    assert base["b"] == 3
    assert base["c"] == [9]


def test_patch_changes_reward_coefficient():
    patch = {"reward": {"reward_scales": {"collision": 8.0}}}
    cfg = _resolve(["--opponent", "scripted"], patch=patch)
    assert cfg["reward"]["reward_scales"]["collision"] == 8.0


def test_explicit_cli_arg_beats_patch():
    patch = {"schedule": {"total_transitions": 999}}
    args, explicit = parse_args(["--total-transitions", "123"])
    merged = {**_BASE_PATCH, **patch}
    # deep-merge schedule into base patch
    merged = dict(_BASE_PATCH)
    _deep_merge(merged, patch)
    cfg = build_config(args, patch=merged, explicit=explicit)
    assert cfg["schedule"]["total_transitions"] == 123


def test_unpassed_cli_flag_does_not_clobber_patch():
    patch = {"reward": {"reward_scales": {"collision": 8.0, "passing": 3.0}}}
    cfg = _resolve(["--opponent", "scripted"], patch=patch)
    assert cfg["reward"]["reward_scales"]["collision"] == 8.0
    assert cfg["reward"]["reward_scales"]["passing"] == 3.0


def test_patch_scalar_field_takes_effect_without_1v1():
    patch = {"model": {"batch_size": 4096}, "schedule": {"total_transitions": 123}}
    args, explicit = parse_args([])
    merged = dict(_BASE_PATCH)
    _deep_merge(merged, patch)
    cfg = build_config(args, patch=merged, explicit=explicit)
    assert cfg["model"]["batch_size"] == 4096
    assert cfg["schedule"]["total_transitions"] == 123
    assert args.batch_size == 4096
    assert args.total_transitions == 123


def test_default_1v1_uses_lee_reward_scales():
    cfg = _resolve(["--opponent", "scripted"])
    scales = cfg["reward"]["reward_scales"]
    assert scales["passing"] == DEFAULT_CONFIG["reward"]["reward_scales"]["passing"]
    assert scales["collision"] == DEFAULT_CONFIG["reward"]["reward_scales"]["collision"]
    assert scales["rear_end"] == DEFAULT_CONFIG["reward"]["reward_scales"]["rear_end"]


def test_training_defaults_use_10hz_1024_envs_and_3m_replay():
    args, _ = parse_args([])
    control_dt = (
        DEFAULT_CONFIG["env"]["sim_dt"] * DEFAULT_CONFIG["env"]["control_interval"]
    )
    assert control_dt == pytest.approx(0.1)
    assert args.num_envs == 1024
    assert DEFAULT_CONFIG["model"]["replay_buffer_limit"] == 3_000_000


def test_default_steering_reward_scales_and_constants():
    reward = DEFAULT_CONFIG["reward"]
    scales = reward["reward_scales"]
    assert scales["steering_change"] == 0.5
    assert scales["steering_history"] == 0.5
    assert reward["wall_contact_coefficient"] == 20.0
    assert "oob_penalty" not in scales
    assert "oob_impact" not in scales
    assert "boundary_contact" not in scales
    assert "smoothness" not in scales
    assert "overtake" not in scales
    assert "terminal_oob_skip_seconds" not in reward
    assert "boundary_contact_penalty" not in reward
    assert reward["steering_history_c_s"] == 182.883569
    cfg = _resolve([])
    assert cfg["env"]["steering_action_mode"] == "delta"
    assert "term_oob_mode" not in cfg["env"]
    assert cfg["env"]["reset_stationary_probability"] == 0.10
    assert cfg["reward"]["wall_contact_coefficient"] == 20.0
    assert "overtake" not in cfg["reward"]["reward_scales"]


def test_patch_can_disable_steering_reward_scales():
    patch = {
        "reward": {
            "reward_scales": {"steering_change": 0.0, "steering_history": 0.0}
        }
    }
    validate_config_patch(patch)
    cfg = _resolve([], patch=patch)
    assert cfg["reward"]["reward_scales"]["steering_change"] == 0.0
    assert cfg["reward"]["reward_scales"]["steering_history"] == 0.0


def test_validate_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key 'reward.nope'"):
        validate_config_patch({"reward": {"nope": 1.0}})


def test_validate_rejects_unknown_reward_scale_key():
    with pytest.raises(
        ValueError, match="unknown config key 'reward.reward_scales.bogus'"
    ):
        validate_config_patch({"reward": {"reward_scales": {"bogus": 1.0}}})


def test_validate_rejects_type_mismatch():
    with pytest.raises(ValueError, match="type mismatch for config key 'reward'"):
        validate_config_patch({"reward": 1.0})
    with pytest.raises(
        ValueError, match="type mismatch for config key 'model.batch_size'"
    ):
        validate_config_patch({"model": {"batch_size": {"nested": 1}}})


def test_fixed_opponents_defaults_empty():
    cfg = _resolve([])
    assert cfg["fixed_opponents"]["entries"] == []
    assert cfg["env"]["opponent_mix"]["scripted_weight"] == 0.5
    assert cfg["env"]["opponent_mix"]["policy_weight"] == 0.5
    assert cfg["env"]["opponent_mix"]["policy_speed_cap_range"] == [5.0, 7.0]


def test_fixed_opponents_patch_merges_entries():
    patch = {
        "fixed_opponents": {
            "entries": [{"checkpoint": "/tmp/a.pt", "weight": 2.0}]
        }
    }
    validate_config_patch(patch)
    cfg = _resolve([], patch=patch)
    assert cfg["fixed_opponents"]["entries"][0]["checkpoint"] == "/tmp/a.pt"


def test_build_config_rejects_invalid_patch():
    args, explicit = parse_args([])
    with pytest.raises(ValueError, match="unknown config key"):
        build_config(args, patch={"bogus": 1}, explicit=explicit)


def test_snapshot_contains_resolved_config_and_provenance(tmp_path):
    patch_body = {"reward": {"reward_scales": {"collision": 8.0}}}
    patch_file = tmp_path / "patch.json"
    raw = json.dumps(patch_body).encode("utf-8")
    patch_file.write_bytes(raw)

    patch, patch_meta = load_config_patch(str(patch_file))
    assert patch == patch_body
    assert patch_meta["path"] == str(patch_file.resolve())
    assert patch_meta["sha256"] == hashlib.sha256(raw).hexdigest()
    assert patch_meta["contents"] == patch_body

    args, explicit = parse_args(
        ["--opponent", "scripted", "--total-transitions", "42"]
    )
    merged = dict(_BASE_PATCH)
    _deep_merge(merged, patch)
    cfg = build_config(args, patch=merged, explicit=explicit)
    provenance = config_provenance(patch_meta, explicit)
    snapshot = build_run_snapshot("run123", tmp_path, args, cfg, provenance)

    assert snapshot["config"]["reward"]["reward_scales"]["collision"] == 8.0
    assert snapshot["config"]["schedule"]["total_transitions"] == 42
    contact = snapshot["effective_wall_contact"]
    assert contact["geometry"] == "first projected footprint edge intersects mapped wall"
    assert contact["coefficient_per_m"] == 20.0
    assert contact["control_dt_s"] == pytest.approx(0.1)
    assert contact["formula"] == "-coefficient * speed_mps"
    prov = snapshot["config_provenance"]
    assert prov["precedence"] == ["DEFAULT_CONFIG", "config_patch", "cli_args"]
    assert prov["patch"]["path"] == str(patch_file.resolve())
    assert "total_transitions" in prov["explicit_cli_args"]
    json.dumps(snapshot, default=str)


def test_provenance_without_patch_is_recorded():
    provenance = config_provenance(None, {"seed"})
    assert provenance["patch"] is None
    assert provenance["precedence"] == ["DEFAULT_CONFIG", "config_patch", "cli_args"]
    assert provenance["explicit_cli_args"] == ["seed"]


def test_asymmetric_build_config_disables_frenet_obs_latency_noise():
    """Trainer path zeros generic Frenet obs DR so the critic sees current state."""
    patch = {
        "env": {
            "domain_randomization": {
                "obs_latency_steps_range": [0, 2],
                "obs_noise_std_range": [0.01, 0.05],
            }
        }
    }
    cfg = _resolve([], patch=patch)
    dr = cfg["env"]["domain_randomization"]
    assert dr["enabled"] is True
    assert dr["obs_latency_steps_range"] == [0, 0]
    assert dr["obs_noise_std_range"] == [0.0, 0.0]
    assert cfg["obs"]["num_actor_obs"] != cfg["obs"]["num_obs"]


def test_build_config_merges_root_sensor_patch():
    patch = {
        "sensor": {
            "num_beams": 541,
            "beam_decimation": 2,
            "max_march_steps": 128,
        }
    }
    validate_config_patch(patch)
    cfg = _resolve([], patch=patch)
    assert cfg["sensor"]["num_beams"] == 541
    assert cfg["sensor"]["beam_decimation"] == 2
    assert cfg["sensor"]["max_march_steps"] == 128
    assert "sensor" not in cfg["env"]


def test_build_env_cfg_forwards_root_sensor():
    patch = {
        "sensor": {
            "num_beams": 541,
            "beam_decimation": 4,
            "max_march_steps": 64,
        }
    }
    cfg = _resolve([], patch=patch)
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": 8},
    )
    assert env_cfg["launch_strategy"] == "uniform_jittered"
    assert env_cfg["launch_strategy_data"] == {"num_cars": 8}
    assert env_cfg["track"] == cfg["env"]["track"]
    assert env_cfg["sensor"] is cfg["sensor"]
    assert env_cfg["sensor"]["num_beams"] == 541
    assert env_cfg["sensor"]["beam_decimation"] == 4
    assert env_cfg["sensor"]["max_march_steps"] == 64

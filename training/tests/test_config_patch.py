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

# Skip the filesystem track-length derivation so build_config stays hermetic.
_BASE_ARGV = ["--episode-length", "60"]


def _resolve(argv, patch=None):
    args, explicit = parse_args(_BASE_ARGV + argv)
    return build_config(args, patch=patch, explicit=explicit)


def test_deep_merge_recurses_and_replaces():
    base = {"a": {"x": 1, "y": 2}, "b": 3, "c": [1, 2]}
    out = _deep_merge(base, {"a": {"y": 20, "z": 30}, "c": [9]})
    assert out is base
    # Nested mapping is merged key-by-key; siblings are preserved.
    assert base["a"] == {"x": 1, "y": 20, "z": 30}
    assert base["b"] == 3
    # Non-mapping values (lists) are replaced wholesale.
    assert base["c"] == [9]


def test_patch_changes_reward_coefficient():
    patch = {"reward": {"reward_scales": {"collision": 8.0}}}
    cfg = _resolve(["--opponent", "scripted"], patch=patch)
    assert cfg["reward"]["reward_scales"]["collision"] == 8.0


def test_explicit_cli_arg_beats_patch():
    patch = {"reward": {"reward_scales": {"collision": 8.0}}}
    cfg = _resolve(
        ["--opponent", "scripted", "--collision-scale", "2.0"], patch=patch
    )
    assert cfg["reward"]["reward_scales"]["collision"] == 2.0


def test_unpassed_cli_flag_does_not_clobber_patch():
    # The key correctness property: an argparse default must never overwrite a
    # value the JSON patch set, when the corresponding flag was not passed.
    patch = {"reward": {"reward_scales": {"collision": 8.0, "passing": 3.0}}}
    cfg = _resolve(["--opponent", "scripted"], patch=patch)
    assert cfg["reward"]["reward_scales"]["collision"] == 8.0
    assert cfg["reward"]["reward_scales"]["passing"] == 3.0


def test_patch_scalar_field_takes_effect_without_1v1():
    patch = {"model": {"batch_size": 4096}, "schedule": {"total_transitions": 123}}
    args, explicit = parse_args(_BASE_ARGV)
    cfg = build_config(args, patch=patch, explicit=explicit)
    assert cfg["model"]["batch_size"] == 4096
    assert cfg["schedule"]["total_transitions"] == 123
    # Config-mapped scalars are mirrored back onto args (the training loop reads
    # several of them directly), so the patch actually takes effect at runtime.
    assert args.batch_size == 4096
    assert args.total_transitions == 123


def test_default_1v1_uses_maggiore_reward_scales():
    # No patch, no explicit reward flags: the resolved 1v1 config must equal the
    # DEFAULT_CONFIG Maggiore coefficients rather than legacy CLI defaults.
    cfg = _resolve(["--opponent", "scripted"])
    scales = cfg["reward"]["reward_scales"]
    assert scales["passing"] == DEFAULT_CONFIG["reward"]["reward_scales"]["passing"]
    assert scales["collision"] == DEFAULT_CONFIG["reward"]["reward_scales"]["collision"]
    assert scales["rear_end"] == DEFAULT_CONFIG["reward"]["reward_scales"]["rear_end"]


def test_validate_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key 'reward.nope'"):
        validate_config_patch({"reward": {"nope": 1.0}})


def test_patch_enables_overtake_reward_scale():
    # 'overtake' is a gated reward scale absent from DEFAULT_CONFIG (presence
    # enables the term), so a patch setting it must validate and merge through.
    patch = {"reward": {"reward_scales": {"overtake": 2.5}}}
    validate_config_patch(patch)
    cfg = _resolve(["--opponent", "scripted"], patch=patch)
    assert cfg["reward"]["reward_scales"]["overtake"] == 2.5


def test_validate_rejects_unknown_reward_scale_key():
    with pytest.raises(
        ValueError, match="unknown config key 'reward.reward_scales.bogus'"
    ):
        validate_config_patch({"reward": {"reward_scales": {"bogus": 1.0}}})


def test_validate_rejects_mapping_for_optional_reward_scale():
    with pytest.raises(
        ValueError, match="type mismatch for config key 'reward.reward_scales.overtake'"
    ):
        validate_config_patch(
            {"reward": {"reward_scales": {"overtake": {"nested": 1}}}}
        )


def test_validate_rejects_type_mismatch():
    with pytest.raises(ValueError, match="type mismatch for config key 'reward'"):
        validate_config_patch({"reward": 1.0})
    with pytest.raises(
        ValueError, match="type mismatch for config key 'model.batch_size'"
    ):
        validate_config_patch({"model": {"batch_size": {"nested": 1}}})


def test_selfplay_anchor_patch_resolves_and_cli_wins():
    # Anchor controls resolve DEFAULT_CONFIG < patch < explicit CLI.
    patch = {"selfplay": {"anchor_ckpt": "/tmp/anchor.pt", "anchor_prob": 0.4}}
    cfg = _resolve([], patch=patch)
    assert cfg["selfplay"]["anchor_ckpt"] == "/tmp/anchor.pt"
    assert cfg["selfplay"]["anchor_prob"] == 0.4

    cfg = _resolve(["--selfplay-anchor-prob", "0.75"], patch=patch)
    assert cfg["selfplay"]["anchor_prob"] == 0.75
    assert cfg["selfplay"]["anchor_ckpt"] == "/tmp/anchor.pt"


def test_selfplay_anchor_defaults_off():
    cfg = _resolve([])
    assert cfg["selfplay"]["anchor_ckpt"] is None
    assert cfg["selfplay"]["anchor_prob"] == 0.0


def test_build_config_rejects_invalid_patch():
    args, explicit = parse_args(_BASE_ARGV)
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
        _BASE_ARGV + ["--opponent", "scripted", "--collision-scale", "2.0"]
    )
    cfg = build_config(args, patch=patch, explicit=explicit)
    provenance = config_provenance(patch_meta, explicit)
    snapshot = build_run_snapshot("run123", tmp_path, args, cfg, provenance)

    assert snapshot["config"]["reward"]["reward_scales"]["collision"] == 2.0
    prov = snapshot["config_provenance"]
    assert prov["precedence"] == ["DEFAULT_CONFIG", "config_patch", "cli_args"]
    assert prov["patch"]["path"] == str(patch_file.resolve())
    assert prov["patch"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert prov["patch"]["contents"] == patch_body
    assert "collision_scale" in prov["explicit_cli_args"]
    # The snapshot must round-trip through JSON exactly as the trainer writes it.
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
    # Sensor stays at the root; do not require a duplicated env.sensor block.
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

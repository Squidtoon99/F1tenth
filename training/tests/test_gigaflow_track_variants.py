"""CPU tests for Gigaflow public-track affine variants (arXiv 2502.03349v1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from f1tenth_env.utils import (
    GIGAFLOW_AFFINE_VARIANTS,
    GIGAFLOW_BLOCKED_TRACKS,
    GIGAFLOW_PUBLIC_TRACKS,
    apply_gigaflow_affine_variant,
    gigaflow_corridor_polygon,
    gigaflow_variant_ops,
    load_gigaflow_track_arrays,
    polygon_self_intersects,
    sample_gigaflow_layout,
)
from standalone_trainer import (
    build_config,
    parse_args,
    resolve_training_scenarios,
    validate_config_patch,
)

_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_SMALL_TRACK = "Oschersleben"
_PUBLIC_NAMES = (
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "Mexico City",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "YasMarina",
    "Zandvoort",
)


def test_public_track_list_matches_pinned_names():
    assert GIGAFLOW_PUBLIC_TRACKS == _PUBLIC_NAMES
    assert len(GIGAFLOW_PUBLIC_TRACKS) == 23
    assert GIGAFLOW_BLOCKED_TRACKS.isdisjoint(GIGAFLOW_PUBLIC_TRACKS)
    assert GIGAFLOW_AFFINE_VARIANTS == (
        "identity",
        "mirror",
        "width_0.8",
        "width_1.2",
        "mirror_width_0.8",
        "mirror_width_1.2",
    )


def test_blocked_tracks_are_rejected():
    for name in sorted(GIGAFLOW_BLOCKED_TRACKS):
        with pytest.raises(ValueError, match="blocked"):
            load_gigaflow_track_arrays(name, str(_TRAINING_ROOT))


def test_each_affine_variant_on_oschersleben_is_simple_and_scaled():
    centerline, w_left, w_right = load_gigaflow_track_arrays(
        _SMALL_TRACK, str(_TRAINING_ROOT)
    )
    for variant_id in GIGAFLOW_AFFINE_VARIANTS:
        mirrored, scale = gigaflow_variant_ops(variant_id)
        point, left, right = apply_gigaflow_affine_variant(
            centerline, w_left, w_right, variant_id
        )
        if mirrored:
            assert np.allclose(point, centerline[::-1])
            assert np.allclose(left, w_right[::-1] * scale)
            assert np.allclose(right, w_left[::-1] * scale)
        else:
            assert np.allclose(point, centerline)
            assert np.allclose(left, w_left * scale)
            assert np.allclose(right, w_right * scale)
        width_scale = float(
            np.mean(np.concatenate([left / w_right[::-1], right / w_left[::-1]]))
            if mirrored
            else np.mean(np.concatenate([left / w_left, right / w_right]))
        )
        assert width_scale == pytest.approx(scale, rel=1e-5, abs=1e-6)
        ring = gigaflow_corridor_polygon(point, left, right)
        assert not polygon_self_intersects(ring), variant_id


def test_sample_gigaflow_layout_is_seeded_and_public():
    first = sample_gigaflow_layout(np.random.default_rng(0))
    second = sample_gigaflow_layout(np.random.default_rng(0))
    assert first == second
    rng = np.random.default_rng(0)
    for _ in range(64):
        track, variant = sample_gigaflow_layout(rng)
        assert track in GIGAFLOW_PUBLIC_TRACKS
        assert track not in GIGAFLOW_BLOCKED_TRACKS
        assert variant in GIGAFLOW_AFFINE_VARIANTS


def test_gigaflow_v2_config_uses_v2_plant_and_sampling():
    path = _TRAINING_ROOT / "configs" / "gigaflow_v2.json"
    patch = json.loads(path.read_text())
    validate_config_patch(patch)
    args, explicit = parse_args([])
    cfg = build_config(args, patch=patch, explicit=explicit)
    env = cfg["env"]
    assert env["gigaflow_track_sampling"] is True
    assert env["f_drive_max"] == pytest.approx(21.46877492171339)
    assert env["f_brake_max"] == pytest.approx(9.139188067139935)
    assert env["steer_gain"] == pytest.approx(1.0511988660159353)
    assert env["warp_sim"]["c_coast"] == pytest.approx(4.804781294740983)
    assert env["warp_sim"]["lat_stiffness_scale"] == pytest.approx(5.329306089973615)
    assert cfg["sensor"]["lidar_map_yaml"] is None
    names = [scenario["name"] for scenario in cfg["scenarios"]]
    assert names == ["solo_flying", "solo_standing", "selfplay"]
    assert env["gigaflow_conditioning"] is True
    assert cfg["obs"]["num_actor_obs"] == 1107
    assert cfg["obs"]["num_obs"] == 402
    assert cfg["obs"]["enable_opponent_obs"] is False
    assert cfg["ppo"]["value_clip"] == 0.0
    assert cfg["ppo"]["actor_lr"] == pytest.approx(5e-4)
    assert cfg["ppo"]["lr_schedule"] == "cosine"
    assert cfg["ppo"]["gamma"] == pytest.approx(0.999)
    assert cfg["ppo"]["advantage_filter_mode"] == "running_max"
    assert cfg["ppo"]["orthogonal_init"] is True
    assert "tal" not in cfg["reward"]["reward_scales"]
    assert cfg["scenarios"][2]["env"]["opponent_strategy"] == "policy"
    blob = path.read_text()
    for blocked in GIGAFLOW_BLOCKED_TRACKS:
        assert blocked not in blob
        assert blocked not in json.dumps(cfg["scenarios"])


def test_self_play_cli_keeps_solo_scenarios_without_opponents():
    path = _TRAINING_ROOT / "configs" / "gigaflow_v2.json"
    patch = json.loads(path.read_text())
    args, explicit = parse_args(["--self-play"])
    args.num_envs = 8
    cfg = build_config(args, patch=patch, explicit=explicit)
    assert cfg["env"]["opponent_strategy"] == "policy"
    resolved = resolve_training_scenarios(cfg, 8, str(_TRAINING_ROOT))
    by_name = {
        spec["name"]: spec["cfg"]["env"]["opponent_strategy"] for spec in resolved
    }
    assert by_name["solo_flying"] is None
    assert by_name["solo_standing"] is None
    assert by_name["selfplay"] == "policy"

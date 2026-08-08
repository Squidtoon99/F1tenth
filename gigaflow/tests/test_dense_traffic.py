"""Focused tests for dense-traffic max-agents experiment helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gigaflow_f1tenth.config import estimate_memory_bytes, load_config
from gigaflow_f1tenth.dense_traffic import (
    DenseTrafficMetrics,
    H100_REFERENCE_CONFIG_PATH,
    estimate_h100_vram_gib,
    finalize_gates,
    measure_variant,
    promotion_gates,
    recommend_first_variant,
    slot_matched_worlds,
)
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import (
    build_track_arrays,
    load_atlas,
    make_synthetic_oval_table,
    pack_built_tracks,
    rebuild_atlas_capacity,
    sample_active_counts,
)

ROOT = Path(__file__).resolve().parents[1]


def _metrics(**overrides) -> DenseTrafficMetrics:
    base = dict(
        max_agents=8,
        realized_cars_per_world_mean=5.0,
        realized_cars_per_dense_world_mean=6.0,
        dense_world_frac=0.3,
        solo_world_frac=0.05,
        close_pair_rate=0.10,
        mean_nearest_opponent_m=1.5,
        overtake_proxy=1.0,
        collision_events=2.0,
        oob_events=1.0,
        collision_per_km=1.0,
        oob_per_km=0.5,
        lidar_occlusion_proxy=0.20,
        track_visibility_proxy=0.40,
        spawn_requested=40.0,
        spawn_realized=40.0,
        spawn_rejects=1.0,
        spawn_reject_rate=0.025,
        sim_world_ticks_per_s=100.0,
        learner_updates_per_s=1.0,
        learner_transitions_per_s=1000.0,
        vram_est_gib=63.9,
        startup_estimate_bytes=1,
        ppo_policy_loss=0.1,
        ppo_value_loss=0.2,
        ppo_entropy=0.5,
        ppo_approx_kl=0.01,
        ppo_finite=True,
        head_to_head_cars=2.0,
        extras={"vram_slot_matched_gib": 63.9, "slot_matched_worlds": 1024.0},
    )
    extras = dict(base["extras"])
    extras.update(overrides.pop("extras", {}))
    base.update(overrides)
    base["extras"] = extras
    return DenseTrafficMetrics(**base)


def test_cpu_experiment_configs_load_and_preserve_mix():
    for n in (8, 10, 12):
        cfg = load_config(
            ROOT / f"configs/experiments/dense_traffic_cpu_max{n}.yaml"
        )
        assert cfg.worlds.max_agents_per_world == n
        assert cfg.worlds.solo_world_fraction == pytest.approx(0.05)
        assert cfg.worlds.density_bins == ("sparse", "medium", "dense")
        assert cfg.worlds.device == "cpu"
        assert "solo" in cfg.evaluation.suite
        assert "head_to_head" in cfg.evaluation.suite


def test_h100_experiment_configs_do_not_change_production_default():
    prod = load_config(ROOT / "configs/production_h100.yaml")
    default = load_config(ROOT / "configs/default.yaml")
    assert prod.worlds.max_agents_per_world == 8
    assert default.worlds.max_agents_per_world == 8
    for n in (8, 10, 12):
        cfg = load_config(
            ROOT / f"configs/experiments/dense_traffic_h100_max{n}.yaml"
        )
        assert cfg.worlds.max_agents_per_world == n
        assert cfg.worlds.solo_world_fraction == prod.worlds.solo_world_fraction
        assert cfg.worlds.density_bins == prod.worlds.density_bins


def test_rebuild_atlas_capacity_scales_and_refuses_inplace(tmp_path: Path):
    table = make_synthetic_oval_table(radius=20.0, n=120, width=2.0)
    built = build_track_arrays(
        table,
        name="oval",
        source_sha="d" * 64,
        lut_resolution=0.5,
        edt_resolution=0.25,
        max_agents=8,
        license_id="local-fixture",
        optional_local=True,
    )
    atlas = pack_built_tracks([built])
    src = tmp_path / "src"
    src.mkdir()
    from gigaflow_f1tenth.tracks import _save_atlas_npz, _write_manifest

    _write_manifest(
        src / "manifest.json",
        atlas.manifest(),
        pin={"commit": "x", "ref": "y", "repo": "z", "license": "local"},
        sampling="stratified_shuffle",
    )
    _save_atlas_npz(src / "atlas.npz", atlas.view())

    with pytest.raises(Exception, match="in-place"):
        rebuild_atlas_capacity(src, src, max_agents=10)

    dst10 = tmp_path / "max10"
    out10 = rebuild_atlas_capacity(src, dst10, max_agents=10)
    dst12 = tmp_path / "max12"
    out12 = rebuild_atlas_capacity(src, dst12, max_agents=12)
    assert int(out10.view().capacity[0]) >= int(atlas.view().capacity[0])
    assert int(out12.view().capacity[0]) >= int(out10.view().capacity[0])
    assert int(out12.view().capacity[0]) <= 12
    # Source untouched.
    src_reload = load_atlas(str(src), device="cpu")
    assert int(src_reload.view().capacity[0]) == int(atlas.view().capacity[0])


def test_solo_and_pair_distributions_stable_when_max_agents_rises():
    track_ids = np.zeros(2000, dtype=np.int32)
    solos = []
    for max_a, cap in ((8, 8), (10, 10), (12, 12)):
        capacities = np.asarray([cap], dtype=np.int32)
        counts = sample_active_counts(
            track_ids,
            capacities,
            density_bins=("sparse", "medium", "dense"),
            solo_world_fraction=0.05,
            max_agents=max_a,
            seed=0,
        )
        solos.append(float(np.mean(counts == 1)))
        pair = sample_active_counts(
            track_ids,
            capacities,
            density_bins=("pair",),
            solo_world_fraction=0.0,
            max_agents=min(2, max_a),
            seed=0,
        )
        assert set(pair.tolist()) == {2}
    # Forced solo_world_fraction is identical; realized singles stay in band.
    assert min(solos) >= 0.05
    assert max(solos) - min(solos) <= 0.08


def test_promotion_gates_require_interaction_lift_without_blowups():
    baseline = _metrics(max_agents=8)
    good = _metrics(
        max_agents=10,
        realized_cars_per_dense_world_mean=7.5,
        close_pair_rate=0.15,
        spawn_reject_rate=0.03,
        lidar_occlusion_proxy=0.25,
        collision_per_km=1.2,
        oob_per_km=0.6,
        sim_world_ticks_per_s=90.0,
        vram_est_gib=79.0,
        extras={"vram_slot_matched_gib": 63.0, "slot_matched_worlds": 832.0},
    )
    gates = finalize_gates(promotion_gates(baseline, good))
    assert gates["all_pass"]

    bad_occ = _metrics(
        max_agents=12,
        realized_cars_per_dense_world_mean=9.0,
        close_pair_rate=0.20,
        lidar_occlusion_proxy=0.50,  # >1.8x of 0.20
        vram_est_gib=70.0,
    )
    g_bad = finalize_gates(promotion_gates(baseline, bad_occ))
    assert g_bad["occlusion_bounded"] is False
    assert g_bad["all_pass"] is False


def test_vram_estimate_delegates_to_the_validated_memory_estimator():
    """dense_traffic must not keep a second memory model that can drift.

    The corrected estimator concluded 1024 worlds x 8 agents does not fit an
    H100 (production_h100.yaml runs 448 instead); the delegated estimate must
    agree with that conclusion rather than the old anchor-based linear model.
    """
    from dataclasses import replace as dc_replace

    ref = load_config(H100_REFERENCE_CONFIG_PATH)
    cfg_1024x8 = dc_replace(
        ref, worlds=dc_replace(ref.worlds, num_worlds=1024, max_agents_per_world=8)
    )
    expected_1024x8 = estimate_memory_bytes(cfg_1024x8) / 1024**3
    assert estimate_h100_vram_gib(num_worlds=1024, max_agents=8) == pytest.approx(
        expected_1024x8
    )
    assert estimate_h100_vram_gib(num_worlds=1024, max_agents=8) > 72.0
    assert estimate_h100_vram_gib(num_worlds=448, max_agents=8) < 72.0
    assert estimate_h100_vram_gib(
        num_worlds=1024, max_agents=10
    ) > estimate_h100_vram_gib(num_worlds=1024, max_agents=8)
    assert slot_matched_worlds(10) == 832
    assert slot_matched_worlds(12) == 704


def test_recommend_try_max10_first():
    results = {
        8: _metrics(max_agents=8),
        10: _metrics(max_agents=10, realized_cars_per_dense_world_mean=7.0),
        12: _metrics(max_agents=12, vram_est_gib=90.0),
    }
    gates = {
        10: finalize_gates(
            promotion_gates(results[8], results[10])
        ),
        12: {"all_pass": False, "vram_under_budget": False},
    }
    rec = recommend_first_variant(results, gates)
    assert rec["try_first"] == 10


def test_measure_variant_cpu_smoke():
    cfg = load_config(ROOT / "configs/experiments/dense_traffic_cpu_max8.yaml")
    atlas = make_synthetic_oval_atlas(max_agents=8)
    m = measure_variant(
        cfg, atlas=atlas, device="cpu", sim_steps=8, learner_updates=1, seed=0
    )
    assert m.max_agents == 8
    assert m.realized_cars_per_dense_world_mean >= 1.0
    assert m.head_to_head_cars == pytest.approx(2.0)
    assert m.ppo_finite
    assert 0.0 <= m.solo_world_frac <= 1.0
    with pytest.raises(RuntimeError, match="refuses CUDA"):
        measure_variant(cfg, atlas=atlas, device="cuda", sim_steps=1, learner_updates=1)

"""Focused N-agent simulator correctness tests (CPU Warp + CPU reference)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.config import config_from_dict, load_config
from gigaflow_f1tenth.evaluation import _suite_world_overrides
from gigaflow_f1tenth.kernels import (
    build_simulator,
    simulator_interface_gaps,
    slot_index,
    world_slot_layout,
)
from gigaflow_f1tenth.sim.contact import resolve_pair_contact_numpy
from gigaflow_f1tenth.sim.geometry import (
    build_sim_geometry_from_atlas,
    make_synthetic_oval_atlas,
    make_two_track_atlas,
    project_frenet_numpy,
)
from gigaflow_f1tenth.tracks import compute_track_capacity
from gigaflow_f1tenth.sim.reference_cpu import (
    overlapping_boxes_should_contact,
    ray_hits_box_ahead,
)
from gigaflow_f1tenth.sim.spawn import assign_world_tracks, mix_seed, sample_active_counts

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(SMOKE)


@pytest.fixture(scope="module")
def atlas():
    return make_synthetic_oval_atlas()


@pytest.fixture(scope="module")
def sim(cfg, atlas):
    return build_simulator(cfg, atlas, "cpu")


def test_interface_gaps_resolved():
    gaps = simulator_interface_gaps()
    assert gaps == ()


def test_capacity_and_spawn_deterministic(cfg, atlas):
    geom = build_sim_geometry_from_atlas(atlas)
    assert geom.capacity[0] >= 1
    tracks_a = assign_world_tracks(cfg, geom, seed=0)
    tracks_b = assign_world_tracks(cfg, geom, seed=0)
    assert np.array_equal(tracks_a, tracks_b)
    counts_a = sample_active_counts(cfg, geom, tracks_a, seed=1)
    counts_b = sample_active_counts(cfg, geom, tracks_a, seed=1)
    assert np.array_equal(counts_a, counts_b)
    assert mix_seed(0, 1, 2, 3) == mix_seed(0, 1, 2, 3)


def test_frenet_projection_on_centerline(cfg, atlas):
    geom = build_sim_geometry_from_atlas(atlas)
    a, b = int(geom.offsets[0]), int(geom.offsets[1])
    p = geom.centerline_xy[a]
    fr = project_frenet_numpy(
        p,
        0,
        geom.centerline_xy[a:b],
        geom.tangents_xy[a:b],
        geom.normals_xy[a:b],
        geom.widths_rl[a:b],
        geom.cum_length[a:b],
        geom.segment_length[a:b],
    )
    assert abs(fr["ey"]) < 0.05
    assert fr["boundary_distance"] > 0.5


def test_cpu_contact_reference_overlap_and_gap(cfg):
    L, W = cfg.agents.car_length_m, cfg.agents.car_width_m
    assert overlapping_boxes_should_contact(0.1, L, W)
    assert not overlapping_boxes_should_contact(2.0, L, W)
    hit = resolve_pair_contact_numpy(
        0.0, 0.0, 0.0, 0.2, 0.0, 0.0, L, W, avx=2.0, avy=0.0, bvx=0.0, bvy=0.0
    )
    assert hit["contact"] == 1
    assert float(hit["closing_speed"]) >= 0.0


def test_cpu_ray_obb_reference():
    dist = ray_hits_box_ahead()
    assert 1.5 < dist < 2.0


def test_simulator_builds_soa_and_masks(sim, cfg):
    st = sim.state()
    layout = world_slot_layout(cfg)
    assert st.layout.num_slots == layout.num_slots
    assert int(st.active.sum().item()) >= 1
    assert st.arrays["sensor_obs"].shape[-1] == 1097
    assert slot_index(1, 1, 2) == 3


def test_step_returns_contract_and_finite(sim, cfg):
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), dtype=torch.float32)
    actions[:, 0] = 0.2
    out = sim.step(actions)
    for key in ("rewards", "done", "timeout", "reset_mask", "sensor_obs", "critic_state"):
        assert key in out
    assert out["sensor_obs"].shape == (n, 1097)
    assert torch.isfinite(out["sensor_obs"]).all()
    assert torch.isfinite(out["rewards"]).all()
    # critic_state must carry track_id: the track-preview sampler needs it to
    # know which atlas track each slot's frenet_s/pose belong to.
    assert out["critic_state"]["track_id"].shape == (n,)


def test_cross_world_isolation(cfg):
    atlas = make_two_track_atlas()
    # Force two worlds, two cars each.
    from dataclasses import replace

    worlds = replace(cfg.worlds, num_worlds=2, max_agents_per_world=2)
    cfg2 = replace(cfg, worlds=worlds)
    sim = build_simulator(cfg2, atlas, "cpu")
    st = sim.state()
    n = st.layout.num_slots
    out = sim.step(torch.zeros((n, 2)))
    counterpart = sim.buffers.torch_arrays.contact_counterpart
    world_id = sim.buffers.torch_arrays.world_id
    contact = sim.buffers.torch_arrays.contact
    for i in range(n):
        if int(contact[i].item()) == 0:
            continue
        j = int(counterpart[i].item())
        if j < 0:
            continue
        assert int(world_id[i].item()) == int(world_id[j].item())
    overflow = out["broadphase_overflow"]
    overflow_i = int(overflow.item()) if hasattr(overflow, "item") else int(overflow)
    assert overflow_i == 0


def test_async_reset_mask_clears_only_targets(sim, cfg):
    n = world_slot_layout(cfg).num_slots
    mask = np.zeros(n, dtype=np.uint8)
    mask[0] = 1
    before = sim.state().arrays["episode_id"].clone()
    sim.reset_agents(mask, seed=123)
    after = sim.state().arrays["episode_id"]
    assert int(after[0].item()) == int(before[0].item()) + 1
    if n > 1 and int(sim.state().active[1].item()) == 1:
        assert int(after[1].item()) == int(before[1].item())


def test_lidar_occlusion_prefers_nearer_car(cfg, atlas):
    from dataclasses import replace

    worlds = replace(cfg.worlds, num_worlds=1, max_agents_per_world=2)
    agents = replace(cfg.agents, async_respawn=False)
    cfg2 = replace(cfg, worlds=worlds, agents=agents)
    sim = build_simulator(cfg2, atlas, "cpu")
    t = sim.buffers.torch_arrays
    # Place both cars on the synthetic oval centerline, nose-to-tail.
    p0 = atlas.centerline_xy[0]
    p1 = atlas.centerline_xy[4]
    yaw = float(
        np.arctan2(
            atlas.tangents_xy[0, 1],
            atlas.tangents_xy[0, 0],
        )
    )
    t.active[:] = 0
    t.trainable[:] = 0
    t.active[0] = 1
    t.active[1] = 1
    t.trainable[0] = 1
    t.trainable[1] = 1
    t.track_id[:] = 0
    t.x[0], t.y[0], t.yaw[0] = float(p0[0]), float(p0[1]), yaw
    t.x[1], t.y[1], t.yaw[1] = float(p1[0]), float(p1[1]), yaw
    t.vx[:] = 0.0
    t.frenet_segment[:] = 0
    t.lidar_range_noise_std[:] = 0.0
    t.lidar_dropout_prob[:] = 0.0
    t.lidar_far_dropout_prob[:] = 0.0
    t.lidar_sector_width[:] = 0
    t.lidar_angle_bias[:] = 0.0
    t.lidar_extrinsic_x[:] = 0.0
    t.lidar_extrinsic_y[:] = 0.0
    t.lidar_extrinsic_yaw[:] = 0.0
    out = sim.step(torch.zeros((2, 2)))
    center = out["sensor_obs"][0, 540].item()
    assert center < 8.0


def test_estimate_capacity_scales_with_length():
    wr = np.full(8, 1.0)
    wl = np.full(8, 1.0)
    short = compute_track_capacity(20.0, wr, wl, car_length_m=0.568, car_width_m=0.296)
    long = compute_track_capacity(200.0, wr, wl, car_length_m=0.568, car_width_m=0.296)
    assert long >= short


def test_head_to_head_suite_spawns_exactly_two_active(cfg):
    raw = _suite_world_overrides("head_to_head", cfg)
    assert raw["worlds"]["density_bins"] == ["pair"]
    assert int(raw["worlds"]["max_agents_per_world"]) == 2
    assert float(raw["worlds"]["solo_world_fraction"]) == 0.0
    raw["worlds"]["num_worlds"] = 2
    raw["evaluation"]["num_worlds"] = 2
    suite_cfg = config_from_dict(raw)
    atlas = make_two_track_atlas(max_agents=2)
    assert int(np.min(atlas.capacity)) >= 2
    sim = build_simulator(suite_cfg, atlas, "cpu", sync_no_respawn=True)
    for seed in (0, 1, 3, 11, 29):
        sim.reset_all(seed)
        active = sim.state().active.detach().cpu().numpy().astype(np.int32)
        assert active.shape[0] == 4
        assert int(active.sum()) == 4
        assert int(active[:2].sum()) == 2
        assert int(active[2:].sum()) == 2

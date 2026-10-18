"""Multi-track Frenet lifecycle regressions (async respawn / reward geom / pack)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from gigaflow_f1tenth.buffers import STATE_INDEX, pack_compact_agent_state
from gigaflow_f1tenth.config import AGENT_STATE_DIM, load_config
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.rewards import (
    compute_rewards_from_sim_arrays,
    deployment_style,
    reward_boundary,
)
from gigaflow_f1tenth.sim.geometry import (
    FRENET_WINDOW,
    build_sim_geometry_from_atlas,
    make_offset_mismatch_atlas,
    make_two_track_atlas,
    project_frenet_numpy,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _two_world_cfg(*, async_respawn: bool = True):
    cfg = load_config(SMOKE)
    worlds = replace(cfg.worlds, num_worlds=2, max_agents_per_world=2)
    agents = replace(cfg.agents, async_respawn=async_respawn)
    return replace(cfg, worlds=worlds, agents=agents)


def _place_on_track(sim, geom, *, tid: int, local_seg: int, slot: int = 0) -> None:
    t = sim.buffers.torch_arrays
    a = int(geom.offsets[tid])
    b = int(geom.offsets[tid + 1])
    count = b - a
    assert 0 <= local_seg < count
    glob = a + local_seg
    p = geom.centerline_xy[glob]
    yaw = float(
        np.arctan2(geom.tangents_xy[glob, 1], geom.tangents_xy[glob, 0])
    )
    t.active[:] = 0
    t.trainable[:] = 0
    t.reset_mask[:] = 0
    t.done[:] = 0
    t.active[slot] = 1
    t.trainable[slot] = 1
    t.track_id[slot] = tid
    t.x[slot] = float(p[0])
    t.y[slot] = float(p[1])
    t.yaw[slot] = yaw
    t.vx[slot] = 1.0
    t.vy[slot] = 0.0
    t.yaw_rate[slot] = 0.0
    t.frenet_segment[slot] = local_seg
    t.frenet_s[slot] = float(geom.cum_length[glob])
    t.prev_s[slot] = float(geom.cum_length[glob])
    t.frenet_ey[slot] = 0.0
    t.boundary_distance[slot] = 1.0
    t.wall_contact[slot] = 0
    t.contact[slot] = 0
    t.stalled_steps[slot] = 0


def test_offset_mismatch_atlas_exposes_window_gap():
    atlas = make_offset_mismatch_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    a = int(geom.offsets[1])
    count = int(geom.offsets[2] - geom.offsets[1])
    assert a > 0
    assert count > 2 * FRENET_WINDOW + 1
    phase = a % count
    gap = min(phase, count - phase)
    assert gap > FRENET_WINDOW


def test_async_respawn_seeds_project_window_with_local_segment():
    """Warp async respawn must seed project_window with track-local indices."""
    cfg = _two_world_cfg(async_respawn=True)
    atlas = make_offset_mismatch_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu")
    t = sim.buffers.torch_arrays
    tid = 1
    a = int(geom.offsets[tid])
    b = int(geom.offsets[tid + 1])
    count = b - a

    mismatches = 0
    for ep in range(48):
        t.active[:] = 0
        t.reset_mask[:] = 0
        t.track_id[0] = tid
        t.active[0] = 1
        t.reset_mask[0] = 1
        t.episode_id[0] = ep
        sim._launch_async_respawn(seed=4242)
        seg = int(t.frenet_segment[0].item())
        assert 0 <= seg < count
        xy = np.array([float(t.x[0]), float(t.y[0])], dtype=np.float32)
        fr = project_frenet_numpy(
            xy,
            seg,
            geom.centerline_xy[a:b],
            geom.tangents_xy[a:b],
            geom.normals_xy[a:b],
            geom.widths_rl[a:b],
            geom.cum_length[a:b],
            geom.segment_length[a:b],
        )
        # Nearest centerline vertex as independent reference.
        d2 = np.sum((geom.centerline_xy[a:b] - xy) ** 2, axis=1)
        nearest = int(np.argmin(d2))
        fr_n = project_frenet_numpy(
            xy,
            nearest,
            geom.centerline_xy[a:b],
            geom.tangents_xy[a:b],
            geom.normals_xy[a:b],
            geom.widths_rl[a:b],
            geom.cum_length[a:b],
            geom.segment_length[a:b],
        )
        err_s = abs(float(t.frenet_s[0]) - float(fr_n["s"]))
        err_ey = abs(float(t.frenet_ey[0]) - float(fr_n["ey"]))
        seg_gap = min((seg - int(fr_n["segment"])) % count, (int(fr_n["segment"]) - seg) % count)
        if err_s > 0.05 or err_ey > 0.05 or seg_gap > 2:
            mismatches += 1
        assert abs(float(t.frenet_s[0]) - float(fr["s"])) < 1e-4
        assert abs(float(t.frenet_ey[0]) - float(fr["ey"])) < 1e-4
    assert mismatches == 0


def test_reward_geometry_uses_local_segment_on_nonzero_offset():
    cfg = _two_world_cfg(async_respawn=False)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu")
    tid = 1
    local_seg = 10
    a = int(geom.offsets[tid])
    glob = a + local_seg
    _place_on_track(sim, geom, tid=tid, local_seg=local_seg)
    true_txy = geom.tangents_xy[glob]
    true_yaw = float(np.arctan2(true_txy[1], true_txy[0]))
    true_hw = 0.5 * (float(geom.widths_rl[glob, 0]) + float(geom.widths_rl[glob, 1]))
    # Pre-fix bug clamped local<offset to atlas index `a` (track start).
    clamp_txy = geom.tangents_xy[a]
    clamp_yaw = float(np.arctan2(clamp_txy[1], clamp_txy[0]))
    assert abs(true_yaw - clamp_yaw) > 0.1

    sim._launch_reward_kernels()
    got_yaw = float(wp.to_torch(sim._wp_tangent_yaw)[0])
    got_hw = float(wp.to_torch(sim._wp_half_width)[0])
    assert got_yaw == pytest.approx(true_yaw, abs=1e-5)
    assert got_hw == pytest.approx(true_hw, abs=1e-5)


def test_compact_state_roundtrip_preserves_frenet_segment_and_step_parity():
    assert STATE_INDEX["frenet_segment"] == AGENT_STATE_DIM - 1
    cfg = _two_world_cfg(async_respawn=False)
    atlas = make_offset_mismatch_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim_a = build_simulator(cfg, atlas, "cpu")
    sim_b = build_simulator(cfg, atlas, "cpu")
    tid = 1
    local_seg = 20
    _place_on_track(sim_a, geom, tid=tid, local_seg=local_seg)
    _place_on_track(sim_b, geom, tid=tid, local_seg=local_seg)
    # Nudge along-track so the next projection uses the seed window.
    tang = geom.tangents_xy[int(geom.offsets[tid]) + local_seg]
    sim_a.buffers.torch_arrays.x[0] += 0.02 * float(tang[0])
    sim_a.buffers.torch_arrays.y[0] += 0.02 * float(tang[1])
    sim_b.buffers.torch_arrays.x[0] = float(sim_a.buffers.torch_arrays.x[0])
    sim_b.buffers.torch_arrays.y[0] = float(sim_a.buffers.torch_arrays.y[0])

    packed = sim_a.pack_state()
    assert packed.shape[-1] == AGENT_STATE_DIM
    assert float(packed[0, STATE_INDEX["frenet_segment"]]) == float(local_seg)

    # Corrupt live segment, then restore — must recover projection lock.
    sim_a.buffers.torch_arrays.frenet_segment[0] = 0
    sim_a.restore_state(packed)
    assert int(sim_a.buffers.torch_arrays.frenet_segment[0].item()) == local_seg

    n = sim_a.buffers.torch_arrays.x.shape[0]
    actions = torch.zeros((n, 2), dtype=torch.float32)
    actions[:, 0] = 0.15
    out_a = sim_a.step(actions)
    out_b = sim_b.step(actions)
    ta = sim_a.buffers.torch_arrays
    tb = sim_b.buffers.torch_arrays
    assert int(ta.frenet_segment[0].item()) == int(tb.frenet_segment[0].item())
    assert float(ta.frenet_s[0]) == pytest.approx(float(tb.frenet_s[0]), abs=1e-5)
    assert float(ta.frenet_ey[0]) == pytest.approx(float(tb.frenet_ey[0]), abs=1e-5)
    assert torch.isfinite(out_a["rewards"]).all()
    assert torch.isfinite(out_b["rewards"]).all()
    assert float(out_a["rewards"][0]) == pytest.approx(float(out_b["rewards"][0]), abs=1e-5)


def test_offroad_boundary_indicator_and_conservative_full_oob_terminal():
    """Disclosed off-road: geometric wall flag * alpha_boundary; keep full-OOB done."""
    cfg = _two_world_cfg(async_respawn=False)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    # Eval-style: no respawn so `done` remains observable after the step.
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=True)
    t = sim.buffers.torch_arrays
    tid = 1
    local_seg = 8
    a = int(geom.offsets[tid])
    glob = a + local_seg
    half_w = 0.5 * float(cfg.agents.car_width_m)
    nxy = geom.normals_xy[glob]
    # Soft boundary: footprint crosses wall, center still inside corridor.
    _place_on_track(sim, geom, tid=tid, local_seg=local_seg)
    lateral = float(geom.widths_rl[glob, 1]) - 0.5 * half_w
    t.x[0] = float(geom.centerline_xy[glob, 0] + lateral * nxy[0])
    t.y[0] = float(geom.centerline_xy[glob, 1] + lateral * nxy[1])
    t.vx[0] = 0.5  # below catastrophic speed
    sim._torch_alpha["alpha_boundary"][0] = 2.5
    sim._style_tensors = {
        name: sim._torch_alpha[name] for name in sim._style_alpha_names
    }

    n = t.x.shape[0]
    out = sim.step(torch.zeros((n, 2)))
    assert int(t.wall_contact[0].item()) == 1
    assert not bool(out["done"][0].item()), "soft wall alone must not terminate"
    bnd = float(out["reward_terms"]["boundary"][0])
    assert bnd == pytest.approx(-2.5, abs=1e-4)
    assert reward_boundary(torch.tensor([1.0]), torch.tensor([2.5])).item() == pytest.approx(
        -2.5
    )

    # Full OOB: center past corridor by > half-width → conservative terminal.
    _place_on_track(sim, geom, tid=tid, local_seg=local_seg)
    oob = float(geom.widths_rl[glob, 1]) + half_w + 0.05
    t.x[0] = float(geom.centerline_xy[glob, 0] + oob * nxy[0])
    t.y[0] = float(geom.centerline_xy[glob, 1] + oob * nxy[1])
    t.vx[0] = 1.0
    t.active[0] = 1
    t.trainable[0] = 1
    sim._torch_alpha["alpha_boundary"][0] = 3.0
    out = sim.step(torch.zeros((n, 2)))
    assert bool(out["done"][0].item()), "full OOB must terminate (conservative)"
    assert int(t.active[0].item()) == 0, "eval path deactivates without respawn"
    terms = out["reward_terms"]
    total = float(out["rewards"][0])
    assert np.isfinite(total)
    assert -100.0 - 1e-6 <= total <= 100.0 + 1e-6
    for key in ("boundary", "lane_center", "progress", "total"):
        assert torch.isfinite(terms[key]).all()
    assert float(terms["boundary"][0]) <= 0.0

    # Training async path: same full-OOB pose schedules respawn (reset_mask).
    sim_async = build_simulator(
        _two_world_cfg(async_respawn=True), atlas, "cpu", sync_no_respawn=False
    )
    ta = sim_async.buffers.torch_arrays
    _place_on_track(sim_async, geom, tid=tid, local_seg=local_seg)
    ta.x[0] = float(geom.centerline_xy[glob, 0] + oob * nxy[0])
    ta.y[0] = float(geom.centerline_xy[glob, 1] + oob * nxy[1])
    ta.vx[0] = 1.0
    sim_async._torch_alpha["alpha_boundary"][0] = 3.0
    sim_async._style_tensors = {
        name: sim_async._torch_alpha[name] for name in sim_async._style_alpha_names
    }
    out_a = sim_async.step(torch.zeros((ta.x.shape[0], 2)))
    # Transition-time done must survive async respawn clearing of slot flags.
    assert bool(out_a["done"][0].item()), "async OOB must expose done for PPO/GAE"
    assert not bool(out_a["timeout"][0].item())
    assert bool(out_a["reset_mask"][0].item())
    assert int(ta.active[0].item()) == 1, "async path respawns in-place"
    assert int(ta.done[0].item()) == 0, "live slot done cleared after respawn"
    assert np.isfinite(float(out_a["rewards"][0]))
    assert abs(float(out_a["rewards"][0])) <= 100.0 + 1e-6


def test_passing_gate_cleared_only_for_respawned_rows():
    """Fail-before: any respawn wiped every world's latched passing gate."""
    cfg = _two_world_cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    sim = build_simulator(cfg, atlas, "cpu")
    t = sim.buffers.torch_arrays
    gate = sim._passing_gate
    gate.fill_(1)
    t.active[:] = 1
    t.reset_mask[:] = 0
    t.reset_mask[0] = 1

    sim._launch_async_respawn(seed=99)

    assert int(gate[0].sum().item()) == 0
    # Same world, untouched slot keeps its latch; so does the other world.
    assert int(gate[1].sum().item()) == gate.shape[1]
    assert int(gate[2:].sum().item()) == gate[2:].numel()


def test_passing_gate_and_totals_match_torch_mirror_over_steps():
    cfg = _two_world_cfg(async_respawn=False)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    style = deployment_style(cfg.evaluation.conservative_deployment_style)
    sim.apply_styles([style for _ in range(n)])
    for world in range(2):
        for local, seg in enumerate((4, 12)):
            slot = world * 2 + local
            _place_on_track(sim, geom, tid=world, local_seg=seg, slot=slot)
    # _place_on_track deactivates the other slots; re-arm the full grid.
    t.active[:] = 1
    t.trainable[:] = 1

    dt = 1.0 / float(cfg.agents.control_hz)
    actions = torch.zeros((n, 2))
    actions[:, 0] = 0.1
    prev_gate = None
    for _ in range(4):
        out = sim.step(actions)
        assert not bool(out["done"].any().item())
        warp_gate = wp.to_torch(sim._wp_gate).clone().bool()
        # The Torch mirror reads progress from arrays.rewards, which the Warp
        # kernel has already overwritten with the composed total.
        saved = t.rewards.clone()
        t.rewards.copy_(out["reward_terms"]["progress"])
        terms, gate = compute_rewards_from_sim_arrays(
            t,
            styles=sim._torch_alpha,
            track_length=wp.to_torch(sim._wp_track_length_slot),
            half_width=wp.to_torch(sim._wp_half_width),
            tangent_yaw=wp.to_torch(sim._wp_tangent_yaw),
            dt=dt,
            prev_passing_gate=prev_gate,
            max_agents_per_world=cfg.worlds.max_agents_per_world,
        )
        t.rewards.copy_(saved)
        assert torch.equal(gate, warp_gate)
        assert torch.allclose(
            terms.passing, out["reward_terms"]["passing"], atol=1e-5
        )
        assert torch.allclose(terms.total, out["reward_terms"]["total"], atol=1e-5)
        prev_gate = gate


def test_pack_includes_frenet_segment_channel():
    cfg = _two_world_cfg()
    sim = build_simulator(cfg, make_two_track_atlas(), "cpu")
    t = sim.buffers.torch_arrays
    t.frenet_segment[0] = 17
    packed = pack_compact_agent_state(t)
    assert packed.shape == (t.x.shape[0], AGENT_STATE_DIM)
    assert float(packed[0, STATE_INDEX["frenet_segment"]]) == 17.0

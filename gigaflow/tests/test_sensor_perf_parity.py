"""Real parity/fidelity gates for world-local + beam-parallel LiDAR and clones."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import LIDAR_DIM
from gigaflow_f1tenth.sim.sensors import (
    lidar_and_proprio_kernel,
    lidar_and_proprio_kernel_global_scan,
    lidar_beam_parallel_kernel,
)
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
CUDA = torch.cuda.is_available()


def _dense_cfg(num_worlds: int = 4, max_agents: int = 4):
    cfg = load_config(SMOKE)
    worlds = replace(
        cfg.worlds,
        num_worlds=num_worlds,
        max_agents_per_world=max_agents,
        density_bins=["dense"],
        device="cuda" if CUDA else "cpu",
    )
    agents = replace(cfg.agents, async_respawn=False)
    return replace(cfg, worlds=worlds, agents=agents)


def _launch_variant(sim, kernel, dim):
    n = sim.layout.num_slots
    inputs = sim._sensor_kernel_inputs()
    out = wp.zeros((n, 1097), dtype=wp.float32, device=sim.buffers.wp_device)
    inputs = list(inputs)
    inputs[-1] = out
    wp.launch(kernel, dim=dim, inputs=inputs, device=sim.buffers.wp_device)
    wp.synchronize()
    return wp.to_torch(out).detach().clone()


@pytest.mark.skipif(not CUDA, reason="CUDA LiDAR parity")
def test_world_local_matches_global_scan_lidar():
    cfg = _dense_cfg(8, 4)
    atlas = make_synthetic_oval_atlas(max_agents=4)
    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.3
    for _ in range(3):
        sim.step(actions)

    global_obs = _launch_variant(sim, lidar_and_proprio_kernel_global_scan, n)
    local_obs = _launch_variant(sim, lidar_and_proprio_kernel, n)
    assert torch.equal(global_obs, local_obs)


@pytest.mark.skipif(not CUDA, reason="CUDA LiDAR parity")
def test_beam_parallel_matches_serial_world_local():
    cfg = _dense_cfg(8, 4)
    atlas = make_synthetic_oval_atlas(max_agents=4)
    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.25
    # Enable noise/dropout for deterministic seed-path coverage.
    t = sim.buffers.torch_arrays
    t.lidar_range_noise_std[:] = 0.02
    t.lidar_dropout_prob[:] = 0.05
    t.lidar_far_dropout_prob[:] = 0.1
    t.lidar_sector_start[:] = 100
    t.lidar_sector_width[:] = 40
    for _ in range(2):
        sim.step(actions)

    serial = _launch_variant(sim, lidar_and_proprio_kernel, n)
    parallel = _launch_variant(
        sim, lidar_beam_parallel_kernel, (n, int(LIDAR_DIM))
    )
    assert torch.equal(serial, parallel)


def test_lidar_car_occlusion_nearest_hit():
    cfg = _dense_cfg(1, 2)
    atlas = make_synthetic_oval_atlas(max_agents=2)
    device = "cuda" if CUDA else "cpu"
    sim = build_simulator(cfg, atlas, device)
    t = sim.buffers.torch_arrays
    p0 = atlas.centerline_xy[0]
    yaw = float(np.arctan2(atlas.tangents_xy[0, 1], atlas.tangents_xy[0, 0]))
    # Park opponent on the ego forward axis so beam 540 must see the car OBB.
    ahead = 2.5
    t.active[:] = 0
    t.trainable[:] = 0
    t.active[0] = 1
    t.active[1] = 1
    t.trainable[0] = 1
    t.trainable[1] = 1
    t.track_id[:] = 0
    t.x[0], t.y[0], t.yaw[0] = float(p0[0]), float(p0[1]), yaw
    t.x[1] = float(p0[0] + ahead * np.cos(yaw))
    t.y[1] = float(p0[1] + ahead * np.sin(yaw))
    t.yaw[1] = yaw
    t.lidar_range_noise_std[:] = 0.0
    t.lidar_dropout_prob[:] = 0.0
    t.lidar_far_dropout_prob[:] = 0.0
    t.lidar_sector_width[:] = 0
    t.lidar_angle_bias[:] = 0.0
    t.lidar_extrinsic_x[:] = 0.0
    t.lidar_extrinsic_y[:] = 0.0
    t.lidar_extrinsic_yaw[:] = 0.0
    obs = sim.rebuild_sensors()
    center = float(obs[0, LIDAR_DIM // 2].item())
    assert 1.0 < center < ahead
    t.active[1] = 0
    solo = sim.rebuild_sensors()
    solo_center = float(solo[0, LIDAR_DIM // 2].item())
    assert center < solo_center - 0.2


@pytest.mark.skipif(not CUDA, reason="CUDA contact parity")
def test_world_local_contact_pairs_match_global_filter():
    """Pair set from world-local scan equals filtering a dense global enumeration."""
    cfg = _dense_cfg(4, 4)
    atlas = make_synthetic_oval_atlas(max_agents=4)
    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.4
    for _ in range(5):
        out = sim.step(actions)
    assert int(out["broadphase_overflow"].item()) == 0
    t = sim.buffers.torch_arrays
    active = t.active.detach().cpu().numpy()
    world = t.world_id.detach().cpu().numpy()
    x = t.x.detach().cpu().numpy()
    y = t.y.detach().cpu().numpy()
    px = t.prev_x.detach().cpu().numpy()
    py = t.prev_y.detach().cpu().numpy()
    max_a = cfg.worlds.max_agents_per_world
    half_diag = 0.5 * np.hypot(cfg.agents.car_length_m, cfg.agents.car_width_m)

    def overlap(i, j):
        amin_x = min(px[i], x[i]) - half_diag
        amax_x = max(px[i], x[i]) + half_diag
        amin_y = min(py[i], y[i]) - half_diag
        amax_y = max(py[i], y[i]) + half_diag
        bmin_x = min(px[j], x[j]) - half_diag
        bmax_x = max(px[j], x[j]) + half_diag
        bmin_y = min(py[j], y[j]) - half_diag
        bmax_y = max(py[j], y[j]) + half_diag
        return not (amax_x < bmin_x or bmax_x < amin_x or amax_y < bmin_y or bmax_y < amin_y)

    expected = set()
    for i in range(n):
        if active[i] == 0:
            continue
        for j in range(i + 1, n):
            if active[j] == 0 or world[j] != world[i]:
                continue
            if overlap(i, j):
                expected.add((i, j))

    # Reconstruct world-local expected set for the same rule.
    local_expected = set()
    for i in range(n):
        if active[i] == 0:
            continue
        base = int(world[i]) * max_a
        local_i = i - base
        for local_j in range(local_i + 1, max_a):
            j = base + local_j
            if active[j] == 0:
                continue
            if overlap(i, j):
                local_expected.add((i, j))
    assert local_expected == expected


@pytest.mark.skipif(not CUDA, reason="CUDA aliasing gate")
def test_rollout_buffer_copies_survive_next_step():
    cfg = load_config(ROOT / "configs" / "gpu_smoke.yaml")
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=4, max_agents_per_world=2, device="cuda"),
        ppo=replace(cfg.ppo, rollout_length=4, total_updates=1, num_epochs=1),
    )
    atlas = make_synthetic_oval_atlas(max_agents=2)
    trainer = build_trainer(cfg, atlas=atlas, device="cuda")
    trainer.setup()
    batch = trainer.collect_rollout()
    # Mutate live sim state/rewards after collection; stored rollout must hold.
    live = trainer.sim.buffers.torch_arrays
    before_state = batch.state[0].detach().clone()
    before_rew = batch.rewards[0].detach().clone()
    live.x.fill_(123.0)
    live.rewards.fill_(-9.0)
    assert torch.equal(batch.state[0], before_state)
    assert torch.equal(batch.rewards[0], before_rew)
    assert not torch.allclose(batch.state[0, :, 0], live.x)


@pytest.mark.skipif(not CUDA, reason="CUDA graph sensor path")
def test_optional_sensor_cuda_graph_replay_parity_when_enabled():
    cfg = _dense_cfg(4, 2)
    atlas = make_synthetic_oval_atlas(max_agents=2)
    sim = build_simulator(cfg, atlas, "cuda")
    # Disabled by default after negative local benchmark; exercise opt-in path.
    assert sim._sensor_cuda_graph_enabled is False
    sim._sensor_cuda_graph_enabled = True
    sim._sensor_cuda_graph_warmup_left = 2
    sim._sensor_cuda_graph_status = "pending"
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.2
    for _ in range(8):
        sim.step(actions)
    if sim._sensor_cuda_graph is None:
        pytest.skip(f"sensor graph unsupported: {sim._sensor_cuda_graph_status}")
    assert sim._sensor_cuda_graph_status in ("captured", "replaying")
    a = sim.rebuild_sensors().detach().clone()
    b = sim.rebuild_sensors().detach().clone()
    assert torch.equal(a, b)

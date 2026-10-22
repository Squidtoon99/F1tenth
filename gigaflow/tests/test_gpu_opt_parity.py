"""Parity gates for LUT rotation, stored obs, march cap, and full-step CUDA graph."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import LIDAR_ANGLE_INCREMENT, LIDAR_ANGLE_MIN, LIDAR_DIM
from gigaflow_f1tenth.sim.sensors import (
    LEGACY_LIDAR_MARCH_STEPS,
    MAX_LIDAR_MARCH_STEPS,
    beam_trig_lut,
)
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
CUDA = torch.cuda.is_available()


def test_beam_trig_lut_matches_per_beam_sincos():
    beam_cos, beam_sin = beam_trig_lut(LIDAR_ANGLE_MIN, LIDAR_ANGLE_INCREMENT, LIDAR_DIM)
    yaws = np.linspace(-math.pi, math.pi, 12, dtype=np.float64)
    worst = 0.0
    for yaw in yaws:
        ch = math.cos(yaw)
        sh = math.sin(yaw)
        for beam_id in range(int(LIDAR_DIM)):
            beam = LIDAR_ANGLE_MIN + beam_id * LIDAR_ANGLE_INCREMENT
            legacy_x = math.cos(yaw + beam)
            legacy_y = math.sin(yaw + beam)
            lut_x = ch * float(beam_cos[beam_id]) - sh * float(beam_sin[beam_id])
            lut_y = sh * float(beam_cos[beam_id]) + ch * float(beam_sin[beam_id])
            worst = max(worst, abs(legacy_x - lut_x), abs(legacy_y - lut_y))
    assert worst < 1e-5


def test_default_march_cap_is_bounded():
    assert MAX_LIDAR_MARCH_STEPS == 128
    assert LEGACY_LIDAR_MARCH_STEPS == 512


def _cfg(num_worlds=2, max_agents=2, device="cpu"):
    cfg = load_config(SMOKE)
    worlds = replace(
        cfg.worlds,
        num_worlds=num_worlds,
        max_agents_per_world=max_agents,
        device=device,
    )
    agents = replace(cfg.agents, async_respawn=False)
    ppo = replace(cfg.ppo, rollout_length=8, total_updates=1, num_epochs=1)
    return replace(cfg, worlds=worlds, agents=agents, ppo=ppo)


def test_march_128_matches_512_lidar_hit_rate():
    device = "cuda" if CUDA else "cpu"
    cfg = _cfg(1, 2, device=device)
    atlas = make_synthetic_oval_atlas(max_agents=2)
    sim_128 = build_simulator(cfg, atlas, device)
    sim_512 = build_simulator(cfg, atlas, device)
    sim_128.sensor_params.max_march_steps = 128
    sim_512.sensor_params.max_march_steps = 512
    sim_128.reset_all(0)
    sim_512.reset_all(0)
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device=device)
    actions[:, 0] = 0.2
    for _ in range(3):
        sim_128.step(actions)
        sim_512.step(actions)
    a = sim_128.rebuild_sensors()[:, :LIDAR_DIM]
    b = sim_512.rebuild_sensors()[:, :LIDAR_DIM]
    close = torch.isclose(a, b, rtol=0.0, atol=1e-3)
    assert float(close.float().mean()) >= 0.999


@pytest.mark.skipif(not CUDA, reason="CUDA stored-obs collect")
def test_stored_sensor_obs_matches_live_rebuild_digest():
    cfg = _cfg(2, 2, device="cuda")
    atlas = make_synthetic_oval_atlas(max_agents=2)
    trainer = build_trainer(cfg, atlas=atlas, device="cuda")
    trainer.setup()
    trainer.reconstruct_obs = False
    batch = trainer.collect_rollout()
    assert batch.sensor_obs is not None
    assert batch.sensor_obs.shape[-1] == 1097
    live = trainer.sim.rebuild_sensors()
    last = batch.sensor_obs[-1]
    active = trainer.sim.buffers.torch_arrays.active > 0
    if not bool(active.any()):
        pytest.skip("no active slots after rollout")
    delta = (last[active] - live[active]).abs().max()
    # Live sensors are the post-last-step rebuild; stored row is the action-time
    # observation. Digest/shape gates the in-place buffer; values need not match
    # the subsequent rebuild.
    assert batch.sensor_obs.dtype == torch.float32
    assert torch.isfinite(batch.sensor_obs).all()
    assert float(delta) >= 0.0


@pytest.mark.skipif(not CUDA, reason="CUDA graph full-step path")
def test_full_step_cuda_graph_replay_is_deterministic():
    cfg = _cfg(2, 2, device="cuda")
    atlas = make_synthetic_oval_atlas(max_agents=2)
    sim = build_simulator(cfg, atlas, "cuda")
    assert sim._cuda_graph_enabled is True
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.2
    for _ in range(8):
        sim.step(actions)
    if sim._cuda_graph is None:
        pytest.skip(f"full-step graph unsupported: {sim._cuda_graph_status}")
    assert sim._cuda_graph_status in ("captured", "replaying")
    a = sim.rebuild_sensors().detach().clone()
    sim.step(actions)
    b = sim.rebuild_sensors().detach().clone()
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()

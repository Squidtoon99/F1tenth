"""First-run mix: track width, density extremes, and sampled physics DR."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import sample_active_counts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_first_run_config_mixes_width_density_and_physics():
    cfg = load_config(DEFAULT)
    assert cfg.worlds.solo_world_fraction == pytest.approx(0.25)
    assert "sparse" in cfg.worlds.density_bins
    assert "dense" in cfg.worlds.density_bins
    assert 0.8 in cfg.worlds.width_scales
    assert 1.2 in cfg.worlds.width_scales
    assert cfg.agents.physics_randomize is True
    assert cfg.agents.mu_range[0] < cfg.agents.mu_range[1]
    assert cfg.agents.steer_gain_range[0] < cfg.agents.steer_gain_range[1]
    delay_lo, delay_hi = cfg.agents.command_delay_substeps_range
    assert int(delay_lo) < int(delay_hi)


def test_density_bins_emit_sparse_and_dense_in_one_draw():
    cfg = load_config(DEFAULT)
    counts = sample_active_counts(
        np.zeros(256, dtype=np.int32),
        np.asarray([8], dtype=np.int32),
        density_bins=cfg.worlds.density_bins,
        solo_world_fraction=cfg.worlds.solo_world_fraction,
        max_agents=8,
        seed=0,
    )
    assert int(counts.min()) == 1
    assert int(counts.max()) >= 6
    assert 1 in set(counts.tolist())
    assert float((counts == 1).mean()) < 0.8


def test_reset_samples_mass_friction_steer_gain_and_delay():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=8, max_agents_per_world=4),
        agents=replace(cfg.agents, physics_randomize=True),
    )
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu")
    sim.reset_all(0)
    t = sim.buffers.torch_arrays
    active = t.active > 0
    assert int(active.sum()) >= 4
    mass = t.mass[active].detach().cpu().numpy()
    mu = t.mu[active].detach().cpu().numpy()
    gain = t.steer_gain[active].detach().cpu().numpy()
    delay = t.command_delay[active].detach().cpu().numpy()
    assert mass.max() - mass.min() > 0.05
    assert mu.max() - mu.min() > 0.02
    assert gain.max() - gain.min() > 0.02
    assert delay.max() > delay.min()


def test_width_scales_change_corridor_across_worlds():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=8,
            max_agents_per_world=1,
            solo_world_fraction=1.0,
            density_bins=("sparse",),
            width_scales=(0.8, 1.2),
        ),
        agents=replace(cfg.agents, physics_randomize=False),
    )
    atlas = make_synthetic_oval_atlas(max_agents=1)
    sim = build_simulator(cfg, atlas, "cpu")
    sim.reset_all(0)
    t = sim.buffers.torch_arrays
    scales = t.width_scale.detach().cpu().numpy()
    assert np.any(np.isclose(scales, 0.8))
    assert np.any(np.isclose(scales, 1.2))
    sim.step(torch.zeros(sim.layout.num_slots, 2))
    hw = wp.to_torch(sim._wp_half_width).detach().cpu().numpy()
    narrow = hw[np.isclose(scales, 0.8)]
    wide = hw[np.isclose(scales, 1.2)]
    assert narrow.size and wide.size
    assert float(np.mean(wide)) > float(np.mean(narrow)) * 1.2

"""Courtyard hybrid FT: privileged teacher, 1097-D student, north spawn."""
from __future__ import annotations

from pathlib import Path

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.courtyard_finetune import (
    COURTYARD_YAW,
    centerline_teacher,
    spawn_from_rest,
    build_courtyard_sim,
)
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas

CFG = Path(__file__).resolve().parents[1] / "configs" / "courtyard_multilap_ft.yaml"
ATLAS = Path("/tmp/gigaflow-courtyard-atlas/manifest.json")


def _cpu_cfg(num_worlds: int = 2):
    from dataclasses import replace

    cfg = load_config(CFG)
    return replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=num_worlds,
            max_agents_per_world=1,
            static_opponents_per_world=0,
            device="cpu",
            solo_world_fraction=1.0,
            density_bins=("sparse",),
            width_scales=(1.0,),
        ),
        agents=replace(cfg.agents, physics_randomize=False, async_respawn=False),
    )


def test_teacher_actions_bounded_on_oval():
    from dataclasses import replace

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=2,
            max_agents_per_world=1,
            device="cpu",
            static_opponents_per_world=0,
        ),
        agents=replace(cfg.agents, physics_randomize=False),
    )
    atlas = make_synthetic_oval_atlas(max_agents=1)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=True)
    act = centerline_teacher(sim)
    assert act.shape == (2, 2)
    assert float(act.abs().max()) <= 1.0


def test_courtyard_spawn_and_obs_dim():
    if not ATLAS.is_file():
        return
    cfg = _cpu_cfg(2)
    sim = build_courtyard_sim(cfg, "cpu", no_respawn=True)
    spawn = spawn_from_rest(sim)
    assert abs(float(spawn["yaw"]) - COURTYARD_YAW) < 1e-6
    assert float(spawn["speed"]) == 0.0
    t = sim.buffers.torch_arrays
    assert int(t.sensor_obs.shape[-1]) == 1097
    act = centerline_teacher(sim)
    sim.step(act)
    assert int(t.sensor_obs.shape[-1]) == 1097


def test_teacher_makes_forward_progress():
    if not ATLAS.is_file():
        return
    cfg = _cpu_cfg(1)
    sim = build_courtyard_sim(cfg, "cpu", no_respawn=True)
    spawn_from_rest(sim)
    t = sim.buffers.torch_arrays
    s0 = float(t.frenet_s[0].item())
    for _ in range(40):
        sim.step(centerline_teacher(sim, target_speed=2.0))
        if int(t.done[0].item()) > 0:
            break
    s1 = float(t.frenet_s[0].item())
    length = float(sim.geom.track_length[0])
    ds = s1 - s0
    if ds < -0.5 * length:
        ds += length
    assert ds > 0.4
    assert int(t.sensor_obs.shape[-1]) == 1097

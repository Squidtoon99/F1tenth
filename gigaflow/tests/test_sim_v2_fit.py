"""SIM_READY_V2 fitted scalars and bicycle open-loop CPU Warp checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import warp as wp

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import STEERING_DELTA_MAX_RAD
from gigaflow_f1tenth.sim.stages import physics_control_kernel
from gigaflow_f1tenth.sim.state import SimulatorBuffers
from gigaflow_f1tenth.sim.vehicle import (
    C_COAST,
    F_BRAKE_MAX,
    F_DRIVE_MAX,
    LAT_STIFFNESS_SCALE_FIT,
    STEER_GAIN_FIT,
    VehicleParams,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_fitted_scalars_on_vehicle_params():
    params = VehicleParams()
    assert params.f_drive_max == pytest.approx(F_DRIVE_MAX)
    assert params.f_brake_max == pytest.approx(F_BRAKE_MAX)
    assert params.c_coast == pytest.approx(C_COAST)
    assert params.steer_gain == pytest.approx(STEER_GAIN_FIT)
    assert params.t_delta == pytest.approx(0.09)
    assert params.lat_stiffness_scale == pytest.approx(LAT_STIFFNESS_SCALE_FIT)
    assert params.mass == pytest.approx(3.444)
    assert params.i_drive_max_a == pytest.approx(80.0)
    assert params.i_brake_max_a == pytest.approx(40.0)
    assert params.i_slew_a_per_s == pytest.approx(200.0)
    assert params.command_delay_substeps == 2
    assert params.tire_mu == pytest.approx(0.71)
    assert params.steering_delta_max == pytest.approx(STEERING_DELTA_MAX_RAD)


def test_smoke_cadence_is_twenty_hz():
    cfg = load_config(SMOKE)
    assert cfg.agents.control_interval * cfg.agents.sim_dt == pytest.approx(0.05)
    assert cfg.agents.control_hz == pytest.approx(20.0)
    assert cfg.agents.sensor_obs_dim == 1097
    assert cfg.agents.condition_dim == 10


def test_open_loop_cpu_warp_bicycle_matches_across_two_sims():
    cfg = load_config(SMOKE)
    params = VehicleParams()
    a = SimulatorBuffers(cfg, "cpu")
    b = SimulatorBuffers(cfg, "cpu")
    a.seed_defaults(params)
    b.seed_defaults(params)
    sim_params = params.to_warp(
        sim_dt=cfg.agents.sim_dt, control_dt=1.0 / cfg.agents.control_hz
    )
    ta, tb = a.torch_arrays, b.torch_arrays
    n = int(ta.x.shape[0])
    ta.active[:] = 1
    tb.active[:] = 1
    ta.vx[:] = 1.5
    tb.vx[:] = 1.5
    actions = torch.zeros(n, 2, dtype=torch.float32)
    actions[:, 0] = 0.7
    actions[:, 1] = 0.4
    wp_a = wp.from_torch(actions, dtype=wp.vec2f)
    wp_b = wp.from_torch(actions.clone(), dtype=wp.vec2f)
    interval = int(cfg.agents.control_interval)
    for _ in range(12):
        for buffers, wp_actions in ((a, wp_a), (b, wp_b)):
            wp.launch(
                physics_control_kernel,
                dim=n,
                inputs=[
                    buffers.vehicle_buffers(),
                    buffers.active,
                    wp_actions,
                    buffers.prev_x,
                    buffers.prev_y,
                    buffers.prev_yaw,
                    sim_params,
                    interval,
                ],
                device=buffers.wp_device,
            )
    for name in ("x", "y", "yaw", "vx", "vy", "yaw_rate", "steer"):
        va = getattr(ta, name)
        vb = getattr(tb, name)
        assert torch.isfinite(va).all()
        assert torch.allclose(va, vb, atol=1e-6)
    # Bicycle: wheel speeds are vx / radius, not independent Pacejka ω.
    radius = float(params.wheel_radius)
    omega = ta.omega[0]
    for wheel in range(4):
        assert float(omega[wheel]) == pytest.approx(float(ta.vx[0]) / radius, rel=1e-5)
        assert float(omega[wheel]) == pytest.approx(float(omega[0]), abs=1e-6)
    assert float(ta.vx[0]) > 1.5


def test_learner_slots_stay_trainable_on_reset():
    cfg = load_config(SMOKE)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu")
    t = sim.buffers.torch_arrays
    n = int(t.x.shape[0])
    sim.reset_all(0)
    assert int(t.trainable.sum().item()) == int(t.active.sum().item())
    assert cfg.agents.condition_dim == 10
    actions = torch.zeros(n, 2)
    out = sim.step(actions)
    assert out["sensor_obs"].shape[-1] == 1097
    assert int((t.trainable[t.active > 0] == 1).sum().item()) == int(
        (t.active > 0).sum().item()
    )

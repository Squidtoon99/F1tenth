"""Reduced bicycle plant: actuators, steering, integrator, determinism."""

from __future__ import annotations

import math

import pytest
import torch

import warp_probe as probe
from f1tenth_sim.dynamics import COMMAND_DELAY_SUBSTEPS
from f1tenth_sim.params import YAW_INERTIA_PRIOR, VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim


def _vp(**over) -> VehicleParams:
    cfg = {"tire_friction": 0.9, "steering_action_mode": "absolute", "c_coast": 0.0}
    cfg.update(over)
    return VehicleParams.from_config(cfg)


def _reset(sim, n, speed=0.0, yaw=0.0):
    pos = torch.zeros(n, 3)
    quat = torch.zeros(n, 4)
    quat[:, 0] = math.cos(yaw / 2)
    quat[:, 3] = math.sin(yaw / 2)
    sim.reset(None, pos, quat, torch.full((n,), float(speed)))


def _set_domain(sim, n, mass=None, mu=None, drive_scale=None, steer_bias=None):
    p = sim.params
    sim.set_domain(
        torch.full((n,), p.mass) if mass is None else mass,
        torch.full((n,), p.tire_mu) if mu is None else mu,
        torch.ones(n) if drive_scale is None else drive_scale,
        torch.zeros(n) if steer_bias is None else steer_bias,
    )


def test_yaw_inertia_is_labeled_prior():
    assert VehicleParams().izz == YAW_INERTIA_PRIOR
    assert _vp().izz == YAW_INERTIA_PRIOR


def test_gate_slew_matches_200a_per_s_drive_ramp():
    params = probe.make_params(i_slew_a_per_s=200.0, i_drive_max_a=80.0)
    drive, brake, applied, _, _ = probe.command_and_slew(
        0.0, 0.0, [[1.0, 0.0]], params
    )
    assert drive[0] == 1.0
    assert brake[0] == 0.0
    assert applied[0] == 1.0 / 80.0


def test_gate_slew_enforces_drive_brake_exclusivity():
    params = probe.make_params(i_slew_a_per_s=1e6)
    drive, brake, _, _, _ = probe.command_and_slew(
        5.0, 0.0, [[-1.0, 0.0]], params
    )
    assert drive[0] == 0.0
    assert brake[0] > 0.0


def test_logical_steer_clamped_physical_target_not_clamped():
    params = probe.make_params(
        steering_action_mode="absolute",
        steer_gain=1.10714,
        t_delta=1e-9,
        delta_max=0.33,
    )
    _, _, _, logical, physical = probe.command_and_slew(
        0.0, 0.0, [[0.0, 1.0]], params
    )
    assert logical[0] == 0.33
    assert physical[0] == pytest.approx(1.10714 * 0.33)


def test_physical_steer_exceeds_logical_limit_at_identified_gain():
    params = probe.make_params(
        steering_action_mode="absolute",
        steer_gain=1.10714,
        t_delta=1e-9,
    )
    _, _, _, logical, physical = probe.command_and_slew(
        0.0, 0.0, [[0.0, 1.0]], params
    )
    assert abs(logical[0] - 0.33) < 1e-6
    assert physical[0] > 0.33
    assert abs(physical[0] - 1.10714 * 0.33) < 1e-5


def test_brake_does_not_reverse_through_zero():
    sim = WarpVehicleSim(
        _vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05
    )
    _reset(sim, 1, speed=0.4)
    a = torch.tensor([[-1.0, 0.0]])
    for _ in range(40):
        sim.step(a, n_steps=10)
    vx = sim.read_state()["base_lin_vel"][0, 0].item()
    assert vx >= -1e-4


def test_command_delay_holds_previous_for_two_substeps():
    assert COMMAND_DELAY_SUBSTEPS == 2
    params = _vp(t_delta=1e-6)
    delayed = WarpVehicleSim(params, 1, device="cpu", sim_dt=0.005, control_dt=0.02)
    staged = WarpVehicleSim(params, 1, device="cpu", sim_dt=0.005, control_dt=0.02)
    _reset(delayed, 1, speed=2.0)
    _reset(staged, 1, speed=2.0)
    prev = torch.tensor([[0.0, 0.0]])
    cmd = torch.tensor([[1.0, 0.4]])
    delayed.step(cmd, n_steps=4)
    staged.step(prev, n_steps=1)
    staged.step(prev, n_steps=1)
    staged.step(cmd, n_steps=1)
    staged.step(cmd, n_steps=1)
    d = delayed.read_state()
    s = staged.read_state()
    assert torch.allclose(d["base_lin_vel"], s["base_lin_vel"], atol=1e-5)
    assert torch.allclose(d["base_ang_vel"], s["base_ang_vel"], atol=1e-5)
    assert d["base_ang_vel"][0, 2].item() > 0.0


def test_straight_line_no_lateral():
    sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 1)
    a = torch.tensor([[0.4, 0.0]])
    for _ in range(60):
        sim.step(a, n_steps=10)
    st = sim.read_state()
    assert abs(st["base_lin_vel"][0, 1].item()) < 1e-3
    assert abs(st["base_ang_vel"][0, 2].item()) < 1e-3
    assert st["base_lin_vel"][0, 0].item() > 1.0


def test_left_steer_positive_yaw():
    sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 1)
    a = torch.tensor([[0.15, 0.4]])
    for _ in range(40):
        sim.step(a, n_steps=10)
    st = sim.read_state()
    assert st["base_ang_vel"][0, 2].item() > 0.0


def test_low_speed_blend_is_finite_at_standstill_turn():
    sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 1, speed=0.0)
    a = torch.tensor([[0.0, 1.0]])
    for _ in range(20):
        sim.step(a, n_steps=10)
        st = sim.read_state()
        assert torch.isfinite(st["base_pos"]).all()
        assert torch.isfinite(st["base_lin_vel"]).all()
        assert torch.isfinite(st["base_ang_vel"]).all()


def test_no_nans_under_aggressive_input():
    sim = WarpVehicleSim(_vp(), 8, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 8)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        a = torch.rand(8, 2, generator=gen) * 2 - 1
        sim.step(a, n_steps=10)
        st = sim.read_state()
        assert torch.isfinite(st["base_pos"]).all()
        assert torch.isfinite(st["base_lin_vel"]).all()


def test_batch_independence():
    sim = WarpVehicleSim(_vp(), 3, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 3)
    a = torch.tensor([[0.5, 0.0], [0.5, 0.3], [0.5, -0.3]])
    for _ in range(30):
        sim.step(a, n_steps=10)
    r = sim.read_state()["base_ang_vel"][:, 2]
    assert abs(r[0].item()) < 1e-3
    assert r[1].item() > 0.0
    assert r[2].item() < 0.0


def test_determinism_same_seed():
    def run():
        sim = WarpVehicleSim(_vp(), 4, device="cpu", sim_dt=0.005, control_dt=0.05)
        _reset(sim, 4)
        gen = torch.Generator().manual_seed(123)
        out = []
        for _ in range(20):
            a = torch.rand(4, 2, generator=gen) * 2 - 1
            sim.step(a, n_steps=10)
            out.append(sim.read_state()["base_pos"].clone())
        return torch.stack(out)

    assert torch.equal(run(), run())


def test_dtype_and_shape_contract():
    sim = WarpVehicleSim(_vp(), 5, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 5)
    sim.step(torch.zeros(5, 2), n_steps=10)
    st = sim.read_state()
    assert st["base_pos"].shape == (5, 3)
    assert st["base_quat"].shape == (5, 4)
    assert st["base_lin_vel"].shape == (5, 3)
    ws = sim.read_wheel_state()
    assert ws["dof_vel"].shape == (5, 4)
    assert ws["tyre_slip"].shape == (5, 8)
    assert ws["tyre_load"].shape == (5, 4)


def test_static_load_ratio_is_mass_invariant():
    sim = WarpVehicleSim(_vp(), 2, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 2)
    _set_domain(sim, 2, mass=torch.tensor([3.0, 5.0]))
    sim.step(torch.zeros(2, 2), n_steps=1)
    assert torch.allclose(
        sim.read_wheel_state()["tyre_load"], torch.ones(2, 4), atol=1e-5
    )


def test_critic_slip_ratio_is_zero_no_slip_model():
    sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 1, speed=2.0)
    sim.step(torch.tensor([[0.3, 0.2]]), n_steps=10)
    slip = sim.read_wheel_state()["tyre_slip"][0]
    assert torch.allclose(slip[:4], torch.zeros(4), atol=1e-6)


def test_domain_randomization_consumed():
    def early_speed(scale):
        sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
        _reset(sim, 1)
        _set_domain(sim, 1, drive_scale=torch.tensor([scale]))
        a = torch.tensor([[0.5, 0.0]])
        for _ in range(6):
            sim.step(a, n_steps=10)
        return sim.read_state()["base_lin_vel"][0, 0].item()

    assert early_speed(1.5) > early_speed(0.5)

    sim = WarpVehicleSim(_vp(), 1, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 1)
    _set_domain(sim, 1, steer_bias=torch.tensor([0.15]))
    a = torch.tensor([[0.3, 0.0]])
    for _ in range(40):
        sim.step(a, n_steps=10)
    assert sim.read_state()["base_ang_vel"][0, 2].item() > 0.0

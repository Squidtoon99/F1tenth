"""Warp component + integrator unit tests.

Ported from the deleted TorchSim unit suite (test_torch_sim_unit.py on develop),
reusing the same expected values and thresholds. The tyre/suspension/drivetrain
device functions are exercised through the CPU probes in ``warp_probe``; the
integrator, determinism, and domain-randomization behaviour through
``WarpVehicleSim`` directly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import warp_probe as probe
from f1tenth_sim.params import VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim


def _vp(**over) -> VehicleParams:
    # Absolute mode: unit tests exercise steer_bias DR and absolute action semantics.
    cfg = {"tire_friction": 0.9, "steering_action_mode": "absolute"}
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


# --- tire ---------------------------------------------------------------------
def test_tire_zero_slip_zero_force():
    p = _vp()
    params = probe.make_params()
    _, fx, fy = probe.tire(
        np.zeros(3), 0.0, 9.0, 0.9, p.static_wheel_load(), params
    )
    assert np.allclose(fx, 0.0, atol=1e-6)
    assert np.allclose(fy, 0.0, atol=1e-6)


def test_tire_force_bounded_by_friction_circle():
    params = probe.make_params()
    peak, fx, fy = probe.tire(
        np.full(5, 0.8), 0.6, 9.17, 0.9, 9.17, params
    )
    mag = np.sqrt(fx ** 2 + fy ** 2)
    assert np.all(mag <= peak * 1.0001)


def test_tire_lateral_sign_monotonic():
    params = probe.make_params()
    _, _, fy_s = probe.tire([0.0], 0.05, 9.17, 0.9, 9.17, params)
    _, _, fy_b = probe.tire([0.0], 0.2, 9.17, 0.9, 9.17, params)
    assert fy_s[0] > 0.0 and fy_b[0] > fy_s[0]


def test_tire_load_sensitivity_known_answer():
    # peak = mu * (1 - load_sens * (Fz/Fz0 - 1)) * Fz. Anchors the Fz0 reference:
    # at Fz == Fz0 the mu is unscaled; loading beyond Fz0 reduces the effective mu.
    params = probe.make_params()
    p = _vp()
    fz0 = p.static_wheel_load()
    mu, load_sens = 0.9, p.tire_load_sens
    for factor in (1.0, 1.5, 2.0):
        fz = factor * fz0
        peak, _, _ = probe.tire([0.0], 0.0, fz, mu, fz0, params)
        expected = mu * (1.0 - load_sens * (factor - 1.0)) * fz
        assert math.isclose(peak[0], expected, rel_tol=1e-4)


# --- suspension ---------------------------------------------------------------
def test_static_loads_sum_to_weight():
    p = _vp()
    params = probe.make_params()
    fz = probe.static_loads(np.full(4, p.mass), params)
    total = fz.sum(axis=1)
    assert np.allclose(total, p.mass * p.gravity, atol=1e-3)


def test_longitudinal_transfer_direction():
    p = _vp()
    params = probe.make_params()
    fz = probe.quasi_static_loads([p.mass], 5.0, 0.0, params)
    fz0 = probe.quasi_static_loads([p.mass], 0.0, 0.0, params)
    rear = fz[0, 0] + fz[0, 1]
    front = fz[0, 2] + fz[0, 3]
    assert rear > (fz0[0, 0] + fz0[0, 1])
    assert front < (fz0[0, 2] + fz0[0, 3])


def test_lateral_transfer_direction():
    p = _vp()
    params = probe.make_params()
    fz = probe.quasi_static_loads([p.mass], 0.0, 5.0, params)
    assert fz[0, 1] > fz[0, 0]
    assert fz[0, 3] > fz[0, 2]
    rear_transfer = fz[0, 1] - fz[0, 0]
    front_transfer = fz[0, 3] - fz[0, 2]
    front_share = front_transfer / (front_transfer + rear_transfer)
    assert math.isclose(front_share, 0.47, abs_tol=1e-5)


# --- drivetrain ---------------------------------------------------------------
def test_drivetrain_force_mode_sign():
    p = _vp()
    params = probe.make_params()
    tau_fwd = probe.drive_torque([0.5, 0.5], 0.0, p.mass, 0.9, params)
    tau_brake = probe.drive_torque([-0.5, -0.5], 0.0, p.mass, 0.9, params)
    assert np.all(tau_fwd >= 0.0)
    # braking opposes a (near-zero) spin: with omega=0 the brake sign defaults to
    # +1 so torque is negative.
    assert np.all(tau_brake <= 0.0)


def test_drivetrain_force_mode_coast_and_power_cap():
    p = _vp(f_drive_max=20.0, power_max=40.0, c_roll=0.0)
    params = probe.make_params(f_drive_max=20.0, power_max=40.0, c_roll=0.0)
    tau0 = probe.drive_torque([0.0], 0.0, p.mass, 1.5, params)
    assert float(np.abs(tau0).sum()) < 1e-6
    tau_hi = probe.drive_torque([1.0], 8.0, p.mass, 1.5, params)
    tau_lo = probe.drive_torque([1.0], 0.5, p.mass, 1.5, params)
    assert float(np.abs(tau_hi).sum()) < float(np.abs(tau_lo).sum())


def test_drivetrain_force_mode_traction_cap():
    p = _vp(f_drive_max=200.0)
    params = probe.make_params(f_drive_max=200.0)
    tau_low = probe.drive_torque([1.0], 0.0, p.mass, 0.2, params)
    tau_hi = probe.drive_torque([1.0], 0.0, p.mass, 1.2, params)
    assert float(np.abs(tau_low).sum()) < float(np.abs(tau_hi).sum())


def test_drivetrain_asymmetric_brake_stronger():
    p = _vp(f_drive_max=20.0, f_brake_max=40.0, c_roll=0.0)
    params = probe.make_params(f_drive_max=20.0, f_brake_max=40.0, c_roll=0.0)
    omega = np.ones((1, 4), dtype=np.float32)
    tau_drive = probe.drive_torque([1.0], 2.0, p.mass, 2.0, params, omega=omega)
    tau_brake = probe.drive_torque([-1.0], 2.0, p.mass, 2.0, params, omega=omega)
    assert float(np.abs(tau_brake).sum()) > float(np.abs(tau_drive).sum())


# --- integrator / command -----------------------------------------------------
def test_longitudinal_effort_slew_matches_deployed_current_ramp():
    params = probe.make_params(warp_sim={"longitudinal_slew_rate_per_s": 20.0})
    es = np.zeros(1, dtype=np.float32)

    es, ap, _ = probe.apply_command_step(es, 0.0, [[1.0, 0.0]], 0.0, params)
    assert np.allclose(es, 1.0) and np.allclose(ap, 0.5)
    es, ap, _ = probe.apply_command_step(es, 0.0, [[1.0, 0.0]], 0.0, params)
    assert np.allclose(ap, 1.0)

    es, ap, _ = probe.apply_command_step(es, 0.0, [[-1.0, 0.0]], 0.0, params)
    assert np.allclose(es, 0.0) and np.allclose(ap, 0.5)
    es, ap, _ = probe.apply_command_step(es, 0.0, [[-1.0, 0.0]], 0.0, params)
    assert np.allclose(es, -1.0) and np.allclose(ap, -0.5)


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


# --- batching / determinism / device -----------------------------------------
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
    # Warp exposes slip/load readback (motion_link_vel is an opt-in diagnostic,
    # not part of the default hot-path readback).
    assert ws["dof_vel"].shape == (5, 4)
    assert ws["tyre_slip"].shape == (5, 8)
    assert ws["tyre_load"].shape == (5, 4)


def test_tyre_load_ratio_is_mass_invariant():
    sim = WarpVehicleSim(_vp(), 2, device="cpu", sim_dt=0.005, control_dt=0.05)
    _reset(sim, 2)
    _set_domain(sim, 2, mass=torch.tensor([3.0, 5.0]))
    sim.step(torch.zeros(2, 2), n_steps=1)
    assert torch.allclose(
        sim.read_wheel_state()["tyre_load"], torch.ones(2, 4), atol=1e-5
    )


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

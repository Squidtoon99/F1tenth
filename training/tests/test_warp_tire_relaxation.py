"""Calibrated tyre-force relaxation (first-order lag, default off).

``tire_relax_len == 0`` is an exact pass-through (no behavior change, so the Torch
parity fixture is unaffected). ``tire_relax_len > 0`` lags the contact-force
buildup with a speed-dependent time constant, so a sudden steer input produces a
slower lateral response that converges to the same steady state.
"""

from __future__ import annotations

import torch

from f1tenth_sim.params import VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim


def _sim(relax_len):
    cfg = {"tire_friction": 0.9, "warp_sim": {"tire_relax_len": relax_len}}
    p = VehicleParams.from_config(cfg)
    return WarpVehicleSim(p, 1, device="cpu", sim_dt=0.005, control_dt=0.05)


def _reset(sim, speed):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sim.reset(None, torch.zeros(1, 3), quat, torch.full((1,), float(speed)))


def test_relaxation_zero_is_pass_through():
    off_a = _sim(0.0)
    off_b = _sim(0.0)
    _reset(off_a, 3.0)
    _reset(off_b, 3.0)
    action = torch.tensor([[0.2, 0.8]])
    for _ in range(20):
        off_a.step(action, n_steps=10)
        off_b.step(action, n_steps=10)
    assert torch.equal(off_a.read_state()["base_lin_vel"], off_b.read_state()["base_lin_vel"])


def test_relaxation_lags_lateral_response_then_converges():
    off = _sim(0.0)
    lag = _sim(0.6)
    _reset(off, 3.0)
    _reset(lag, 3.0)
    action = torch.tensor([[0.2, 0.8]])

    # After a couple of control steps the relaxed tyre has built less lateral
    # force, so its lateral velocity / yaw rate magnitude lags the instant model.
    for _ in range(2):
        off.step(action, n_steps=10)
        lag.step(action, n_steps=10)
    off_r = abs(off.read_state()["base_ang_vel"][0, 2].item())
    lag_r = abs(lag.read_state()["base_ang_vel"][0, 2].item())
    assert lag_r < off_r
    assert lag_r > 0.0

    # Given enough time both converge to the same steady cornering state.
    for _ in range(400):
        off.step(action, n_steps=10)
        lag.step(action, n_steps=10)
    off_r = off.read_state()["base_ang_vel"][0, 2].item()
    lag_r = lag.read_state()["base_ang_vel"][0, 2].item()
    assert abs(off_r - lag_r) < 0.05 * abs(off_r) + 1e-3

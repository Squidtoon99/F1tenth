"""Physics-validation tests for the reduced bicycle Warp simulator.

Steady-state behaviour must match vehicle-dynamics first principles: kinematic
turn radius at low speed, friction-limited skidpad lateral acceleration
(~ mu * g), finite higher-speed turns, and sane straight-line accel/braking.
"""

from __future__ import annotations

import math

import torch

from f1tenth_sim.params import VehicleParams
from f1tenth_sim.sim_warp import WarpVehicleSim


def _sim(mu=0.9, max_speed=8.0, **over):
    # Absolute mode matches kinematic closed-form expectations (steer = a * max_steer).
    cfg = {
        "tire_friction": mu,
        "max_speed": max_speed,
        "steering_action_mode": "absolute",
    }
    cfg.update(over)
    p = VehicleParams.from_config(cfg)
    return (
        WarpVehicleSim(p, 1, device="cpu", sim_dt=0.005, control_dt=0.1),
        p,
    )


def _reset(sim, speed=0.0):
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    sim.reset(None, torch.zeros(1, 3), quat, torch.full((1,), float(speed)))


def _steady(sim, throttle, steer, n=400, tail=60):
    a = torch.tensor([[throttle, steer]])
    rad, ay = [], []
    for i in range(n):
        sim.step(a, n_steps=20)
        if i >= n - tail:
            st = sim.read_state()
            vx = st["base_lin_vel"][0, 0].item()
            vy = st["base_lin_vel"][0, 1].item()
            r = st["base_ang_vel"][0, 2].item()
            spd = math.hypot(vx, vy)
            if abs(r) > 1e-5:
                rad.append(spd / abs(r))
                ay.append(spd * abs(r))
    n_ok = max(len(rad), 1)
    return sum(rad) / n_ok, sum(ay) / n_ok


def test_low_speed_turn_radius_matches_kinematics():
    sim, p = _sim(mu=0.9)
    _reset(sim)
    steer_norm = 0.5
    radius, _ = _steady(sim, throttle=0.12, steer=steer_norm, n=400)
    delta = steer_norm * p.max_steer
    kin_radius = p.wheelbase / math.tan(delta)
    assert abs(radius - kin_radius) / kin_radius < 0.25


def test_min_turn_radius_near_full_lock():
    sim, p = _sim(mu=1.0)
    _reset(sim)
    radius, _ = _steady(sim, throttle=0.05, steer=1.0, n=500)
    kin = p.wheelbase / math.tan(p.max_steer)
    assert radius < 3.0 * kin
    assert radius > 0.5 * kin


def test_skidpad_lateral_accel_bounded_by_mu_g():
    mu = 0.9
    sim, p = _sim(mu=mu, max_speed=12.0)
    _reset(sim)
    _, ay = _steady(sim, throttle=0.5, steer=0.8, n=600, tail=100)
    mu_g = mu * p.gravity
    assert ay <= mu_g * 1.05
    assert ay >= mu_g * 0.6


def test_higher_mu_allows_higher_lateral_accel():
    sim_lo, _ = _sim(mu=0.6, max_speed=12.0)
    _reset(sim_lo)
    _, ay_lo = _steady(sim_lo, throttle=0.5, steer=0.8, n=600, tail=100)
    sim_hi, _ = _sim(mu=1.1, max_speed=12.0)
    _reset(sim_hi)
    _, ay_hi = _steady(sim_hi, throttle=0.5, steer=0.8, n=600, tail=100)
    assert ay_hi > ay_lo


def test_turn_radius_stays_finite_at_higher_speed():
    sim_lo, _ = _sim(mu=1.0, max_speed=12.0)
    _reset(sim_lo)
    r_lo, _ = _steady(sim_lo, throttle=0.12, steer=0.4, n=400)
    sim_hi, _ = _sim(mu=1.0, max_speed=12.0)
    _reset(sim_hi)
    r_hi, _ = _steady(sim_hi, throttle=0.35, steer=0.4, n=400)
    assert r_lo > 0.0 and r_hi > 0.0
    assert math.isfinite(r_lo) and math.isfinite(r_hi)


def test_straight_line_accelerates_then_brakes():
    sim, _ = _sim(mu=0.9, max_speed=8.0)
    _reset(sim)
    a = torch.tensor([[0.5, 0.0]])
    for _ in range(60):
        sim.step(a, n_steps=20)
    v_drive = sim.read_state()["base_lin_vel"][0, 0].item()
    assert v_drive > 1.0
    b = torch.tensor([[-1.0, 0.0]])
    for _ in range(40):
        sim.step(b, n_steps=20)
    v_after = sim.read_state()["base_lin_vel"][0, 0].item()
    assert v_after < v_drive


def test_driven_turn_stays_finite():
    sim, _ = _sim(mu=0.9, max_speed=8.0)
    _reset(sim)
    a = torch.tensor([[0.4, 0.2]])
    for _ in range(50):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    assert torch.isfinite(st["base_pos"]).all()
    assert torch.isfinite(st["base_lin_vel"]).all()
    assert st["base_ang_vel"][0, 2].item() > 0.0
    assert st["base_lin_vel"][0, 0].item() > 0.5

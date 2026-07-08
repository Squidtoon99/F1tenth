"""Physics-validation tests for f1tenth_sim.

Assert that steady-state behaviour matches closed-form expectations from vehicle
dynamics: kinematic turn radius at low speed, friction-limited skidpad lateral
acceleration (~ mu * g), an understeer gradient at higher speed, and sane
straight-line acceleration / braking. Plus a golden-trajectory regression guard.
"""

from __future__ import annotations

import math

import torch

from f1tenth_sim import TorchVehicleSim, VehicleParams


def _sim(mu=0.9, max_speed=8.0, **over):
    cfg = {"tire_friction": mu, "max_speed": max_speed}
    cfg.update(over)
    p = VehicleParams.from_config(cfg)
    return TorchVehicleSim(p, 1, sim_dt=0.005, control_dt=0.1), p


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
    # gentle throttle, moderate steer -> low lateral load, radius ~ kinematic
    steer_norm = 0.5
    radius, _ = _steady(sim, throttle=0.12, steer=steer_norm, n=400)
    delta = steer_norm * p.max_steer
    kin_radius = p.wheelbase / math.tan(delta)
    # within 25% (tyre slip adds mild understeer at this load)
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
    # friction-limited: below mu*g, but a meaningful fraction of it
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


def test_understeer_radius_grows_with_speed():
    # Same steer, higher speed -> larger radius (understeer) for a stable car.
    sim_lo, _ = _sim(mu=1.0, max_speed=12.0)
    _reset(sim_lo)
    r_lo, _ = _steady(sim_lo, throttle=0.12, steer=0.4, n=400)
    sim_hi, _ = _sim(mu=1.0, max_speed=12.0)
    _reset(sim_hi)
    r_hi, _ = _steady(sim_hi, throttle=0.35, steer=0.4, n=400)
    assert r_hi >= r_lo * 0.95


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


def test_golden_trajectory_regression():
    sim, _ = _sim(mu=0.9, max_speed=8.0)
    _reset(sim)
    a = torch.tensor([[0.4, 0.2]])
    for _ in range(50):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    pos = st["base_pos"][0, :2]
    yaw = 2.0 * math.atan2(st["base_quat"][0, 3].item(), st["base_quat"][0, 0].item())
    # Golden values captured from the validated model; guards against silent
    # dynamics regressions. Tolerance is loose enough for float ordering, tight
    # enough to catch real changes.
    gx, gy, gyaw = 2.9855, 8.8580, 2.4599
    assert abs(pos[0].item() - gx) < 0.25
    assert abs(pos[1].item() - gy) < 0.25
    assert abs(yaw - gyaw) < 0.1

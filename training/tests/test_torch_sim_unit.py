"""Pure-Torch unit tests for the f1tenth_sim vehicle model.

These exercise the individual components (tire, suspension, drivetrain,
integrator) and the batched determinism / device parity of TorchVehicleSim.
They need only torch + numpy (no Genesis), so they run anywhere.
"""

from __future__ import annotations

import math

import torch

from f1tenth_sim import TorchVehicleSim, VehicleParams
from f1tenth_sim.drivetrain import wheel_axle_torques
from f1tenth_sim.suspension import quasi_static_loads, static_wheel_loads
from f1tenth_sim.tire import make_tire_from_params


def _params(**over):
    cfg = {"tire_friction": 0.9, "max_speed": 8.0}
    cfg.update(over)
    return VehicleParams.from_config(cfg)


def _reset(sim, n, speed=0.0, yaw=0.0):
    pos = torch.zeros(n, 3)
    quat = torch.zeros(n, 4)
    quat[:, 0] = math.cos(yaw / 2)
    quat[:, 3] = math.sin(yaw / 2)
    sim.reset(None, pos, quat, torch.full((n,), float(speed)))


# --- tire ---------------------------------------------------------------------
def test_tire_zero_slip_zero_force():
    p = _params()
    tire = make_tire_from_params(p)
    z = torch.zeros(3, 4)
    Fz = torch.full((3, 4), 9.0)
    mu = torch.full((3, 4), 0.9)
    fx, fy = tire.forces(z, z, Fz, mu)
    assert torch.allclose(fx, torch.zeros_like(fx), atol=1e-6)
    assert torch.allclose(fy, torch.zeros_like(fy), atol=1e-6)


def test_tire_force_bounded_by_friction_circle():
    p = _params()
    tire = make_tire_from_params(p)
    kappa = torch.full((5, 4), 0.8)
    alpha = torch.full((5, 4), 0.6)
    Fz = torch.full((5, 4), 9.17)
    mu = torch.full((5, 4), 0.9)
    fx, fy = tire.forces(kappa, alpha, Fz, mu)
    mag = torch.sqrt(fx ** 2 + fy ** 2)
    peak = tire.load_scaled_mu(Fz, mu) * Fz
    assert torch.all(mag <= peak * 1.0001)


def test_tire_lateral_sign_monotonic():
    p = _params()
    tire = make_tire_from_params(p)
    Fz = torch.full((1, 1), 9.17)
    mu = torch.full((1, 1), 0.9)
    a_small = torch.tensor([[0.05]])
    a_big = torch.tensor([[0.2]])
    _, fy_s = tire.forces(torch.zeros(1, 1), a_small, Fz, mu)
    _, fy_b = tire.forces(torch.zeros(1, 1), a_big, Fz, mu)
    assert fy_s.item() > 0.0 and fy_b.item() > fy_s.item()


# --- suspension ---------------------------------------------------------------
def test_static_loads_sum_to_weight():
    p = _params()
    fz = static_wheel_loads(p, 4, torch.device("cpu"), torch.float32)
    total = fz.sum(dim=1)
    assert torch.allclose(total, torch.full((4,), p.mass * p.gravity), atol=1e-3)


def test_longitudinal_transfer_direction():
    p = _params()
    ax = torch.tensor([5.0])  # forward accel -> load to rear
    ay = torch.tensor([0.0])
    fz = quasi_static_loads(p, ax, ay)
    rear = fz[0, 0] + fz[0, 1]
    front = fz[0, 2] + fz[0, 3]
    fz0 = quasi_static_loads(p, torch.zeros(1), torch.zeros(1))
    assert rear > (fz0[0, 0] + fz0[0, 1])
    assert front < (fz0[0, 2] + fz0[0, 3])


def test_lateral_transfer_direction():
    p = _params()
    # +ay (left turn) -> load onto right wheels (RR idx1, RF idx3)
    fz = quasi_static_loads(p, torch.zeros(1), torch.tensor([5.0]))
    assert fz[0, 1] > fz[0, 0]
    assert fz[0, 3] > fz[0, 2]


# --- drivetrain ---------------------------------------------------------------
def test_drivetrain_force_mode_sign():
    p = _params(throttle_mode="force")
    omega = torch.zeros(2, 4)
    mass = torch.full((2,), p.mass)
    mu = torch.full((2,), 0.9)
    v = torch.zeros(2)
    tau_fwd = wheel_axle_torques(p, torch.full((2,), 0.5), v, omega, mu, mass)
    tau_brake = wheel_axle_torques(p, torch.full((2,), -0.5), v, omega, mu, mass)
    assert torch.all(tau_fwd >= 0.0)
    # braking opposes a (near-zero) spin: with omega=0 the brake sign defaults to
    # +1 so torque is negative.
    assert torch.all(tau_brake <= 0.0)


def test_drivetrain_force_mode_coast_and_power_cap():
    p = _params(throttle_mode="force", f_drive_max=20.0, power_max=40.0, c_roll=0.0)
    omega = torch.zeros(1, 4)
    mass = torch.full((1,), p.mass)
    mu = torch.full((1,), 1.5)  # high mu so traction is not the limiter
    # Coast: zero throttle -> near-zero axle torque.
    tau0 = wheel_axle_torques(p, torch.zeros(1), torch.zeros(1), omega, mu, mass)
    assert float(tau0.abs().sum()) < 1e-6
    # At high speed, power_max / v limits drive force below f_drive_max.
    v_hi = torch.tensor([8.0])
    tau_hi = wheel_axle_torques(p, torch.ones(1), v_hi, omega, mu, mass)
    tau_lo = wheel_axle_torques(p, torch.ones(1), torch.tensor([0.5]), omega, mu, mass)
    assert float(tau_hi.abs().sum()) < float(tau_lo.abs().sum())


def test_drivetrain_force_mode_traction_cap():
    p = _params(throttle_mode="force", f_drive_max=200.0)
    omega = torch.zeros(1, 4)
    mass = torch.full((1,), p.mass)
    mu_low = torch.full((1,), 0.2)
    mu_hi = torch.full((1,), 1.2)
    v = torch.zeros(1)
    tau_low = wheel_axle_torques(p, torch.ones(1), v, omega, mu_low, mass)
    tau_hi = wheel_axle_torques(p, torch.ones(1), v, omega, mu_hi, mass)
    assert float(tau_low.abs().sum()) < float(tau_hi.abs().sum())


def test_drivetrain_speed_mode_regulates():
    p = _params(throttle_mode="speed", max_speed=8.0)
    omega = torch.zeros(1, 4)
    mass = torch.full((1,), p.mass)
    mu = torch.full((1,), 0.9)
    # below setpoint -> positive drive; above setpoint -> negative (brake)
    tau_lo = wheel_axle_torques(p, torch.tensor([0.5]), torch.tensor([1.0]),
                                omega, mu, mass)
    tau_hi = wheel_axle_torques(p, torch.tensor([0.5]), torch.tensor([7.9]),
                                omega, mu, mass)
    assert tau_lo.sum() > 0.0
    assert tau_hi.sum() < tau_lo.sum()


# --- integrator / dynamics ----------------------------------------------------
def test_straight_line_no_lateral():
    sim = TorchVehicleSim(_params(), 1, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 1)
    a = torch.tensor([[0.4, 0.0]])
    for _ in range(60):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    assert abs(st["base_lin_vel"][0, 1].item()) < 1e-3
    assert abs(st["base_ang_vel"][0, 2].item()) < 1e-3
    assert st["base_lin_vel"][0, 0].item() > 1.0


def test_left_steer_positive_yaw():
    sim = TorchVehicleSim(_params(), 1, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 1)
    a = torch.tensor([[0.15, 0.4]])
    for _ in range(40):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    assert st["base_ang_vel"][0, 2].item() > 0.0


def test_no_nans_under_aggressive_input():
    sim = TorchVehicleSim(_params(), 8, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 8)
    torch.manual_seed(0)
    for _ in range(50):
        a = torch.rand(8, 2) * 2 - 1
        sim.step(a, n_steps=20)
        st = sim.read_state()
        assert torch.isfinite(st["base_pos"]).all()
        assert torch.isfinite(st["base_lin_vel"]).all()


# --- batching / determinism / device -----------------------------------------
def test_batch_independence():
    sim = TorchVehicleSim(_params(), 3, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 3)
    a = torch.tensor([[0.5, 0.0], [0.5, 0.3], [0.5, -0.3]])
    for _ in range(30):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    r = st["base_ang_vel"][:, 2]
    assert abs(r[0].item()) < 1e-3          # straight
    assert r[1].item() > 0.0                # left
    assert r[2].item() < 0.0                # right


def test_determinism_same_seed():
    def run():
        sim = TorchVehicleSim(_params(), 4, sim_dt=0.005, control_dt=0.1)
        _reset(sim, 4)
        torch.manual_seed(123)
        out = []
        for _ in range(20):
            a = torch.rand(4, 2) * 2 - 1
            sim.step(a, n_steps=20)
            out.append(sim.read_state()["base_pos"].clone())
        return torch.stack(out)

    assert torch.equal(run(), run())


def test_dtype_and_shape_contract():
    sim = TorchVehicleSim(_params(), 5, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 5)
    sim.step(torch.zeros(5, 2), n_steps=20)
    st = sim.read_state()
    assert st["base_pos"].shape == (5, 3)
    assert st["base_quat"].shape == (5, 4)
    assert st["base_lin_vel"].shape == (5, 3)
    ws = sim.read_wheel_state()
    assert ws["motion_link_vel"].shape == (5, 4, 3)
    assert ws["dof_vel"].shape == (5, 4)


def test_domain_randomization_consumed():
    # drive_scale should scale forward drive effort; steer_bias should induce yaw
    # even with zero steering command. Measure early acceleration (before the speed
    # loop settles) so the check is valid for both throttle modes.
    def early_speed(scale):
        sim = TorchVehicleSim(_params(), 1, sim_dt=0.005, control_dt=0.1)
        _reset(sim, 1)
        sim.set_domain(torch.ones(1, dtype=torch.bool),
                       drive_scale=torch.tensor([scale]))
        a = torch.tensor([[0.5, 0.0]])
        for _ in range(6):
            sim.step(a, n_steps=20)
        return sim.read_state()["base_lin_vel"][0, 0].item()

    assert early_speed(1.5) > early_speed(0.5)

    sim = TorchVehicleSim(_params(), 1, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 1)
    sim.set_domain(torch.ones(1, dtype=torch.bool),
                   steer_bias=torch.tensor([0.15]))
    a = torch.tensor([[0.3, 0.0]])
    for _ in range(40):
        sim.step(a, n_steps=20)
    assert sim.read_state()["base_ang_vel"][0, 2].item() > 0.0


def test_kinematic_model_runs():
    p = _params()
    p.model = "kinematic"
    sim = TorchVehicleSim(p, 2, sim_dt=0.005, control_dt=0.1)
    _reset(sim, 2)
    a = torch.tensor([[0.6, 0.2], [0.6, -0.2]])
    for _ in range(30):
        sim.step(a, n_steps=20)
    st = sim.read_state()
    assert torch.isfinite(st["base_pos"]).all()
    assert st["base_ang_vel"][0, 2].item() > 0.0
    assert st["base_ang_vel"][1, 2].item() < 0.0

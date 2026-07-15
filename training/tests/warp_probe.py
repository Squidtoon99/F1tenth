"""CPU ``@wp.kernel`` probes that expose the device ``@wp.func`` physics to tests.

The tire/suspension/drivetrain/command primitives in ``f1tenth_sim`` are Warp
device functions, callable only from inside a kernel. These thin one-thread-per-
element wrappers let the component tests assert their behaviour directly (the Warp
analogue of importing the Torch functions), reusing the exact expected values from
the deleted TorchSim unit suite.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from f1tenth_sim.drivetrain import direct_drive_torque
from f1tenth_sim.dynamics import VehicleLocal, apply_command
from f1tenth_sim.params import SimParams, VehicleParams
from f1tenth_sim.suspension import static_wheel_load, warp_quasi_static_loads
from f1tenth_sim.tire import combined_pacejka

wp.init()


@wp.kernel(enable_backward=False)
def _probe_tire(
    kappa: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    fz: wp.array(dtype=wp.float32),
    mu: wp.array(dtype=wp.float32),
    fz0: wp.float32,
    params: SimParams,
    out_peak: wp.array(dtype=wp.float32),
    out_fx: wp.array(dtype=wp.float32),
    out_fy: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    tire = combined_pacejka(kappa[i], alpha[i], fz[i], mu[i], fz0, params)
    out_peak[i] = tire.peak
    out_fx[i] = tire.fx
    out_fy[i] = tire.fy


@wp.kernel(enable_backward=False)
def _probe_quasi_static(
    mass: wp.array(dtype=wp.float32),
    ax: wp.array(dtype=wp.float32),
    ay: wp.array(dtype=wp.float32),
    params: SimParams,
    out: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    out[i] = warp_quasi_static_loads(mass[i], ax[i], ay[i], params)


@wp.kernel(enable_backward=False)
def _probe_static_load(
    mass: wp.array(dtype=wp.float32),
    params: SimParams,
    out: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    out[i] = wp.vec4f(
        static_wheel_load(mass[i], 0, params),
        static_wheel_load(mass[i], 1, params),
        static_wheel_load(mass[i], 2, params),
        static_wheel_load(mass[i], 3, params),
    )


@wp.kernel(enable_backward=False)
def _probe_drive(
    effort: wp.array(dtype=wp.float32),
    vx: wp.array(dtype=wp.float32),
    omega: wp.array(dtype=wp.vec4f),
    mass: wp.array(dtype=wp.float32),
    mu: wp.array(dtype=wp.float32),
    drive_scale: wp.array(dtype=wp.float32),
    params: SimParams,
    out: wp.array(dtype=wp.vec4f),
):
    i = wp.tid()
    out[i] = direct_drive_torque(
        effort[i], vx[i], omega[i], mass[i], mu[i], drive_scale[i], params
    )


@wp.kernel(enable_backward=False)
def _probe_apply_command(
    effort_state: wp.array(dtype=wp.float32),
    steer: wp.array(dtype=wp.float32),
    action: wp.array(dtype=wp.vec2f),
    steer_bias: wp.array(dtype=wp.float32),
    params: SimParams,
    out_effort_state: wp.array(dtype=wp.float32),
    out_applied: wp.array(dtype=wp.float32),
    out_steer: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    vehicle = VehicleLocal()
    vehicle.effort_state = effort_state[i]
    vehicle.steer = steer[i]
    vehicle = apply_command(vehicle, action[i], steer_bias[i], params)
    out_effort_state[i] = vehicle.effort_state
    out_applied[i] = vehicle.applied_effort
    out_steer[i] = vehicle.steer


def make_params(sim_dt: float = 0.005, control_dt: float = 0.05, **cfg) -> SimParams:
    base = {"tire_friction": 0.9}
    base.update(cfg)
    return VehicleParams.from_config(base).to_warp(sim_dt=sim_dt, control_dt=control_dt)


def _f32(values) -> wp.array:
    return wp.from_torch(torch.as_tensor(values, dtype=torch.float32).contiguous())


def _vec4(values) -> wp.array:
    arr = torch.as_tensor(values, dtype=torch.float32).contiguous()
    return wp.from_torch(arr, dtype=wp.vec4f)


def tire(kappa, alpha, fz, mu, fz0, params: SimParams):
    """Return (peak, fx, fy) numpy arrays from ``combined_pacejka``."""
    kappa = np.atleast_1d(np.asarray(kappa, dtype=np.float32))
    n = kappa.shape[0]
    peak = torch.zeros(n)
    fx = torch.zeros(n)
    fy = torch.zeros(n)
    op, ofx, ofy = _f32(peak), _f32(fx), _f32(fy)
    wp.launch(
        _probe_tire,
        dim=n,
        inputs=[
            _f32(kappa),
            _f32(np.broadcast_to(alpha, kappa.shape).copy()),
            _f32(np.broadcast_to(fz, kappa.shape).copy()),
            _f32(np.broadcast_to(mu, kappa.shape).copy()),
            float(fz0),
            params,
            op,
            ofx,
            ofy,
        ],
        device="cpu",
    )
    return peak.numpy(), fx.numpy(), fy.numpy()


def quasi_static_loads(mass, ax, ay, params: SimParams) -> np.ndarray:
    mass = np.atleast_1d(np.asarray(mass, dtype=np.float32))
    n = mass.shape[0]
    out = torch.zeros(n, 4)
    wp.launch(
        _probe_quasi_static,
        dim=n,
        inputs=[
            _f32(mass),
            _f32(np.broadcast_to(ax, mass.shape).copy()),
            _f32(np.broadcast_to(ay, mass.shape).copy()),
            params,
            _vec4(out),
        ],
        device="cpu",
    )
    return out.numpy()


def static_loads(mass, params: SimParams) -> np.ndarray:
    mass = np.atleast_1d(np.asarray(mass, dtype=np.float32))
    n = mass.shape[0]
    out = torch.zeros(n, 4)
    wp.launch(
        _probe_static_load,
        dim=n,
        inputs=[_f32(mass), params, _vec4(out)],
        device="cpu",
    )
    return out.numpy()


def drive_torque(effort, vx, mass, mu, params: SimParams, omega=None, drive_scale=None):
    effort = np.atleast_1d(np.asarray(effort, dtype=np.float32))
    n = effort.shape[0]
    if omega is None:
        omega = np.zeros((n, 4), dtype=np.float32)
    if drive_scale is None:
        drive_scale = np.ones(n, dtype=np.float32)
    out = torch.zeros(n, 4)
    wp.launch(
        _probe_drive,
        dim=n,
        inputs=[
            _f32(effort),
            _f32(np.broadcast_to(vx, effort.shape).copy()),
            _vec4(np.asarray(omega, dtype=np.float32)),
            _f32(np.broadcast_to(mass, effort.shape).copy()),
            _f32(np.broadcast_to(mu, effort.shape).copy()),
            _f32(np.broadcast_to(drive_scale, effort.shape).copy()),
            params,
            _vec4(out),
        ],
        device="cpu",
    )
    return out.numpy()


def apply_command_step(effort_state, steer, action, steer_bias, params: SimParams):
    effort_state = np.atleast_1d(np.asarray(effort_state, dtype=np.float32))
    n = effort_state.shape[0]
    action = np.asarray(action, dtype=np.float32).reshape(n, 2)
    action_wp = wp.from_torch(
        torch.as_tensor(action, dtype=torch.float32).contiguous(), dtype=wp.vec2f
    )
    out_es = torch.zeros(n)
    out_ap = torch.zeros(n)
    out_st = torch.zeros(n)
    wp.launch(
        _probe_apply_command,
        dim=n,
        inputs=[
            _f32(effort_state),
            _f32(np.broadcast_to(steer, effort_state.shape).copy()),
            action_wp,
            _f32(np.broadcast_to(steer_bias, effort_state.shape).copy()),
            params,
            _f32(out_es),
            _f32(out_ap),
            _f32(out_st),
        ],
        device="cpu",
    )
    return out_es.numpy(), out_ap.numpy(), out_st.numpy()

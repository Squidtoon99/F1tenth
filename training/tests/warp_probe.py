"""CPU Warp probes for the reduced bicycle plant."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from f1tenth_sim.dynamics import (
    VehicleLocal,
    apply_command,
    apply_current_slew,
    apply_steering_lag,
)
from f1tenth_sim.params import SimParams, VehicleParams

wp.init()


@wp.kernel(enable_backward=False)
def _probe_command_and_slew(
    i_drive: wp.array(dtype=wp.float32),
    i_brake: wp.array(dtype=wp.float32),
    steer_logical: wp.array(dtype=wp.float32),
    steer: wp.array(dtype=wp.float32),
    action: wp.array(dtype=wp.vec2f),
    steer_bias: wp.array(dtype=wp.float32),
    params: SimParams,
    out_drive: wp.array(dtype=wp.float32),
    out_brake: wp.array(dtype=wp.float32),
    out_applied: wp.array(dtype=wp.float32),
    out_logical: wp.array(dtype=wp.float32),
    out_steer: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    vehicle = VehicleLocal()
    vehicle.i_drive = i_drive[i]
    vehicle.i_brake = i_brake[i]
    vehicle.steer_logical = steer_logical[i]
    vehicle.steer = steer[i]
    vehicle = apply_command(vehicle, action[i], steer_bias[i], params)
    vehicle = apply_current_slew(vehicle, params)
    vehicle = apply_steering_lag(vehicle, params)
    out_drive[i] = vehicle.i_drive
    out_brake[i] = vehicle.i_brake
    out_applied[i] = vehicle.applied_effort
    out_logical[i] = vehicle.steer_logical
    out_steer[i] = vehicle.steer


def make_params(sim_dt: float = 0.005, control_dt: float = 0.05, **cfg) -> SimParams:
    base = {"tire_friction": 0.9}
    base.update(cfg)
    return VehicleParams.from_config(base).to_warp(sim_dt=sim_dt, control_dt=control_dt)


def _f32(values) -> wp.array:
    return wp.from_torch(torch.as_tensor(values, dtype=torch.float32).contiguous())


def command_and_slew(
    i_drive,
    i_brake,
    action,
    params: SimParams,
    steer_logical=0.0,
    steer=0.0,
    steer_bias=0.0,
):
    i_drive = np.atleast_1d(np.asarray(i_drive, dtype=np.float32))
    n = i_drive.shape[0]
    action = np.asarray(action, dtype=np.float32).reshape(n, 2)
    action_wp = wp.from_torch(
        torch.as_tensor(action, dtype=torch.float32).contiguous(), dtype=wp.vec2f
    )
    out_drive = torch.zeros(n)
    out_brake = torch.zeros(n)
    out_applied = torch.zeros(n)
    out_logical = torch.zeros(n)
    out_steer = torch.zeros(n)
    wp.launch(
        _probe_command_and_slew,
        dim=n,
        inputs=[
            _f32(i_drive),
            _f32(np.broadcast_to(i_brake, i_drive.shape).copy()),
            _f32(np.broadcast_to(steer_logical, i_drive.shape).copy()),
            _f32(np.broadcast_to(steer, i_drive.shape).copy()),
            action_wp,
            _f32(np.broadcast_to(steer_bias, i_drive.shape).copy()),
            params,
            _f32(out_drive),
            _f32(out_brake),
            _f32(out_applied),
            _f32(out_logical),
            _f32(out_steer),
        ],
        device="cpu",
    )
    return (
        out_drive.numpy(),
        out_brake.numpy(),
        out_applied.numpy(),
        out_logical.numpy(),
        out_steer.numpy(),
    )

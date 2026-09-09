from __future__ import annotations

import torch
import warp as wp

from .dynamics import (
    VehicleBuffers,
    apply_command_and_integrate,
    load_vehicle,
    store_vehicle,
)
from .params import SimParams, VehicleParams


@wp.kernel(enable_backward=False)
def vehicle_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    buffers: VehicleBuffers,
    params: SimParams,
    substeps: wp.int32,
):
    env_id = wp.tid()
    vehicle = load_vehicle(buffers, env_id)
    vehicle = apply_command_and_integrate(
        vehicle,
        actions[env_id],
        buffers.steer_bias[env_id],
        buffers.mass[env_id],
        buffers.mu[env_id],
        buffers.drive_scale[env_id],
        params,
        substeps,
    )
    store_vehicle(buffers, env_id, vehicle)


class WarpVehicleSim:
    def __init__(
        self,
        params: VehicleParams,
        num_envs: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sim_dt: float = 0.005,
        control_dt: float = 0.05,
    ):
        if dtype != torch.float32:
            raise ValueError("WarpVehicleSim supports float32 only")
        self.params = params
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("WarpVehicleSim supports CPU and CUDA devices")
        self.dtype = dtype
        self.sim_dt = float(sim_dt)
        self.control_dt = float(control_dt)
        self.wp_device = wp.get_device(
            "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        )
        self._params = params.to_warp(sim_dt=self.sim_dt, control_dt=self.control_dt)
        self.s = self._allocate_state()
        self._buffers = self._make_buffers()

    def _zeros(self, *shape):
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def _allocate_state(self):
        n = self.num_envs
        state = {
            "X": self._zeros(n),
            "Y": self._zeros(n),
            "z": self._zeros(n),
            "yaw": self._zeros(n),
            "vx": self._zeros(n),
            "vy": self._zeros(n),
            "r": self._zeros(n),
            "steer": self._zeros(n),
            "longitudinal_effort": self._zeros(n),
            "throttle": self._zeros(n),
            "ax": self._zeros(n),
            "ay": self._zeros(n),
            "omega": self._zeros(n, 4),
            "slip_ratio": self._zeros(n, 4),
            "slip_angle": self._zeros(n, 4),
            "load_ratio": torch.ones(n, 4, device=self.device),
            "fx_lag": self._zeros(n, 4),
            "fy_lag": self._zeros(n, 4),
            "mass": torch.full((n,), self.params.mass, device=self.device),
            "mu": torch.full((n,), self.params.tire_mu, device=self.device),
            "drive_scale": torch.ones(n, device=self.device),
            "steer_bias": self._zeros(n),
        }
        return state

    def _make_buffers(self):
        buffers = VehicleBuffers()
        scalar_fields = {
            "x": "X",
            "y": "Y",
            "yaw": "yaw",
            "vx": "vx",
            "vy": "vy",
            "yaw_rate": "r",
            "steer": "steer",
            "effort_state": "longitudinal_effort",
            "applied_effort": "throttle",
            "ax": "ax",
            "ay": "ay",
            "mass": "mass",
            "mu": "mu",
            "drive_scale": "drive_scale",
            "steer_bias": "steer_bias",
        }
        for field, key in scalar_fields.items():
            setattr(buffers, field, wp.from_torch(self.s[key]))
        for field in (
            "omega",
            "slip_ratio",
            "slip_angle",
            "load_ratio",
            "fx_lag",
            "fy_lag",
        ):
            setattr(
                buffers,
                field,
                wp.from_torch(self.s[field], dtype=wp.vec4f),
            )
        return buffers

    def _stream(self):
        if self.device.type == "cuda":
            return wp.stream_from_torch(torch.cuda.current_stream(self.device))
        return None

    def reset(self, mask, pos, quat, speed):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        pos = pos.to(device=self.device, dtype=torch.float32)
        quat = quat.to(device=self.device, dtype=torch.float32)
        speed = speed.to(device=self.device, dtype=torch.float32)
        sin_yaw = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cos_yaw = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        yaw = torch.atan2(sin_yaw, cos_yaw)
        self.s["X"][mask] = pos[mask, 0]
        self.s["Y"][mask] = pos[mask, 1]
        self.s["z"][mask] = pos[mask, 2]
        self.s["yaw"][mask] = yaw[mask]
        self.s["vx"][mask] = speed[mask]
        for key in (
            "vy",
            "r",
            "steer",
            "longitudinal_effort",
            "throttle",
            "ax",
            "ay",
            "omega",
            "slip_ratio",
            "slip_angle",
            "fx_lag",
            "fy_lag",
        ):
            self.s[key][mask] = 0.0
        self.s["omega"][mask] = speed[mask, None] / self.params.wheel_radius
        self.s["load_ratio"][mask] = 1.0

    def set_domain(self, mass, mu, drive_scale, steer_bias):
        self.s["mass"].copy_(mass.to(self.device, torch.float32))
        self.s["mu"].copy_(mu.to(self.device, torch.float32))
        self.s["drive_scale"].copy_(drive_scale.to(self.device, torch.float32))
        self.s["steer_bias"].copy_(steer_bias.to(self.device, torch.float32))

    def step(self, actions, n_steps: int = 10):
        actions = actions.to(device=self.device, dtype=torch.float32).contiguous()
        wp.launch(
            vehicle_step_kernel,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(actions, dtype=wp.vec2f),
                self._buffers,
                self._params,
                int(n_steps),
            ],
            device=self.wp_device,
            stream=self._stream(),
        )

    def read_state(self):
        yaw = self.s["yaw"]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        vx = self.s["vx"]
        vy = self.s["vy"]
        return {
            "base_pos": torch.stack((self.s["X"], self.s["Y"], self.s["z"]), dim=1),
            "base_quat": torch.stack(
                (
                    torch.cos(0.5 * yaw),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(0.5 * yaw),
                ),
                dim=1,
            ),
            "base_vel_world": torch.stack(
                (cosine * vx - sine * vy, sine * vx + cosine * vy, torch.zeros_like(vx)),
                dim=1,
            ),
            "base_lin_vel": torch.stack((vx, vy, torch.zeros_like(vx)), dim=1),
            "base_ang_vel": torch.stack(
                (torch.zeros_like(vx), torch.zeros_like(vx), self.s["r"]), dim=1
            ),
            "base_lin_acc": torch.stack(
                (self.s["ax"], self.s["ay"], torch.zeros_like(vx)), dim=1
            ),
        }

    def read_wheel_state(self):
        return {
            "dof_vel": self.s["omega"],
            "tyre_slip": torch.cat(
                (self.s["slip_ratio"], self.s["slip_angle"]), dim=1
            ),
            "tyre_load": self.s["load_ratio"],
        }

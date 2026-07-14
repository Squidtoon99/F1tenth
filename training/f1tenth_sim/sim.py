"""TorchVehicleSim: the batched state container + step/reset/readback surface.

Owns all per-env state tensors and exposes exactly the buffers the training env
needs, so ``TorchSimBackend`` is a thin adapter.

Wheel order is ``[LR, RR, LF, RF]``. Quaternions are ``wxyz``.
"""

from __future__ import annotations

import torch

from . import dynamics
from .params import VehicleParams
from .suspension import SuspensionFilter, static_wheel_loads
from .tire import make_tire_from_params


class TorchVehicleSim:
    def __init__(
        self,
        params: VehicleParams,
        num_envs: int,
        device=None,
        dtype=torch.float32,
        sim_dt: float = 0.005,
        control_dt: float = 0.1,
        internal_substeps: int = 1,
    ):
        self.params = params
        self.num_envs = int(num_envs)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.sim_dt = float(sim_dt)
        self.control_dt = float(control_dt)
        self.internal_substeps = max(int(internal_substeps), 1)
        self.steer_lag_alpha = self.control_dt / (params.t_delta + self.control_dt)

        self.tire = make_tire_from_params(params)
        self._step_dynamic = (
            torch.compile(dynamics.step_dynamic, mode="default")
            if self.device.type == "cuda"
            else dynamics.step_dynamic
        )
        self.susp = (
            SuspensionFilter(params, self.num_envs, self.device, dtype)
            if params.suspension_mode == "dynamic"
            else None
        )

        z = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        z4 = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        self.s = {
            "X": z.clone(), "Y": z.clone(), "z": z.clone(), "yaw": z.clone(),
            "vx": z.clone(), "vy": z.clone(), "r": z.clone(),
            "omega": z4.clone(), "steer": z.clone(), "throttle": z.clone(),
            "longitudinal_effort": z.clone(),
            "ax": z.clone(), "ay": z.clone(),
            "mass": torch.full((self.num_envs,), params.mass, device=self.device,
                               dtype=dtype),
            "mu": torch.full((self.num_envs,), params.tire_mu, device=self.device,
                             dtype=dtype),
            "drive_scale": torch.ones((self.num_envs,), device=self.device,
                                      dtype=dtype),
            "steer_bias": z.clone(),
        }
        if params.tire_relax_len > 0.0:
            self.s["fx_lag"] = z4.clone()
            self.s["fy_lag"] = z4.clone()

        # Native tyre readback (single source for the slip observation): per-wheel
        # slip ratio + geometric slip angle from the tyre model, and normal-load
        # ratio Fz / Fz_static. Filled each substep; defaults are the at-rest values
        # (no slip, static load) so a readback before the first step is well-defined.
        self._tyre_slip = torch.zeros((self.num_envs, 8), device=self.device,
                                      dtype=dtype)
        self._tyre_load = torch.ones((self.num_envs, 4), device=self.device,
                                     dtype=dtype)

    # --- lifecycle ---------------------------------------------------------
    def reset(self, mask, pos, quat, speed):
        mask = self._as_mask(mask)
        m = mask
        m1 = mask.unsqueeze(1)
        yaw = _quat_to_yaw(quat)
        v = speed.to(self.dtype)
        self.s["X"] = torch.where(m, pos[:, 0].to(self.dtype), self.s["X"])
        self.s["Y"] = torch.where(m, pos[:, 1].to(self.dtype), self.s["Y"])
        self.s["z"] = torch.where(m, pos[:, 2].to(self.dtype), self.s["z"])
        self.s["yaw"] = torch.where(m, yaw.to(self.dtype), self.s["yaw"])
        self.s["vx"] = torch.where(m, v, self.s["vx"])
        self.s["vy"] = torch.where(m, torch.zeros_like(v), self.s["vy"])
        self.s["r"] = torch.where(m, torch.zeros_like(v), self.s["r"])
        self.s["ax"] = torch.where(m, torch.zeros_like(v), self.s["ax"])
        self.s["ay"] = torch.where(m, torch.zeros_like(v), self.s["ay"])
        self.s["steer"] = torch.where(m, torch.zeros_like(v), self.s["steer"])
        self.s["throttle"] = torch.where(m, torch.zeros_like(v), self.s["throttle"])
        self.s["longitudinal_effort"] = torch.where(
            m, torch.zeros_like(v), self.s["longitudinal_effort"]
        )
        omega0 = (v / max(self.params.wheel_radius, 1e-6)).unsqueeze(1).expand(-1, 4)
        self.s["omega"] = torch.where(m1, omega0, self.s["omega"])
        for key in ("fx_lag", "fy_lag"):
            if key in self.s:
                self.s[key] = torch.where(m1, torch.zeros_like(self.s[key]),
                                          self.s[key])
        self._tyre_slip = torch.where(m1, torch.zeros_like(self._tyre_slip),
                                      self._tyre_slip)
        self._tyre_load = torch.where(m1, torch.ones_like(self._tyre_load),
                                      self._tyre_load)
        if self.susp is not None:
            self.susp.reset(mask, self.params)

    def set_domain(self, mask, mass=None, mu=None, drive_scale=None, steer_bias=None):
        mask = self._as_mask(mask)
        if mass is not None:
            self.s["mass"] = torch.where(mask, mass.to(self.dtype), self.s["mass"])
        if mu is not None:
            self.s["mu"] = torch.where(mask, mu.to(self.dtype), self.s["mu"])
        if drive_scale is not None:
            self.s["drive_scale"] = torch.where(
                mask, drive_scale.to(self.dtype), self.s["drive_scale"]
            )
        if steer_bias is not None:
            self.s["steer_bias"] = torch.where(
                mask, steer_bias.to(self.dtype), self.s["steer_bias"]
            )

    # --- actuation / integration ------------------------------------------
    def apply_actions(self, exec_actions):
        target = exec_actions[:, 0].to(self.dtype)
        rate = self.params.longitudinal_slew_rate_per_s
        if rate <= 0.0:
            effort = target
            throttle = target
        else:
            start = self.s["longitudinal_effort"]
            delta = target - start
            max_step = rate * self.control_dt
            effort = start + delta.clamp(-max_step, max_step)
            # Use the exact interval-average of a linear ramp that holds the target
            # after reaching it. This models current slew without per-substep kernels.
            reaches_target = delta.abs() <= max_step
            ramp_then_hold = target - delta * delta.abs() / (2.0 * max_step)
            full_interval_ramp = 0.5 * (start + effort)
            throttle = torch.where(
                reaches_target, ramp_then_hold, full_interval_ramp
            )
        self.s["longitudinal_effort"] = effort
        steer_norm = exec_actions[:, 1].to(self.dtype)
        delta_cmd = (steer_norm * self.params.max_steer + self.s["steer_bias"]).clamp(
            -self.params.max_steer, self.params.max_steer
        )
        self.s["steer"] = self.s["steer"] + self.steer_lag_alpha * (
            delta_cmd - self.s["steer"]
        )
        self.s["throttle"] = throttle

    def substep(self, n_steps: int = 1):
        dt = self.sim_dt / self.internal_substeps
        total = int(n_steps) * self.internal_substeps
        diag = None
        if self.params.model == "kinematic":
            for _ in range(total):
                diag = dynamics.step_kinematic(self.s, self.params, dt)
        else:
            for _ in range(total):
                diag = self._step_dynamic(
                    self.s, self.params, self.tire, dt, self.susp
                )
        self._update_tyre_readback(diag)

    def _update_tyre_readback(self, diag) -> None:
        if diag is None:
            return
        kappa = diag.get("kappa")
        slip_angle = diag.get("slip_angle_obs")
        if kappa is not None and slip_angle is not None:
            self._tyre_slip = torch.cat([kappa, slip_angle], dim=-1)
        Fz = diag.get("Fz")
        if Fz is not None:
            static_load = static_wheel_loads(
                self.params,
                self.num_envs,
                self.device,
                self.dtype,
                self.s["mass"],
            ).clamp_min(1e-6)
            self._tyre_load = Fz / static_load

    def step(self, actions, n_steps: int = 1):
        self.apply_actions(actions)
        self.substep(n_steps)

    # --- readback ----------------------------------------------------------
    def read_state(self) -> dict:
        s = self.s
        cos_y = torch.cos(s["yaw"])
        sin_y = torch.sin(s["yaw"])
        base_pos = torch.stack([s["X"], s["Y"], s["z"]], dim=-1)
        base_quat = _yaw_to_quat_wxyz(s["yaw"])
        vworld = torch.stack(
            [s["vx"] * cos_y - s["vy"] * sin_y,
             s["vx"] * sin_y + s["vy"] * cos_y,
             torch.zeros_like(s["vx"])],
            dim=-1,
        )
        zeros = torch.zeros_like(s["vx"])
        base_lin_vel = torch.stack([s["vx"], s["vy"], zeros], dim=-1)
        base_ang_vel = torch.stack([zeros, zeros, s["r"]], dim=-1)
        base_lin_acc = torch.stack([s["ax"], s["ay"], zeros], dim=-1)
        return {
            "base_pos": base_pos,
            "base_quat": base_quat,
            "base_vel_world": vworld,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "base_lin_acc": base_lin_acc,
        }

    def read_wheel_state(self) -> dict:
        s = self.s
        delta_lr = dynamics.ackermann_wheel_angles(self.params, s["steer"])
        delta_wheel = torch.zeros_like(s["omega"])
        delta_wheel[:, 2] = delta_lr[:, 0]
        delta_wheel[:, 3] = delta_lr[:, 1]
        offsets = dynamics.wheel_offsets(self.params, self.device, self.dtype)
        v_long, v_lat = dynamics.wheel_frame_velocities(
            self.params, s["vx"], s["vy"], s["r"], delta_wheel, offsets
        )
        motion = torch.stack([v_long, v_lat, torch.zeros_like(v_long)], dim=-1)
        return {
            "motion_link_vel": motion,
            "dof_vel": s["omega"],
            "tyre_slip": self._tyre_slip,
            "tyre_load": self._tyre_load,
        }

    # --- helpers -----------------------------------------------------------
    def _as_mask(self, mask) -> torch.Tensor:
        if mask is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask = torch.as_tensor(mask, device=self.device)
        if mask.dtype != torch.bool:
            out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            out[mask.long()] = True
            return out
        return mask


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat

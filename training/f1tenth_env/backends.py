"""Torch vehicle simulation adapter used by :class:`F1tenthEnv`."""

from __future__ import annotations

import torch

from . import runtime as rt
from .terminations import obb_overlap_mtv


class TorchSimBackend:
    def __init__(
        self,
        *,
        num_envs,
        env_cfg,
        device,
    ):
        from f1tenth_sim import TorchVehicleSim, VehicleParams

        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.device = device or torch.device("cpu")
        self.dtype = rt.tc_float
        self.has_opponent = env_cfg.get("opponent_strategy") is not None

        self.dt = float(env_cfg.get("sim_dt", 0.005))
        self.control_interval = int(env_cfg.get("control_interval", 20))
        self.control_dt = self.dt * self.control_interval
        self._internal = int(
            (env_cfg.get("torch_sim") or {}).get("internal_substeps", 1)
        )
        self.params = VehicleParams.from_config(env_cfg)
        self._TorchVehicleSim = TorchVehicleSim
        self._car_len = float(env_cfg.get("car_length", 0.568))
        self._car_wid = float(env_cfg.get("car_width", 0.296))
        self._restitution = float(
            (env_cfg.get("torch_sim") or {}).get("contact_restitution", 0.1)
        )

    def build(self) -> None:
        self.sim = self._TorchVehicleSim(
            self.params,
            self.num_envs,
            device=self.device,
            dtype=self.dtype,
            sim_dt=self.dt,
            control_dt=self.control_dt,
            internal_substeps=self._internal,
        )
        self.opp_sim = (
            self._TorchVehicleSim(
                self.params,
                self.num_envs,
                device=self.device,
                dtype=self.dtype,
                sim_dt=self.dt,
                control_dt=self.control_dt,
                internal_substeps=self._internal,
            )
            if self.has_opponent
            else None
        )

    def _domain(self, mask, dr):
        if not dr.get("enabled"):
            return None
        return {
            "mass": (dr["vehicle_mass"] * dr["mass_scale"]).to(self.dtype),
            "mu": dr["tire_friction"].to(self.dtype),
            "drive_scale": dr.get("drive_scale"),
            "steer_bias": dr.get("steer_bias"),
        }

    def reset(self, mask, pos, quat, speed, opp_pos, opp_quat, opp_speed, dr) -> None:
        self.sim.reset(mask, pos, quat, speed)
        domain = self._domain(mask, dr)
        if domain is not None:
            self.sim.set_domain(mask, **domain)
        if self.opp_sim is not None and opp_pos is not None:
            speed = opp_speed if opp_speed is not None else torch.zeros_like(speed)
            self.opp_sim.reset(mask, opp_pos, opp_quat, speed)
            if domain is not None:
                self.opp_sim.set_domain(mask, **domain)

    def apply_ego_actions(self, exec_actions) -> None:
        self.sim.apply_actions(exec_actions)

    def apply_opp_actions(self, opp_actions) -> None:
        if self.opp_sim is not None:
            self.opp_sim.apply_actions(opp_actions)

    def substep(self, n_steps: int) -> None:
        self.sim.substep(n_steps)
        if self.opp_sim is not None:
            self.opp_sim.substep(n_steps)
            self._resolve_contact()

    def _resolve_contact(self) -> None:
        a = self.sim.s
        b = self.opp_sim.s
        pa = torch.stack([a["X"], a["Y"]], dim=-1)
        pb = torch.stack([b["X"], b["Y"]], dim=-1)
        overlap, normal, depth = obb_overlap_mtv(
            pa, a["yaw"], pb, b["yaw"], self._car_len, self._car_wid
        )
        nx = normal[:, 0]
        ny = normal[:, 1]
        half = 0.5 * depth
        a["X"] = a["X"] + half * nx
        a["Y"] = a["Y"] + half * ny
        b["X"] = b["X"] - half * nx
        b["Y"] = b["Y"] - half * ny

        vax, vay = _body_to_world_vel(a)
        vbx, vby = _body_to_world_vel(b)
        rel = (vax - vbx) * nx + (vay - vby) * ny
        closing = rel.clamp_max(0.0) * overlap.to(rel.dtype)
        impulse = -(1.0 + self._restitution) * closing * 0.5
        vax = vax + impulse * nx
        vay = vay + impulse * ny
        vbx = vbx - impulse * nx
        vby = vby - impulse * ny
        _world_to_body_vel(a, vax, vay)
        _world_to_body_vel(b, vbx, vby)

    def read_state(self) -> dict[str, torch.Tensor]:
        out = self.sim.read_state()
        if self.opp_sim is not None:
            opp = self.opp_sim.read_state()
            out["opp_base_pos"] = opp["base_pos"]
            out["opp_base_quat"] = opp["base_quat"]
            out["opp_vel_world"] = opp["base_vel_world"]
            out["opp_ang_world"] = opp["base_ang_vel"]
        return out

    def read_wheel_state(self, which: str = "ego") -> dict[str, torch.Tensor]:
        sim = self.opp_sim if which == "opp" else self.sim
        return sim.read_wheel_state()

    def close(self) -> None:
        pass


def _body_to_world_vel(state):
    cos_yaw = torch.cos(state["yaw"])
    sin_yaw = torch.sin(state["yaw"])
    return (
        state["vx"] * cos_yaw - state["vy"] * sin_yaw,
        state["vx"] * sin_yaw + state["vy"] * cos_yaw,
    )


def _world_to_body_vel(state, vel_x, vel_y):
    cos_yaw = torch.cos(state["yaw"])
    sin_yaw = torch.sin(state["yaw"])
    state["vx"] = vel_x * cos_yaw + vel_y * sin_yaw
    state["vy"] = -vel_x * sin_yaw + vel_y * cos_yaw

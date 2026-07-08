"""Physics-backend seam for :class:`F1tenthEnv`.

The env owns reset sampling, observation/reward/termination wiring and the
opponent controller; the *backend* owns the vehicle physics: entity/state setup,
actuation, stepping and state readback. Two interchangeable implementations:

- :class:`GenesisBackend` -- the existing Genesis rigid-body engine (default).
- :class:`TorchSimBackend` -- the pure-Torch :class:`f1tenth_sim.TorchVehicleSim`.

Both present the same small surface (see :class:`VehicleBackend`) so the env logic
and the QRSAC trainer are identical regardless of backend, enabling A/B comparison.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

from .car import (
    URDF_PATH,
    ackermann_left_right,
    compute_dissipative_force_world,
    compute_wheel_torques,
    setup_entity_controls,
)
from .domain_randomization import apply_dr_physics
from .utils import draw_track_boundaries_debug


class VehicleBackend(Protocol):
    has_opponent: bool

    def build(self) -> None: ...

    def reset(
        self,
        mask: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        speed: torch.Tensor,
        opp_pos: torch.Tensor | None,
        opp_quat: torch.Tensor | None,
        dr: dict[str, Any],
    ) -> None: ...

    def apply_ego_actions(
        self, exec_actions: torch.Tensor, base_lin_vel: torch.Tensor,
        dr: dict[str, Any],
    ) -> None: ...

    def apply_opp_actions(
        self, opp_actions: torch.Tensor, opp_body_vel: torch.Tensor,
        dr: dict[str, Any],
    ) -> None: ...

    def substep(self, n_steps: int) -> None: ...

    def read_state(self) -> dict[str, torch.Tensor]: ...

    def read_wheel_state(self, which: str = "ego") -> dict[str, torch.Tensor]: ...


def make_backend(kind: str, **kwargs) -> VehicleBackend:
    kind = (kind or "genesis").lower()
    if kind == "torch":
        return TorchSimBackend(**kwargs)
    if kind == "genesis":
        return GenesisBackend(**kwargs)
    raise ValueError(f"unknown physics backend: {kind!r} (expected genesis|torch)")


class GenesisBackend:
    """Wraps the Genesis scene/entities and the actuation extracted from env.py."""

    def __init__(self, *, num_envs, env_cfg, obs_cfg, reward_cfg, track_state,
                 device, show_viewer=False, enable_recording=False):
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.track_state = track_state
        self.device = device
        self.show_viewer = show_viewer
        self.enable_recording = enable_recording
        self.has_opponent = env_cfg.get("opponent_strategy") is not None

        self.dt = float(env_cfg.get("sim_dt", 0.01))
        self.control_interval = int(env_cfg.get("control_interval", 10))
        self.control_dt = self.dt * self.control_interval
        t_delta = float(env_cfg.get("t_delta", 0.1))
        self.steer_lag_alpha = self.control_dt / (t_delta + self.control_dt)
        self.spawn_z = float(env_cfg.get("car_spawn_pos", (0.0, 0.0, 0.01))[2])

    def build(self) -> None:
        env_cfg = self.env_cfg
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -5.0, 3.5),
                camera_lookat=(0.4, 0.0, 0.2),
                camera_fov=35,
                res=(960, 640),
                max_FPS=int(1.0 / self.dt),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                batch_links_info=True,
                constraint_solver=gs.constraint_solver.Newton,
                constraint_timeconst=float(
                    env_cfg.get("constraint_timeconst", max(0.02, 2.0 * self.dt))
                ),
                iterations=int(env_cfg.get("solver_iterations", 50)),
                ls_iterations=int(env_cfg.get("solver_ls_iterations", 50)),
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt, substeps=int(env_cfg.get("sim_substeps", 10))
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=bool(env_cfg.get("show_fps", False)),
            ),
            show_viewer=self.show_viewer,
        )

        self.ground = self.scene.add_entity(gs.morphs.Plane())
        self.ground.set_friction(float(env_cfg.get("tire_friction", 0.7)))

        self.car = self.scene.add_entity(
            gs.morphs.URDF(
                file=URDF_PATH,
                pos=env_cfg["car_spawn_pos"],
                euler=env_cfg["car_spawn_rot"],
                recompute_inertia=False,
                default_armature=0.0,
            )
        )
        if self.has_opponent:
            self.opponent = self.scene.add_entity(
                gs.morphs.URDF(
                    file=URDF_PATH,
                    pos=env_cfg["car_spawn_pos"],
                    euler=env_cfg["car_spawn_rot"],
                    recompute_inertia=False,
                )
            )
        else:
            self.opponent = None

        if self.scene.viewer and self.show_viewer:
            self.scene.viewer.follow_entity(self.car)

        if self.enable_recording:
            self.cam1 = self.scene.add_camera(
                res=(1024, 1024), pos=(2, 0, 1), lookat=(0, 0, 0.5), debug=True
            )
        else:
            self.cam1 = None

        self.scene.build(n_envs=self.num_envs)
        if self.show_viewer or self.enable_recording:
            draw_track_boundaries_debug(
                scene=self.scene,
                centerline=self.track_state["centerline"],
                w_tr_left=self.track_state["w_tr_left"],
                w_tr_right=self.track_state["w_tr_right"],
                reward_cfg=self.reward_cfg,
            )
        if self.cam1 is not None:
            self.cam1.start_recording()

        self.wheel_dofs, self.steer_dofs = setup_entity_controls(self.car, env_cfg)
        if self.opponent is not None:
            setup_entity_controls(self.opponent, env_cfg)

        self.base_link_idx = self.car.get_link("base_link").idx
        self.base_link_idx_local = self.car.get_link("base_link").idx_local
        self.dr_wheel_link_ids = [
            self.car.get_link(name).idx_local
            for name in (
                "base_link", "left_rear_wheel", "right_rear_wheel",
                "left_front_wheel", "right_front_wheel",
            )
        ]
        self.root_dof_vel_idx = list(
            self.car.get_joint("root_joint").dofs_idx_local[3:6]
        )
        self.slip_motion_link_idx = [
            self.car.get_link("left_rear_wheel").idx_local,
            self.car.get_link("right_rear_wheel").idx_local,
            self.car.get_link("left_front_wheel").idx_local,
            self.car.get_link("right_front_wheel").idx_local,
        ]
        self.slip_frame_link_idx = [
            self.car.get_link("base_link").idx_local,
            self.car.get_link("base_link").idx_local,
            self.car.get_link("left_steering_hinge").idx_local,
            self.car.get_link("right_steering_hinge").idx_local,
        ]

        self.steer_state = torch.zeros((self.num_envs,), dtype=gs.tc_float,
                                       device=gs.device)
        self.opp_steer_state = torch.zeros((self.num_envs,), dtype=gs.tc_float,
                                           device=gs.device)
        self._drag_force = None

    def reset(self, mask, pos, quat, speed, opp_pos, opp_quat, dr) -> None:
        self.car.set_pos(pos, envs_idx=mask, zero_velocity=True, relative=False)
        self.car.set_quat(quat, envs_idx=mask, zero_velocity=True, relative=False)

        if self.opponent is not None and opp_pos is not None:
            self.opponent.set_pos(opp_pos, envs_idx=mask, zero_velocity=True,
                                  relative=False)
            self.opponent.set_quat(opp_quat, envs_idx=mask, zero_velocity=True,
                                   relative=False)
            self.opp_steer_state = torch.where(
                mask, torch.zeros_like(self.opp_steer_state), self.opp_steer_state
            )

        v_min = min(speed.min().item(), 0.0) if speed.numel() else 0.0
        launch = bool((speed.abs() > 1e-8).any().item())
        self.steer_state = torch.where(
            mask, torch.zeros_like(self.steer_state), self.steer_state
        )
        if launch:
            self._apply_launch_velocity(mask, quat, speed)

        if dr.get("enabled"):
            apply_dr_physics(self.car, self.opponent, dr, mask, self.dr_wheel_link_ids)
            if mask.any():
                gf = float(dr["ground_friction"][mask].mean().item())
                self.ground.set_friction(gf)
        _ = v_min

    def _apply_launch_velocity(self, mask, quat, speed) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        n = int(env_ids.numel())
        if n == 0:
            return
        wheel_radius = max(float(self.env_cfg.get("wheel_radius", 0.05)), 1e-6)
        sp = speed[env_ids]
        wheel_omega = (sp / wheel_radius).unsqueeze(1).expand(
            n, len(self.wheel_dofs)
        ).contiguous()
        steer_zeros = torch.zeros((n, len(self.steer_dofs)), dtype=gs.tc_float,
                                  device=self.device)
        yaw = gu.quat_to_xyz(quat[env_ids], rpy=True, degrees=False)[:, 2]
        root_vel = torch.stack(
            [sp * torch.cos(yaw), sp * torch.sin(yaw), torch.zeros_like(sp)], dim=-1
        )
        self.car.set_dofs_position(steer_zeros, self.steer_dofs, envs_idx=env_ids,
                                   zero_velocity=False)
        self.car.set_dofs_velocity(wheel_omega, self.wheel_dofs, envs_idx=env_ids)
        self.car.set_dofs_velocity(root_vel, self.root_dof_vel_idx, envs_idx=env_ids)
        if self.opponent is not None:
            self.opponent.set_dofs_position(steer_zeros, self.steer_dofs,
                                            envs_idx=env_ids, zero_velocity=False)
            self.opponent.set_dofs_velocity(wheel_omega, self.wheel_dofs,
                                            envs_idx=env_ids)
            self.opponent.set_dofs_velocity(root_vel, self.root_dof_vel_idx,
                                            envs_idx=env_ids)

    def _actuate(self, entity, exec_actions, steer_state, base_vel_body, dr):
        throttle_cmd = exec_actions[:, 0]
        steer = exec_actions[:, 1]
        delta_max = float(
            self.env_cfg.get("delta_max", self.env_cfg.get("max_steer", 0.44))
        )
        delta_cmd = torch.clamp(steer * delta_max, min=-delta_max, max=delta_max)
        steer_state = steer_state + self.steer_lag_alpha * (delta_cmd - steer_state)
        steer_targets = ackermann_left_right(
            delta_center=steer_state,
            L=self.env_cfg.get("wheelbase", 0.325),
            W=self.env_cfg.get("track_width", 0.20),
        )
        wheel_dof_vel = entity.get_dofs_velocity(dofs_idx_local=self.wheel_dofs)
        wheel_torques = compute_wheel_torques(
            throttle_cmd=throttle_cmd,
            base_lin_vel_body=base_vel_body,
            wheel_dof_vel=wheel_dof_vel,
            env_cfg=self.env_cfg,
            vehicle_mass=dr["vehicle_mass"] * dr["mass_scale"],
            tire_friction=dr["tire_friction"],
        )
        entity.control_dofs_force(wheel_torques.to(device=gs.device), self.wheel_dofs)
        entity.control_dofs_position(steer_targets.to(device=gs.device),
                                     self.steer_dofs)
        return steer_state

    def apply_ego_actions(self, exec_actions, base_lin_vel, dr) -> None:
        self.steer_state = self._actuate(
            self.car, exec_actions, self.steer_state, base_lin_vel, dr
        )
        self._drag_force = self._compute_drag()

    def apply_opp_actions(self, opp_actions, opp_body_vel, dr) -> None:
        if self.opponent is None:
            return
        self.opp_steer_state = self._actuate(
            self.opponent, opp_actions, self.opp_steer_state, opp_body_vel, dr
        )

    def _dissipative_enabled(self) -> bool:
        return bool(self.env_cfg.get("enable_aero_drag", False)) or (
            float(self.env_cfg.get("c_roll", 0.0)) > 0.0
        )

    def _compute_drag(self):
        if not self._dissipative_enabled():
            return None
        lin_vel_world = self.car.get_vel()
        return compute_dissipative_force_world(lin_vel_world, self.env_cfg)

    def _apply_drag(self, force) -> None:
        self.car._solver.apply_links_external_force(
            force, links_idx=(self.base_link_idx,), ref="link_com", local=False
        )

    def substep(self, n_steps: int) -> None:
        import rerun as rr

        for _ in range(n_steps):
            if self._drag_force is not None:
                self._apply_drag(self._drag_force)
            self.scene.step()
            if self.cam1 is not None:
                position = self.car.get_pos(envs_idx=[0])[0]
                self.cam1.set_pose(
                    lookat=position.cpu() + np.array([0.0, 0.0, 0.5]),
                    pos=position.cpu() + np.array([3.0, 0.0, 7.0]),
                )
                rgb, *_ = self.cam1.render()
                rr.log("image", rr.Image(rgb))

    def read_state(self) -> dict[str, torch.Tensor]:
        car = self.car
        quat = car.get_quat()
        vel_world = car.get_vel()
        out = {
            "base_pos": car.get_pos(),
            "base_quat": quat,
            "base_vel_world": vel_world,
            "base_lin_vel": gu.inv_transform_by_quat(vel_world, quat),
            "base_ang_vel": gu.inv_transform_by_quat(car.get_ang(), quat),
            "base_lin_acc": gu.inv_transform_by_quat(
                car.get_links_acc(links_idx_local=[self.base_link_idx_local])[:, 0, :],
                quat,
            ),
        }
        if self.opponent is not None:
            opp = self.opponent
            oq = opp.get_quat()
            out["opp_base_pos"] = opp.get_pos()
            out["opp_base_quat"] = oq
            out["opp_vel_world"] = opp.get_vel()
            out["opp_ang_world"] = opp.get_ang()
        return out

    def read_wheel_state(self, which: str = "ego") -> dict[str, torch.Tensor]:
        entity = self.opponent if which == "opp" else self.car
        return dict(
            motion_link_vel=entity.get_links_vel(
                links_idx_local=self.slip_motion_link_idx, ref="link_com"
            ),
            frame_quat=entity.get_links_quat(links_idx_local=self.slip_frame_link_idx),
            dof_vel=entity.get_dofs_velocity(dofs_idx_local=self.wheel_dofs),
        )

    def close(self) -> None:
        self.scene.destroy()


class TorchSimBackend:
    """Wraps :class:`f1tenth_sim.TorchVehicleSim` behind the backend surface."""

    def __init__(self, *, num_envs, env_cfg, obs_cfg, reward_cfg, track_state,
                 device, show_viewer=False, enable_recording=False):
        from f1tenth_sim import TorchVehicleSim, VehicleParams

        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.device = device or torch.device("cpu")
        self.dtype = getattr(gs, "tc_float", None) or torch.float32
        self.has_opponent = env_cfg.get("opponent_strategy") is not None

        self.dt = float(env_cfg.get("sim_dt", 0.005))
        self.control_interval = int(env_cfg.get("control_interval", 20))
        self.control_dt = self.dt * self.control_interval
        internal = int((env_cfg.get("torch_sim") or {}).get("internal_substeps", 1))

        self.params = VehicleParams.from_config(env_cfg)
        self._TorchVehicleSim = TorchVehicleSim
        self._internal = internal
        car_len = float(env_cfg.get("car_length", 0.46))
        car_wid = float(env_cfg.get("car_width", 0.30))
        self._contact_radius = 0.5 * (car_len + car_wid) / 2.0
        self._restitution = float((env_cfg.get("torch_sim") or {}).get(
            "contact_restitution", 0.1))

    def build(self) -> None:
        self.sim = self._TorchVehicleSim(
            self.params, self.num_envs, device=self.device, dtype=self.dtype,
            sim_dt=self.dt, control_dt=self.control_dt,
            internal_substeps=self._internal,
        )
        self.opp_sim = (
            self._TorchVehicleSim(
                self.params, self.num_envs, device=self.device, dtype=self.dtype,
                sim_dt=self.dt, control_dt=self.control_dt,
                internal_substeps=self._internal,
            )
            if self.has_opponent else None
        )

    def _domain(self, mask, dr):
        if not dr.get("enabled"):
            return None
        return dict(
            mass=(dr["vehicle_mass"] * dr["mass_scale"]).to(self.dtype),
            mu=dr["tire_friction"].to(self.dtype),
            drive_scale=dr.get("drive_scale"),
            steer_bias=dr.get("steer_bias"),
        )

    def reset(self, mask, pos, quat, speed, opp_pos, opp_quat, dr) -> None:
        self.sim.reset(mask, pos, quat, speed)
        dom = self._domain(mask, dr)
        if dom is not None:
            self.sim.set_domain(mask, **dom)
        if self.opp_sim is not None and opp_pos is not None:
            self.opp_sim.reset(mask, opp_pos, opp_quat, torch.zeros_like(speed))
            if dom is not None:
                self.opp_sim.set_domain(mask, **dom)

    def apply_ego_actions(self, exec_actions, base_lin_vel, dr) -> None:
        self.sim.apply_actions(exec_actions)

    def apply_opp_actions(self, opp_actions, opp_body_vel, dr) -> None:
        if self.opp_sim is not None:
            self.opp_sim.apply_actions(opp_actions)

    def substep(self, n_steps: int) -> None:
        self.sim.substep(n_steps)
        if self.opp_sim is not None:
            self.opp_sim.substep(n_steps)
            self._resolve_contact()

    def _resolve_contact(self) -> None:
        """Soft ego-opponent contact: separate overlaps + damp the closing speed."""
        a = self.sim.s
        b = self.opp_sim.s
        dx = a["X"] - b["X"]
        dy = a["Y"] - b["Y"]
        dist = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
        min_dist = 2.0 * self._contact_radius
        overlap = dist < min_dist
        if not bool(overlap.any()):
            return
        nx = dx / dist
        ny = dy / dist
        pen = (min_dist - dist).clamp_min(0.0) * overlap.to(dist.dtype)
        half = 0.5 * pen
        a["X"] = a["X"] + half * nx
        a["Y"] = a["Y"] + half * ny
        b["X"] = b["X"] - half * nx
        b["Y"] = b["Y"] - half * ny

        vax, vay = _body_to_world_vel(a)
        vbx, vby = _body_to_world_vel(b)
        rel = (vax - vbx) * nx + (vay - vby) * ny
        closing = rel.clamp_max(0.0) * overlap.to(dist.dtype)
        j = -(1.0 + self._restitution) * closing * 0.5
        vax = vax + j * nx
        vay = vay + j * ny
        vbx = vbx - j * nx
        vby = vby - j * ny
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


def _body_to_world_vel(s):
    cy = torch.cos(s["yaw"])
    sy = torch.sin(s["yaw"])
    return s["vx"] * cy - s["vy"] * sy, s["vx"] * sy + s["vy"] * cy


def _world_to_body_vel(s, vwx, vwy):
    cy = torch.cos(s["yaw"])
    sy = torch.sin(s["yaw"])
    s["vx"] = vwx * cy + vwy * sy
    s["vy"] = -vwx * sy + vwy * cy

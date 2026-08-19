"""Vehicle parameters for the Torch simulator, seeded from calibrated config + URDF.

``VehicleParams`` is a plain dataclass of scalar physical constants. Per-env
randomized quantities (mass, friction) live on the simulator state, not here, so
this object stays a small immutable description of the nominal car.

Wheel ordering everywhere in this package is ``[LR, RR, LF, RF]`` (left-rear,
right-rear, left-front, right-front) to match the observation/reward pipeline.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import warp as wp

GRAVITY = 9.81

_DEFAULT_URDF = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "F110.export.urdf")
)


@wp.struct
class SimParams:
    sim_dt: wp.float32
    control_dt: wp.float32
    gravity: wp.float32
    izz: wp.float32
    wheelbase: wp.float32
    lf: wp.float32
    lr: wp.float32
    track_width: wp.float32
    h_cg: wp.float32
    wheel_radius: wp.float32
    wheel_inertia: wp.float32
    tire_b_long: wp.float32
    tire_c_long: wp.float32
    tire_e_long: wp.float32
    tire_b_lat: wp.float32
    tire_c_lat: wp.float32
    tire_e_lat: wp.float32
    tire_load_sens: wp.float32
    fz0_ref: wp.float32
    tire_relax_len: wp.float32
    low_speed_blend: wp.float32
    f_drive_max: wp.float32
    f_brake_max: wp.float32
    power_max: wp.float32
    k_drive_front: wp.float32
    v_eps: wp.float32
    drive_torque_sign: wp.float32
    effort_slew_rate: wp.float32
    max_steer: wp.float32
    steer_time_constant: wp.float32
    steering_action_mode: wp.int32
    steering_delta_max: wp.float32
    slip_min_lat: wp.float32
    slip_min_active_long: wp.float32
    slip_min_passive_long: wp.float32
    roll_stiffness_front: wp.float32
    drag_coeff: wp.float32
    enable_aero_drag: wp.int32
    wheel_x: wp.vec4f
    wheel_y: wp.vec4f


@dataclass
class VehicleParams:
    # --- chassis / geometry ---
    mass: float = 3.74
    izz: float = 0.13
    wheelbase: float = 0.325
    lf: float = 0.1584
    lr: float = 0.1666
    track_width: float = 0.253
    h_cg: float = 0.05

    # --- wheels ---
    wheel_radius: float = 0.053
    wheel_inertia: float = 4.12e-4

    # --- tire (Pacejka Magic Formula) ---
    tire_mu: float = 0.65
    tire_B_long: float = 11.0
    tire_C_long: float = 1.55
    tire_E_long: float = 0.55
    tire_B_lat: float = 11.0
    tire_C_lat: float = 1.45
    tire_E_lat: float = 0.6
    # Load sensitivity: mu(Fz) = mu * (1 - load_sens * (Fz/Fz0 - 1)), Fz0 = per-wheel static.
    tire_load_sens: float = 0.15
    # Longitudinal relaxation length (m); 0 disables tire lag (Tier 1 default).
    tire_relax_len: float = 0.0

    # --- drivetrain (force/brake effort; ADR 0006) ---
    f_drive_max: float = 23.0
    f_brake_max: float = 5.2
    power_max: float = 320.0
    k_drive_front: float = 0.5
    v_eps: float = 0.1
    low_speed_blend: float = 1.0
    c_roll: float = 0.0
    drive_torque_sign: float = 1.0
    # Normalized effort slew: 200 A/s physical / 80 A full-scale command envelope.
    longitudinal_slew_rate_per_s: float = 2.5
    # Soft accel clamp for the Tier-0 kinematic fallback only.
    kinematic_accel_limit: float = 12.0

    # --- steering ---
    max_steer: float = 0.33
    t_delta: float = 0.1
    steering_action_mode: str = "delta"
    steering_delta_max: float = math.pi / 60.0

    # --- tyre slip (modern PhysX denominators, m/s; scaled for the 1/10 car) ---
    slip_min_lat: float = 0.2
    slip_min_active_long: float = 0.1
    slip_min_passive_long: float = 0.4

    # --- suspension (Tier 2 spring-damper; Tier 1 uses roll_stiffness_front only) ---
    roll_stiffness_front: float = 0.47
    susp_stiffness: float = 4000.0

    # --- aero / environment ---
    enable_aero_drag: bool = True
    dragcoeff: float = 0.075
    gravity: float = GRAVITY

    # Per-wheel signed body-frame (x, y) offsets from the CoM, order [LR, RR, LF, RF].
    wheel_xy: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (-0.1666, 0.1265),
            (-0.1666, -0.1265),
            (0.1584, 0.1265),
            (0.1584, -0.1265),
        )
    )

    def static_wheel_load(self) -> float:
        """Mean per-wheel static normal load (reference Fz0 for load sensitivity)."""
        return self.mass * self.gravity / 4.0

    def to_warp(self, *, sim_dt: float, control_dt: float) -> SimParams:
        params = SimParams()
        params.sim_dt = sim_dt
        params.control_dt = control_dt
        params.gravity = self.gravity
        params.izz = self.izz
        params.wheelbase = self.wheelbase
        params.lf = self.lf
        params.lr = self.lr
        params.track_width = self.track_width
        params.h_cg = self.h_cg
        params.wheel_radius = self.wheel_radius
        params.wheel_inertia = self.wheel_inertia
        params.tire_b_long = self.tire_B_long
        params.tire_c_long = self.tire_C_long
        params.tire_e_long = self.tire_E_long
        params.tire_b_lat = self.tire_B_lat
        params.tire_c_lat = self.tire_C_lat
        params.tire_e_lat = self.tire_E_lat
        params.tire_load_sens = self.tire_load_sens
        params.fz0_ref = self.static_wheel_load()
        params.tire_relax_len = self.tire_relax_len
        params.low_speed_blend = self.low_speed_blend
        params.f_drive_max = self.f_drive_max
        params.f_brake_max = self.f_brake_max
        params.power_max = self.power_max
        params.k_drive_front = self.k_drive_front
        params.v_eps = self.v_eps
        params.drive_torque_sign = self.drive_torque_sign
        params.effort_slew_rate = self.longitudinal_slew_rate_per_s
        params.max_steer = self.max_steer
        params.steer_time_constant = self.t_delta
        params.steering_action_mode = int(self.steering_action_mode == "delta")
        params.steering_delta_max = self.steering_delta_max
        params.slip_min_lat = self.slip_min_lat
        params.slip_min_active_long = self.slip_min_active_long
        params.slip_min_passive_long = self.slip_min_passive_long
        params.roll_stiffness_front = self.roll_stiffness_front
        params.drag_coeff = self.dragcoeff
        params.enable_aero_drag = int(self.enable_aero_drag)
        params.wheel_x = wp.vec4f(*(xy[0] for xy in self.wheel_xy))
        params.wheel_y = wp.vec4f(*(xy[1] for xy in self.wheel_xy))
        return params

    @classmethod
    def from_config(cls, env_cfg: dict[str, Any] | None = None,
                    urdf_path: str | None = None) -> "VehicleParams":
        """Build URDF inertial priors, overridden by calibrated ``env_cfg`` values."""
        params = cls.from_urdf(urdf_path or _DEFAULT_URDF)
        if env_cfg:
            params.apply_env_cfg(env_cfg)
        return params

    def apply_env_cfg(self, env_cfg: dict[str, Any]) -> None:
        """Override drivetrain/steer/tire scalars from the training env config."""
        g = env_cfg.get
        self.wheelbase = float(g("wheelbase", self.wheelbase))
        self.track_width = float(g("track_width", self.track_width))
        self.wheel_radius = float(g("wheel_radius", self.wheel_radius))
        self.max_steer = float(g("delta_max", g("max_steer", self.max_steer)))
        self.t_delta = float(g("t_delta", self.t_delta))
        self.steering_action_mode = str(
            g("steering_action_mode", self.steering_action_mode)
        )
        if self.steering_action_mode not in ("absolute", "delta"):
            raise ValueError(
                "steering_action_mode must be 'absolute' or 'delta', got "
                f"{self.steering_action_mode!r}"
            )
        self.steering_delta_max = float(
            g("steering_delta_max_rad", self.steering_delta_max)
        )
        if self.steering_delta_max <= 0.0:
            raise ValueError("steering_delta_max_rad must be positive")
        self.slip_min_lat = float(g("slip_min_lat", self.slip_min_lat))
        self.slip_min_active_long = float(
            g("slip_min_active_long", self.slip_min_active_long)
        )
        self.slip_min_passive_long = float(
            g("slip_min_passive_long", self.slip_min_passive_long)
        )
        self.f_drive_max = float(g("f_drive_max", self.f_drive_max))
        self.f_brake_max = float(g("f_brake_max", self.f_brake_max))
        self.power_max = float(g("power_max", self.power_max))
        self.k_drive_front = float(g("k_drive_front", self.k_drive_front))
        self.v_eps = float(g("v_eps", self.v_eps))
        self.c_roll = float(g("c_roll", self.c_roll))
        self.drive_torque_sign = float(g("drive_torque_sign", self.drive_torque_sign))
        self.tire_mu = float(g("tire_friction", self.tire_mu))
        self.enable_aero_drag = bool(g("enable_aero_drag", self.enable_aero_drag))
        self.dragcoeff = float(g("dragcoeff", self.dragcoeff))
        sim = g("warp_sim") or {}
        for key in (
            "izz", "h_cg", "wheel_inertia", "tire_B_long", "tire_C_long",
            "tire_E_long", "tire_B_lat", "tire_C_lat", "tire_E_lat",
            "tire_load_sens", "tire_relax_len", "roll_stiffness_front",
            "susp_stiffness", "kinematic_accel_limit",
            "longitudinal_slew_rate_per_s",
        ):
            if key in sim:
                setattr(self, key, float(sim[key]))
        self._recompute_lf_lr()

    def _recompute_lf_lr(self) -> None:
        if abs(self.lf + self.lr - self.wheelbase) > 1e-6:
            frac = self.lr / max(self.lf + self.lr, 1e-9)
            self.lr = frac * self.wheelbase
            self.lf = self.wheelbase - self.lr
        half_track = 0.5 * self.track_width
        self.wheel_xy = (
            (-self.lr, half_track),
            (-self.lr, -half_track),
            (self.lf, half_track),
            (self.lf, -half_track),
        )

    @classmethod
    def from_urdf(cls, path: str) -> "VehicleParams":
        """Parse mass, CoM, yaw inertia, wheelbase and track from a URDF."""
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return cls()

        joints: dict[str, dict[str, Any]] = {}
        child_to_joint: dict[str, str] = {}
        for j in root.findall("joint"):
            name = j.get("name", "")
            parent = j.find("parent")
            child = j.find("child")
            origin = j.find("origin")
            xyz = _parse_xyz(origin.get("xyz") if origin is not None else None)
            pj = parent.get("link") if parent is not None else None
            cj = child.get("link") if child is not None else None
            joints[name] = {"parent": pj, "child": cj, "xyz": xyz}
            if cj is not None:
                child_to_joint[cj] = name

        def link_world_xy(link: str) -> tuple[float, float]:
            x, y = 0.0, 0.0
            visited = set()
            cur = link
            while cur in child_to_joint and cur not in visited:
                visited.add(cur)
                jname = child_to_joint[cur]
                jx, jy, _ = joints[jname]["xyz"]
                x += jx
                y += jy
                cur = joints[jname]["parent"]
            return x, y

        links: list[dict[str, Any]] = []
        for lk in root.findall("link"):
            inertial = lk.find("inertial")
            if inertial is None:
                continue
            mass_el = inertial.find("mass")
            mass = float(mass_el.get("value", "0.0")) if mass_el is not None else 0.0
            if mass <= 0.0:
                continue
            io = inertial.find("origin")
            iox, ioy, _ = _parse_xyz(io.get("xyz") if io is not None else None)
            inertia = inertial.find("inertia")
            izz = float(inertia.get("izz", "0.0")) if inertia is not None else 0.0
            wx, wy = link_world_xy(lk.get("name", ""))
            links.append({
                "name": lk.get("name", ""),
                "mass": mass,
                "x": wx + iox,
                "y": wy + ioy,
                "izz": izz,
            })

        total_mass = sum(lk["mass"] for lk in links)
        if total_mass <= 0.0:
            return cls()
        com_x = sum(lk["mass"] * lk["x"] for lk in links) / total_mass
        com_y = sum(lk["mass"] * lk["y"] for lk in links) / total_mass
        izz_total = sum(
            lk["izz"] + lk["mass"] * ((lk["x"] - com_x) ** 2 + (lk["y"] - com_y) ** 2)
            for lk in links
        )

        rear_x, front_x, track_ys = [], [], []
        wheel_pos: dict[str, tuple[float, float]] = {}
        wheel_radius = cls.wheel_radius
        wheel_inertia = cls.wheel_inertia
        for lk in root.findall("link"):
            name = lk.get("name", "")
            if not name.endswith("_wheel"):
                continue
            wx, wy = link_world_xy(name)
            wheel_pos[name] = (wx, wy)
            track_ys.append(abs(wy))
            if "front" in name:
                front_x.append(wx)
            elif "rear" in name:
                rear_x.append(wx)
            col = lk.find("collision")
            if col is not None:
                cyl = col.find("geometry/cylinder")
                if cyl is not None:
                    wheel_radius = float(cyl.get("radius", wheel_radius))
            inertial = lk.find("inertial")
            if inertial is not None:
                inertia = inertial.find("inertia")
                if inertia is not None:
                    wheel_inertia = float(inertia.get("izz", wheel_inertia))

        rear_axle_x = sum(rear_x) / len(rear_x) if rear_x else 0.0
        front_axle_x = sum(front_x) / len(front_x) if front_x else 0.325
        wheelbase = abs(front_axle_x - rear_axle_x)
        track_width = 2.0 * (sum(track_ys) / len(track_ys)) if track_ys else 0.253
        lr = abs(com_x - rear_axle_x)
        lf = abs(front_axle_x - com_x)

        order = ["left_rear_wheel", "right_rear_wheel",
                 "left_front_wheel", "right_front_wheel"]
        wheel_xy = tuple(
            (wheel_pos.get(n, (0.0, 0.0))[0] - com_x,
             wheel_pos.get(n, (0.0, 0.0))[1] - com_y)
            for n in order
        )

        return cls(
            mass=total_mass,
            izz=izz_total,
            wheelbase=wheelbase,
            lf=lf,
            lr=lr,
            track_width=track_width,
            wheel_radius=wheel_radius,
            wheel_inertia=wheel_inertia,
            wheel_xy=wheel_xy,
        )


def _parse_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    parts = [float(v) for v in text.split()]
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0], parts[1], parts[2]

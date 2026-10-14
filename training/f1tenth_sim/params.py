"""Vehicle parameters for the reduced bicycle simulator.

``VehicleParams`` is a plain dataclass of scalar physical constants. Per-env
randomized quantities (mass, friction, Izz, C_α, steer lag/rate/gain,
drive scale) live on the simulator state, not here. Yaw inertia is a
labeled prior, not a bag measurement.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import warp as wp

GRAVITY = 9.81
YAW_INERTIA_PRIOR = 0.13
STEER_GAIN_SKIDPAD = 1.10714
STEER_LAG_S = 0.09
LOW_SPEED_BLEND_MPS = 1.0
LAT_STIFFNESS_SCALE_PRIOR = 11.0
STEER_RATE_UNBOUNDED = 1.0e6

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
    wheel_radius: wp.float32
    tire_mu: wp.float32
    lat_stiffness_scale: wp.float32
    f_drive_max: wp.float32
    f_brake_max: wp.float32
    i_drive_max: wp.float32
    i_brake_max: wp.float32
    i_slew: wp.float32
    c_coast: wp.float32
    v_eps: wp.float32
    low_speed_blend: wp.float32
    drive_torque_sign: wp.float32
    max_steer: wp.float32
    steer_time_constant: wp.float32
    steer_gain: wp.float32
    steering_action_mode: wp.int32
    steering_delta_max: wp.float32
    command_delay_substeps: wp.int32
    history_stride: wp.int32


@dataclass
class VehicleParams:
    mass: float = 3.444
    izz: float = YAW_INERTIA_PRIOR
    wheelbase: float = 0.325
    lf: float = 0.1584
    lr: float = 0.1666
    track_width: float = 0.253
    h_cg: float = 0.05
    wheel_radius: float = 0.053

    tire_mu: float = 0.71
    lat_stiffness_scale: float = LAT_STIFFNESS_SCALE_PRIOR

    f_drive_max: float = 26.5
    f_brake_max: float = 23.1
    i_drive_max_a: float = 80.0
    i_brake_max_a: float = 40.0
    i_slew_a_per_s: float = 200.0
    v_eps: float = 0.1
    low_speed_blend: float = LOW_SPEED_BLEND_MPS
    c_coast: float = 0.0
    drive_torque_sign: float = 1.0
    command_delay_substeps: int = 2

    max_steer: float = 0.33
    t_delta: float = STEER_LAG_S
    steer_gain: float = STEER_GAIN_SKIDPAD
    steering_action_mode: str = "delta"
    steering_delta_max: float = math.pi / 120.0

    gravity: float = GRAVITY

    wheel_xy: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (-0.1666, 0.1265),
            (-0.1666, -0.1265),
            (0.1584, 0.1265),
            (0.1584, -0.1265),
        )
    )

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
        params.wheel_radius = self.wheel_radius
        params.tire_mu = self.tire_mu
        params.lat_stiffness_scale = self.lat_stiffness_scale
        params.f_drive_max = self.f_drive_max
        params.f_brake_max = self.f_brake_max
        params.i_drive_max = self.i_drive_max_a
        params.i_brake_max = self.i_brake_max_a
        params.i_slew = self.i_slew_a_per_s
        params.c_coast = self.c_coast
        params.v_eps = self.v_eps
        params.low_speed_blend = self.low_speed_blend
        params.drive_torque_sign = self.drive_torque_sign
        params.max_steer = self.max_steer
        params.steer_time_constant = self.t_delta
        params.steer_gain = self.steer_gain
        params.steering_action_mode = int(self.steering_action_mode == "delta")
        params.steering_delta_max = self.steering_delta_max
        params.command_delay_substeps = int(self.command_delay_substeps)
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        params.history_stride = max(1, int(round(0.1 / float(control_dt))))
        return params

    def axle_stiffness(self, mass: float | None = None) -> tuple[float, float]:
        m = self.mass if mass is None else float(mass)
        weight = m * self.gravity
        denom = max(self.wheelbase, 1.0e-9)
        scale = self.lat_stiffness_scale * weight / denom
        return scale * self.lr, scale * self.lf

    @classmethod
    def from_config(
        cls,
        env_cfg: dict[str, Any] | None = None,
        urdf_path: str | None = None,
    ) -> "VehicleParams":
        params = cls.from_urdf(urdf_path or _DEFAULT_URDF)
        params.izz = YAW_INERTIA_PRIOR
        if env_cfg:
            params.apply_env_cfg(env_cfg)
        return params

    def apply_env_cfg(self, env_cfg: dict[str, Any]) -> None:
        g = env_cfg.get
        self.wheelbase = float(g("wheelbase", self.wheelbase))
        self.track_width = float(g("track_width", self.track_width))
        self.wheel_radius = float(g("wheel_radius", self.wheel_radius))
        self.max_steer = float(g("delta_max", g("max_steer", self.max_steer)))
        self.t_delta = float(g("t_delta", self.t_delta))
        self.steer_gain = float(g("steer_gain", self.steer_gain))
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
        self.f_drive_max = float(g("f_drive_max", self.f_drive_max))
        self.f_brake_max = float(g("f_brake_max", self.f_brake_max))
        self.i_drive_max_a = float(g("i_drive_max_a", self.i_drive_max_a))
        self.i_brake_max_a = float(g("i_brake_max_a", self.i_brake_max_a))
        self.i_slew_a_per_s = float(g("i_slew_a_per_s", self.i_slew_a_per_s))
        self.v_eps = float(g("v_eps", self.v_eps))
        self.c_coast = float(g("c_coast", g("c_roll", self.c_coast)))
        self.drive_torque_sign = float(g("drive_torque_sign", self.drive_torque_sign))
        self.tire_mu = float(g("tire_friction", self.tire_mu))
        if g("vehicle_mass") is not None:
            self.mass = float(g("vehicle_mass"))
        elif g("mass") is not None:
            self.mass = float(g("mass"))
        sim = g("warp_sim") or {}
        for key in (
            "izz",
            "h_cg",
            "lat_stiffness_scale",
            "steer_gain",
            "c_coast",
            "low_speed_blend",
        ):
            if key in sim:
                setattr(self, key, float(sim[key]))
        if "izz" not in sim:
            self.izz = YAW_INERTIA_PRIOR
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
            wx, wy = link_world_xy(lk.get("name", ""))
            links.append(
                {
                    "name": lk.get("name", ""),
                    "mass": mass,
                    "x": wx + iox,
                    "y": wy + ioy,
                }
            )

        total_mass = sum(lk["mass"] for lk in links)
        if total_mass <= 0.0:
            return cls()
        com_x = sum(lk["mass"] * lk["x"] for lk in links) / total_mass
        com_y = sum(lk["mass"] * lk["y"] for lk in links) / total_mass

        rear_x, front_x, track_ys = [], [], []
        wheel_pos: dict[str, tuple[float, float]] = {}
        wheel_radius = cls.wheel_radius
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

        rear_axle_x = sum(rear_x) / len(rear_x) if rear_x else 0.0
        front_axle_x = sum(front_x) / len(front_x) if front_x else 0.325
        wheelbase = abs(front_axle_x - rear_axle_x)
        track_width = 2.0 * (sum(track_ys) / len(track_ys)) if track_ys else 0.253
        lr = abs(com_x - rear_axle_x)
        lf = abs(front_axle_x - com_x)

        order = [
            "left_rear_wheel",
            "right_rear_wheel",
            "left_front_wheel",
            "right_front_wheel",
        ]
        wheel_xy = tuple(
            (
                wheel_pos.get(n, (0.0, 0.0))[0] - com_x,
                wheel_pos.get(n, (0.0, 0.0))[1] - com_y,
            )
            for n in order
        )

        return cls(
            mass=total_mass,
            izz=YAW_INERTIA_PRIOR,
            wheelbase=wheelbase,
            lf=lf,
            lr=lr,
            track_width=track_width,
            wheel_radius=wheel_radius,
            wheel_xy=wheel_xy,
        )


def _parse_xyz(text: str | None) -> tuple[float, float, float]:
    if not text:
        return 0.0, 0.0, 0.0
    parts = [float(v) for v in text.split()]
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0], parts[1], parts[2]

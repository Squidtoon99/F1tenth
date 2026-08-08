"""Calibrated F1TENTH vehicle params and Warp dynamics formulas."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from gigaflow_f1tenth.sim.layout_local import (
    GRAVITY,
    MAX_STEER_RAD,
    STEERING_DELTA_MAX_RAD,
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


@wp.struct
class TireForce:
    peak: wp.float32
    fx: wp.float32
    fy: wp.float32


@wp.struct
class VehicleLocal:
    x: wp.float32
    y: wp.float32
    yaw: wp.float32
    vx: wp.float32
    vy: wp.float32
    yaw_rate: wp.float32
    steer: wp.float32
    effort_state: wp.float32
    applied_effort: wp.float32
    ax: wp.float32
    ay: wp.float32
    omega: wp.vec4f
    slip_ratio: wp.vec4f
    slip_angle: wp.vec4f
    load_ratio: wp.vec4f
    fx_lag: wp.vec4f
    fy_lag: wp.vec4f


@wp.struct
class VehicleBuffers:
    x: wp.array(dtype=wp.float32)
    y: wp.array(dtype=wp.float32)
    yaw: wp.array(dtype=wp.float32)
    vx: wp.array(dtype=wp.float32)
    vy: wp.array(dtype=wp.float32)
    yaw_rate: wp.array(dtype=wp.float32)
    steer: wp.array(dtype=wp.float32)
    effort_state: wp.array(dtype=wp.float32)
    applied_effort: wp.array(dtype=wp.float32)
    ax: wp.array(dtype=wp.float32)
    ay: wp.array(dtype=wp.float32)
    omega: wp.array(dtype=wp.vec4f)
    slip_ratio: wp.array(dtype=wp.vec4f)
    slip_angle: wp.array(dtype=wp.vec4f)
    load_ratio: wp.array(dtype=wp.vec4f)
    fx_lag: wp.array(dtype=wp.vec4f)
    fy_lag: wp.array(dtype=wp.vec4f)
    mass: wp.array(dtype=wp.float32)
    mu: wp.array(dtype=wp.float32)
    drive_scale: wp.array(dtype=wp.float32)
    steer_scale: wp.array(dtype=wp.float32)
    accel_scale: wp.array(dtype=wp.float32)
    vmax_scale: wp.array(dtype=wp.float32)


@dataclass
class VehicleParams:
    mass: float = 3.74
    izz: float = 0.13
    wheelbase: float = 0.325
    lf: float = 0.1584
    lr: float = 0.1666
    track_width: float = 0.253
    h_cg: float = 0.05
    wheel_radius: float = 0.053
    wheel_inertia: float = 4.12e-4
    tire_mu: float = 0.65
    tire_B_long: float = 11.0
    tire_C_long: float = 1.55
    tire_E_long: float = 0.55
    tire_B_lat: float = 11.0
    tire_C_lat: float = 1.45
    tire_E_lat: float = 0.6
    tire_load_sens: float = 0.15
    tire_relax_len: float = 0.0
    f_drive_max: float = 23.0
    f_brake_max: float = 5.2
    power_max: float = 320.0
    k_drive_front: float = 0.5
    v_eps: float = 0.1
    low_speed_blend: float = 1.0
    drive_torque_sign: float = 1.0
    longitudinal_slew_rate_per_s: float = 4.444444444444445
    max_steer: float = MAX_STEER_RAD
    t_delta: float = 0.1
    steering_action_mode: str = "delta"
    steering_delta_max: float = STEERING_DELTA_MAX_RAD
    slip_min_lat: float = 0.2
    slip_min_active_long: float = 0.1
    slip_min_passive_long: float = 0.4
    roll_stiffness_front: float = 0.47
    enable_aero_drag: bool = True
    dragcoeff: float = 0.075
    gravity: float = GRAVITY
    wheel_xy: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: (
            (-0.1666, 0.1265),
            (-0.1666, -0.1265),
            (0.1584, 0.1265),
            (0.1584, -0.1265),
        )
    )

    def static_wheel_load(self) -> float:
        return self.mass * self.gravity / 4.0

    def to_warp(self, *, sim_dt: float, control_dt: float) -> SimParams:
        params = SimParams()
        params.sim_dt = float(sim_dt)
        params.control_dt = float(control_dt)
        params.gravity = float(self.gravity)
        params.izz = float(self.izz)
        params.wheelbase = float(self.wheelbase)
        params.lf = float(self.lf)
        params.lr = float(self.lr)
        params.track_width = float(self.track_width)
        params.h_cg = float(self.h_cg)
        params.wheel_radius = float(self.wheel_radius)
        params.wheel_inertia = float(self.wheel_inertia)
        params.tire_b_long = float(self.tire_B_long)
        params.tire_c_long = float(self.tire_C_long)
        params.tire_e_long = float(self.tire_E_long)
        params.tire_b_lat = float(self.tire_B_lat)
        params.tire_c_lat = float(self.tire_C_lat)
        params.tire_e_lat = float(self.tire_E_lat)
        params.tire_load_sens = float(self.tire_load_sens)
        params.fz0_ref = float(self.static_wheel_load())
        params.tire_relax_len = float(self.tire_relax_len)
        params.low_speed_blend = float(self.low_speed_blend)
        params.f_drive_max = float(self.f_drive_max)
        params.f_brake_max = float(self.f_brake_max)
        params.power_max = float(self.power_max)
        params.k_drive_front = float(self.k_drive_front)
        params.v_eps = float(self.v_eps)
        params.drive_torque_sign = float(self.drive_torque_sign)
        params.effort_slew_rate = float(self.longitudinal_slew_rate_per_s)
        params.max_steer = float(self.max_steer)
        params.steer_time_constant = float(self.t_delta)
        params.steering_action_mode = int(self.steering_action_mode == "delta")
        params.steering_delta_max = float(self.steering_delta_max)
        params.slip_min_lat = float(self.slip_min_lat)
        params.slip_min_active_long = float(self.slip_min_active_long)
        params.slip_min_passive_long = float(self.slip_min_passive_long)
        params.roll_stiffness_front = float(self.roll_stiffness_front)
        params.drag_coeff = float(self.dragcoeff)
        params.enable_aero_drag = int(self.enable_aero_drag)
        params.wheel_x = wp.vec4f(*(xy[0] for xy in self.wheel_xy))
        params.wheel_y = wp.vec4f(*(xy[1] for xy in self.wheel_xy))
        return params


@wp.func
def magic_formula(
    slip: wp.float32,
    b: wp.float32,
    c: wp.float32,
    e: wp.float32,
) -> wp.float32:
    bx = b * slip
    return wp.sin(c * wp.atan(bx - e * (bx - wp.atan(bx))))


@wp.func
def combined_pacejka(
    kappa: wp.float32,
    alpha: wp.float32,
    fz: wp.float32,
    mu: wp.float32,
    fz0: wp.float32,
    params: SimParams,
) -> TireForce:
    out = TireForce()
    normal_load = wp.max(fz, 0.0)
    relative_load = normal_load / wp.max(fz0, 1.0e-6) - 1.0
    load_mu = mu * (1.0 - params.tire_load_sens * relative_load)
    out.peak = wp.max(load_mu, 1.0e-4) * normal_load
    fx0 = out.peak * magic_formula(
        kappa, params.tire_b_long, params.tire_c_long, params.tire_e_long
    )
    fy0 = out.peak * magic_formula(
        alpha, params.tire_b_lat, params.tire_c_lat, params.tire_e_lat
    )
    peak_safe = wp.max(out.peak, 1.0e-6)
    combined = wp.sqrt(
        (fx0 / peak_safe) * (fx0 / peak_safe)
        + (fy0 / peak_safe) * (fy0 / peak_safe)
    )
    scale = 1.0 / wp.max(combined, 1.0)
    out.fx = fx0 * scale
    out.fy = fy0 * scale
    return out


@wp.func
def static_wheel_load(
    mass: wp.float32,
    wheel: wp.int32,
    params: SimParams,
) -> wp.float32:
    axle = mass * params.gravity * params.lf / params.wheelbase
    if wheel > 1:
        axle = mass * params.gravity * params.lr / params.wheelbase
    return 0.5 * axle


@wp.func
def warp_quasi_static_loads(
    mass: wp.float32,
    ax: wp.float32,
    ay: wp.float32,
    params: SimParams,
) -> wp.vec4f:
    weight = mass * params.gravity
    front = weight * params.lr / params.wheelbase
    rear = weight * params.lf / params.wheelbase
    longitudinal = mass * ax * params.h_cg / params.wheelbase
    front = front - longitudinal
    rear = rear + longitudinal
    lateral = mass * ay * params.h_cg / wp.max(params.track_width, 1.0e-6)
    lateral_front = params.roll_stiffness_front * lateral
    lateral_rear = (1.0 - params.roll_stiffness_front) * lateral
    return wp.vec4f(
        wp.max(0.5 * rear - lateral_rear, 0.0),
        wp.max(0.5 * rear + lateral_rear, 0.0),
        wp.max(0.5 * front - lateral_front, 0.0),
        wp.max(0.5 * front + lateral_front, 0.0),
    )


@wp.func
def direct_drive_torque(
    effort: wp.float32,
    body_vx: wp.float32,
    omega: wp.vec4f,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    vmax_scale: wp.float32,
    params: SimParams,
) -> wp.vec4f:
    drive = wp.max(effort, 0.0)
    brake = wp.max(-effort, 0.0)
    # vmax_scale raises/lowers the force and power ceilings together, so it sets
    # the speed a car can hold rather than trimming its response.
    drive_force = drive * params.f_drive_max * vmax_scale
    drive_force = wp.min(
        drive_force,
        params.power_max * vmax_scale / wp.max(wp.abs(body_vx), params.v_eps),
    )
    drive_force = wp.min(drive_force, mu * mass * params.gravity)
    drive_force = drive_force * drive_scale * params.drive_torque_sign

    rear_torque = (
        0.5 * (1.0 - params.k_drive_front) * drive_force * params.wheel_radius
    )
    front_torque = (
        0.5 * params.k_drive_front * drive_force * params.wheel_radius
    )
    out = wp.vec4f(rear_torque, rear_torque, front_torque, front_torque)
    if brake > 1.0e-3:
        brake_torque = 0.25 * brake * params.f_brake_max * params.wheel_radius
        for wheel in range(4):
            direction = wp.float32(1.0)
            if omega[wheel] < 0.0:
                direction = -1.0
            out[wheel] = -direction * brake_torque
    elif drive <= 1.0e-3:
        out = wp.vec4f(0.0)
    return out


@wp.func
def load_vehicle(buffers: VehicleBuffers, index: wp.int32) -> VehicleLocal:
    vehicle = VehicleLocal()
    vehicle.x = buffers.x[index]
    vehicle.y = buffers.y[index]
    vehicle.yaw = buffers.yaw[index]
    vehicle.vx = buffers.vx[index]
    vehicle.vy = buffers.vy[index]
    vehicle.yaw_rate = buffers.yaw_rate[index]
    vehicle.steer = buffers.steer[index]
    vehicle.effort_state = buffers.effort_state[index]
    vehicle.applied_effort = buffers.applied_effort[index]
    vehicle.ax = buffers.ax[index]
    vehicle.ay = buffers.ay[index]
    vehicle.omega = buffers.omega[index]
    vehicle.slip_ratio = buffers.slip_ratio[index]
    vehicle.slip_angle = buffers.slip_angle[index]
    vehicle.load_ratio = buffers.load_ratio[index]
    vehicle.fx_lag = buffers.fx_lag[index]
    vehicle.fy_lag = buffers.fy_lag[index]
    return vehicle


@wp.func
def store_vehicle(
    buffers: VehicleBuffers,
    index: wp.int32,
    vehicle: VehicleLocal,
):
    buffers.x[index] = vehicle.x
    buffers.y[index] = vehicle.y
    buffers.yaw[index] = vehicle.yaw
    buffers.vx[index] = vehicle.vx
    buffers.vy[index] = vehicle.vy
    buffers.yaw_rate[index] = vehicle.yaw_rate
    buffers.steer[index] = vehicle.steer
    buffers.effort_state[index] = vehicle.effort_state
    buffers.applied_effort[index] = vehicle.applied_effort
    buffers.ax[index] = vehicle.ax
    buffers.ay[index] = vehicle.ay
    buffers.omega[index] = vehicle.omega
    buffers.slip_ratio[index] = vehicle.slip_ratio
    buffers.slip_angle[index] = vehicle.slip_angle
    buffers.load_ratio[index] = vehicle.load_ratio
    buffers.fx_lag[index] = vehicle.fx_lag
    buffers.fy_lag[index] = vehicle.fy_lag


@wp.func
def apply_command(
    vehicle: VehicleLocal,
    action: wp.vec2f,
    steer_scale: wp.float32,
    accel_scale: wp.float32,
    params: SimParams,
) -> VehicleLocal:
    target = wp.clamp(action[0], -1.0, 1.0)
    # accel_scale sets how fast the longitudinal effort may ramp.
    max_step = params.effort_slew_rate * params.control_dt * accel_scale
    if max_step > 0.0:
        delta = target - vehicle.effort_state
        next_effort = vehicle.effort_state + wp.clamp(delta, -max_step, max_step)
        if wp.abs(delta) <= max_step:
            vehicle.applied_effort = (
                target - delta * wp.abs(delta) / (2.0 * max_step)
            )
        else:
            vehicle.applied_effort = 0.5 * (vehicle.effort_state + next_effort)
        vehicle.effort_state = next_effort
    else:
        vehicle.effort_state = target
        vehicle.applied_effort = target

    # steer_scale is a symmetric gain on the commanded steering, so left and
    # right authority stay equal for every sampled scale.
    if params.steering_action_mode != 0:
        vehicle.steer = wp.clamp(
            vehicle.steer + action[1] * params.steering_delta_max * steer_scale,
            -params.max_steer,
            params.max_steer,
        )
    else:
        steer_target = wp.clamp(
            action[1] * params.max_steer * steer_scale,
            -params.max_steer,
            params.max_steer,
        )
        steer_alpha = params.control_dt / (
            params.steer_time_constant + params.control_dt
        )
        vehicle.steer = vehicle.steer + steer_alpha * (steer_target - vehicle.steer)
    return vehicle


@wp.func
def ackermann_angles(steer: wp.float32, params: SimParams) -> wp.vec2f:
    if wp.abs(steer) < 1.0e-6:
        return wp.vec2f(0.0)
    radius = params.wheelbase / wp.tan(steer)
    half_track = 0.5 * params.track_width
    return wp.vec2f(
        wp.atan(params.wheelbase / (radius - half_track)),
        wp.atan(params.wheelbase / (radius + half_track)),
    )


@wp.func
def integrate_vehicle_substep(
    vehicle: VehicleLocal,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    vmax_scale: wp.float32,
    params: SimParams,
) -> VehicleLocal:
    front_angles = ackermann_angles(vehicle.steer, params)
    wheel_angle = wp.vec4f(0.0, 0.0, front_angles[0], front_angles[1])
    wheel_long = wp.vec4f(0.0)
    wheel_lat = wp.vec4f(0.0)
    for wheel in range(4):
        patch_vx = vehicle.vx - vehicle.yaw_rate * params.wheel_y[wheel]
        patch_vy = vehicle.vy + vehicle.yaw_rate * params.wheel_x[wheel]
        cosine = wp.cos(wheel_angle[wheel])
        sine = wp.sin(wheel_angle[wheel])
        wheel_long[wheel] = cosine * patch_vx + sine * patch_vy
        wheel_lat[wheel] = -sine * patch_vx + cosine * patch_vy

    loads = warp_quasi_static_loads(mass, vehicle.ax, vehicle.ay, params)
    axle_torque = direct_drive_torque(
        vehicle.applied_effort,
        vehicle.vx,
        vehicle.omega,
        mass,
        mu,
        drive_scale,
        vmax_scale,
        params,
    )
    force_x = wp.float32(0.0)
    force_y = wp.float32(0.0)
    yaw_moment = wp.float32(0.0)
    active_min = params.slip_min_passive_long
    if wp.abs(vehicle.applied_effort) > params.v_eps:
        active_min = params.slip_min_active_long

    for wheel in range(4):
        long_abs = wp.abs(wheel_long[wheel])
        denominator = long_abs + active_min
        kappa = (
            params.wheel_radius * vehicle.omega[wheel] - wheel_long[wheel]
        ) / denominator
        alpha_force = wp.atan2(-wheel_lat[wheel], long_abs + params.slip_min_lat)
        static_load = static_wheel_load(mass, wheel, params)
        tire = combined_pacejka(
            kappa,
            alpha_force,
            loads[wheel],
            mu,
            params.fz0_ref,
            params,
        )

        if params.tire_relax_len > 0.0:
            v_blend = wp.max(params.low_speed_blend, params.v_eps)
            tau = params.tire_relax_len / wp.max(long_abs, v_blend)
            beta = params.sim_dt / (tau + params.sim_dt)
            fx_relaxed = vehicle.fx_lag[wheel] + beta * (
                tire.fx - vehicle.fx_lag[wheel]
            )
            fy_relaxed = vehicle.fy_lag[wheel] + beta * (
                tire.fy - vehicle.fy_lag[wheel]
            )
            vehicle.fx_lag[wheel] = fx_relaxed
            vehicle.fy_lag[wheel] = fy_relaxed
            tire.fx = fx_relaxed
            tire.fy = fy_relaxed

        reaction = axle_torque[wheel] - params.wheel_radius * tire.fx
        slope = (
            tire.peak
            * params.tire_b_long
            * params.tire_c_long
            * params.wheel_radius
            * params.wheel_radius
            / denominator
        )
        vehicle.omega[wheel] = vehicle.omega[wheel] + (
            params.sim_dt * reaction / params.wheel_inertia
        ) / (1.0 + params.sim_dt * slope / params.wheel_inertia)

        cosine = wp.cos(wheel_angle[wheel])
        sine = wp.sin(wheel_angle[wheel])
        body_fx = cosine * tire.fx - sine * tire.fy
        body_fy = sine * tire.fx + cosine * tire.fy
        force_x = force_x + body_fx
        force_y = force_y + body_fy
        yaw_moment = (
            yaw_moment
            + params.wheel_x[wheel] * body_fy
            - params.wheel_y[wheel] * body_fx
        )
        vehicle.slip_ratio[wheel] = kappa
        vehicle.slip_angle[wheel] = wp.atan2(
            wheel_lat[wheel], long_abs + params.slip_min_lat
        )
        vehicle.load_ratio[wheel] = loads[wheel] / wp.max(static_load, 1.0e-6)

    speed = wp.sqrt(vehicle.vx * vehicle.vx + vehicle.vy * vehicle.vy)
    if params.enable_aero_drag != 0:
        force_x = force_x - params.drag_coeff * speed * vehicle.vx
        force_y = force_y - params.drag_coeff * speed * vehicle.vy

    ax = force_x / mass
    ay = force_y / mass
    yaw_acceleration = yaw_moment / params.izz
    vx_next = vehicle.vx + params.sim_dt * (ax + vehicle.yaw_rate * vehicle.vy)
    vy_next = vehicle.vy + params.sim_dt * (ay - vehicle.yaw_rate * vehicle.vx)
    yaw_rate_next = vehicle.yaw_rate + params.sim_dt * yaw_acceleration
    yaw_next = vehicle.yaw + params.sim_dt * yaw_rate_next
    vehicle.x = vehicle.x + params.sim_dt * (
        vx_next * wp.cos(yaw_next) - vy_next * wp.sin(yaw_next)
    )
    vehicle.y = vehicle.y + params.sim_dt * (
        vx_next * wp.sin(yaw_next) + vy_next * wp.cos(yaw_next)
    )
    vehicle.yaw = yaw_next
    vehicle.vx = vx_next
    vehicle.vy = vy_next
    vehicle.yaw_rate = yaw_rate_next
    vehicle.ax = ax
    vehicle.ay = ay
    return vehicle


@wp.func
def vehicle_is_finite(vehicle: VehicleLocal) -> wp.bool:
    return (
        wp.isfinite(vehicle.x)
        and wp.isfinite(vehicle.y)
        and wp.isfinite(vehicle.yaw)
        and wp.isfinite(vehicle.vx)
        and wp.isfinite(vehicle.vy)
        and wp.isfinite(vehicle.yaw_rate)
        and wp.isfinite(vehicle.ax)
        and wp.isfinite(vehicle.ay)
    )


@wp.func
def clear_vehicle(vehicle: VehicleLocal) -> VehicleLocal:
    vehicle.vy = 0.0
    vehicle.yaw_rate = 0.0
    vehicle.steer = 0.0
    vehicle.effort_state = 0.0
    vehicle.applied_effort = 0.0
    vehicle.ax = 0.0
    vehicle.ay = 0.0
    vehicle.omega = wp.vec4f(0.0)
    vehicle.slip_ratio = wp.vec4f(0.0)
    vehicle.slip_angle = wp.vec4f(0.0)
    vehicle.load_ratio = wp.vec4f(1.0)
    vehicle.fx_lag = wp.vec4f(0.0)
    vehicle.fy_lag = wp.vec4f(0.0)
    return vehicle


@wp.func
def body_velocity_world(vehicle: VehicleLocal) -> wp.vec2f:
    c = wp.cos(vehicle.yaw)
    s = wp.sin(vehicle.yaw)
    return wp.vec2f(
        c * vehicle.vx - s * vehicle.vy,
        s * vehicle.vx + c * vehicle.vy,
    )


def empty_vehicle_numpy() -> dict[str, float | np.ndarray]:
    return {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "yaw_rate": 0.0,
        "steer": 0.0,
        "effort_state": 0.0,
        "applied_effort": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "omega": np.zeros(4, dtype=np.float32),
        "slip_ratio": np.zeros(4, dtype=np.float32),
        "slip_angle": np.zeros(4, dtype=np.float32),
        "load_ratio": np.ones(4, dtype=np.float32),
        "fx_lag": np.zeros(4, dtype=np.float32),
        "fy_lag": np.zeros(4, dtype=np.float32),
    }


def wrap_angle_np(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

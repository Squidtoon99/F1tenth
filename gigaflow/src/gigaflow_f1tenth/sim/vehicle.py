"""SIM_READY_V2 bicycle params and Warp dynamics (copied; no training/ import)."""

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

YAW_INERTIA_PRIOR = 0.13
STEER_LAG_S = 0.09
LOW_SPEED_BLEND_MPS = 1.0
STEER_RATE_UNBOUNDED = 1.0e6
COMMAND_DELAY_SUBSTEPS = 2
# Compile-time fused-loop bound; live cadence is `substeps` (≤ this).
_MAX_INTEGRATE_SUBSTEPS = 20

# Frozen PR #80 SIM_READY_V2 bag-fit (do not retune).
F_DRIVE_MAX = 21.46877492171339
F_BRAKE_MAX = 9.139188067139935
C_COAST = 4.804781294740983
STEER_GAIN_FIT = 1.0511988660159353
LAT_STIFFNESS_SCALE_FIT = 5.329306089973615


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


@wp.struct
class VehicleLocal:
    x: wp.float32
    y: wp.float32
    yaw: wp.float32
    vx: wp.float32
    vy: wp.float32
    yaw_rate: wp.float32
    steer: wp.float32
    steer_logical: wp.float32
    i_drive: wp.float32
    i_brake: wp.float32
    i_drive_cmd: wp.float32
    i_brake_cmd: wp.float32
    applied_effort: wp.float32
    ax: wp.float32
    ay: wp.float32
    imu_gz: wp.float32
    omega: wp.vec4f
    slip_ratio: wp.vec4f
    slip_angle: wp.vec4f
    load_ratio: wp.vec4f


@wp.struct
class VehicleBuffers:
    x: wp.array(dtype=wp.float32)
    y: wp.array(dtype=wp.float32)
    yaw: wp.array(dtype=wp.float32)
    vx: wp.array(dtype=wp.float32)
    vy: wp.array(dtype=wp.float32)
    yaw_rate: wp.array(dtype=wp.float32)
    steer: wp.array(dtype=wp.float32)
    steer_logical: wp.array(dtype=wp.float32)
    i_drive: wp.array(dtype=wp.float32)
    i_brake: wp.array(dtype=wp.float32)
    i_drive_cmd: wp.array(dtype=wp.float32)
    i_brake_cmd: wp.array(dtype=wp.float32)
    applied_effort: wp.array(dtype=wp.float32)
    ax: wp.array(dtype=wp.float32)
    ay: wp.array(dtype=wp.float32)
    imu_gz: wp.array(dtype=wp.float32)
    omega: wp.array(dtype=wp.vec4f)
    slip_ratio: wp.array(dtype=wp.vec4f)
    slip_angle: wp.array(dtype=wp.vec4f)
    load_ratio: wp.array(dtype=wp.vec4f)
    mass: wp.array(dtype=wp.float32)
    mu: wp.array(dtype=wp.float32)
    drive_scale: wp.array(dtype=wp.float32)
    steer_scale: wp.array(dtype=wp.float32)
    accel_scale: wp.array(dtype=wp.float32)
    vmax_scale: wp.array(dtype=wp.float32)
    izz: wp.array(dtype=wp.float32)
    c_f: wp.array(dtype=wp.float32)
    c_r: wp.array(dtype=wp.float32)
    steer_lag: wp.array(dtype=wp.float32)
    steer_rate: wp.array(dtype=wp.float32)
    steer_gain: wp.array(dtype=wp.float32)
    command_delay: wp.array(dtype=wp.int32)


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
    lat_stiffness_scale: float = LAT_STIFFNESS_SCALE_FIT

    f_drive_max: float = F_DRIVE_MAX
    f_brake_max: float = F_BRAKE_MAX
    i_drive_max_a: float = 80.0
    i_brake_max_a: float = 40.0
    i_slew_a_per_s: float = 200.0
    v_eps: float = 0.1
    low_speed_blend: float = LOW_SPEED_BLEND_MPS
    c_coast: float = C_COAST
    drive_torque_sign: float = 1.0
    command_delay_substeps: int = COMMAND_DELAY_SUBSTEPS

    max_steer: float = MAX_STEER_RAD
    t_delta: float = STEER_LAG_S
    steer_gain: float = STEER_GAIN_FIT
    steering_action_mode: str = "delta"
    steering_delta_max: float = STEERING_DELTA_MAX_RAD

    gravity: float = GRAVITY
    # HUD / viewer speed-axis only; the bicycle plant has no aero or power cap.
    power_max: float = 320.0
    dragcoeff: float = 0.075
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
        params.sim_dt = float(sim_dt)
        params.control_dt = float(control_dt)
        params.gravity = float(self.gravity)
        params.izz = float(self.izz)
        params.wheelbase = float(self.wheelbase)
        params.lf = float(self.lf)
        params.lr = float(self.lr)
        params.track_width = float(self.track_width)
        params.wheel_radius = float(self.wheel_radius)
        params.tire_mu = float(self.tire_mu)
        params.lat_stiffness_scale = float(self.lat_stiffness_scale)
        params.f_drive_max = float(self.f_drive_max)
        params.f_brake_max = float(self.f_brake_max)
        params.i_drive_max = float(self.i_drive_max_a)
        params.i_brake_max = float(self.i_brake_max_a)
        params.i_slew = float(self.i_slew_a_per_s)
        params.c_coast = float(self.c_coast)
        params.v_eps = float(self.v_eps)
        params.low_speed_blend = float(self.low_speed_blend)
        params.drive_torque_sign = float(self.drive_torque_sign)
        params.max_steer = float(self.max_steer)
        params.steer_time_constant = float(self.t_delta)
        params.steer_gain = float(self.steer_gain)
        params.steering_action_mode = int(self.steering_action_mode == "delta")
        params.steering_delta_max = float(self.steering_delta_max)
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
    vehicle.steer_logical = buffers.steer_logical[index]
    vehicle.i_drive = buffers.i_drive[index]
    vehicle.i_brake = buffers.i_brake[index]
    vehicle.i_drive_cmd = buffers.i_drive_cmd[index]
    vehicle.i_brake_cmd = buffers.i_brake_cmd[index]
    vehicle.applied_effort = buffers.applied_effort[index]
    vehicle.ax = buffers.ax[index]
    vehicle.ay = buffers.ay[index]
    vehicle.imu_gz = buffers.imu_gz[index]
    vehicle.omega = buffers.omega[index]
    vehicle.slip_ratio = buffers.slip_ratio[index]
    vehicle.slip_angle = buffers.slip_angle[index]
    vehicle.load_ratio = buffers.load_ratio[index]
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
    buffers.steer_logical[index] = vehicle.steer_logical
    buffers.i_drive[index] = vehicle.i_drive
    buffers.i_brake[index] = vehicle.i_brake
    buffers.i_drive_cmd[index] = vehicle.i_drive_cmd
    buffers.i_brake_cmd[index] = vehicle.i_brake_cmd
    buffers.applied_effort[index] = vehicle.applied_effort
    buffers.ax[index] = vehicle.ax
    buffers.ay[index] = vehicle.ay
    buffers.imu_gz[index] = vehicle.imu_gz
    buffers.omega[index] = vehicle.omega
    buffers.slip_ratio[index] = vehicle.slip_ratio
    buffers.slip_angle[index] = vehicle.slip_angle
    buffers.load_ratio[index] = vehicle.load_ratio


@wp.func
def _slew_channel(prev: wp.float32, tgt: wp.float32, max_step: wp.float32) -> wp.float32:
    delta = tgt - prev
    if wp.abs(delta) <= max_step:
        return tgt
    return prev + wp.clamp(delta, -max_step, max_step)


@wp.func
def apply_current_slew(
    vehicle: VehicleLocal,
    params: SimParams,
    accel_scale: wp.float32,
) -> VehicleLocal:
    max_step = wp.abs(params.i_slew) * params.sim_dt * wp.max(accel_scale, 0.0)
    i_drive = _slew_channel(vehicle.i_drive, vehicle.i_drive_cmd, max_step)
    i_brake = _slew_channel(vehicle.i_brake, vehicle.i_brake_cmd, max_step)
    if i_drive > 0.0 and i_brake > 0.0:
        if vehicle.i_drive_cmd >= vehicle.i_brake_cmd:
            i_brake = 0.0
        else:
            i_drive = 0.0
    vehicle.i_drive = i_drive
    vehicle.i_brake = i_brake
    drive_frac = i_drive / wp.max(params.i_drive_max, 1.0e-6)
    brake_frac = i_brake / wp.max(params.i_brake_max, 1.0e-6)
    vehicle.applied_effort = drive_frac - brake_frac
    return vehicle


@wp.func
def apply_command(
    vehicle: VehicleLocal,
    action: wp.vec2f,
    params: SimParams,
) -> VehicleLocal:
    longitudinal = wp.clamp(action[0], -1.0, 1.0)
    if longitudinal >= 0.0:
        vehicle.i_drive_cmd = longitudinal * params.i_drive_max
        vehicle.i_brake_cmd = 0.0
    else:
        vehicle.i_drive_cmd = 0.0
        vehicle.i_brake_cmd = -longitudinal * params.i_brake_max

    if params.steering_action_mode != 0:
        vehicle.steer_logical = wp.clamp(
            vehicle.steer_logical + action[1] * params.steering_delta_max,
            -params.max_steer,
            params.max_steer,
        )
    else:
        vehicle.steer_logical = wp.clamp(
            action[1] * params.max_steer,
            -params.max_steer,
            params.max_steer,
        )
    return vehicle


@wp.func
def apply_steering_lag(
    vehicle: VehicleLocal,
    params: SimParams,
    steer_lag: wp.float32,
    steer_rate: wp.float32,
    steer_gain: wp.float32,
) -> VehicleLocal:
    target = steer_gain * vehicle.steer_logical
    lag = wp.max(steer_lag, 0.0)
    alpha = params.sim_dt / (lag + params.sim_dt)
    next_steer = vehicle.steer + alpha * (target - vehicle.steer)
    max_step = wp.max(steer_rate, 0.0) * params.sim_dt
    vehicle.steer = vehicle.steer + wp.clamp(
        next_steer - vehicle.steer, -max_step, max_step
    )
    return vehicle


@wp.func
def _smoothstep01(value: wp.float32) -> wp.float32:
    t = wp.clamp(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _saturate_lateral(
    fy: wp.float32,
    fx: wp.float32,
    peak: wp.float32,
) -> wp.float32:
    fy_max_sq = peak * peak - fx * fx
    if fy_max_sq <= 0.0:
        return 0.0
    fy_max = wp.sqrt(fy_max_sq)
    return wp.clamp(fy, -fy_max, fy_max)


@wp.func
def integrate_vehicle_substep(
    vehicle: VehicleLocal,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    vmax_scale: wp.float32,
    accel_scale: wp.float32,
    izz: wp.float32,
    c_f: wp.float32,
    c_r: wp.float32,
    steer_lag: wp.float32,
    steer_rate: wp.float32,
    steer_gain: wp.float32,
    params: SimParams,
) -> VehicleLocal:
    vehicle = apply_current_slew(vehicle, params, accel_scale)
    vehicle = apply_steering_lag(
        vehicle, params, steer_lag, steer_rate, steer_gain
    )

    weight = mass * params.gravity
    fz_f = weight * params.lr / params.wheelbase
    fz_r = weight * params.lf / params.wheelbase

    drive_force = (
        vehicle.i_drive
        * params.f_drive_max
        * vmax_scale
        / wp.max(params.i_drive_max, 1.0e-6)
        * drive_scale
        * params.drive_torque_sign
    )
    brake_force = (
        vehicle.i_brake * params.f_brake_max / wp.max(params.i_brake_max, 1.0e-6)
    )
    coast = params.c_coast
    vx = vehicle.vx
    if vx > params.v_eps:
        f_long = drive_force - brake_force - coast
    elif vx < -params.v_eps:
        f_long = drive_force + brake_force + coast
    else:
        f_long = drive_force

    fx_f = f_long * fz_f / wp.max(weight, 1.0e-6)
    fx_r = f_long * fz_r / wp.max(weight, 1.0e-6)

    alpha_f = wp.atan2(vehicle.vy + params.lf * vehicle.yaw_rate, vx) - vehicle.steer
    alpha_r = wp.atan2(vehicle.vy - params.lr * vehicle.yaw_rate, vx)
    fy_f = _saturate_lateral(-c_f * alpha_f, fx_f, mu * fz_f)
    fy_r = _saturate_lateral(-c_r * alpha_r, fx_r, mu * fz_r)

    cosine = wp.cos(vehicle.steer)
    sine = wp.sin(vehicle.steer)
    fx_front = fx_f * cosine - fy_f * sine
    fy_front = fx_f * sine + fy_f * cosine
    force_x = fx_front + fx_r
    force_y = fy_front + fy_r
    yaw_moment = params.lf * fy_front - params.lr * fy_r

    ax = force_x / mass
    ay = force_y / mass
    vx_dyn = vx + params.sim_dt * (ax + vehicle.yaw_rate * vehicle.vy)
    vy_dyn = vehicle.vy + params.sim_dt * (ay - vehicle.yaw_rate * vx)
    r_dyn = vehicle.yaw_rate + params.sim_dt * (
        yaw_moment / wp.max(izz, 1.0e-6)
    )

    if vx * vx_dyn < 0.0 and vx_dyn * drive_force <= 0.0:
        vx_dyn = 0.0

    r_kin = vx * wp.tan(vehicle.steer) / wp.max(params.wheelbase, 1.0e-6)
    vy_kin = r_kin * params.lr
    blend = _smoothstep01(wp.abs(vx) / wp.max(params.low_speed_blend, 1.0e-6))
    vy_next = (1.0 - blend) * vy_kin + blend * vy_dyn
    r_next = (1.0 - blend) * r_kin + blend * r_dyn
    yaw_next = vehicle.yaw + params.sim_dt * r_next
    vehicle.x = vehicle.x + params.sim_dt * (
        vx_dyn * wp.cos(yaw_next) - vy_next * wp.sin(yaw_next)
    )
    vehicle.y = vehicle.y + params.sim_dt * (
        vx_dyn * wp.sin(yaw_next) + vy_next * wp.cos(yaw_next)
    )
    vehicle.yaw = yaw_next
    vehicle.vx = vx_dyn
    vehicle.vy = vy_next
    vehicle.yaw_rate = r_next
    vehicle.ax = vehicle.ax + ax
    vehicle.ay = vehicle.ay + ay
    vehicle.imu_gz = vehicle.imu_gz + r_next

    wheel_speed = vx_dyn / wp.max(params.wheel_radius, 1.0e-6)
    vehicle.omega = wp.vec4f(wheel_speed, wheel_speed, wheel_speed, wheel_speed)
    vehicle.slip_ratio = wp.vec4f(0.0)
    vehicle.slip_angle = wp.vec4f(alpha_r, alpha_r, alpha_f, alpha_f)
    vehicle.load_ratio = wp.vec4f(1.0, 1.0, 1.0, 1.0)
    return vehicle


@wp.func
def apply_command_and_integrate(
    vehicle: VehicleLocal,
    action: wp.vec2f,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    vmax_scale: wp.float32,
    accel_scale: wp.float32,
    izz: wp.float32,
    c_f: wp.float32,
    c_r: wp.float32,
    steer_lag: wp.float32,
    steer_rate: wp.float32,
    steer_gain: wp.float32,
    params: SimParams,
    substeps: wp.int32,
    command_delay: wp.int32,
) -> VehicleLocal:
    delay = command_delay
    if delay >= substeps:
        delay = wp.int32(0)
    vehicle.ax = 0.0
    vehicle.ay = 0.0
    vehicle.imu_gz = 0.0
    if delay == 0:
        vehicle = apply_command(vehicle, action, params)
    count = wp.float32(0.0)
    for step in range(_MAX_INTEGRATE_SUBSTEPS):
        if step >= substeps:
            break
        if delay > 0 and step == delay:
            vehicle = apply_command(vehicle, action, params)
        vehicle = integrate_vehicle_substep(
            vehicle,
            mass,
            mu,
            drive_scale,
            vmax_scale,
            accel_scale,
            izz,
            c_f,
            c_r,
            steer_lag,
            steer_rate,
            steer_gain,
            params,
        )
        count = count + 1.0
    inv = 1.0 / wp.max(count, 1.0)
    vehicle.ax = vehicle.ax * inv
    vehicle.ay = vehicle.ay * inv
    vehicle.imu_gz = vehicle.imu_gz * inv
    return vehicle


@wp.func
def apply_command_and_integrate_env(
    vehicle: VehicleLocal,
    action: wp.vec2f,
    buffers: VehicleBuffers,
    index: wp.int32,
    params: SimParams,
    substeps: wp.int32,
) -> VehicleLocal:
    return apply_command_and_integrate(
        vehicle,
        action,
        buffers.mass[index],
        buffers.mu[index],
        buffers.drive_scale[index],
        buffers.vmax_scale[index],
        buffers.accel_scale[index],
        buffers.izz[index],
        buffers.c_f[index],
        buffers.c_r[index],
        buffers.steer_lag[index],
        buffers.steer_rate[index],
        buffers.steer_gain[index] * buffers.steer_scale[index],
        params,
        substeps,
        buffers.command_delay[index],
    )


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
        and wp.isfinite(vehicle.steer)
        and wp.isfinite(vehicle.steer_logical)
    )


@wp.func
def clear_vehicle(vehicle: VehicleLocal) -> VehicleLocal:
    vehicle.vy = 0.0
    vehicle.yaw_rate = 0.0
    vehicle.steer = 0.0
    vehicle.steer_logical = 0.0
    vehicle.i_drive = 0.0
    vehicle.i_brake = 0.0
    vehicle.i_drive_cmd = 0.0
    vehicle.i_brake_cmd = 0.0
    vehicle.applied_effort = 0.0
    vehicle.ax = 0.0
    vehicle.ay = 0.0
    vehicle.imu_gz = 0.0
    vehicle.omega = wp.vec4f(0.0)
    vehicle.slip_ratio = wp.vec4f(0.0)
    vehicle.slip_angle = wp.vec4f(0.0)
    vehicle.load_ratio = wp.vec4f(1.0, 1.0, 1.0, 1.0)
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
        "steer_logical": 0.0,
        "i_drive": 0.0,
        "i_brake": 0.0,
        "i_drive_cmd": 0.0,
        "i_brake_cmd": 0.0,
        "applied_effort": 0.0,
        "ax": 0.0,
        "ay": 0.0,
        "imu_gz": 0.0,
        "omega": np.zeros(4, dtype=np.float32),
        "slip_ratio": np.zeros(4, dtype=np.float32),
        "slip_angle": np.zeros(4, dtype=np.float32),
        "load_ratio": np.ones(4, dtype=np.float32),
    }


def wrap_angle_np(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

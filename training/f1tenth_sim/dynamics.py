import warp as wp

from .drivetrain import direct_drive_torque
from .params import SimParams
from .suspension import static_wheel_load, warp_quasi_static_loads
from .tire import combined_pacejka


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
    steer_bias: wp.array(dtype=wp.float32)


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
    steer_bias: wp.float32,
    params: SimParams,
) -> VehicleLocal:
    target = wp.clamp(action[0], -1.0, 1.0)
    max_step = params.effort_slew_rate * params.control_dt
    if max_step > 0.0:
        delta = target - vehicle.effort_state
        next_effort = vehicle.effort_state + wp.clamp(
            delta, -max_step, max_step
        )
        if wp.abs(delta) <= max_step:
            vehicle.applied_effort = (
                target - delta * wp.abs(delta) / (2.0 * max_step)
            )
        else:
            vehicle.applied_effort = 0.5 * (
                vehicle.effort_state + next_effort
            )
        vehicle.effort_state = next_effort
    else:
        vehicle.effort_state = target
        vehicle.applied_effort = target

    steer_target = wp.clamp(
        action[1] * params.max_steer + steer_bias,
        -params.max_steer,
        params.max_steer,
    )
    steer_alpha = params.control_dt / (
        params.steer_time_constant + params.control_dt
    )
    vehicle.steer = vehicle.steer + steer_alpha * (
        steer_target - vehicle.steer
    )
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
        alpha_force = wp.atan2(
            -wheel_lat[wheel], long_abs + params.slip_min_lat
        )
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
        vehicle.load_ratio[wheel] = loads[wheel] / wp.max(
            static_load, 1.0e-6
        )

    speed = wp.sqrt(vehicle.vx * vehicle.vx + vehicle.vy * vehicle.vy)
    if params.enable_aero_drag != 0:
        force_x = force_x - params.drag_coeff * speed * vehicle.vx
        force_y = force_y - params.drag_coeff * speed * vehicle.vy

    ax = force_x / mass
    ay = force_y / mass
    yaw_acceleration = yaw_moment / params.izz
    vx_next = vehicle.vx + params.sim_dt * (
        ax + vehicle.yaw_rate * vehicle.vy
    )
    vy_next = vehicle.vy + params.sim_dt * (
        ay - vehicle.yaw_rate * vehicle.vx
    )
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

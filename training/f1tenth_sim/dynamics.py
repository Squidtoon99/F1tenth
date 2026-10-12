import warp as wp

from .params import SimParams

# Teleop→VESC p50 is ~10 ms (one 50 Hz gate tick). At sim_dt=0.005 that is
# two substeps, shared by Warp replay and the training env.
COMMAND_DELAY_SUBSTEPS = 2


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
def apply_current_slew(vehicle: VehicleLocal, params: SimParams) -> VehicleLocal:
    max_step = wp.abs(params.i_slew) * params.sim_dt
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
    steer_bias: wp.float32,
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
            action[1] * params.max_steer + steer_bias,
            -params.max_steer,
            params.max_steer,
        )
    return vehicle


@wp.func
def apply_steering_lag(vehicle: VehicleLocal, params: SimParams) -> VehicleLocal:
    target = params.steer_gain * vehicle.steer_logical
    alpha = params.sim_dt / (params.steer_time_constant + params.sim_dt)
    vehicle.steer = vehicle.steer + alpha * (target - vehicle.steer)
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
    params: SimParams,
) -> VehicleLocal:
    vehicle = apply_current_slew(vehicle, params)
    vehicle = apply_steering_lag(vehicle, params)

    weight = mass * params.gravity
    fz_f = weight * params.lr / params.wheelbase
    fz_r = weight * params.lf / params.wheelbase
    c_f = params.lat_stiffness_scale * fz_f
    c_r = params.lat_stiffness_scale * fz_r

    drive_force = (
        vehicle.i_drive
        * params.f_drive_max
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
    r_dyn = vehicle.yaw_rate + params.sim_dt * (yaw_moment / params.izz)

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
    steer_bias: wp.float32,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    params: SimParams,
    substeps: wp.int32,
) -> VehicleLocal:
    delay = wp.int32(COMMAND_DELAY_SUBSTEPS)
    if delay >= substeps:
        delay = wp.int32(0)
    vehicle.ax = 0.0
    vehicle.ay = 0.0
    vehicle.imu_gz = 0.0
    if delay == 0:
        vehicle = apply_command(vehicle, action, steer_bias, params)
    count = wp.float32(substeps)
    for step in range(substeps):
        if delay > 0 and step == delay:
            vehicle = apply_command(vehicle, action, steer_bias, params)
        vehicle = integrate_vehicle_substep(
            vehicle, mass, mu, drive_scale, params
        )
    inv = 1.0 / wp.max(count, 1.0)
    vehicle.ax = vehicle.ax * inv
    vehicle.ay = vehicle.ay * inv
    vehicle.imu_gz = vehicle.imu_gz * inv
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
        and wp.isfinite(vehicle.steer)
        and wp.isfinite(vehicle.steer_logical)
    )

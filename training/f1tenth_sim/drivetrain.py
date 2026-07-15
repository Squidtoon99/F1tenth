import warp as wp

from .params import SimParams


@wp.func
def direct_drive_torque(
    effort: wp.float32,
    body_vx: wp.float32,
    omega: wp.vec4f,
    mass: wp.float32,
    mu: wp.float32,
    drive_scale: wp.float32,
    params: SimParams,
) -> wp.vec4f:
    drive = wp.max(effort, 0.0)
    brake = wp.max(-effort, 0.0)
    drive_force = drive * params.f_drive_max
    drive_force = wp.min(
        drive_force,
        params.power_max / wp.max(wp.abs(body_vx), params.v_eps),
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

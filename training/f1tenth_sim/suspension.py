import warp as wp

from .params import SimParams


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

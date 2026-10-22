from __future__ import annotations

import numpy as np
import warp as wp

from f1tenth_policy.layout import (  # noqa: F401
    ACTOR_OBS_DIM,
    IMU_DIM,
    IMU_START as ACTOR_IMU_START,
    LIDAR_DIM as ACTOR_LIDAR_DIM,
    NATIVE_NUM_BEAMS,
    PROPRIO_DIM as ACTOR_PROPRIO_DIM,
    STEER_DELTA0 as ACTOR_STEER_DELTA0,
    STEER_DELTA1 as ACTOR_STEER_DELTA1,
    STEER_DELTA2 as ACTOR_STEER_DELTA2,
    STEER_T as ACTOR_STEER_T,
    STEER_T1 as ACTOR_STEER_T1,
    STEER_T2 as ACTOR_STEER_T2,
    THROTTLE_CURRENT as ACTOR_THROTTLE_CURRENT,
    THROTTLE_PRED as ACTOR_THROTTLE_PRED,
    VESC_CURRENT as ACTOR_VESC_CURRENT,
    VESC_SPEED as ACTOR_VESC_SPEED,
)
from f1tenth_sim.dynamics import VehicleBuffers, VehicleLocal, load_vehicle
from f1tenth_sim.params import SimParams

from .kernel import (
    CorridorDistanceField,
    EnvBuffers,
    OpponentParams,
    ResetParams,
    SensorParams,
    map_variant_id,
    observation_noise,
    sample_corridor_distance,
)

wp.set_module_options({"fast_math": True, "enable_backward": False})

MAX_LIDAR_MARCH_STEPS = 128
LEGACY_LIDAR_MARCH_STEPS = 512
ACTOR_IMU_DIM = IMU_DIM
_EDT_HIT_FRAC = 0.55
VIEW_EGO = 0
VIEW_OPPONENT = 1
_VIEW_SEED_PRIME = 668265263
_VESC_SPEED_NOISE_FEATURE = 10_000
_VESC_SPEED_DROPOUT_FEATURE = 10_002
_VESC_CURRENT_NOISE_FEATURE = 10_001


def beam_trig_lut(
    angle_min: float,
    angle_increment: float,
    num_beams: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(num_beams)
    angles = np.float32(angle_min) + np.arange(n, dtype=np.float32) * np.float32(
        angle_increment
    )
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def attach_sensor_trig(params: SensorParams, num_envs: int, device: str) -> SensorParams:
    beam_cos, beam_sin = beam_trig_lut(
        float(params.angle_min),
        float(params.angle_increment),
        int(params.num_beams),
    )
    params.beam_cos = wp.array(beam_cos, dtype=wp.float32, device=device)
    params.beam_sin = wp.array(beam_sin, dtype=wp.float32, device=device)
    params.heading_cos = wp.zeros(int(num_envs), dtype=wp.float32, device=device)
    params.heading_sin = wp.zeros(int(num_envs), dtype=wp.float32, device=device)
    return params


@wp.func
def beam_direction_from_lut(
    heading_cos: wp.float32,
    heading_sin: wp.float32,
    beam_cos: wp.float32,
    beam_sin: wp.float32,
) -> wp.vec2f:
    return wp.vec2f(
        heading_cos * beam_cos - heading_sin * beam_sin,
        heading_sin * beam_cos + heading_cos * beam_sin,
    )


@wp.kernel(enable_backward=False)
def cache_lidar_heading_kernel(
    vehicles: VehicleBuffers,
    env: EnvBuffers,
    sensor: SensorParams,
):
    env_id = wp.tid()
    heading = (
        vehicles.yaw[env_id]
        + sensor.lidar_offset_yaw
        + env.lidar_extrinsic_yaw[env_id]
        + env.lidar_angle_bias[env_id]
    )
    sensor.heading_cos[env_id] = wp.cos(heading)
    sensor.heading_sin[env_id] = wp.sin(heading)


@wp.func
def ray_obb_distance(
    origin: wp.vec2f,
    direction: wp.vec2f,
    box_x: wp.float32,
    box_y: wp.float32,
    box_yaw: wp.float32,
    half_length: wp.float32,
    half_width: wp.float32,
) -> wp.float32:
    cosine = wp.cos(box_yaw)
    sine = wp.sin(box_yaw)
    dx = origin[0] - box_x
    dy = origin[1] - box_y
    local_o = wp.vec2f(cosine * dx + sine * dy, -sine * dx + cosine * dy)
    local_d = wp.vec2f(
        cosine * direction[0] + sine * direction[1],
        -sine * direction[0] + cosine * direction[1],
    )
    t_min = wp.float32(-1.0e30)
    t_max = wp.float32(1.0e30)
    for axis in range(2):
        origin_a = local_o[axis]
        dir_a = local_d[axis]
        half = half_length
        if axis == 1:
            half = half_width
        if wp.abs(dir_a) < 1.0e-12:
            if origin_a < -half or origin_a > half:
                return wp.float32(1.0e30)
        else:
            inv = 1.0 / dir_a
            t0 = (-half - origin_a) * inv
            t1 = (half - origin_a) * inv
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp
            if t0 > t_min:
                t_min = t0
            if t1 < t_max:
                t_max = t1
            if t_min > t_max:
                return wp.float32(1.0e30)
    if t_max < 0.0:
        return wp.float32(1.0e30)
    if t_min < 0.0:
        return wp.float32(0.0)
    return t_min


@wp.func
def sphere_trace_walls(
    origin: wp.vec2f,
    direction: wp.vec2f,
    field: CorridorDistanceField,
    variant_id: wp.int32,
    range_max: wp.float32,
    max_march_steps: wp.int32,
) -> wp.float32:
    traveled = wp.float32(0.0)
    hit_eps = _EDT_HIT_FRAC * field.resolution
    min_step = 0.5 * field.resolution
    steps = max_march_steps
    if steps <= 0:
        steps = wp.int32(MAX_LIDAR_MARCH_STEPS)
    for _ in range(steps):
        if traveled >= range_max:
            return range_max
        px = origin[0] + traveled * direction[0]
        py = origin[1] + traveled * direction[1]
        dist = sample_corridor_distance(field, px, py, variant_id)
        if dist < 0.0:
            return range_max
        if dist <= hit_eps:
            return wp.min(traveled, range_max)
        step = wp.max(dist, min_step)
        traveled = traveled + step
    return range_max


@wp.func
def sensor_view_seed(seed: wp.int32, view_id: wp.int32) -> wp.int32:
    return seed ^ (view_id * _VIEW_SEED_PRIME)


@wp.func
def lidar_beam_range(
    ego: VehicleLocal,
    opponent: VehicleLocal,
    env_id: wp.int32,
    beam_id: wp.int32,
    env: EnvBuffers,
    field: CorridorDistanceField,
    sensor: SensorParams,
    opponent_params: OpponentParams,
    reset: ResetParams,
    has_opponent: wp.int32,
    view_id: wp.int32,
) -> wp.float32:
    beam_time = wp.float32(0.0)
    mount_x = sensor.lidar_offset_x + env.lidar_extrinsic_x[env_id]
    mount_y = sensor.lidar_offset_y + env.lidar_extrinsic_y[env_id]
    cosine = wp.cos(ego.yaw)
    sine = wp.sin(ego.yaw)
    origin = wp.vec2f(
        ego.x + cosine * mount_x - sine * mount_y,
        ego.y + sine * mount_x + cosine * mount_y,
    )
    origin = origin + beam_time * wp.vec2f(0.0, 0.0)
    direction = beam_direction_from_lut(
        sensor.heading_cos[env_id],
        sensor.heading_sin[env_id],
        sensor.beam_cos[beam_id],
        sensor.beam_sin[beam_id],
    )

    variant_id = map_variant_id(
        reset.seed, env_id, env.episode_id[env_id], field.variant_count
    )

    hit = sphere_trace_walls(
        origin,
        direction,
        field,
        variant_id,
        sensor.range_max,
        sensor.max_march_steps,
    )
    if has_opponent != 0:
        half_length = 0.5 * opponent_params.car_length
        half_width = 0.5 * opponent_params.car_width
        car_bound_radius = wp.sqrt(
            half_length * half_length + half_width * half_width
        )
        dx = opponent.x - origin[0]
        dy = opponent.y - origin[1]
        dir_x = direction[0]
        dir_y = direction[1]
        reach = hit + car_bound_radius
        if dx * dx + dy * dy <= reach * reach:
            if dx * dir_x + dy * dir_y >= -car_bound_radius:
                opp_hit = ray_obb_distance(
                    origin,
                    direction,
                    opponent.x,
                    opponent.y,
                    opponent.yaw,
                    half_length,
                    half_width,
                )
                if opp_hit < hit:
                    hit = opp_hit

    if hit < sensor.range_min:
        hit = sensor.range_max
    if hit > sensor.range_max:
        hit = sensor.range_max

    view_seed = sensor_view_seed(reset.seed, view_id)
    noise_std = env.lidar_range_noise_std[env_id]
    if noise_std > 0.0:
        hit = hit + observation_noise(
            view_seed,
            env_id,
            env.episode_id[env_id],
            env.episode_step[env_id],
            beam_id,
            noise_std,
        )
        hit = wp.clamp(hit, sensor.range_min, sensor.range_max)

    dropout = env.lidar_dropout_prob[env_id]
    if hit > sensor.reliable_range:
        dropout = wp.max(dropout, env.lidar_far_dropout_prob[env_id])
    if dropout > 0.0:
        bin_m = sensor.lidar_dropout_bin_m
        if bin_m > 0.0:
            hx = origin[0] + hit * direction[0]
            hy = origin[1] + hit * direction[1]
            bx = wp.int32(wp.floor(hx / bin_m))
            by = wp.int32(wp.floor(hy / bin_m))
            mixed = (
                view_seed
                ^ (env_id * 73244475)
                ^ (env.episode_id[env_id] * 295075153)
                ^ (bx * 104395301)
                ^ (by * 122949829)
            )
        else:
            mixed = (
                view_seed
                ^ (env_id * 73244475)
                ^ (env.episode_id[env_id] * 295075153)
                ^ (env.episode_step[env_id] * 104395301)
                ^ ((beam_id + sensor.num_beams) * 122949829)
            )
        random = wp.rand_init(mixed & 2147483647)
        if wp.randf(random) < dropout:
            hit = sensor.range_max

    return hit


@wp.kernel(enable_backward=False)
def lidar_solo_kernel(
    ego_buffers: VehicleBuffers,
    env: EnvBuffers,
    field: CorridorDistanceField,
    sensor: SensorParams,
    opponent_params: OpponentParams,
    reset: ResetParams,
    ranges: wp.array2d(dtype=wp.float32),
):
    env_id, beam_id = wp.tid()
    if beam_id >= sensor.num_beams:
        return
    ego = load_vehicle(ego_buffers, env_id)
    opponent = VehicleLocal()
    ranges[env_id, beam_id] = lidar_beam_range(
        ego,
        opponent,
        env_id,
        beam_id,
        env,
        field,
        sensor,
        opponent_params,
        reset,
        0,
        VIEW_EGO,
    )


@wp.kernel(enable_backward=False)
def lidar_stage_kernel(
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    field: CorridorDistanceField,
    sensor: SensorParams,
    opponent_params: OpponentParams,
    reset: ResetParams,
    view_id: wp.int32,
    ranges: wp.array2d(dtype=wp.float32),
):
    env_id, beam_id = wp.tid()
    if beam_id >= sensor.num_beams:
        return
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    ranges[env_id, beam_id] = lidar_beam_range(
        ego,
        opponent,
        env_id,
        beam_id,
        env,
        field,
        sensor,
        opponent_params,
        reset,
        1,
        view_id,
    )


@wp.func
def write_imu_columns(
    ego: VehicleLocal,
    env: EnvBuffers,
    reset: ResetParams,
    sim: SimParams,
    env_id: wp.int32,
    view_id: wp.int32,
    output: wp.array2d(dtype=wp.float32),
    col0: wp.int32,
):
    ax = ego.ax
    ay = ego.ay
    az = sim.gravity
    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = ego.imu_gz

    misalign = env.imu_axis_misalign[env_id]
    if misalign != 0.0:
        cosine = wp.cos(misalign)
        sine = wp.sin(misalign)
        ax_r = cosine * ax - sine * ay
        ay_r = sine * ax + cosine * ay
        ax = ax_r
        ay = ay_r
        gx_r = cosine * gx - sine * gy
        gy_r = sine * gx + cosine * gy
        gx = gx_r
        gy = gy_r

    ax = ax + env.imu_accel_bias_x[env_id]
    ay = ay + env.imu_accel_bias_y[env_id]
    gz = gz + env.imu_gyro_bias_z[env_id]

    view_seed = sensor_view_seed(reset.seed, view_id)
    accel_std = env.imu_accel_noise_std[env_id]
    gyro_std = env.imu_gyro_noise_std[env_id]
    episode = env.episode_id[env_id]
    step = env.episode_step[env_id]
    if accel_std > 0.0:
        ax = ax + observation_noise(view_seed, env_id, episode, step, 0, accel_std)
        ay = ay + observation_noise(view_seed, env_id, episode, step, 1, accel_std)
    if gyro_std > 0.0:
        gz = gz + observation_noise(view_seed, env_id, episode, step, 5, gyro_std)

    output[env_id, col0 + 0] = ax
    output[env_id, col0 + 1] = ay
    output[env_id, col0 + 2] = az
    output[env_id, col0 + 3] = gx
    output[env_id, col0 + 4] = gy
    output[env_id, col0 + 5] = gz


@wp.func
def write_vesc_and_commands(
    vehicle: VehicleBuffers,
    env: EnvBuffers,
    reset: ResetParams,
    sim: SimParams,
    env_id: wp.int32,
    use_opponent_commands: wp.int32,
    view_id: wp.int32,
    output: wp.array2d(dtype=wp.float32),
):
    raw_speed = vehicle.vx[env_id]
    speed = raw_speed
    current = vehicle.applied_effort[env_id]
    speed = speed + env.vesc_speed_bias[env_id]
    current = current + env.vesc_current_bias[env_id]
    view_seed = sensor_view_seed(reset.seed, view_id)
    episode = env.episode_id[env_id]
    step = env.episode_step[env_id]
    speed_std = env.vesc_speed_noise_std[env_id]
    if speed_std > 0.0:
        speed = speed + observation_noise(
            view_seed,
            env_id,
            episode,
            step,
            _VESC_SPEED_NOISE_FEATURE,
            speed_std,
        )
    if raw_speed * speed < 0.0 and wp.abs(raw_speed) > 0.5:
        speed = raw_speed
    dropout = env.vesc_speed_dropout_prob[env_id]
    if dropout > 0.0:
        mixed = (
            view_seed
            ^ (env_id * 73244475)
            ^ (episode * 295075153)
            ^ (step * 104395301)
            ^ (_VESC_SPEED_DROPOUT_FEATURE * 122949829)
        )
        random = wp.rand_init(mixed & 2147483647)
        if wp.randf(random) < dropout:
            speed = 0.0
    current_std = env.vesc_current_noise_std[env_id]
    if current_std > 0.0:
        current = current + observation_noise(
            view_seed,
            env_id,
            episode,
            step,
            _VESC_CURRENT_NOISE_FEATURE,
            current_std,
        )
    output[env_id, ACTOR_VESC_SPEED] = speed
    output[env_id, ACTOR_VESC_CURRENT] = current

    stride = sim.history_stride
    if stride < 1:
        stride = wp.int32(1)
    if use_opponent_commands != 0:
        longitudinal = env.opponent_executed_longitudinal_history[env_id, 0]
        longitudinal_predecessor = env.opponent_executed_longitudinal_history[
            env_id, stride
        ]
        steer_t = env.opponent_executed_steer_history[env_id, 0]
        steer_t1 = env.opponent_executed_steer_history[env_id, stride]
        steer_t2 = env.opponent_executed_steer_history[env_id, 2 * stride]
        steer_t3 = env.opponent_executed_steer_history[env_id, 3 * stride]
    else:
        longitudinal = env.executed_longitudinal_history[env_id, 0]
        longitudinal_predecessor = env.executed_longitudinal_history[
            env_id, stride
        ]
        steer_t = env.executed_steer_history[env_id, 0]
        steer_t1 = env.executed_steer_history[env_id, stride]
        steer_t2 = env.executed_steer_history[env_id, 2 * stride]
        steer_t3 = env.executed_steer_history[env_id, 3 * stride]
    output[env_id, ACTOR_THROTTLE_CURRENT] = longitudinal
    output[env_id, ACTOR_THROTTLE_PRED] = longitudinal_predecessor
    output[env_id, ACTOR_STEER_T] = steer_t
    output[env_id, ACTOR_STEER_T1] = steer_t1
    output[env_id, ACTOR_STEER_T2] = steer_t2
    output[env_id, ACTOR_STEER_DELTA0] = steer_t - steer_t1
    output[env_id, ACTOR_STEER_DELTA1] = steer_t1 - steer_t2
    output[env_id, ACTOR_STEER_DELTA2] = steer_t2 - steer_t3


@wp.kernel(enable_backward=False)
def imu_stage_kernel(
    ego_buffers: VehicleBuffers,
    env: EnvBuffers,
    reset: ResetParams,
    sim: SimParams,
    view_id: wp.int32,
    imu: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    ego = load_vehicle(ego_buffers, env_id)
    write_imu_columns(ego, env, reset, sim, env_id, view_id, imu, 0)


@wp.kernel(enable_backward=False)
def sensor_actor_solo_kernel(
    ego_buffers: VehicleBuffers,
    env: EnvBuffers,
    field: CorridorDistanceField,
    sensor: SensorParams,
    opponent_params: OpponentParams,
    reset: ResetParams,
    sim: SimParams,
    output: wp.array2d(dtype=wp.float32),
):
    env_id, beam_id = wp.tid()
    if beam_id >= sensor.num_beams:
        return
    ego = load_vehicle(ego_buffers, env_id)
    opponent = VehicleLocal()
    output[env_id, beam_id] = lidar_beam_range(
        ego,
        opponent,
        env_id,
        beam_id,
        env,
        field,
        sensor,
        opponent_params,
        reset,
        0,
        VIEW_EGO,
    )
    if beam_id == 0:
        write_imu_columns(
            ego, env, reset, sim, env_id, VIEW_EGO, output, ACTOR_IMU_START
        )
        write_vesc_and_commands(
            ego_buffers,
            env,
            reset,
            sim,
            env_id,
            0,
            VIEW_EGO,
            output,
        )


@wp.kernel(enable_backward=False)
def sensor_actor_stage_kernel(
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    field: CorridorDistanceField,
    sensor: SensorParams,
    opponent_params: OpponentParams,
    reset: ResetParams,
    sim: SimParams,
    use_opponent_commands: wp.int32,
    view_id: wp.int32,
    output: wp.array2d(dtype=wp.float32),
):
    env_id, beam_id = wp.tid()
    if beam_id >= sensor.num_beams:
        return
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    output[env_id, beam_id] = lidar_beam_range(
        ego,
        opponent,
        env_id,
        beam_id,
        env,
        field,
        sensor,
        opponent_params,
        reset,
        1,
        view_id,
    )
    if beam_id == 0:
        write_imu_columns(
            ego, env, reset, sim, env_id, view_id, output, ACTOR_IMU_START
        )
        write_vesc_and_commands(
            ego_buffers,
            env,
            reset,
            sim,
            env_id,
            use_opponent_commands,
            view_id,
            output,
        )

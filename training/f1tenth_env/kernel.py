from __future__ import annotations

import warp as wp

from f1tenth_sim.dynamics import (
    VehicleBuffers,
    VehicleLocal,
    apply_command,
    integrate_vehicle_substep,
    load_vehicle,
    store_vehicle,
    vehicle_is_finite,
)
from f1tenth_sim.params import SimParams

FRENET_WINDOW = 40
FUTURE_SAMPLES = 60
OBS_DIM = 392
OBS_FUTURE_START = 12
OBS_SLIP_RATIO_START = 372
OBS_SLIP_ANGLE_START = 376
OBS_LOAD_START = 380
OBS_OPPONENT_START = 384
MAX_FORWARD_SEGMENTS = 2048
ACTION_HISTORY = 4
STEER_HISTORY = 4


@wp.struct
class TrackData:
    point: wp.array(dtype=wp.vec2f)
    tangent: wp.array(dtype=wp.vec2f)
    normal: wp.array(dtype=wp.vec2f)
    segment_length: wp.array(dtype=wp.float32)
    cumulative_length: wp.array(dtype=wp.float32)
    width_left: wp.array(dtype=wp.float32)
    width_right: wp.array(dtype=wp.float32)
    nearest_segment_lut: wp.array(dtype=wp.int32)
    count: wp.int32
    length: wp.float32
    lut_width: wp.int32
    lut_height: wp.int32
    lut_origin: wp.vec2f
    lut_resolution: wp.float32


@wp.struct
class FrenetState:
    segment: wp.int32
    t: wp.float32
    s: wp.float32
    ey: wp.float32
    width_left: wp.float32
    width_right: wp.float32
    boundary_distance: wp.float32
    distance_sq: wp.float32
    projection: wp.vec2f
    tangent: wp.vec2f


@wp.func
def wrap_segment(index: wp.int32, count: wp.int32) -> wp.int32:
    wrapped = index % count
    if wrapped < 0:
        wrapped = wrapped + count
    return wrapped


@wp.func
def project_window(
    position: wp.vec2f,
    seed_segment: wp.int32,
    track: TrackData,
) -> FrenetState:
    best = FrenetState()
    best.distance_sq = wp.float32(1.0e30)
    for relative in range(-FRENET_WINDOW, FRENET_WINDOW + 1):
        segment = wrap_segment(seed_segment + relative, track.count)
        following = wrap_segment(segment + 1, track.count)
        start = track.point[segment]
        edge = track.point[following] - start
        alpha = wp.clamp(
            wp.dot(position - start, edge)
            / wp.max(wp.dot(edge, edge), 1.0e-10),
            0.0,
            1.0,
        )
        projection = start + alpha * edge
        delta = position - projection
        distance_sq = wp.dot(delta, delta)
        if distance_sq < best.distance_sq - 1.0e-10 or (
            wp.abs(distance_sq - best.distance_sq) <= 1.0e-10
            and segment < best.segment
        ):
            tangent = track.tangent[segment]
            normal = track.normal[segment]
            width_left = track.width_left[segment] + alpha * (
                track.width_left[following] - track.width_left[segment]
            )
            width_right = track.width_right[segment] + alpha * (
                track.width_right[following] - track.width_right[segment]
            )
            ey = wp.dot(delta, normal)
            best.segment = segment
            best.t = alpha
            best.s = (
                track.cumulative_length[segment]
                + alpha * track.segment_length[segment]
            )
            best.ey = ey
            best.width_left = width_left
            best.width_right = width_right
            best.boundary_distance = wp.min(
                width_left - ey, width_right + ey
            )
            best.distance_sq = distance_sq
            best.projection = projection
            best.tangent = tangent
    return best


@wp.func
def recovery_seed(position: wp.vec2f, track: TrackData) -> wp.int32:
    grid_x = wp.int32(
        wp.floor((position[0] - track.lut_origin[0]) / track.lut_resolution)
    )
    grid_y = wp.int32(
        wp.floor((position[1] - track.lut_origin[1]) / track.lut_resolution)
    )
    grid_x = wp.clamp(grid_x, 0, track.lut_width - 1)
    grid_y = wp.clamp(grid_y, 0, track.lut_height - 1)
    return track.nearest_segment_lut[grid_y * track.lut_width + grid_x]


@wp.func
def project_track(
    position: wp.vec2f,
    seed_segment: wp.int32,
    track: TrackData,
) -> FrenetState:
    result = project_window(position, seed_segment, track)
    if result.distance_sq > 0.25:
        result = project_window(position, recovery_seed(position, track), track)
    return result


@wp.func
def world_to_body_point(
    point: wp.vec2f,
    position: wp.vec2f,
    cosine: wp.float32,
    sine: wp.float32,
) -> wp.vec2f:
    delta = point - position
    return wp.vec2f(
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
    )


@wp.func
def write_vec2(
    output: wp.array2d(dtype=wp.float32),
    row: wp.int32,
    column: wp.int32,
    value: wp.vec2f,
):
    output[row, column] = value[0]
    output[row, column + 1] = value[1]


@wp.func
def write_future_track(
    env_id: wp.int32,
    position: wp.vec2f,
    yaw: wp.float32,
    speed: wp.float32,
    frenet: FrenetState,
    track: TrackData,
    output: wp.array2d(dtype=wp.float32),
    horizon_seconds: wp.float32,
    minimum_lookahead: wp.float32,
):
    lookahead = wp.max(speed * horizon_seconds, minimum_lookahead)
    cosine = wp.cos(yaw)
    sine = wp.sin(yaw)
    segment = frenet.segment
    segment_t = frenet.t
    covered = wp.float32(0.0)
    advanced = wp.int32(0)
    for sample in range(FUTURE_SAMPLES):
        target = (
            wp.float32(sample + 1) / wp.float32(FUTURE_SAMPLES)
        ) * lookahead
        available = (1.0 - segment_t) * track.segment_length[segment]
        for _ in range(MAX_FORWARD_SEGMENTS):
            if covered + available >= target:
                break
            covered = covered + available
            segment = wrap_segment(segment + 1, track.count)
            segment_t = 0.0
            available = track.segment_length[segment]
            advanced = advanced + 1
        alpha = segment_t + (target - covered) / wp.max(
            track.segment_length[segment], 1.0e-8
        )
        alpha = wp.clamp(alpha, 0.0, 1.0)
        following = wrap_segment(segment + 1, track.count)
        center = track.point[segment] + alpha * (
            track.point[following] - track.point[segment]
        )
        normal = track.normal[segment]
        width_left = track.width_left[segment] + alpha * (
            track.width_left[following] - track.width_left[segment]
        )
        width_right = track.width_right[segment] + alpha * (
            track.width_right[following] - track.width_right[segment]
        )
        left = center + width_left * normal
        right = center - width_right * normal
        body_position = wp.vec2f(position[0], position[1])
        write_vec2(
            output,
            env_id,
            OBS_FUTURE_START + 2 * sample,
            world_to_body_point(center, body_position, cosine, sine),
        )
        write_vec2(
            output,
            env_id,
            OBS_FUTURE_START + 2 * (FUTURE_SAMPLES + sample),
            world_to_body_point(left, body_position, cosine, sine),
        )
        write_vec2(
            output,
            env_id,
            OBS_FUTURE_START + 2 * (2 * FUTURE_SAMPLES + sample),
            world_to_body_point(right, body_position, cosine, sine),
        )


@wp.kernel(enable_backward=False)
def project_track_kernel(
    positions: wp.array(dtype=wp.vec2f),
    seeds: wp.array(dtype=wp.int32),
    track: TrackData,
    segments: wp.array(dtype=wp.int32),
    s: wp.array(dtype=wp.float32),
    ey: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    result = project_track(positions[env_id], seeds[env_id], track)
    segments[env_id] = result.segment
    s[env_id] = result.s
    ey[env_id] = result.ey
    boundary_distance[env_id] = result.boundary_distance


@wp.struct
class ObsParams:
    future_horizon_seconds: wp.float32
    future_minimum_lookahead: wp.float32
    clip: wp.float32
    opponent_ahead: wp.float32
    opponent_behind: wp.float32
    contact_margin: wp.float32


@wp.struct
class SensorParams:
    num_beams: wp.int32
    angle_min: wp.float32
    angle_increment: wp.float32
    range_min: wp.float32
    range_max: wp.float32
    reliable_range: wp.float32
    lidar_offset_x: wp.float32
    lidar_offset_y: wp.float32
    lidar_offset_yaw: wp.float32
    max_march_steps: wp.int32


@wp.struct
class CorridorDistanceField:
    distance: wp.array(dtype=wp.float32)
    width: wp.int32
    height: wp.int32
    origin: wp.vec2f
    resolution: wp.float32


@wp.struct
class RewardParams:
    progress_forward: wp.float32
    progress_backward: wp.float32
    progress_max_lateral: wp.float32
    wall_contact_coefficient: wp.float32
    control_dt: wp.float32
    boundary_contact_coefficient: wp.float32
    oob_impact_coefficient: wp.float32
    steering_change: wp.float32
    steering_history: wp.float32
    max_steer: wp.float32
    steering_c_s: wp.float32
    steering_c_o: wp.float32
    steering_c_d: wp.float32
    passing: wp.float32
    passing_ahead: wp.float32
    passing_behind: wp.float32
    collision: wp.float32
    rear_end: wp.float32
    global_scale: wp.float32
    rear_end_any_contact: wp.int32
    wall_cost_mode: wp.int32


@wp.struct
class TerminationParams:
    maximum_episode_steps: wp.int32
    maximum_stopped_steps: wp.int32
    minimum_stopped_step: wp.int32
    speed_threshold: wp.float32
    minimum_progress: wp.float32
    maximum_heading_error: wp.float32
    collision_speed: wp.float32
    terminate_on_collision: wp.int32
    recoverable_boundary: wp.int32


@wp.struct
class ResetParams:
    seed: wp.int32
    has_opponent: wp.int32
    spawn_margin: wp.float32
    speed_min: wp.float32
    speed_max: wp.float32
    stationary_probability: wp.float32
    opponent_gap_min: wp.float32
    opponent_gap_max: wp.float32
    opponent_behind_probability: wp.float32
    opponent_speed_min: wp.float32
    opponent_speed_max: wp.float32
    mass_min: wp.float32
    mass_max: wp.float32
    friction_min: wp.float32
    friction_max: wp.float32
    drive_scale_min: wp.float32
    drive_scale_max: wp.float32
    steer_bias_min: wp.float32
    steer_bias_max: wp.float32
    action_latency_min: wp.int32
    action_latency_max: wp.int32
    observation_latency_min: wp.int32
    observation_latency_max: wp.int32
    observation_noise_min: wp.float32
    observation_noise_max: wp.float32
    spawn_yaw_jitter: wp.float32
    opponent_lateral_spawn: wp.int32
    lidar_range_noise_std_min: wp.float32
    lidar_range_noise_std_max: wp.float32
    lidar_far_dropout_prob_min: wp.float32
    lidar_far_dropout_prob_max: wp.float32
    lidar_dropout_prob_min: wp.float32
    lidar_dropout_prob_max: wp.float32
    lidar_angle_bias_min: wp.float32
    lidar_angle_bias_max: wp.float32
    lidar_extrinsic_xy_min: wp.float32
    lidar_extrinsic_xy_max: wp.float32
    lidar_extrinsic_yaw_min: wp.float32
    lidar_extrinsic_yaw_max: wp.float32
    imu_accel_bias_min: wp.float32
    imu_accel_bias_max: wp.float32
    imu_gyro_bias_min: wp.float32
    imu_gyro_bias_max: wp.float32
    imu_accel_noise_std_min: wp.float32
    imu_accel_noise_std_max: wp.float32
    imu_gyro_noise_std_min: wp.float32
    imu_gyro_noise_std_max: wp.float32
    imu_axis_misalign_min: wp.float32
    imu_axis_misalign_max: wp.float32
    vesc_speed_bias_min: wp.float32
    vesc_speed_bias_max: wp.float32
    vesc_current_bias_min: wp.float32
    vesc_current_bias_max: wp.float32
    vesc_speed_noise_std_min: wp.float32
    vesc_speed_noise_std_max: wp.float32
    vesc_current_noise_std_min: wp.float32
    vesc_current_noise_std_max: wp.float32


@wp.struct
class OpponentParams:
    strategy: wp.int32
    policy_probability: wp.float32
    kp_lateral: wp.float32
    kp_heading: wp.float32
    kp_speed: wp.float32
    target_speed_min: wp.float32
    target_speed_max: wp.float32
    lateral_offset_max: wp.float32
    policy_speed_cap_prob: wp.float32
    policy_speed_cap_lo: wp.float32
    policy_speed_cap_hi: wp.float32
    car_length: wp.float32
    car_width: wp.float32
    restitution: wp.float32


@wp.struct
class EnvBuffers:
    reward: wp.array(dtype=wp.float32)
    reward_progress: wp.array(dtype=wp.float32)
    reward_wall_contact: wp.array(dtype=wp.float32)
    reward_boundary_contact: wp.array(dtype=wp.float32)
    reward_oob_impact: wp.array(dtype=wp.float32)
    reward_steering_change: wp.array(dtype=wp.float32)
    reward_steering_history: wp.array(dtype=wp.float32)
    reward_passing: wp.array(dtype=wp.float32)
    reward_collision: wp.array(dtype=wp.float32)
    reward_rear_end: wp.array(dtype=wp.float32)
    done: wp.array(dtype=wp.bool)
    done_flags: wp.array(dtype=wp.int32)
    term_timeout: wp.array(dtype=wp.bool)
    term_oob: wp.array(dtype=wp.bool)
    term_stopped: wp.array(dtype=wp.bool)
    term_invalid: wp.array(dtype=wp.bool)
    term_collision: wp.array(dtype=wp.bool)
    episode_step: wp.array(dtype=wp.int32)
    completed_episode_steps: wp.array(dtype=wp.int32)
    episode_id: wp.array(dtype=wp.int32)
    lap_count: wp.array(dtype=wp.int32)
    lap_cross: wp.array(dtype=wp.float32)
    ego_segment: wp.array(dtype=wp.int32)
    opponent_segment: wp.array(dtype=wp.int32)
    prev_s: wp.array(dtype=wp.float32)
    prev_opponent_s: wp.array(dtype=wp.float32)
    prev_wall_contact: wp.array(dtype=wp.int32)
    prev_opponent_ahead: wp.array(dtype=wp.int32)
    prev_opponent_in_window: wp.array(dtype=wp.int32)
    stopped_streak: wp.array(dtype=wp.int32)
    last_action: wp.array(dtype=wp.vec2f)
    opponent_last_action: wp.array(dtype=wp.vec2f)
    prev_steer_delta: wp.array(dtype=wp.float32)
    current_action: wp.array(dtype=wp.vec2f)
    current_opponent_action: wp.array(dtype=wp.vec2f)
    action_history: wp.array2d(dtype=wp.vec2f)
    executed_longitudinal_history: wp.array2d(dtype=wp.float32)
    opponent_executed_longitudinal_history: wp.array2d(dtype=wp.float32)
    executed_steer_history: wp.array2d(dtype=wp.float32)
    opponent_executed_steer_history: wp.array2d(dtype=wp.float32)
    action_head: wp.array(dtype=wp.int32)
    action_latency: wp.array(dtype=wp.int32)
    observation_latency: wp.array(dtype=wp.int32)
    observation_noise: wp.array(dtype=wp.float32)
    lidar_extrinsic_x: wp.array(dtype=wp.float32)
    lidar_extrinsic_y: wp.array(dtype=wp.float32)
    lidar_extrinsic_yaw: wp.array(dtype=wp.float32)
    lidar_angle_bias: wp.array(dtype=wp.float32)
    lidar_range_noise_std: wp.array(dtype=wp.float32)
    lidar_dropout_prob: wp.array(dtype=wp.float32)
    lidar_far_dropout_prob: wp.array(dtype=wp.float32)
    imu_accel_bias_x: wp.array(dtype=wp.float32)
    imu_accel_bias_y: wp.array(dtype=wp.float32)
    imu_accel_bias_z: wp.array(dtype=wp.float32)
    imu_gyro_bias_x: wp.array(dtype=wp.float32)
    imu_gyro_bias_y: wp.array(dtype=wp.float32)
    imu_gyro_bias_z: wp.array(dtype=wp.float32)
    imu_accel_noise_std: wp.array(dtype=wp.float32)
    imu_gyro_noise_std: wp.array(dtype=wp.float32)
    imu_axis_misalign: wp.array(dtype=wp.float32)
    vesc_speed_bias: wp.array(dtype=wp.float32)
    vesc_current_bias: wp.array(dtype=wp.float32)
    vesc_speed_noise_std: wp.array(dtype=wp.float32)
    vesc_current_noise_std: wp.array(dtype=wp.float32)
    opponent_mode: wp.array(dtype=wp.int32)
    contact: wp.array(dtype=wp.int32)
    valid: wp.array(dtype=wp.int32)
    opponent_target_speed: wp.array(dtype=wp.float32)
    opponent_lateral_offset: wp.array(dtype=wp.float32)
    opponent_speed_cap: wp.array(dtype=wp.float32)
    terminal_x: wp.array(dtype=wp.float32)
    terminal_y: wp.array(dtype=wp.float32)
    terminal_s: wp.array(dtype=wp.float32)
    metric_progress: wp.array(dtype=wp.float32)
    metric_wall: wp.array(dtype=wp.float32)
    metric_boundary: wp.array(dtype=wp.float32)
    metric_lateral: wp.array(dtype=wp.float32)
    metric_speed: wp.array(dtype=wp.float32)
    metric_opponent_speed: wp.array(dtype=wp.float32)
    metric_nonfinite: wp.array(dtype=wp.float32)
    contact_closing_speed: wp.array(dtype=wp.float32)


@wp.struct
class PhysicsBuffers:
    opponent_segment: wp.array(dtype=wp.int32)
    episode_step: wp.array(dtype=wp.int32)
    action_history: wp.array2d(dtype=wp.vec2f)
    executed_longitudinal_history: wp.array2d(dtype=wp.float32)
    opponent_executed_longitudinal_history: wp.array2d(dtype=wp.float32)
    executed_steer_history: wp.array2d(dtype=wp.float32)
    opponent_executed_steer_history: wp.array2d(dtype=wp.float32)
    action_head: wp.array(dtype=wp.int32)
    action_latency: wp.array(dtype=wp.int32)
    opponent_mode: wp.array(dtype=wp.int32)
    opponent_target_speed: wp.array(dtype=wp.float32)
    opponent_lateral_offset: wp.array(dtype=wp.float32)
    opponent_speed_cap: wp.array(dtype=wp.float32)
    current_action: wp.array(dtype=wp.vec2f)
    current_opponent_action: wp.array(dtype=wp.vec2f)
    contact: wp.array(dtype=wp.int32)
    contact_closing_speed: wp.array(dtype=wp.float32)
    valid: wp.array(dtype=wp.int32)


@wp.struct
class ContactResult:
    contact: wp.int32
    normal: wp.vec2f
    depth: wp.float32
    closing_speed: wp.float32


@wp.struct
class RewardResult:
    total: wp.float32
    progress: wp.float32
    wall_contact: wp.float32
    boundary_contact: wp.float32
    oob_impact: wp.float32
    steering_change: wp.float32
    steering_history: wp.float32
    passing: wp.float32
    collision: wp.float32
    rear_end: wp.float32
    done: wp.bool
    done_flags: wp.int32


@wp.func
def wrap_angle(angle: wp.float32) -> wp.float32:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def wrapped_delta(
    current: wp.float32,
    previous: wp.float32,
    length: wp.float32,
) -> wp.float32:
    delta = current - previous
    if delta > 0.5 * length:
        delta = delta - length
    elif delta < -0.5 * length:
        delta = delta + length
    return delta


@wp.func
def body_velocity_world(vehicle: VehicleLocal) -> wp.vec2f:
    cosine = wp.cos(vehicle.yaw)
    sine = wp.sin(vehicle.yaw)
    return wp.vec2f(
        cosine * vehicle.vx - sine * vehicle.vy,
        sine * vehicle.vx + cosine * vehicle.vy,
    )


@wp.func
def body_accel_world(vehicle: VehicleLocal) -> wp.vec2f:
    cosine = wp.cos(vehicle.yaw)
    sine = wp.sin(vehicle.yaw)
    return wp.vec2f(
        cosine * vehicle.ax - sine * vehicle.ay,
        sine * vehicle.ax + cosine * vehicle.ay,
    )


@wp.func
def axis_for_box(yaw: wp.float32, axis_index: wp.int32) -> wp.vec2f:
    if axis_index == 0:
        return wp.vec2f(wp.cos(yaw), wp.sin(yaw))
    return wp.vec2f(-wp.sin(yaw), wp.cos(yaw))


@wp.func
def resolve_pair_contact(
    ego: VehicleLocal,
    opponent: VehicleLocal,
    params: OpponentParams,
) -> ContactResult:
    result = ContactResult()
    delta = wp.vec2f(ego.x - opponent.x, ego.y - opponent.y)
    ego_x = axis_for_box(ego.yaw, 0)
    ego_y = axis_for_box(ego.yaw, 1)
    opponent_x = axis_for_box(opponent.yaw, 0)
    opponent_y = axis_for_box(opponent.yaw, 1)
    half_length = 0.5 * params.car_length
    half_width = 0.5 * params.car_width
    best_depth = wp.float32(1.0e30)
    best_normal = wp.vec2f(0.0)
    separated = wp.int32(0)
    for axis_index in range(4):
        axis = ego_x
        if axis_index == 1:
            axis = ego_y
        elif axis_index == 2:
            axis = opponent_x
        elif axis_index == 3:
            axis = opponent_y
        projection = wp.dot(delta, axis)
        ego_radius = (
            half_length * wp.abs(wp.dot(ego_x, axis))
            + half_width * wp.abs(wp.dot(ego_y, axis))
        )
        opponent_radius = (
            half_length * wp.abs(wp.dot(opponent_x, axis))
            + half_width * wp.abs(wp.dot(opponent_y, axis))
        )
        depth = ego_radius + opponent_radius - wp.abs(projection)
        if depth <= 0.0:
            separated = 1
        if depth < best_depth:
            direction = wp.float32(1.0)
            if projection < 0.0:
                direction = -1.0
            best_depth = depth
            best_normal = direction * axis
    if separated == 0:
        result.contact = 1
        result.normal = best_normal
        result.depth = wp.max(best_depth, 0.0)
        relative_velocity = body_velocity_world(ego) - body_velocity_world(opponent)
        result.closing_speed = wp.max(-wp.dot(relative_velocity, best_normal), 0.0)
    return result


@wp.func
def apply_contact(
    ego: VehicleLocal,
    opponent: VehicleLocal,
    contact: ContactResult,
    restitution: wp.float32,
) -> tuple[VehicleLocal, VehicleLocal]:
    if contact.contact != 0:
        correction = 0.5 * contact.depth * contact.normal
        ego.x = ego.x + correction[0]
        ego.y = ego.y + correction[1]
        opponent.x = opponent.x - correction[0]
        opponent.y = opponent.y - correction[1]
        ego_world = body_velocity_world(ego)
        opponent_world = body_velocity_world(opponent)
        relative = wp.dot(ego_world - opponent_world, contact.normal)
        closing = wp.min(relative, 0.0)
        impulse = -0.5 * (1.0 + restitution) * closing
        ego_world = ego_world + impulse * contact.normal
        opponent_world = opponent_world - impulse * contact.normal
        ego_cosine = wp.cos(ego.yaw)
        ego_sine = wp.sin(ego.yaw)
        opponent_cosine = wp.cos(opponent.yaw)
        opponent_sine = wp.sin(opponent.yaw)
        ego.vx = ego_cosine * ego_world[0] + ego_sine * ego_world[1]
        ego.vy = -ego_sine * ego_world[0] + ego_cosine * ego_world[1]
        opponent.vx = (
            opponent_cosine * opponent_world[0]
            + opponent_sine * opponent_world[1]
        )
        opponent.vy = (
            -opponent_sine * opponent_world[0]
            + opponent_cosine * opponent_world[1]
        )
    return ego, opponent


@wp.func
def resolve_and_apply_pair_contact(
    ego: VehicleLocal,
    opponent: VehicleLocal,
    opponent_params: OpponentParams,
) -> tuple[VehicleLocal, VehicleLocal, ContactResult]:
    contact = resolve_pair_contact(ego, opponent, opponent_params)
    if contact.contact != 0:
        ego, opponent = apply_contact(
            ego,
            opponent,
            contact,
            opponent_params.restitution,
        )
    return ego, opponent, contact


@wp.func
def reset_random(seed: wp.int32, env_id: wp.int32, episode_id: wp.int32):
    mixed = seed ^ (env_id * 73244475) ^ (episode_id * 295075153)
    return wp.rand_init(mixed & 2147483647)


@wp.func
def random_range(state: wp.uint32, minimum: wp.float32, maximum: wp.float32):
    return minimum + (maximum - minimum) * wp.randf(state)


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
def spawn_on_track(
    segment: wp.int32,
    alpha: wp.float32,
    lateral: wp.float32,
    speed: wp.float32,
    yaw_offset: wp.float32,
    track: TrackData,
    sim: SimParams,
) -> VehicleLocal:
    following = wrap_segment(segment + 1, track.count)
    center = track.point[segment] + alpha * (
        track.point[following] - track.point[segment]
    )
    normal = track.normal[segment]
    vehicle = VehicleLocal()
    vehicle.x = center[0] + lateral * normal[0]
    vehicle.y = center[1] + lateral * normal[1]
    vehicle.yaw = wp.atan2(
        track.tangent[segment][1], track.tangent[segment][0]
    ) + yaw_offset
    vehicle.vx = speed
    vehicle = clear_vehicle(vehicle)
    vehicle.vx = speed
    wheel_speed = speed / sim.wheel_radius
    vehicle.omega = wp.vec4f(wheel_speed)
    return vehicle


@wp.func
def reset_pair(
    env_id: wp.int32,
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    sim: SimParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
) -> tuple[VehicleLocal, VehicleLocal, FrenetState, FrenetState]:
    next_episode = env.episode_id[env_id] + 1
    random = reset_random(reset.seed, env_id, next_episode)
    segment = wp.min(
        wp.int32(wp.randf(random) * wp.float32(track.count)),
        track.count - 1,
    )
    alpha = wp.randf(random)
    width_left = track.width_left[segment] - reset.spawn_margin
    width_right = track.width_right[segment] - reset.spawn_margin
    lateral = random_range(random, -wp.max(width_right, 0.0), wp.max(width_left, 0.0))
    speed = random_range(random, reset.speed_min, reset.speed_max)
    if (
        reset.stationary_probability > 0.0
        and wp.randf(random) < reset.stationary_probability
    ):
        speed = 0.0
    ego_yaw_jitter = random_range(
        random, -reset.spawn_yaw_jitter, reset.spawn_yaw_jitter
    )
    ego = spawn_on_track(segment, alpha, lateral, speed, ego_yaw_jitter, track, sim)

    gap = random_range(random, reset.opponent_gap_min, reset.opponent_gap_max)
    if wp.randf(random) < reset.opponent_behind_probability:
        gap = -gap
    opponent_s = (
        track.cumulative_length[segment]
        + alpha * track.segment_length[segment]
        + gap
    )
    while opponent_s < 0.0:
        opponent_s = opponent_s + track.length
    while opponent_s >= track.length:
        opponent_s = opponent_s - track.length
    opponent_segment = segment
    for _ in range(MAX_FORWARD_SEGMENTS):
        start_s = track.cumulative_length[opponent_segment]
        end_s = start_s + track.segment_length[opponent_segment]
        if opponent_s >= start_s and opponent_s <= end_s:
            break
        opponent_segment = wrap_segment(opponent_segment + 1, track.count)
    opponent_alpha = (
        opponent_s - track.cumulative_length[opponent_segment]
    ) / wp.max(track.segment_length[opponent_segment], 1.0e-8)
    opponent_speed = random_range(
        random, reset.opponent_speed_min, reset.opponent_speed_max
    )
    opponent_lateral = wp.float32(0.0)
    if reset.opponent_lateral_spawn != 0:
        opp_half = wp.min(
            track.width_left[opponent_segment],
            track.width_right[opponent_segment],
        ) - reset.spawn_margin
        opp_half = wp.max(opp_half, 0.0)
        opponent_lateral = random_range(random, -opp_half, opp_half)
    opponent_yaw_jitter = random_range(
        random, -reset.spawn_yaw_jitter, reset.spawn_yaw_jitter
    )
    opponent = spawn_on_track(
        opponent_segment,
        opponent_alpha,
        opponent_lateral,
        opponent_speed,
        opponent_yaw_jitter,
        track,
        sim,
    )

    mass = random_range(random, reset.mass_min, reset.mass_max)
    friction = random_range(random, reset.friction_min, reset.friction_max)
    drive_scale = random_range(
        random, reset.drive_scale_min, reset.drive_scale_max
    )
    steer_bias = random_range(
        random, reset.steer_bias_min, reset.steer_bias_max
    )
    ego_buffers.mass[env_id] = mass
    ego_buffers.mu[env_id] = friction
    ego_buffers.drive_scale[env_id] = drive_scale
    ego_buffers.steer_bias[env_id] = steer_bias
    opponent_buffers.mass[env_id] = mass
    opponent_buffers.mu[env_id] = friction
    opponent_buffers.drive_scale[env_id] = drive_scale
    opponent_buffers.steer_bias[env_id] = steer_bias

    action_span = reset.action_latency_max - reset.action_latency_min + 1
    observation_span = (
        reset.observation_latency_max - reset.observation_latency_min + 1
    )
    env.action_latency[env_id] = reset.action_latency_min + wp.int32(
        wp.randf(random) * wp.float32(action_span)
    )
    env.observation_latency[env_id] = (
        reset.observation_latency_min
        + wp.int32(wp.randf(random) * wp.float32(observation_span))
    )
    env.observation_noise[env_id] = random_range(
        random, reset.observation_noise_min, reset.observation_noise_max
    )
    env.opponent_target_speed[env_id] = random_range(
        random,
        opponent_params.target_speed_min,
        opponent_params.target_speed_max,
    )
    env.opponent_lateral_offset[env_id] = random_range(
        random,
        -opponent_params.lateral_offset_max,
        opponent_params.lateral_offset_max,
    )
    if opponent_params.strategy == 2:
        env.opponent_mode[env_id] = wp.int32(
            wp.randf(random) < opponent_params.policy_probability
        )
    else:
        env.opponent_mode[env_id] = opponent_params.strategy

    env.opponent_speed_cap[env_id] = 1.0e30
    if (
        opponent_params.strategy == 2
        and env.opponent_mode[env_id] != 0
        and wp.randf(random) < opponent_params.policy_speed_cap_prob
    ):
        u = wp.randf(random)
        span = opponent_params.policy_speed_cap_hi - opponent_params.policy_speed_cap_lo
        env.opponent_speed_cap[env_id] = (
            opponent_params.policy_speed_cap_hi - span * u * u
        )

    env.lidar_extrinsic_x[env_id] = random_range(
        random, reset.lidar_extrinsic_xy_min, reset.lidar_extrinsic_xy_max
    )
    env.lidar_extrinsic_y[env_id] = random_range(
        random, reset.lidar_extrinsic_xy_min, reset.lidar_extrinsic_xy_max
    )
    env.lidar_extrinsic_yaw[env_id] = random_range(
        random, reset.lidar_extrinsic_yaw_min, reset.lidar_extrinsic_yaw_max
    )
    env.lidar_angle_bias[env_id] = random_range(
        random, reset.lidar_angle_bias_min, reset.lidar_angle_bias_max
    )
    env.lidar_range_noise_std[env_id] = random_range(
        random, reset.lidar_range_noise_std_min, reset.lidar_range_noise_std_max
    )
    env.lidar_dropout_prob[env_id] = random_range(
        random, reset.lidar_dropout_prob_min, reset.lidar_dropout_prob_max
    )
    env.lidar_far_dropout_prob[env_id] = random_range(
        random, reset.lidar_far_dropout_prob_min, reset.lidar_far_dropout_prob_max
    )
    env.imu_accel_bias_x[env_id] = random_range(
        random, reset.imu_accel_bias_min, reset.imu_accel_bias_max
    )
    env.imu_accel_bias_y[env_id] = random_range(
        random, reset.imu_accel_bias_min, reset.imu_accel_bias_max
    )
    env.imu_accel_bias_z[env_id] = random_range(
        random, reset.imu_accel_bias_min, reset.imu_accel_bias_max
    )
    env.imu_gyro_bias_x[env_id] = random_range(
        random, reset.imu_gyro_bias_min, reset.imu_gyro_bias_max
    )
    env.imu_gyro_bias_y[env_id] = random_range(
        random, reset.imu_gyro_bias_min, reset.imu_gyro_bias_max
    )
    env.imu_gyro_bias_z[env_id] = random_range(
        random, reset.imu_gyro_bias_min, reset.imu_gyro_bias_max
    )
    env.imu_accel_noise_std[env_id] = random_range(
        random, reset.imu_accel_noise_std_min, reset.imu_accel_noise_std_max
    )
    env.imu_gyro_noise_std[env_id] = random_range(
        random, reset.imu_gyro_noise_std_min, reset.imu_gyro_noise_std_max
    )
    env.imu_axis_misalign[env_id] = random_range(
        random, reset.imu_axis_misalign_min, reset.imu_axis_misalign_max
    )
    env.vesc_speed_bias[env_id] = random_range(
        random, reset.vesc_speed_bias_min, reset.vesc_speed_bias_max
    )
    env.vesc_current_bias[env_id] = random_range(
        random, reset.vesc_current_bias_min, reset.vesc_current_bias_max
    )
    env.vesc_speed_noise_std[env_id] = random_range(
        random, reset.vesc_speed_noise_std_min, reset.vesc_speed_noise_std_max
    )
    env.vesc_current_noise_std[env_id] = random_range(
        random, reset.vesc_current_noise_std_min, reset.vesc_current_noise_std_max
    )
    env.episode_id[env_id] = next_episode
    env.episode_step[env_id] = 0
    env.lap_count[env_id] = 0
    env.lap_cross[env_id] = 0.0
    env.stopped_streak[env_id] = 0
    env.prev_wall_contact[env_id] = 0
    env.prev_opponent_ahead[env_id] = 1
    env.prev_opponent_in_window[env_id] = 0
    env.last_action[env_id] = wp.vec2f(0.0)
    env.opponent_last_action[env_id] = wp.vec2f(0.0)
    env.prev_steer_delta[env_id] = 0.0
    env.current_action[env_id] = wp.vec2f(0.0)
    env.current_opponent_action[env_id] = wp.vec2f(0.0)
    env.contact[env_id] = 0
    env.contact_closing_speed[env_id] = 0.0
    env.valid[env_id] = 1
    env.action_head[env_id] = 0
    for history_index in range(ACTION_HISTORY):
        env.action_history[env_id, history_index] = wp.vec2f(0.0)
    for longitudinal_index in range(2):
        env.executed_longitudinal_history[env_id, longitudinal_index] = 0.0
        env.opponent_executed_longitudinal_history[env_id, longitudinal_index] = 0.0
    for steer_index in range(STEER_HISTORY):
        env.executed_steer_history[env_id, steer_index] = 0.0
        env.opponent_executed_steer_history[env_id, steer_index] = 0.0

    ego_frenet = project_track(wp.vec2f(ego.x, ego.y), segment, track)
    opponent_frenet = project_track(
        wp.vec2f(opponent.x, opponent.y), opponent_segment, track
    )
    env.ego_segment[env_id] = ego_frenet.segment
    env.opponent_segment[env_id] = opponent_frenet.segment
    env.prev_s[env_id] = ego_frenet.s
    env.prev_opponent_s[env_id] = opponent_frenet.s
    env.metric_progress[env_id] = 0.0
    env.metric_wall[env_id] = 0.0
    env.metric_boundary[env_id] = ego_frenet.boundary_distance
    env.metric_lateral[env_id] = ego_frenet.ey
    env.metric_speed[env_id] = speed
    env.metric_opponent_speed[env_id] = opponent_speed
    env.metric_nonfinite[env_id] = 0.0
    return ego, opponent, ego_frenet, opponent_frenet


@wp.func
def delayed_action(
    env_id: wp.int32,
    action: wp.vec2f,
    env: EnvBuffers,
) -> wp.vec2f:
    head = env.action_head[env_id]
    env.action_history[env_id, head] = action
    latency = wp.clamp(env.action_latency[env_id], 0, ACTION_HISTORY - 1)
    selected = head - latency
    if selected < 0:
        selected = selected + ACTION_HISTORY
    env.action_head[env_id] = (head + 1) % ACTION_HISTORY
    return env.action_history[env_id, selected]


@wp.func
def delayed_physics_action(
    env_id: wp.int32,
    action: wp.vec2f,
    buffers: PhysicsBuffers,
) -> wp.vec2f:
    head = buffers.action_head[env_id]
    buffers.action_history[env_id, head] = action
    latency = wp.clamp(buffers.action_latency[env_id], 0, ACTION_HISTORY - 1)
    selected = head - latency
    if selected < 0:
        selected = selected + ACTION_HISTORY
    buffers.action_head[env_id] = (head + 1) % ACTION_HISTORY
    return buffers.action_history[env_id, selected]


@wp.func
def push_executed_steer(
    history: wp.array2d(dtype=wp.float32),
    env_id: wp.int32,
    steer: wp.float32,
):
    history[env_id, 3] = history[env_id, 2]
    history[env_id, 2] = history[env_id, 1]
    history[env_id, 1] = history[env_id, 0]
    history[env_id, 0] = steer


@wp.func
def push_executed_longitudinal(
    history: wp.array2d(dtype=wp.float32),
    env_id: wp.int32,
    longitudinal: wp.float32,
):
    history[env_id, 1] = history[env_id, 0]
    history[env_id, 0] = longitudinal


@wp.func
def scripted_opponent_action(
    env_id: wp.int32,
    opponent: VehicleLocal,
    frenet: FrenetState,
    env: EnvBuffers,
    sim: SimParams,
    params: OpponentParams,
) -> wp.vec2f:
    heading = wp.atan2(frenet.tangent[1], frenet.tangent[0])
    heading_error = wrap_angle(opponent.yaw - heading)
    steering_target = -(
        params.kp_lateral
        * (frenet.ey - env.opponent_lateral_offset[env_id])
        + params.kp_heading * heading_error
    )
    steering = steering_target / wp.max(sim.max_steer, 1.0e-6)
    steering = (steering_target - opponent.steer) / wp.max(
        sim.steering_delta_max, 1.0e-6
    )
    track_speed = wp.dot(body_velocity_world(opponent), frenet.tangent)
    throttle = params.kp_speed * (
        env.opponent_target_speed[env_id] - track_speed
    )
    return wp.vec2f(
        wp.clamp(throttle, -1.0, 1.0),
        wp.clamp(steering, -1.0, 1.0),
    )


@wp.func
def scripted_physics_opponent_action(
    env_id: wp.int32,
    opponent: VehicleLocal,
    frenet: FrenetState,
    buffers: PhysicsBuffers,
    sim: SimParams,
    params: OpponentParams,
) -> wp.vec2f:
    heading = wp.atan2(frenet.tangent[1], frenet.tangent[0])
    heading_error = wrap_angle(opponent.yaw - heading)
    steering_target = -(
        params.kp_lateral
        * (frenet.ey - buffers.opponent_lateral_offset[env_id])
        + params.kp_heading * heading_error
    )
    steering = steering_target / wp.max(sim.max_steer, 1.0e-6)
    steering = (steering_target - opponent.steer) / wp.max(
        sim.steering_delta_max, 1.0e-6
    )
    track_speed = wp.dot(body_velocity_world(opponent), frenet.tangent)
    throttle = params.kp_speed * (
        buffers.opponent_target_speed[env_id] - track_speed
    )
    return wp.vec2f(
        wp.clamp(throttle, -1.0, 1.0),
        wp.clamp(steering, -1.0, 1.0),
    )


@wp.func
def compute_reward_and_done(
    env_id: wp.int32,
    ego: VehicleLocal,
    opponent: VehicleLocal,
    ego_frenet: FrenetState,
    opponent_frenet: FrenetState,
    action: wp.vec2f,
    contact: ContactResult,
    env: EnvBuffers,
    track: TrackData,
    reward: RewardParams,
    termination: TerminationParams,
    opponent_enabled: wp.int32,
    car_length: wp.float32,
    car_width: wp.float32,
) -> RewardResult:
    out = RewardResult()
    prev_s = env.prev_s[env_id]
    current_s = ego_frenet.s
    raw_delta_s = wrapped_delta(current_s, prev_s, track.length)
    delta_s = wp.clamp(
        raw_delta_s,
        -0.1 * track.length,
        0.1 * track.length,
    )
    opponent_delta_s = wp.clamp(
        wrapped_delta(
            opponent_frenet.s, env.prev_opponent_s[env_id], track.length
        ),
        -0.1 * track.length,
        0.1 * track.length,
    )
    gap = wrapped_delta(
        opponent_frenet.s, ego_frenet.s, track.length
    )
    heading = wp.atan2(ego_frenet.tangent[1], ego_frenet.tangent[0])
    heading_error = wrap_angle(ego.yaw - heading)
    footprint = (
        0.5 * car_length * wp.abs(wp.sin(heading_error))
        + 0.5 * car_width * wp.abs(wp.cos(heading_error))
    )
    wall_contact = (
        ego_frenet.ey + footprint >= ego_frenet.width_left
        or ego_frenet.ey - footprint <= -ego_frenet.width_right
    )
    progress_ds = delta_s
    if wall_contact:
        progress_ds = 0.0

    out.progress = (
        reward.progress_forward * wp.max(progress_ds, 0.0)
        + reward.progress_backward * wp.min(progress_ds, 0.0)
    )
    speed_squared = ego.vx * ego.vx + ego.vy * ego.vy
    if wall_contact:
        if reward.wall_cost_mode == 1:
            out.wall_contact = (
                -reward.wall_contact_coefficient
                * reward.control_dt
                * wp.sqrt(speed_squared)
            )
        elif reward.wall_cost_mode == 2:
            out.wall_contact = (
                -reward.wall_contact_coefficient
                * reward.control_dt
                * speed_squared
            )
        else:
            out.wall_contact = (
                -reward.wall_contact_coefficient
                * wp.sqrt(speed_squared)
            )
    first_wall_contact = wall_contact and env.prev_wall_contact[env_id] == 0
    if first_wall_contact:
        out.boundary_contact = -reward.boundary_contact_coefficient
    vel_world = body_velocity_world(ego)
    theta_t = env.executed_steer_history[env_id, 0]
    theta_prev = env.executed_steer_history[env_id, 1]
    delta_t = theta_t - theta_prev
    prev_delta = env.prev_steer_delta[env_id]
    out.steering_change = -reward.steering_change * wp.abs(delta_t)
    delta_sum = wp.abs(delta_t) + wp.abs(prev_delta)
    m_t = wp.float32(0.0)
    if (
        wp.abs(delta_t) > reward.steering_c_d
        and wp.abs(prev_delta) > reward.steering_c_d
        and delta_t * prev_delta < 0.0
    ):
        m_t = wp.float32(1.0)
    out.steering_history = (
        -reward.steering_history
        * m_t
        * (
            1.0
            + wp.exp(
                -reward.steering_c_s * (delta_sum - reward.steering_c_o)
            )
        )
    )
    env.prev_steer_delta[env_id] = delta_t
    if opponent_enabled != 0:
        in_window = gap <= reward.passing_ahead and gap >= -reward.passing_behind
        if in_window or env.prev_opponent_in_window[env_id] != 0:
            out.passing = reward.passing * (delta_s - opponent_delta_s)
        env.prev_opponent_in_window[env_id] = wp.int32(in_window)
        out.collision = -reward.collision * wp.float32(contact.contact)
        if contact.contact != 0 and (
            reward.rear_end_any_contact != 0 or gap > 0.0
        ):
            relative_velocity = vel_world - body_velocity_world(opponent)
            out.rear_end = -reward.rear_end * wp.dot(
                relative_velocity, relative_velocity
            )
        env.prev_opponent_ahead[env_id] = wp.int32(gap > 0.0)

    stopped_now = (
        speed_squared
        < termination.speed_threshold * termination.speed_threshold
        and wp.abs(raw_delta_s) < termination.minimum_progress
    )
    if stopped_now and env.episode_step[env_id] > termination.minimum_stopped_step:
        env.stopped_streak[env_id] = env.stopped_streak[env_id] + 1
    else:
        env.stopped_streak[env_id] = 0

    timeout = env.episode_step[env_id] >= termination.maximum_episode_steps
    if termination.recoverable_boundary != 0:
        full_out_left = ego_frenet.ey - footprint > ego_frenet.width_left
        full_out_right = ego_frenet.ey + footprint < -ego_frenet.width_right
        oob_done = full_out_left or full_out_right
    else:
        oob_done = wall_contact
    if oob_done:
        out.oob_impact = -reward.oob_impact_coefficient * speed_squared
    stopped = (
        env.stopped_streak[env_id] >= termination.maximum_stopped_steps
    )
    invalid = (
        not vehicle_is_finite(ego)
        or wp.abs(heading_error) > termination.maximum_heading_error
    )
    collision_done = (
        opponent_enabled != 0
        and termination.terminate_on_collision != 0
        and contact.contact != 0
        and contact.closing_speed > termination.collision_speed
    )

    out.total = reward.global_scale * (
        out.progress
        + out.wall_contact
        + out.boundary_contact
        + out.oob_impact
        + out.steering_change
        + out.steering_history
        + out.passing
        + out.collision
        + out.rear_end
    )

    out.done = (
        timeout
        or oob_done
        or stopped
        or invalid
        or collision_done
    )
    if timeout:
        out.done_flags = out.done_flags | 1
    if oob_done:
        out.done_flags = out.done_flags | 2
    if stopped:
        out.done_flags = out.done_flags | 4
    if invalid:
        out.done_flags = out.done_flags | 8
    if collision_done:
        out.done_flags = out.done_flags | 16

    env.lap_cross[env_id] = 0.0
    if (
        prev_s > 0.9 * track.length
        and current_s < 0.1 * track.length
        and raw_delta_s > 0.0
    ):
        env.lap_count[env_id] = env.lap_count[env_id] + 1
        env.lap_cross[env_id] = 1.0
    env.prev_s[env_id] = current_s
    env.prev_opponent_s[env_id] = opponent_frenet.s
    env.prev_wall_contact[env_id] = wp.int32(wall_contact)
    env.metric_progress[env_id] = progress_ds
    env.metric_wall[env_id] = wp.float32(wall_contact)
    env.metric_boundary[env_id] = ego_frenet.boundary_distance
    env.metric_lateral[env_id] = ego_frenet.ey
    env.metric_speed[env_id] = wp.sqrt(speed_squared)
    env.metric_opponent_speed[env_id] = wp.dot(
        body_velocity_world(opponent), opponent_frenet.tangent
    )
    env.metric_nonfinite[env_id] = wp.float32(invalid)
    return out


@wp.func
def observation_noise(
    seed: wp.int32,
    env_id: wp.int32,
    episode_id: wp.int32,
    episode_step: wp.int32,
    feature: wp.int32,
    standard_deviation: wp.float32,
) -> wp.float32:
    mixed = (
        seed
        ^ (env_id * 73244475)
        ^ (episode_id * 295075153)
        ^ (episode_step * 104395301)
        ^ (feature * 122949829)
    )
    random = wp.rand_init(mixed & 2147483647)
    return standard_deviation * wp.randn(random)


@wp.func
def write_raw_observation(
    env_id: wp.int32,
    ego: VehicleLocal,
    opponent: VehicleLocal,
    ego_frenet: FrenetState,
    opponent_frenet: FrenetState,
    prior_action: wp.vec2f,
    track: TrackData,
    obs_params: ObsParams,
    reset: ResetParams,
    output: wp.array2d(dtype=wp.float32),
) -> wp.int32:
    for feature in range(OBS_OPPONENT_START, OBS_DIM):
        output[env_id, feature] = 0.0
    output[env_id, 0] = ego.vx
    output[env_id, 1] = ego.vy
    output[env_id, 2] = ego.yaw_rate
    output[env_id, 3] = ego.ax
    output[env_id, 4] = ego.ay
    output[env_id, 5] = prior_action[0]
    output[env_id, 6] = prior_action[1]
    progress_angle = 2.0 * wp.pi * ego_frenet.s / track.length
    output[env_id, 7] = wp.cos(progress_angle)
    output[env_id, 8] = wp.sin(progress_angle)
    track_heading = wp.atan2(
        ego_frenet.tangent[1], ego_frenet.tangent[0]
    )
    output[env_id, 9] = wrap_angle(ego.yaw - track_heading)
    output[env_id, 10] = ego_frenet.ey
    contact_flag = wp.float32(0.0)
    if ego_frenet.boundary_distance < obs_params.contact_margin:
        contact_flag = 1.0
    output[env_id, 11] = contact_flag
    write_future_track(
        env_id,
        wp.vec2f(ego.x, ego.y),
        ego.yaw,
        wp.sqrt(ego.vx * ego.vx + ego.vy * ego.vy),
        ego_frenet,
        track,
        output,
        obs_params.future_horizon_seconds,
        obs_params.future_minimum_lookahead,
    )
    for wheel in range(4):
        output[env_id, OBS_SLIP_RATIO_START + wheel] = ego.slip_ratio[wheel]
        output[env_id, OBS_SLIP_ANGLE_START + wheel] = ego.slip_angle[wheel]
        output[env_id, OBS_LOAD_START + wheel] = ego.load_ratio[wheel]

    visible = wp.int32(0)
    if reset.has_opponent != 0:
        gap = wrapped_delta(
            opponent_frenet.s, ego_frenet.s, track.length
        )
        if gap <= obs_params.opponent_ahead and gap >= -obs_params.opponent_behind:
            visible = 1
            relative_position = world_to_body_point(
                wp.vec2f(opponent.x, opponent.y),
                wp.vec2f(ego.x, ego.y),
                wp.cos(ego.yaw),
                wp.sin(ego.yaw),
            )
            relative_velocity = body_velocity_world(opponent) - body_velocity_world(ego)
            relative_velocity_body = wp.vec2f(
                wp.cos(ego.yaw) * relative_velocity[0]
                + wp.sin(ego.yaw) * relative_velocity[1],
                -wp.sin(ego.yaw) * relative_velocity[0]
                + wp.cos(ego.yaw) * relative_velocity[1],
            )
            relative_accel = body_accel_world(opponent) - body_accel_world(ego)
            relative_accel_body = wp.vec2f(
                wp.cos(ego.yaw) * relative_accel[0]
                + wp.sin(ego.yaw) * relative_accel[1],
                -wp.sin(ego.yaw) * relative_accel[0]
                + wp.cos(ego.yaw) * relative_accel[1],
            )
            output[env_id, OBS_OPPONENT_START] = relative_position[0]
            output[env_id, OBS_OPPONENT_START + 1] = relative_position[1]
            output[env_id, OBS_OPPONENT_START + 2] = relative_velocity_body[0]
            output[env_id, OBS_OPPONENT_START + 3] = relative_velocity_body[1]
            output[env_id, OBS_OPPONENT_START + 4] = relative_accel_body[0]
            output[env_id, OBS_OPPONENT_START + 5] = relative_accel_body[1]
            output[env_id, OBS_OPPONENT_START + 6] = (
                gap / wp.max(0.5 * track.length, 1.0e-6)
            )
            output[env_id, OBS_OPPONENT_START + 7] = opponent_frenet.ey

    if obs_params.clip > 0.0:
        for feature in range(OBS_DIM):
            output[env_id, feature] = wp.clamp(
                output[env_id, feature], -obs_params.clip, obs_params.clip
            )
    return visible


@wp.func
def publish_observation(
    env_id: wp.int32,
    visible: wp.int32,
    env: EnvBuffers,
    reset: ResetParams,
    raw_previous: wp.array2d(dtype=wp.float32),
    raw_current: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    use_previous = env.observation_latency[env_id] > 0
    published_visible = visible
    if use_previous:
        published_visible = wp.int32(0)
        for feature in range(OBS_OPPONENT_START, OBS_DIM):
            if raw_previous[env_id, feature] != 0.0:
                published_visible = 1
    for feature in range(OBS_DIM):
        value = raw_current[env_id, feature]
        if use_previous:
            value = raw_previous[env_id, feature]
        if env.observation_noise[env_id] > 0.0:
            if feature < OBS_OPPONENT_START or published_visible != 0:
                value = value + observation_noise(
                    reset.seed,
                    env_id,
                    env.episode_id[env_id],
                    env.episode_step[env_id],
                    feature,
                    env.observation_noise[env_id],
                )
        if feature >= OBS_OPPONENT_START and published_visible == 0:
            value = 0.0
        output[env_id, feature] = value


@wp.func
def store_reward(
    env_id: wp.int32,
    result: RewardResult,
    env: EnvBuffers,
):
    env.reward[env_id] = result.total
    env.reward_progress[env_id] = result.progress
    env.reward_wall_contact[env_id] = result.wall_contact
    env.reward_boundary_contact[env_id] = result.boundary_contact
    env.reward_oob_impact[env_id] = result.oob_impact
    env.reward_steering_change[env_id] = result.steering_change
    env.reward_steering_history[env_id] = result.steering_history
    env.reward_passing[env_id] = result.passing
    env.reward_collision[env_id] = result.collision
    env.reward_rear_end[env_id] = result.rear_end
    env.done[env_id] = result.done
    env.done_flags[env_id] = result.done_flags
    env.term_timeout[env_id] = (result.done_flags & 1) != 0
    env.term_oob[env_id] = (result.done_flags & 2) != 0
    env.term_stopped[env_id] = (result.done_flags & 4) != 0
    env.term_invalid[env_id] = (result.done_flags & 8) != 0
    env.term_collision[env_id] = (result.done_flags & 16) != 0
    env.completed_episode_steps[env_id] = 0
    if result.done:
        env.completed_episode_steps[env_id] = env.episode_step[env_id]


@wp.kernel(enable_backward=False)
def physics_solo_kernel(
    actions: wp.array(dtype=wp.vec2f),
    ego_buffers: VehicleBuffers,
    env: PhysicsBuffers,
    sim: SimParams,
    substeps: wp.int32,
):
    env_id = wp.tid()
    action = wp.vec2f(
        wp.clamp(actions[env_id][0], -1.0, 1.0),
        wp.clamp(actions[env_id][1], -1.0, 1.0),
    )
    execution_action = delayed_physics_action(env_id, action, env)
    push_executed_longitudinal(
        env.executed_longitudinal_history, env_id, execution_action[0]
    )
    ego = load_vehicle(ego_buffers, env_id)
    ego = apply_command(
        ego, execution_action, ego_buffers.steer_bias[env_id], sim
    )
    executed_steer = execution_action[1]
    executed_steer = ego.steer
    push_executed_steer(
        env.executed_steer_history, env_id, executed_steer
    )
    for _ in range(substeps):
        ego = integrate_vehicle_substep(
            ego,
            ego_buffers.mass[env_id],
            ego_buffers.mu[env_id],
            ego_buffers.drive_scale[env_id],
            sim,
        )
    env.current_action[env_id] = action
    env.current_opponent_action[env_id] = wp.vec2f(0.0)
    env.contact[env_id] = 0
    env.contact_closing_speed[env_id] = 0.0
    valid = wp.int32(1)
    if not vehicle_is_finite(ego):
        valid = 0
    env.valid[env_id] = valid
    env.episode_step[env_id] = env.episode_step[env_id] + 1
    store_vehicle(ego_buffers, env_id, ego)


@wp.kernel(enable_backward=False)
def physics_stage_kernel(
    actions: wp.array(dtype=wp.vec2f),
    policy_opponent_actions: wp.array(dtype=wp.vec2f),
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: PhysicsBuffers,
    track: TrackData,
    sim: SimParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
    num_envs: wp.int32,
    substeps: wp.int32,
):
    thread_id = wp.tid()
    env_id = thread_id % num_envs
    if thread_id < num_envs:
        action = wp.vec2f(
            wp.clamp(actions[env_id][0], -1.0, 1.0),
            wp.clamp(actions[env_id][1], -1.0, 1.0),
        )
        execution_action = delayed_physics_action(env_id, action, env)
        push_executed_longitudinal(
            env.executed_longitudinal_history, env_id, execution_action[0]
        )
        ego = load_vehicle(ego_buffers, env_id)
        ego = apply_command(
            ego, execution_action, ego_buffers.steer_bias[env_id], sim
        )
        executed_steer = execution_action[1]
        executed_steer = ego.steer
        push_executed_steer(
            env.executed_steer_history, env_id, executed_steer
        )
        for _ in range(substeps):
            ego = integrate_vehicle_substep(
                ego,
                ego_buffers.mass[env_id],
                ego_buffers.mu[env_id],
                ego_buffers.drive_scale[env_id],
                sim,
            )
        env.current_action[env_id] = action
        env.episode_step[env_id] = env.episode_step[env_id] + 1
        store_vehicle(ego_buffers, env_id, ego)
    else:
        opponent = load_vehicle(opponent_buffers, env_id)
        opponent_frenet = project_track(
            wp.vec2f(opponent.x, opponent.y),
            env.opponent_segment[env_id],
            track,
        )
        opponent_action = policy_opponent_actions[env_id]
        if env.opponent_mode[env_id] == 0:
            opponent_action = scripted_physics_opponent_action(
                env_id,
                opponent,
                opponent_frenet,
                env,
                sim,
                opponent_params,
            )
        else:
            forward_speed = wp.dot(
                body_velocity_world(opponent), opponent_frenet.tangent
            )
            if forward_speed > env.opponent_speed_cap[env_id]:
                opponent_action = wp.vec2f(0.0, opponent_action[1])
        push_executed_longitudinal(
            env.opponent_executed_longitudinal_history,
            env_id,
            opponent_action[0],
        )
        opponent = apply_command(
            opponent,
            opponent_action,
            opponent_buffers.steer_bias[env_id],
            sim,
        )
        opponent_executed_steer = opponent_action[1]
        opponent_executed_steer = opponent.steer
        push_executed_steer(
            env.opponent_executed_steer_history,
            env_id,
            opponent_executed_steer,
        )
        for _ in range(substeps):
            opponent = integrate_vehicle_substep(
                opponent,
                opponent_buffers.mass[env_id],
                opponent_buffers.mu[env_id],
                opponent_buffers.drive_scale[env_id],
                sim,
            )
        env.current_opponent_action[env_id] = opponent_action
        store_vehicle(opponent_buffers, env_id, opponent)


@wp.kernel(enable_backward=False)
def contact_stage_kernel(
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: PhysicsBuffers,
    opponent_params: OpponentParams,
):
    env_id = wp.tid()
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    ego, opponent, contact = resolve_and_apply_pair_contact(
        ego, opponent, opponent_params
    )
    if contact.contact != 0:
        store_vehicle(ego_buffers, env_id, ego)
        store_vehicle(opponent_buffers, env_id, opponent)
    env.contact[env_id] = contact.contact
    env.contact_closing_speed[env_id] = contact.closing_speed


@wp.kernel(enable_backward=False)
def transaction_stage_kernel(
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    sim: SimParams,
    reward_params: RewardParams,
    termination: TerminationParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
):
    env_id = wp.tid()
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    contact = ContactResult()
    if reset.has_opponent != 0:
        ego, opponent, contact = resolve_and_apply_pair_contact(
            ego, opponent, opponent_params
        )
        if contact.contact != 0:
            store_vehicle(ego_buffers, env_id, ego)
            store_vehicle(opponent_buffers, env_id, opponent)
    env.contact[env_id] = contact.contact
    env.contact_closing_speed[env_id] = contact.closing_speed
    ego_frenet = project_track(
        wp.vec2f(ego.x, ego.y), env.ego_segment[env_id], track
    )
    opponent_frenet = FrenetState()
    if reset.has_opponent != 0:
        opponent_frenet = project_track(
            wp.vec2f(opponent.x, opponent.y),
            env.opponent_segment[env_id],
            track,
        )
    env.ego_segment[env_id] = ego_frenet.segment
    env.opponent_segment[env_id] = opponent_frenet.segment
    result = compute_reward_and_done(
        env_id,
        ego,
        opponent,
        ego_frenet,
        opponent_frenet,
        env.current_action[env_id],
        contact,
        env,
        track,
        reward_params,
        termination,
        reset.has_opponent,
        opponent_params.car_length,
        opponent_params.car_width,
    )
    valid = vehicle_is_finite(ego)
    if reset.has_opponent != 0 and not vehicle_is_finite(opponent):
        valid = False
    if not valid:
        result.done = True
        result.done_flags = result.done_flags | 8
    store_reward(env_id, result, env)
    if result.done:
        env.terminal_x[env_id] = ego.x
        env.terminal_y[env_id] = ego.y
        env.terminal_s[env_id] = ego_frenet.s
        ego, opponent, ego_frenet, opponent_frenet = reset_pair(
            env_id,
            ego_buffers,
            opponent_buffers,
            env,
            track,
            sim,
            reset,
            opponent_params,
        )
        store_vehicle(ego_buffers, env_id, ego)
        store_vehicle(opponent_buffers, env_id, opponent)


@wp.kernel(enable_backward=False)
def observation_stage_kernel(
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    obs_params: ObsParams,
    reset: ResetParams,
    raw_previous: wp.array2d(dtype=wp.float32),
    raw_current: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
    opponent_output: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    ego_frenet = project_track(
        wp.vec2f(ego.x, ego.y), env.ego_segment[env_id], track
    )
    opponent_frenet = FrenetState()
    if reset.has_opponent != 0:
        opponent_frenet = project_track(
            wp.vec2f(opponent.x, opponent.y),
            env.opponent_segment[env_id],
            track,
        )
    visible = write_raw_observation(
        env_id,
        ego,
        opponent,
        ego_frenet,
        opponent_frenet,
        env.last_action[env_id],
        track,
        obs_params,
        reset,
        raw_current,
    )
    if reset.has_opponent != 0:
        write_raw_observation(
            env_id,
            opponent,
            ego,
            opponent_frenet,
            ego_frenet,
            env.opponent_last_action[env_id],
            track,
            obs_params,
            reset,
            opponent_output,
        )
    publish_observation(
        env_id,
        visible,
        env,
        reset,
        raw_previous,
        raw_current,
        output,
    )
    if not env.done[env_id]:
        env.last_action[env_id] = env.current_action[env_id]
        env.opponent_last_action[env_id] = env.current_opponent_action[env_id]


@wp.kernel(enable_backward=False)
def env_step_kernel(
    actions: wp.array(dtype=wp.vec2f),
    policy_opponent_actions: wp.array(dtype=wp.vec2f),
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    sim: SimParams,
    obs_params: ObsParams,
    reward_params: RewardParams,
    termination: TerminationParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
    substeps: wp.int32,
    raw_previous: wp.array2d(dtype=wp.float32),
    raw_current: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
    opponent_output: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    action = wp.vec2f(
        wp.clamp(actions[env_id][0], -1.0, 1.0),
        wp.clamp(actions[env_id][1], -1.0, 1.0),
    )
    execution_action = delayed_action(env_id, action, env)
    ego = load_vehicle(ego_buffers, env_id)
    opponent = load_vehicle(opponent_buffers, env_id)
    ego = apply_command(
        ego, execution_action, ego_buffers.steer_bias[env_id], sim
    )
    executed_steer = execution_action[1]
    executed_steer = ego.steer
    push_executed_steer(
        env.executed_steer_history, env_id, executed_steer
    )

    opponent_frenet = project_track(
        wp.vec2f(opponent.x, opponent.y),
        env.opponent_segment[env_id],
        track,
    )
    opponent_action = wp.vec2f(0.0)
    if reset.has_opponent != 0:
        opponent_action = policy_opponent_actions[env_id]
        if env.opponent_mode[env_id] == 0:
            opponent_action = scripted_opponent_action(
                env_id,
                opponent,
                opponent_frenet,
                env,
                sim,
                opponent_params,
            )
        opponent = apply_command(
            opponent,
            opponent_action,
            opponent_buffers.steer_bias[env_id],
            sim,
        )
        opponent_executed_steer = opponent_action[1]
        opponent_executed_steer = opponent.steer
        push_executed_steer(
            env.opponent_executed_steer_history,
            env_id,
            opponent_executed_steer,
        )

    contact = ContactResult()
    valid = wp.int32(1)
    for _ in range(substeps):
        if valid != 0:
            ego = integrate_vehicle_substep(
                ego,
                ego_buffers.mass[env_id],
                ego_buffers.mu[env_id],
                ego_buffers.drive_scale[env_id],
                sim,
            )
            if reset.has_opponent != 0:
                opponent = integrate_vehicle_substep(
                    opponent,
                    opponent_buffers.mass[env_id],
                    opponent_buffers.mu[env_id],
                    opponent_buffers.drive_scale[env_id],
                    sim,
                )
                substep_contact = resolve_pair_contact(
                    ego, opponent, opponent_params
                )
                if substep_contact.contact != 0:
                    contact = substep_contact
                    ego, opponent = apply_contact(
                        ego,
                        opponent,
                        substep_contact,
                        opponent_params.restitution,
                    )
            if not vehicle_is_finite(ego):
                valid = 0
            if reset.has_opponent != 0 and not vehicle_is_finite(opponent):
                valid = 0

    env.episode_step[env_id] = env.episode_step[env_id] + 1
    ego_frenet = project_track(
        wp.vec2f(ego.x, ego.y), env.ego_segment[env_id], track
    )
    opponent_frenet = project_track(
        wp.vec2f(opponent.x, opponent.y),
        env.opponent_segment[env_id],
        track,
    )
    env.ego_segment[env_id] = ego_frenet.segment
    env.opponent_segment[env_id] = opponent_frenet.segment
    result = compute_reward_and_done(
        env_id,
        ego,
        opponent,
        ego_frenet,
        opponent_frenet,
        action,
        contact,
        env,
        track,
        reward_params,
        termination,
        reset.has_opponent,
        opponent_params.car_length,
        opponent_params.car_width,
    )
    if valid == 0:
        result.done = True
        result.done_flags = result.done_flags | 8
    store_reward(env_id, result, env)

    if result.done:
        env.terminal_x[env_id] = ego.x
        env.terminal_y[env_id] = ego.y
        env.terminal_s[env_id] = ego_frenet.s
        ego, opponent, ego_frenet, opponent_frenet = reset_pair(
            env_id,
            ego_buffers,
            opponent_buffers,
            env,
            track,
            sim,
            reset,
            opponent_params,
        )

    visible = write_raw_observation(
        env_id,
        ego,
        opponent,
        ego_frenet,
        opponent_frenet,
        env.last_action[env_id],
        track,
        obs_params,
        reset,
        raw_current,
    )
    write_raw_observation(
        env_id,
        opponent,
        ego,
        opponent_frenet,
        ego_frenet,
        env.opponent_last_action[env_id],
        track,
        obs_params,
        reset,
        opponent_output,
    )
    publish_observation(
        env_id,
        visible,
        env,
        reset,
        raw_previous,
        raw_current,
        output,
    )
    if not result.done:
        env.last_action[env_id] = action
        env.opponent_last_action[env_id] = opponent_action
    store_vehicle(ego_buffers, env_id, ego)
    store_vehicle(opponent_buffers, env_id, opponent)


@wp.kernel(enable_backward=False)
def reset_envs_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    sim: SimParams,
    obs_params: ObsParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
    raw_a: wp.array2d(dtype=wp.float32),
    raw_b: wp.array2d(dtype=wp.float32),
    obs_a: wp.array2d(dtype=wp.float32),
    obs_b: wp.array2d(dtype=wp.float32),
    opponent_obs: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset_mask[env_id]:
        ego, opponent, ego_frenet, opponent_frenet = reset_pair(
            env_id,
            ego_buffers,
            opponent_buffers,
            env,
            track,
            sim,
            reset,
            opponent_params,
        )
        visible = write_raw_observation(
            env_id,
            ego,
            opponent,
            ego_frenet,
            opponent_frenet,
            env.last_action[env_id],
            track,
            obs_params,
            reset,
            raw_a,
        )
        write_raw_observation(
            env_id,
            opponent,
            ego,
            opponent_frenet,
            ego_frenet,
            env.opponent_last_action[env_id],
            track,
            obs_params,
            reset,
            opponent_obs,
        )
        for feature in range(OBS_DIM):
            raw_b[env_id, feature] = raw_a[env_id, feature]
        publish_observation(
            env_id, visible, env, reset, raw_a, raw_a, obs_a
        )
        for feature in range(OBS_DIM):
            obs_b[env_id, feature] = obs_a[env_id, feature]
        env.done[env_id] = False
        env.done_flags[env_id] = 0
        env.term_timeout[env_id] = False
        env.term_oob[env_id] = False
        env.term_stopped[env_id] = False
        env.term_invalid[env_id] = False
        env.term_collision[env_id] = False
        env.reward[env_id] = 0.0
        store_vehicle(ego_buffers, env_id, ego)
        store_vehicle(opponent_buffers, env_id, opponent)


@wp.func
def place_fixed(
    pose: wp.vec2f,
    yaw: wp.float32,
    speed: wp.float32,
    sim: SimParams,
) -> VehicleLocal:
    vehicle = VehicleLocal()
    vehicle = clear_vehicle(vehicle)
    vehicle.x = pose[0]
    vehicle.y = pose[1]
    vehicle.yaw = yaw
    vehicle.vx = speed
    vehicle.omega = wp.vec4f(speed / sim.wheel_radius)
    return vehicle


@wp.kernel(enable_backward=False)
def reset_to_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    ego_pose: wp.array(dtype=wp.vec2f),
    ego_yaw: wp.array(dtype=wp.float32),
    ego_speed: wp.array(dtype=wp.float32),
    opponent_pose: wp.array(dtype=wp.vec2f),
    opponent_yaw: wp.array(dtype=wp.float32),
    opponent_speed: wp.array(dtype=wp.float32),
    ego_buffers: VehicleBuffers,
    opponent_buffers: VehicleBuffers,
    env: EnvBuffers,
    track: TrackData,
    sim: SimParams,
    obs_params: ObsParams,
    reset: ResetParams,
    opponent_params: OpponentParams,
    raw_a: wp.array2d(dtype=wp.float32),
    raw_b: wp.array2d(dtype=wp.float32),
    obs_a: wp.array2d(dtype=wp.float32),
    obs_b: wp.array2d(dtype=wp.float32),
    opponent_obs: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if reset_mask[env_id]:
        ego = place_fixed(ego_pose[env_id], ego_yaw[env_id], ego_speed[env_id], sim)
        opponent = place_fixed(
            opponent_pose[env_id],
            opponent_yaw[env_id],
            opponent_speed[env_id],
            sim,
        )
        ego_frenet = project_track(
            wp.vec2f(ego.x, ego.y), env.ego_segment[env_id], track
        )
        opponent_frenet = project_track(
            wp.vec2f(opponent.x, opponent.y), env.opponent_segment[env_id], track
        )
        env.ego_segment[env_id] = ego_frenet.segment
        env.opponent_segment[env_id] = opponent_frenet.segment
        env.prev_s[env_id] = ego_frenet.s
        env.prev_opponent_s[env_id] = opponent_frenet.s
        env.metric_boundary[env_id] = ego_frenet.boundary_distance
        env.metric_lateral[env_id] = ego_frenet.ey
        env.metric_speed[env_id] = ego_speed[env_id]
        env.metric_opponent_speed[env_id] = opponent_speed[env_id]

        visible = write_raw_observation(
            env_id,
            ego,
            opponent,
            ego_frenet,
            opponent_frenet,
            env.last_action[env_id],
            track,
            obs_params,
            reset,
            raw_a,
        )
        write_raw_observation(
            env_id,
            opponent,
            ego,
            opponent_frenet,
            ego_frenet,
            env.opponent_last_action[env_id],
            track,
            obs_params,
            reset,
            opponent_obs,
        )
        for feature in range(OBS_DIM):
            raw_b[env_id, feature] = raw_a[env_id, feature]
        publish_observation(
            env_id, visible, env, reset, raw_a, raw_a, obs_a
        )
        for feature in range(OBS_DIM):
            obs_b[env_id, feature] = obs_a[env_id, feature]
        store_vehicle(ego_buffers, env_id, ego)
        store_vehicle(opponent_buffers, env_id, opponent)

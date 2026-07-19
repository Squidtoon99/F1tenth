"""Warp LiDAR (UST-10LX), 6-axis IMU, and fused 1,093-D actor sensor kernels.

EDT sphere-trace for corridor walls, analytic opponent OBB when ``has_opponent``,
and per-episode / per-step domain randomization. Instantaneous scans only;
``beam_time`` is reserved for a later motion-distortion path.

``sensor_actor_*_kernel`` writes LiDAR + IMU + VESC + causal commands in one
launch per view (beam 0 owns the 12 non-LiDAR slots). Standalone LiDAR/IMU
kernels remain for geometry unit tests.

Frozen actor layout (offsets inclusive of the stated half-open ranges):
LiDAR ``[0:1081]``, IMU ``[1081:1087]``, VESC speed ``[1087]``, signed current
proxy ``[1088]``, current command ``[1089:1091]``, predecessor ``[1091:1093]``.
"""

from __future__ import annotations

import warp as wp

from f1tenth_sim.dynamics import VehicleBuffers, VehicleLocal, load_vehicle
from f1tenth_sim.params import SimParams

from .kernel import (
    CorridorDistanceField,
    EnvBuffers,
    OpponentParams,
    ResetParams,
    SensorParams,
    observation_noise,
)

# Bound sphere-trace iterations for GPU latency. At ~2.5 cm cells and 30 m max
# range, sphere tracing typically finishes far sooner; this is a hard safety cap.
MAX_LIDAR_MARCH_STEPS = 512
# Warp IMU kernel layout: ax, ay, az, gx, gy, gz.
IMU_DIM = 6
_EDT_HIT_FRAC = 0.55

# Frozen asymmetric actor observation layout (native UST-10LX beams).
NATIVE_NUM_BEAMS = 1081
ACTOR_LIDAR_DIM = NATIVE_NUM_BEAMS
ACTOR_IMU_DIM = IMU_DIM
ACTOR_LIDAR_START = 0
ACTOR_IMU_START = ACTOR_LIDAR_DIM
ACTOR_VESC_SPEED = ACTOR_IMU_START + ACTOR_IMU_DIM
ACTOR_VESC_CURRENT = ACTOR_VESC_SPEED + 1
ACTOR_CMD_CURRENT_START = ACTOR_VESC_CURRENT + 1
ACTOR_CMD_PRED_START = ACTOR_CMD_CURRENT_START + 2
ACTOR_OBS_DIM = ACTOR_CMD_PRED_START + 2
VESC_CURRENT_SCALE_A = 10.0
# View ids key independent white-noise / dropout draws for ego vs opponent.
VIEW_EGO = 0
VIEW_OPPONENT = 1
_VIEW_SEED_PRIME = 668265263
# Distinct feature ids for VESC per-step noise (away from IMU 0..5 / lidar beams).
_VESC_SPEED_NOISE_FEATURE = 10_000
_VESC_CURRENT_NOISE_FEATURE = 10_001

assert ACTOR_OBS_DIM == 1093
assert ACTOR_IMU_START == 1081
assert ACTOR_VESC_SPEED == 1087
assert ACTOR_VESC_CURRENT == 1088
assert ACTOR_CMD_CURRENT_START == 1089
assert ACTOR_CMD_PRED_START == 1091


@wp.func
def sample_corridor_distance(
    field: CorridorDistanceField,
    x: wp.float32,
    y: wp.float32,
) -> wp.float32:
    """Bilinear EDT sample. Returns -1 outside the grid (no-return)."""
    gx = (x - field.origin[0]) / field.resolution
    gy = (y - field.origin[1]) / field.resolution
    if (
        gx < 0.0
        or gy < 0.0
        or gx >= wp.float32(field.width - 1)
        or gy >= wp.float32(field.height - 1)
    ):
        return wp.float32(-1.0)
    x0 = wp.int32(wp.floor(gx))
    y0 = wp.int32(wp.floor(gy))
    x1 = x0 + 1
    y1 = y0 + 1
    tx = gx - wp.float32(x0)
    ty = gy - wp.float32(y0)
    i00 = y0 * field.width + x0
    i10 = y0 * field.width + x1
    i01 = y1 * field.width + x0
    i11 = y1 * field.width + x1
    d00 = field.distance[i00]
    d10 = field.distance[i10]
    d01 = field.distance[i01]
    d11 = field.distance[i11]
    return (1.0 - tx) * (1.0 - ty) * d00 + tx * (1.0 - ty) * d10 + (
        1.0 - tx
    ) * ty * d01 + tx * ty * d11


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
    """Ray vs oriented box; returns entry distance or a large miss sentinel."""
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
        dist = sample_corridor_distance(field, px, py)
        if dist < 0.0:
            return range_max
        if dist <= hit_eps:
            return wp.min(traveled, range_max)
        step = wp.max(dist, min_step)
        traveled = traveled + step
    return range_max


@wp.func
def sensor_view_seed(seed: wp.int32, view_id: wp.int32) -> wp.int32:
    """Mix a view id into the episode seed so ego/opponent noise is independent."""
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
    # Instantaneous scan; beam_time reserved for motion-distortion interpolation.
    beam_time = wp.float32(0.0)
    mount_x = sensor.lidar_offset_x + env.lidar_extrinsic_x[env_id]
    mount_y = sensor.lidar_offset_y + env.lidar_extrinsic_y[env_id]
    mount_yaw = (
        sensor.lidar_offset_yaw
        + env.lidar_extrinsic_yaw[env_id]
        + env.lidar_angle_bias[env_id]
    )
    cosine = wp.cos(ego.yaw)
    sine = wp.sin(ego.yaw)
    origin = wp.vec2f(
        ego.x + cosine * mount_x - sine * mount_y,
        ego.y + sine * mount_x + cosine * mount_y,
    )
    origin = origin + beam_time * wp.vec2f(0.0, 0.0)
    beam_angle = (
        ego.yaw
        + mount_yaw
        + sensor.angle_min
        + wp.float32(beam_id) * sensor.angle_increment
    )
    direction = wp.vec2f(wp.cos(beam_angle), wp.sin(beam_angle))

    hit = sphere_trace_walls(
        origin, direction, field, sensor.range_max, sensor.max_march_steps
    )
    if has_opponent != 0:
        half_length = 0.5 * opponent_params.car_length
        half_width = 0.5 * opponent_params.car_width
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
        hit = sensor.range_min
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
    gz = ego.yaw_rate

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
    az = az + env.imu_accel_bias_z[env_id]
    gx = gx + env.imu_gyro_bias_x[env_id]
    gy = gy + env.imu_gyro_bias_y[env_id]
    gz = gz + env.imu_gyro_bias_z[env_id]

    view_seed = sensor_view_seed(reset.seed, view_id)
    accel_std = env.imu_accel_noise_std[env_id]
    gyro_std = env.imu_gyro_noise_std[env_id]
    episode = env.episode_id[env_id]
    step = env.episode_step[env_id]
    if accel_std > 0.0:
        ax = ax + observation_noise(view_seed, env_id, episode, step, 0, accel_std)
        ay = ay + observation_noise(view_seed, env_id, episode, step, 1, accel_std)
        az = az + observation_noise(view_seed, env_id, episode, step, 2, accel_std)
    if gyro_std > 0.0:
        gx = gx + observation_noise(view_seed, env_id, episode, step, 3, gyro_std)
        gy = gy + observation_noise(view_seed, env_id, episode, step, 4, gyro_std)
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
    env_id: wp.int32,
    wheel_radius: wp.float32,
    current_scale: wp.float32,
    use_opponent_commands: wp.int32,
    view_id: wp.int32,
    output: wp.array2d(dtype=wp.float32),
):
    omega = vehicle.omega[env_id]
    mean_omega = 0.25 * (omega[0] + omega[1] + omega[2] + omega[3])
    speed = mean_omega * wheel_radius
    current = current_scale * vehicle.applied_effort[env_id]
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

    if use_opponent_commands != 0:
        command = env.current_opponent_action[env_id]
        predecessor = env.opponent_last_action[env_id]
    else:
        command = env.current_action[env_id]
        predecessor = env.last_action[env_id]
    output[env_id, ACTOR_CMD_CURRENT_START] = command[0]
    output[env_id, ACTOR_CMD_CURRENT_START + 1] = command[1]
    output[env_id, ACTOR_CMD_PRED_START] = predecessor[0]
    output[env_id, ACTOR_CMD_PRED_START + 1] = predecessor[1]


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
    wheel_radius: wp.float32,
    current_scale: wp.float32,
    output: wp.array2d(dtype=wp.float32),
):
    """Fused 1v0 sensor→actor write: each beam fills LiDAR; beam 0 fills the rest.

    Must run on post-transaction state before ``observation_stage_kernel``
    advances ``last_action`` / ``opponent_last_action``.
    """
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
            env_id,
            wheel_radius,
            current_scale,
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
    wheel_radius: wp.float32,
    current_scale: wp.float32,
    use_opponent_commands: wp.int32,
    view_id: wp.int32,
    output: wp.array2d(dtype=wp.float32),
):
    """Fused 1v1 sensor→actor write with optional role-swap via buffer order.

    Must run on post-transaction state before ``observation_stage_kernel``
    advances ``last_action`` / ``opponent_last_action``.
    """
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
            env_id,
            wheel_radius,
            current_scale,
            use_opponent_commands,
            view_id,
            output,
        )

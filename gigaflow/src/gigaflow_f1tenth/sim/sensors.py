"""All-car LiDAR (EDT walls + ray/OBB cars) with beam/sector/noise corruption."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from gigaflow_f1tenth.sim.layout_local import (
    IMU_START,
    LIDAR_ANGLE_INCREMENT,
    LIDAR_ANGLE_MIN,
    LIDAR_DIM,
    LIDAR_RANGE_MAX,
    LIDAR_RANGE_MIN,
    LIDAR_RELIABLE_RANGE,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_T,
    STEER_T1,
    STEER_T2,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    VESC_CURRENT,
    VESC_CURRENT_SCALE_A,
    VESC_SPEED,
)
from gigaflow_f1tenth.sim.vehicle import VehicleBuffers, VehicleLocal, load_vehicle

MAX_LIDAR_MARCH_STEPS = 512
_EDT_HIT_FRAC = 0.55
_VIEW_SEED_PRIME = 668265263


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
    car_length: wp.float32
    car_width: wp.float32
    max_agents_per_world: wp.int32
    vesc_current_scale: wp.float32


@wp.struct
class CorridorFieldView:
    distance: wp.array(dtype=wp.float32)
    width: wp.int32
    height: wp.int32
    origin: wp.vec2f
    resolution: wp.float32
    field_offset: wp.int32


def default_sensor_params(car_length: float, car_width: float, max_agents: int) -> SensorParams:
    params = SensorParams()
    params.num_beams = LIDAR_DIM
    params.angle_min = float(LIDAR_ANGLE_MIN)
    params.angle_increment = float(LIDAR_ANGLE_INCREMENT)
    params.range_min = float(LIDAR_RANGE_MIN)
    params.range_max = float(LIDAR_RANGE_MAX)
    params.reliable_range = float(LIDAR_RELIABLE_RANGE)
    params.lidar_offset_x = 0.15
    params.lidar_offset_y = 0.0
    params.lidar_offset_yaw = 0.0
    params.max_march_steps = MAX_LIDAR_MARCH_STEPS
    params.car_length = float(car_length)
    params.car_width = float(car_width)
    params.max_agents_per_world = int(max_agents)
    params.vesc_current_scale = float(VESC_CURRENT_SCALE_A)
    return params


@wp.func
def sample_corridor_distance(
    field: CorridorFieldView,
    x: wp.float32,
    y: wp.float32,
) -> wp.float32:
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
    base = field.field_offset
    i00 = base + y0 * field.width + x0
    i10 = base + y0 * field.width + x1
    i01 = base + y1 * field.width + x0
    i11 = base + y1 * field.width + x1
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
    field: CorridorFieldView,
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
def _rng_state(
    seed: wp.int32,
    slot: wp.int32,
    episode: wp.int32,
    step: wp.int32,
    feature: wp.int32,
) -> wp.uint32:
    # Two-arg rand_init keeps serial and beam-parallel streams identical;
    # large int32 feature multiplies overflow differently for tid vs range().
    base = (
        seed
        ^ (slot * 73244475)
        ^ (episode * 295075153)
        ^ (step * 104395301)
    )
    return wp.rand_init(base, feature)


@wp.func
def observation_noise(
    seed: wp.int32,
    slot: wp.int32,
    episode: wp.int32,
    step: wp.int32,
    feature: wp.int32,
    std: wp.float32,
) -> wp.float32:
    random = _rng_state(seed, slot, episode, step, feature)
    # Box-Muller-ish: two uniforms -> approx normal via sum.
    u1 = wp.max(wp.randf(random), 1.0e-6)
    u2 = wp.randf(random)
    mag = wp.sqrt(-2.0 * wp.log(u1))
    return std * mag * wp.cos(6.28318530718 * u2)


@wp.func
def _write_proprio(
    vehicles: VehicleBuffers,
    vehicle: VehicleLocal,
    slot: wp.int32,
    sensor: SensorParams,
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    output[slot, IMU_START + 0] = vehicle.ax
    output[slot, IMU_START + 1] = vehicle.ay
    output[slot, IMU_START + 2] = 9.81
    output[slot, IMU_START + 3] = 0.0
    output[slot, IMU_START + 4] = 0.0
    output[slot, IMU_START + 5] = vehicle.yaw_rate
    omega = vehicles.omega[slot]
    mean_omega = 0.25 * (omega[0] + omega[1] + omega[2] + omega[3])
    output[slot, VESC_SPEED] = mean_omega * 0.053
    output[slot, VESC_CURRENT] = sensor.vesc_current_scale * vehicle.applied_effort
    output[slot, THROTTLE_CURRENT] = executed_long_0[slot]
    output[slot, THROTTLE_PRED] = executed_long_1[slot]
    output[slot, STEER_T] = executed_steer_0[slot]
    output[slot, STEER_T1] = executed_steer_1[slot]
    output[slot, STEER_T2] = executed_steer_2[slot]
    output[slot, STEER_DELTA0] = executed_steer_0[slot] - executed_steer_1[slot]
    output[slot, STEER_DELTA1] = executed_steer_1[slot] - executed_steer_2[slot]
    output[slot, STEER_DELTA2] = executed_steer_2[slot] - executed_steer_3[slot]


@wp.func
def _lidar_beam_hit(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    origin: wp.vec2f,
    direction: wp.vec2f,
    field: CorridorFieldView,
    sensor: SensorParams,
    slot: wp.int32,
    beam_id: wp.int32,
    seed: wp.int32,
    episode_id: wp.int32,
    episode_step: wp.int32,
    lidar_range_noise_std: wp.float32,
    lidar_dropout_prob: wp.float32,
    lidar_far_dropout_prob: wp.float32,
    lidar_sector_start: wp.int32,
    lidar_sector_width: wp.int32,
) -> wp.float32:
    hit = sphere_trace_walls(
        origin, direction, field, sensor.range_max, sensor.max_march_steps
    )
    half_l = 0.5 * sensor.car_length
    half_w = 0.5 * sensor.car_width
    # Contiguous (world, slot) layout: only scan agents in this world.
    max_a = sensor.max_agents_per_world
    base = (slot // max_a) * max_a
    for local in range(max_a):
        other = base + local
        if other == slot:
            continue
        if active[other] == 0:
            continue
        other_hit = ray_obb_distance(
            origin,
            direction,
            vehicles.x[other],
            vehicles.y[other],
            vehicles.yaw[other],
            half_l,
            half_w,
        )
        if other_hit < hit:
            hit = other_hit

    if hit < sensor.range_min:
        hit = sensor.range_min
    if hit > sensor.range_max:
        hit = sensor.range_max

    if lidar_range_noise_std > 0.0:
        hit = hit + observation_noise(
            seed, slot, episode_id, episode_step, beam_id, lidar_range_noise_std
        )
        hit = wp.clamp(hit, sensor.range_min, sensor.range_max)

    if lidar_sector_width > 0:
        rel = beam_id - lidar_sector_start
        if rel < 0:
            rel = rel + sensor.num_beams
        if rel < lidar_sector_width:
            hit = sensor.range_max

    dropout = lidar_dropout_prob
    if hit > sensor.reliable_range:
        dropout = wp.max(dropout, lidar_far_dropout_prob)
    if dropout > 0.0:
        random = _rng_state(
            seed, slot, episode_id, episode_step, beam_id + sensor.num_beams
        )
        if wp.randf(random) < dropout:
            hit = sensor.range_max
    return hit


@wp.kernel(enable_backward=False)
def lidar_and_proprio_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    track_id: wp.array(dtype=wp.int32),
    episode_id: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    sensor_noise_seed: wp.array(dtype=wp.int32),
    lidar_range_noise_std: wp.array(dtype=wp.float32),
    lidar_dropout_prob: wp.array(dtype=wp.float32),
    lidar_far_dropout_prob: wp.array(dtype=wp.float32),
    lidar_angle_bias: wp.array(dtype=wp.float32),
    lidar_extrinsic_x: wp.array(dtype=wp.float32),
    lidar_extrinsic_y: wp.array(dtype=wp.float32),
    lidar_extrinsic_yaw: wp.array(dtype=wp.float32),
    lidar_sector_start: wp.array(dtype=wp.int32),
    lidar_sector_width: wp.array(dtype=wp.int32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    edt_distance: wp.array(dtype=wp.float32),
    edt_offsets: wp.array(dtype=wp.int32),
    edt_width: wp.array(dtype=wp.int32),
    edt_height: wp.array(dtype=wp.int32),
    edt_origin: wp.array(dtype=wp.vec2f),
    edt_resolution: wp.array(dtype=wp.float32),
    sensor: SensorParams,
    output: wp.array2d(dtype=wp.float32),
):
    """Serial per-slot reference (world-local opponents). Kept for parity tests."""
    slot = wp.tid()
    if active[slot] == 0:
        return
    vehicle = load_vehicle(vehicles, slot)
    tid = track_id[slot]
    field = CorridorFieldView()
    field.distance = edt_distance
    field.width = edt_width[tid]
    field.height = edt_height[tid]
    field.origin = edt_origin[tid]
    field.resolution = edt_resolution[tid]
    field.field_offset = edt_offsets[tid]

    mount_x = sensor.lidar_offset_x + lidar_extrinsic_x[slot]
    mount_y = sensor.lidar_offset_y + lidar_extrinsic_y[slot]
    mount_yaw = (
        sensor.lidar_offset_yaw + lidar_extrinsic_yaw[slot] + lidar_angle_bias[slot]
    )
    cosine = wp.cos(vehicle.yaw)
    sine = wp.sin(vehicle.yaw)
    origin = wp.vec2f(
        vehicle.x + cosine * mount_x - sine * mount_y,
        vehicle.y + sine * mount_x + cosine * mount_y,
    )
    seed = sensor_noise_seed[slot]
    num_beams = sensor.num_beams
    for beam_id in range(1081):
        if beam_id >= num_beams:
            break
        beam_angle = (
            vehicle.yaw
            + mount_yaw
            + sensor.angle_min
            + wp.float32(beam_id) * sensor.angle_increment
        )
        direction = wp.vec2f(wp.cos(beam_angle), wp.sin(beam_angle))
        hit = _lidar_beam_hit(
            vehicles,
            active,
            origin,
            direction,
            field,
            sensor,
            slot,
            beam_id,
            seed,
            episode_id[slot],
            episode_step[slot],
            lidar_range_noise_std[slot],
            lidar_dropout_prob[slot],
            lidar_far_dropout_prob[slot],
            lidar_sector_start[slot],
            lidar_sector_width[slot],
        )
        output[slot, beam_id] = hit

    _write_proprio(
        vehicles,
        vehicle,
        slot,
        sensor,
        executed_long_0,
        executed_long_1,
        executed_steer_0,
        executed_steer_1,
        executed_steer_2,
        executed_steer_3,
        output,
    )


@wp.kernel(enable_backward=False)
def lidar_and_proprio_kernel_global_scan(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    track_id: wp.array(dtype=wp.int32),
    episode_id: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    sensor_noise_seed: wp.array(dtype=wp.int32),
    lidar_range_noise_std: wp.array(dtype=wp.float32),
    lidar_dropout_prob: wp.array(dtype=wp.float32),
    lidar_far_dropout_prob: wp.array(dtype=wp.float32),
    lidar_angle_bias: wp.array(dtype=wp.float32),
    lidar_extrinsic_x: wp.array(dtype=wp.float32),
    lidar_extrinsic_y: wp.array(dtype=wp.float32),
    lidar_extrinsic_yaw: wp.array(dtype=wp.float32),
    lidar_sector_start: wp.array(dtype=wp.int32),
    lidar_sector_width: wp.array(dtype=wp.int32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    edt_distance: wp.array(dtype=wp.float32),
    edt_offsets: wp.array(dtype=wp.int32),
    edt_width: wp.array(dtype=wp.int32),
    edt_height: wp.array(dtype=wp.int32),
    edt_origin: wp.array(dtype=wp.vec2f),
    edt_resolution: wp.array(dtype=wp.float32),
    sensor: SensorParams,
    output: wp.array2d(dtype=wp.float32),
):
    """Pre-optimization global opponent scan (parity reference only)."""
    slot = wp.tid()
    if active[slot] == 0:
        return
    vehicle = load_vehicle(vehicles, slot)
    tid = track_id[slot]
    field = CorridorFieldView()
    field.distance = edt_distance
    field.width = edt_width[tid]
    field.height = edt_height[tid]
    field.origin = edt_origin[tid]
    field.resolution = edt_resolution[tid]
    field.field_offset = edt_offsets[tid]
    mount_x = sensor.lidar_offset_x + lidar_extrinsic_x[slot]
    mount_y = sensor.lidar_offset_y + lidar_extrinsic_y[slot]
    mount_yaw = (
        sensor.lidar_offset_yaw + lidar_extrinsic_yaw[slot] + lidar_angle_bias[slot]
    )
    cosine = wp.cos(vehicle.yaw)
    sine = wp.sin(vehicle.yaw)
    origin = wp.vec2f(
        vehicle.x + cosine * mount_x - sine * mount_y,
        vehicle.y + sine * mount_x + cosine * mount_y,
    )
    half_l = 0.5 * sensor.car_length
    half_w = 0.5 * sensor.car_width
    n = active.shape[0]
    wid = world_id[slot]
    seed = sensor_noise_seed[slot]
    num_beams = sensor.num_beams
    for beam_id in range(1081):
        if beam_id >= num_beams:
            break
        beam_angle = (
            vehicle.yaw
            + mount_yaw
            + sensor.angle_min
            + wp.float32(beam_id) * sensor.angle_increment
        )
        direction = wp.vec2f(wp.cos(beam_angle), wp.sin(beam_angle))
        hit = sphere_trace_walls(
            origin, direction, field, sensor.range_max, sensor.max_march_steps
        )
        for other in range(n):
            if other == slot:
                continue
            if active[other] == 0:
                continue
            if world_id[other] != wid:
                continue
            other_hit = ray_obb_distance(
                origin,
                direction,
                vehicles.x[other],
                vehicles.y[other],
                vehicles.yaw[other],
                half_l,
                half_w,
            )
            if other_hit < hit:
                hit = other_hit
        if hit < sensor.range_min:
            hit = sensor.range_min
        if hit > sensor.range_max:
            hit = sensor.range_max
        noise_std = lidar_range_noise_std[slot]
        if noise_std > 0.0:
            hit = hit + observation_noise(
                seed, slot, episode_id[slot], episode_step[slot], beam_id, noise_std
            )
            hit = wp.clamp(hit, sensor.range_min, sensor.range_max)
        sec_w = lidar_sector_width[slot]
        if sec_w > 0:
            sec0 = lidar_sector_start[slot]
            rel = beam_id - sec0
            if rel < 0:
                rel = rel + sensor.num_beams
            if rel < sec_w:
                hit = sensor.range_max
        dropout = lidar_dropout_prob[slot]
        if hit > sensor.reliable_range:
            dropout = wp.max(dropout, lidar_far_dropout_prob[slot])
        if dropout > 0.0:
            random = _rng_state(
                seed,
                slot,
                episode_id[slot],
                episode_step[slot],
                beam_id + sensor.num_beams,
            )
            if wp.randf(random) < dropout:
                hit = sensor.range_max
        output[slot, beam_id] = hit
    _write_proprio(
        vehicles,
        vehicle,
        slot,
        sensor,
        executed_long_0,
        executed_long_1,
        executed_steer_0,
        executed_steer_1,
        executed_steer_2,
        executed_steer_3,
        output,
    )


@wp.kernel(enable_backward=False)
def lidar_beam_parallel_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    track_id: wp.array(dtype=wp.int32),
    episode_id: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    sensor_noise_seed: wp.array(dtype=wp.int32),
    lidar_range_noise_std: wp.array(dtype=wp.float32),
    lidar_dropout_prob: wp.array(dtype=wp.float32),
    lidar_far_dropout_prob: wp.array(dtype=wp.float32),
    lidar_angle_bias: wp.array(dtype=wp.float32),
    lidar_extrinsic_x: wp.array(dtype=wp.float32),
    lidar_extrinsic_y: wp.array(dtype=wp.float32),
    lidar_extrinsic_yaw: wp.array(dtype=wp.float32),
    lidar_sector_start: wp.array(dtype=wp.int32),
    lidar_sector_width: wp.array(dtype=wp.int32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    edt_distance: wp.array(dtype=wp.float32),
    edt_offsets: wp.array(dtype=wp.int32),
    edt_width: wp.array(dtype=wp.int32),
    edt_height: wp.array(dtype=wp.int32),
    edt_origin: wp.array(dtype=wp.vec2f),
    edt_resolution: wp.array(dtype=wp.float32),
    sensor: SensorParams,
    output: wp.array2d(dtype=wp.float32),
):
    """One thread per (slot, beam); beam 0 also writes proprio channels."""
    slot, beam_id = wp.tid()
    if active[slot] == 0:
        return
    if beam_id >= sensor.num_beams:
        return
    vehicle = load_vehicle(vehicles, slot)
    tid = track_id[slot]
    field = CorridorFieldView()
    field.distance = edt_distance
    field.width = edt_width[tid]
    field.height = edt_height[tid]
    field.origin = edt_origin[tid]
    field.resolution = edt_resolution[tid]
    field.field_offset = edt_offsets[tid]

    mount_x = sensor.lidar_offset_x + lidar_extrinsic_x[slot]
    mount_y = sensor.lidar_offset_y + lidar_extrinsic_y[slot]
    mount_yaw = (
        sensor.lidar_offset_yaw + lidar_extrinsic_yaw[slot] + lidar_angle_bias[slot]
    )
    cosine = wp.cos(vehicle.yaw)
    sine = wp.sin(vehicle.yaw)
    origin = wp.vec2f(
        vehicle.x + cosine * mount_x - sine * mount_y,
        vehicle.y + sine * mount_x + cosine * mount_y,
    )
    beam_angle = (
        vehicle.yaw
        + mount_yaw
        + sensor.angle_min
        + wp.float32(beam_id) * sensor.angle_increment
    )
    direction = wp.vec2f(wp.cos(beam_angle), wp.sin(beam_angle))
    hit = _lidar_beam_hit(
        vehicles,
        active,
        origin,
        direction,
        field,
        sensor,
        slot,
        beam_id,
        sensor_noise_seed[slot],
        episode_id[slot],
        episode_step[slot],
        lidar_range_noise_std[slot],
        lidar_dropout_prob[slot],
        lidar_far_dropout_prob[slot],
        lidar_sector_start[slot],
        lidar_sector_width[slot],
    )
    output[slot, beam_id] = hit
    if beam_id == 0:
        _write_proprio(
            vehicles,
            vehicle,
            slot,
            sensor,
            executed_long_0,
            executed_long_1,
            executed_steer_0,
            executed_steer_1,
            executed_steer_2,
            executed_steer_3,
            output,
        )


def ray_obb_distance_numpy(
    origin: np.ndarray,
    direction: np.ndarray,
    box_xy: np.ndarray,
    box_yaw: float,
    half_length: float,
    half_width: float,
) -> float:
    c, s = math.cos(box_yaw), math.sin(box_yaw)
    dx = origin[0] - box_xy[0]
    dy = origin[1] - box_xy[1]
    local_o = np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float64)
    local_d = np.array(
        [c * direction[0] + s * direction[1], -s * direction[0] + c * direction[1]],
        dtype=np.float64,
    )
    t_min, t_max = -1.0e30, 1.0e30
    for axis, half in enumerate((half_length, half_width)):
        origin_a = local_o[axis]
        dir_a = local_d[axis]
        if abs(dir_a) < 1.0e-12:
            if origin_a < -half or origin_a > half:
                return 1.0e30
            continue
        inv = 1.0 / dir_a
        t0 = (-half - origin_a) * inv
        t1 = (half - origin_a) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
            return 1.0e30
    if t_max < 0.0:
        return 1.0e30
    if t_min < 0.0:
        return 0.0
    return float(t_min)

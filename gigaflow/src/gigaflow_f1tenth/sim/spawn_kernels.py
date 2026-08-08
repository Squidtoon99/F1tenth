"""Warp kernels for async agent respawn on the production step path."""

from __future__ import annotations

import warp as wp

from gigaflow_f1tenth.rewards import CONDITION_DIM, CONDITION_FIELD_NAMES
from gigaflow_f1tenth.sim.geometry import project_window
from gigaflow_f1tenth.sim.vehicle import (
    VehicleBuffers,
    VehicleLocal,
    clear_vehicle,
    store_vehicle,
)

_STYLE_WIDTH = CONDITION_DIM
_IDX_DRIVE_SCALE = CONDITION_FIELD_NAMES.index("drive_scale")
_IDX_STEER_SCALE = CONDITION_FIELD_NAMES.index("steer_scale")
_IDX_ACCEL_SCALE = CONDITION_FIELD_NAMES.index("accel_scale")
_IDX_VMAX_SCALE = CONDITION_FIELD_NAMES.index("vmax_scale")
_IDX_MASS_KG = CONDITION_FIELD_NAMES.index("mass_kg")


@wp.func
def _mix_seed(global_seed: int, world_id: int, slot: int, episode_id: int) -> int:
    mixed = (
        global_seed
        ^ (world_id * 73244475)
        ^ (slot * 19349663)
        ^ (episode_id * 295075153)
    )
    if mixed < 0:
        mixed = -mixed
    return mixed


@wp.kernel(enable_backward=False)
def async_respawn_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    trainable: wp.array(dtype=wp.uint8),
    done: wp.array(dtype=wp.uint8),
    timeout: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    slot_id: wp.array(dtype=wp.int32),
    track_id: wp.array(dtype=wp.int32),
    episode_id: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    stalled_steps: wp.array(dtype=wp.int32),
    frenet_segment: wp.array(dtype=wp.int32),
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    prev_s: wp.array(dtype=wp.float32),
    progress_s: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
    wall_contact: wp.array(dtype=wp.uint8),
    contact: wp.array(dtype=wp.uint8),
    contact_counterpart: wp.array(dtype=wp.int32),
    contact_closing_speed: wp.array(dtype=wp.float32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
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
    point: wp.array(dtype=wp.vec2f),
    tangent: wp.array(dtype=wp.vec2f),
    normal: wp.array(dtype=wp.vec2f),
    segment_length: wp.array(dtype=wp.float32),
    cumulative_length: wp.array(dtype=wp.float32),
    width_left: wp.array(dtype=wp.float32),
    width_right: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    track_length: wp.array(dtype=wp.float32),
    default_mass: float,
    default_mu: float,
    wheel_radius: float,
    car_length: float,
    car_width: float,
    max_agents_per_world: int,
    global_seed: int,
    spawn_margin: float,
    clearance: float,
    max_rejects: int,
):
    i = wp.tid()
    if reset_mask[i] == 0:
        return

    wid = world_id[i]
    slot = slot_id[i]
    tid = track_id[i]
    ep = episode_id[i] + 1
    episode_id[i] = ep
    seed = _mix_seed(global_seed, wid, slot, ep)
    rng = wp.rand_init(seed)

    a = offsets[tid]
    b = offsets[tid + 1]
    count = b - a
    if count < 2:
        return

    radius = 0.5 * wp.sqrt(car_length * car_length + car_width * car_width) + clearance
    placed = int(0)
    px = float(0.0)
    py = float(0.0)
    pyaw = float(0.0)
    # project_window seeds are track-local; atlas reads use a + local.
    pseg_local = int(0)
    # Keep spawn speed dynamic for Warp (no constant mutated in a dynamic loop).
    speed = wp.randf(rng) * 0.0 + 0.5
    attempt = int(0)
    while attempt < max_rejects:
        if placed != 0:
            attempt = max_rejects
        else:
            segment_local = int(wp.randf(rng) * float(count))
            if segment_local >= count:
                segment_local = count - 1
            segment = a + segment_local
            following = a + ((segment_local + 1) % count)
            alpha = wp.randf(rng)
            wr = width_right[segment] - spawn_margin
            wl = width_left[segment] - spawn_margin
            if wr < 0.0:
                wr = 0.0
            if wl < 0.0:
                wl = 0.0
            lateral = -wr + wp.randf(rng) * (wr + wl)
            p0 = point[segment]
            p1 = point[following]
            center = wp.vec2f(
                p0[0] + alpha * (p1[0] - p0[0]),
                p0[1] + alpha * (p1[1] - p0[1]),
            )
            nxy = normal[segment]
            cand_x = center[0] + lateral * nxy[0]
            cand_y = center[1] + lateral * nxy[1]
            txy = tangent[segment]
            cand_yaw = wp.atan2(txy[1], txy[0]) + (wp.randf(rng) * 0.3 - 0.15)

            clear = int(1)
            base = wid * max_agents_per_world
            other_slot = int(0)
            while other_slot < max_agents_per_world:
                if clear != 0:
                    j = base + other_slot
                    if j != i and active[j] != 0 and reset_mask[j] == 0:
                        dx = cand_x - vehicles.x[j]
                        dy = cand_y - vehicles.y[j]
                        if dx * dx + dy * dy < (2.0 * radius) * (2.0 * radius):
                            clear = int(0)
                other_slot = other_slot + 1
            if clear != 0:
                px = cand_x
                py = cand_y
                pyaw = cand_yaw
                pseg_local = segment_local
                speed = 0.5 + wp.randf(rng) * 2.5
                placed = int(1)
            attempt = attempt + 1

    if placed == 0:
        pseg_local = 0
        px = point[a][0]
        py = point[a][1]
        txy = tangent[a]
        pyaw = wp.atan2(txy[1], txy[0])
        speed = wp.randf(rng) * 0.0 + 0.5

    vehicle = VehicleLocal()
    vehicle.x = px
    vehicle.y = py
    vehicle.yaw = pyaw
    vehicle.vx = speed
    vehicle = clear_vehicle(vehicle)
    vehicle.vx = speed
    store_vehicle(vehicles, i, vehicle)
    vehicles.mass[i] = default_mass
    vehicles.mu[i] = default_mu
    vehicles.drive_scale[i] = 1.0
    vehicles.steer_scale[i] = 1.0
    vehicles.accel_scale[i] = 1.0
    vehicles.vmax_scale[i] = 1.0
    w = speed / wheel_radius
    vehicles.omega[i] = wp.vec4f(w, w, w, w)

    fr = project_window(
        wp.vec2f(px, py),
        pseg_local,
        point,
        tangent,
        normal,
        segment_length,
        cumulative_length,
        width_left,
        width_right,
        count,
        a,
    )
    frenet_segment[i] = fr.segment
    frenet_s[i] = fr.s
    prev_s[i] = fr.s
    frenet_ey[i] = fr.ey
    boundary_distance[i] = fr.boundary_distance
    progress_s[i] = 0.0
    episode_step[i] = 0
    stalled_steps[i] = 0
    done[i] = wp.uint8(0)
    timeout[i] = wp.uint8(0)
    # Keep reset_mask=1 for this tick so GRU clear / style resample see it.
    active[i] = wp.uint8(1)
    trainable[i] = wp.uint8(1)
    contact[i] = wp.uint8(0)
    wall_contact[i] = wp.uint8(0)
    contact_counterpart[i] = -1
    contact_closing_speed[i] = 0.0
    # Proprioception reads these on the first post-respawn observation.
    executed_long_0[i] = 0.0
    executed_long_1[i] = 0.0
    executed_steer_0[i] = 0.0
    executed_steer_1[i] = 0.0
    executed_steer_2[i] = 0.0
    executed_steer_3[i] = 0.0

    sensor_noise_seed[i] = _mix_seed(global_seed, wid, slot, ep + 99)
    lidar_range_noise_std[i] = wp.randf(rng) * 0.05
    lidar_dropout_prob[i] = wp.randf(rng) * 0.02
    lidar_far_dropout_prob[i] = wp.randf(rng) * 0.08
    lidar_angle_bias[i] = wp.randf(rng) * 0.02 - 0.01
    lidar_extrinsic_x[i] = wp.randf(rng) * 0.04 - 0.02
    lidar_extrinsic_y[i] = wp.randf(rng) * 0.04 - 0.02
    lidar_extrinsic_yaw[i] = wp.randf(rng) * 0.04 - 0.02
    if wp.randf(rng) < 0.25:
        lidar_sector_width[i] = 8 + int(wp.randf(rng) * 56.0)
        lidar_sector_start[i] = int(wp.randf(rng) * 1081.0)
    else:
        lidar_sector_width[i] = 0
        lidar_sector_start[i] = 0


@wp.kernel(enable_backward=False)
def style_scatter_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    style_table: wp.array2d(dtype=wp.float32),  # [pool, CONDITION_DIM]
    pool_count: int,
    global_seed: int,
    world_id: wp.array(dtype=wp.int32),
    slot_id: wp.array(dtype=wp.int32),
    episode_id: wp.array(dtype=wp.int32),
    out_styles: wp.array2d(dtype=wp.float32),  # [S, CONDITION_DIM]
    drive_scale: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    steer_scale: wp.array(dtype=wp.float32),
    accel_scale: wp.array(dtype=wp.float32),
    vmax_scale: wp.array(dtype=wp.float32),
):
    """Sample a private style row for each reset slot from a fixed style pool."""
    i = wp.tid()
    if reset_mask[i] == 0:
        return
    if pool_count <= 0:
        return
    seed = _mix_seed(global_seed, world_id[i], slot_id[i], episode_id[i] + 17)
    rng = wp.rand_init(seed)
    idx = int(wp.randf(rng) * float(pool_count))
    if idx >= pool_count:
        idx = pool_count - 1
    for c in range(_STYLE_WIDTH):
        out_styles[i, c] = style_table[idx, c]
    drive_scale[i] = out_styles[i, _IDX_DRIVE_SCALE]
    mass[i] = out_styles[i, _IDX_MASS_KG]
    steer_scale[i] = out_styles[i, _IDX_STEER_SCALE]
    accel_scale[i] = out_styles[i, _IDX_ACCEL_SCALE]
    vmax_scale[i] = out_styles[i, _IDX_VMAX_SCALE]

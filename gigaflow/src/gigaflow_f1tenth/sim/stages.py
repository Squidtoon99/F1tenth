"""Fused physics, progress/wall state, and terminal/reset stage kernels."""

from __future__ import annotations

import warp as wp

from gigaflow_f1tenth.sim.geometry import project_window, wrapped_delta
from gigaflow_f1tenth.sim.vehicle import (
    SimParams,
    VehicleBuffers,
    apply_command_and_integrate_env,
    clear_vehicle,
    load_vehicle,
    store_vehicle,
    vehicle_is_finite,
)

wp.set_module_options({"fast_math": True, "enable_backward": False})


# Compile-time upper bound for fused substeps (config control_interval <= this).
MAX_PHYSICS_SUBSTEPS = 20


@wp.kernel(enable_backward=False)
def physics_control_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    actions: wp.array(dtype=wp.vec2f),
    prev_x: wp.array(dtype=wp.float32),
    prev_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    params: SimParams,
    control_interval: int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    vehicle = load_vehicle(vehicles, i)
    prev_x[i] = vehicle.x
    prev_y[i] = vehicle.y
    prev_yaw[i] = vehicle.yaw
    # Bound is compile-time; live cadence is control_interval (≤ MAX_PHYSICS_SUBSTEPS).
    substeps = control_interval
    if substeps > MAX_PHYSICS_SUBSTEPS:
        substeps = MAX_PHYSICS_SUBSTEPS
    vehicle = apply_command_and_integrate_env(
        vehicle,
        actions[i],
        vehicles,
        i,
        params,
        substeps,
    )
    store_vehicle(vehicles, i, vehicle)


@wp.kernel(enable_backward=False)
def progress_wall_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    track_id: wp.array(dtype=wp.int32),
    frenet_segment: wp.array(dtype=wp.int32),
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    prev_s: wp.array(dtype=wp.float32),
    progress_s: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
    wall_contact: wp.array(dtype=wp.uint8),
    rewards: wp.array(dtype=wp.float32),
    point: wp.array(dtype=wp.vec2f),
    tangent: wp.array(dtype=wp.vec2f),
    normal: wp.array(dtype=wp.vec2f),
    segment_length: wp.array(dtype=wp.float32),
    cumulative_length: wp.array(dtype=wp.float32),
    width_left: wp.array(dtype=wp.float32),
    width_right: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    track_length: wp.array(dtype=wp.float32),
    car_half_width: wp.float32,
    width_scale: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if active[i] == 0:
        rewards[i] = 0.0
        return
    tid = track_id[i]
    offset = offsets[tid]
    count = offsets[tid + 1] - offset
    length = track_length[tid]
    vehicle = load_vehicle(vehicles, i)
    fr = project_window(
        wp.vec2f(vehicle.x, vehicle.y),
        frenet_segment[i],
        point,
        tangent,
        normal,
        segment_length,
        cumulative_length,
        width_left,
        width_right,
        count,
        offset,
    )
    ds = wrapped_delta(fr.s - prev_s[i], length)
    # Match CPU progress_delta: clamp to a fraction of track length per step.
    max_ds = 0.05 * length
    if ds > max_ds:
        ds = max_ds
    if ds < -max_ds:
        ds = -max_ds
    prev_s[i] = fr.s
    frenet_s[i] = fr.s
    frenet_ey[i] = fr.ey
    frenet_segment[i] = fr.segment
    progress_s[i] = progress_s[i] + ds
    scale = width_scale[i]
    boundary_distance[i] = wp.min(
        fr.width_left * scale - fr.ey, fr.width_right * scale + fr.ey
    )
    # Footprint wall contact when scaled corridor clearance < half-width.
    wall = wp.uint8(0)
    if boundary_distance[i] < car_half_width:
        wall = wp.uint8(1)
    wall_contact[i] = wall
    # Baseline progress reward (private conditioning owned by later phase).
    rewards[i] = ds


@wp.kernel(enable_backward=False)
def terminal_mask_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    trainable: wp.array(dtype=wp.uint8),
    done: wp.array(dtype=wp.uint8),
    timeout: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    episode_step: wp.array(dtype=wp.int32),
    episode_horizon: wp.array(dtype=wp.int32),
    boundary_distance: wp.array(dtype=wp.float32),
    wall_contact: wp.array(dtype=wp.uint8),
    contact: wp.array(dtype=wp.uint8),
    contact_closing_speed: wp.array(dtype=wp.float32),
    stalled_steps: wp.array(dtype=wp.int32),
    async_respawn: wp.int32,
    sync_no_respawn: wp.int32,
    car_half_width: wp.float32,
    catastrophic_speed: wp.float32,
    stall_speed: wp.float32,
    max_stall_steps: wp.int32,
):
    i = wp.tid()
    done[i] = wp.uint8(0)
    timeout[i] = wp.uint8(0)
    reset_mask[i] = wp.uint8(0)
    if active[i] == 0:
        return
    # Static opponents (trainable=0) are pinned by the caller every step and
    # never enter the terminal/respawn lifecycle: without this guard an
    # untouched episode_horizon would read as an immediate horizon hit and
    # respawn (move) a slot that must stay put.
    if trainable[i] == 0:
        return
    episode_step[i] = episode_step[i] + 1
    vehicle = load_vehicle(vehicles, i)
    speed = wp.sqrt(vehicle.vx * vehicle.vx + vehicle.vy * vehicle.vy)
    if speed < stall_speed:
        stalled_steps[i] = stalled_steps[i] + 1
    else:
        stalled_steps[i] = 0

    terminal = wp.int32(0)
    if not vehicle_is_finite(vehicle):
        terminal = 1
    # Full OOB: center beyond corridor.
    if boundary_distance[i] < -car_half_width:
        terminal = 1
    if wall_contact[i] != 0 and speed > catastrophic_speed:
        terminal = 1
    if contact[i] != 0 and contact_closing_speed[i] > catastrophic_speed:
        terminal = 1
    if stalled_steps[i] >= max_stall_steps:
        terminal = 1

    hit_horizon = wp.int32(0)
    if episode_step[i] >= episode_horizon[i]:
        hit_horizon = 1

    # Without async respawn the row is deactivated after the reward launch
    # (deactivate_terminal_rows), so the terminal transition keeps its reward.
    if terminal != 0:
        # True terminal wins over horizon truncation (no value bootstrap).
        done[i] = wp.uint8(1)
        timeout[i] = wp.uint8(0)
        if async_respawn != 0 and sync_no_respawn == 0:
            reset_mask[i] = wp.uint8(1)
    elif hit_horizon != 0:
        # Horizon truncation: done+timeout so GAE bootstraps; async respawn.
        done[i] = wp.uint8(1)
        timeout[i] = wp.uint8(1)
        if async_respawn != 0 and sync_no_respawn == 0:
            reset_mask[i] = wp.uint8(1)


@wp.kernel(enable_backward=False)
def deactivate_terminal_rows_kernel(
    active: wp.array(dtype=wp.uint8),
    trainable: wp.array(dtype=wp.uint8),
    done: wp.array(dtype=wp.uint8),
):
    """Evaluation path: retire terminal rows once their reward is recorded."""
    i = wp.tid()
    if done[i] == 0:
        return
    active[i] = wp.uint8(0)
    trainable[i] = wp.uint8(0)


@wp.kernel(enable_backward=False)
def push_command_history_kernel(
    active: wp.array(dtype=wp.uint8),
    applied_effort: wp.array(dtype=wp.float32),
    steer: wp.array(dtype=wp.float32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_long_2: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    executed_steer_4: wp.array(dtype=wp.float32),
    executed_steer_5: wp.array(dtype=wp.float32),
    executed_steer_6: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if active[i] == 0:
        return
    executed_long_2[i] = executed_long_1[i]
    executed_long_1[i] = executed_long_0[i]
    executed_long_0[i] = applied_effort[i]
    executed_steer_6[i] = executed_steer_5[i]
    executed_steer_5[i] = executed_steer_4[i]
    executed_steer_4[i] = executed_steer_3[i]
    executed_steer_3[i] = executed_steer_2[i]
    executed_steer_2[i] = executed_steer_1[i]
    executed_steer_1[i] = executed_steer_0[i]
    executed_steer_0[i] = steer[i]


@wp.func
def deactivate_vehicle(vehicles: VehicleBuffers, i: wp.int32):
    vehicle = load_vehicle(vehicles, i)
    vehicle = clear_vehicle(vehicle)
    vehicle.x = 0.0
    vehicle.y = 0.0
    vehicle.yaw = 0.0
    vehicle.vx = 0.0
    store_vehicle(vehicles, i, vehicle)


@wp.kernel(enable_backward=False)
def pin_static_slots_kernel(
    static_mask: wp.array(dtype=wp.uint8),
    pin_x: wp.array(dtype=wp.float32),
    pin_y: wp.array(dtype=wp.float32),
    pin_yaw: wp.array(dtype=wp.float32),
    pin_segment: wp.array(dtype=wp.int32),
    pin_s: wp.array(dtype=wp.float32),
    pin_ey: wp.array(dtype=wp.float32),
    pin_boundary: wp.array(dtype=wp.float32),
    vehicles: VehicleBuffers,
    prev_x: wp.array(dtype=wp.float32),
    prev_y: wp.array(dtype=wp.float32),
    prev_yaw: wp.array(dtype=wp.float32),
    prev_s: wp.array(dtype=wp.float32),
    frenet_segment: wp.array(dtype=wp.int32),
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
    progress_s: wp.array(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    episode_step: wp.array(dtype=wp.int32),
    stalled_steps: wp.array(dtype=wp.int32),
    done: wp.array(dtype=wp.uint8),
    timeout: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    active: wp.array(dtype=wp.uint8),
    trainable: wp.array(dtype=wp.uint8),
    i_brake_max_a: wp.float32,
):
    i = wp.tid()
    if static_mask[i] == 0:
        return
    vehicles.x[i] = pin_x[i]
    vehicles.y[i] = pin_y[i]
    vehicles.yaw[i] = pin_yaw[i]
    prev_x[i] = pin_x[i]
    prev_y[i] = pin_y[i]
    prev_yaw[i] = pin_yaw[i]
    vehicles.vx[i] = 0.0
    vehicles.vy[i] = 0.0
    vehicles.yaw_rate[i] = 0.0
    vehicles.steer[i] = 0.0
    vehicles.steer_logical[i] = 0.0
    vehicles.i_drive[i] = 0.0
    vehicles.i_brake[i] = i_brake_max_a
    vehicles.i_drive_cmd[i] = 0.0
    vehicles.i_brake_cmd[i] = i_brake_max_a
    vehicles.applied_effort[i] = -1.0
    vehicles.ax[i] = 0.0
    vehicles.ay[i] = 0.0
    frenet_segment[i] = pin_segment[i]
    frenet_s[i] = pin_s[i]
    prev_s[i] = pin_s[i]
    frenet_ey[i] = pin_ey[i]
    boundary_distance[i] = pin_boundary[i]
    progress_s[i] = 0.0
    rewards[i] = 0.0
    episode_step[i] = 0
    stalled_steps[i] = 0
    done[i] = wp.uint8(0)
    timeout[i] = wp.uint8(0)
    reset_mask[i] = wp.uint8(0)
    active[i] = wp.uint8(1)
    trainable[i] = wp.uint8(0)


@wp.kernel(enable_backward=False)
def snapshot_transition_kernel(
    done: wp.array(dtype=wp.uint8),
    timeout: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    contact: wp.array(dtype=wp.uint8),
    wall_contact: wp.array(dtype=wp.uint8),
    snap_done: wp.array(dtype=wp.uint8),
    snap_timeout: wp.array(dtype=wp.uint8),
    snap_reset: wp.array(dtype=wp.uint8),
    snap_contact: wp.array(dtype=wp.uint8),
    snap_wall: wp.array(dtype=wp.uint8),
):
    i = wp.tid()
    snap_done[i] = done[i]
    snap_timeout[i] = timeout[i]
    snap_reset[i] = reset_mask[i]
    snap_contact[i] = contact[i]
    snap_wall[i] = wall_contact[i]


@wp.kernel(enable_backward=False)
def post_contact_and_geom_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    trainable: wp.array(dtype=wp.uint8),
    track_id: wp.array(dtype=wp.int32),
    frenet_segment: wp.array(dtype=wp.int32),
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    prev_s: wp.array(dtype=wp.float32),
    progress_s: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
    wall_contact: wp.array(dtype=wp.uint8),
    rewards: wp.array(dtype=wp.float32),
    point: wp.array(dtype=wp.vec2f),
    tangent: wp.array(dtype=wp.vec2f),
    normal: wp.array(dtype=wp.vec2f),
    segment_length: wp.array(dtype=wp.float32),
    cumulative_length: wp.array(dtype=wp.float32),
    width_left: wp.array(dtype=wp.float32),
    width_right: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    track_length: wp.array(dtype=wp.float32),
    car_half_width: wp.float32,
    width_scale: wp.array(dtype=wp.float32),
    applied_effort: wp.array(dtype=wp.float32),
    steer: wp.array(dtype=wp.float32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_long_2: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    executed_steer_4: wp.array(dtype=wp.float32),
    executed_steer_5: wp.array(dtype=wp.float32),
    executed_steer_6: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.uint8),
    timeout: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    episode_step: wp.array(dtype=wp.int32),
    episode_horizon: wp.array(dtype=wp.int32),
    contact: wp.array(dtype=wp.uint8),
    contact_closing_speed: wp.array(dtype=wp.float32),
    stalled_steps: wp.array(dtype=wp.int32),
    async_respawn: wp.int32,
    sync_no_respawn: wp.int32,
    catastrophic_speed: wp.float32,
    stall_speed: wp.float32,
    max_stall_steps: wp.int32,
    out_track_length: wp.array(dtype=wp.float32),
    out_half_width: wp.array(dtype=wp.float32),
    out_tangent_yaw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    # progress_wall
    if active[i] == 0:
        rewards[i] = 0.0
        done[i] = wp.uint8(0)
        timeout[i] = wp.uint8(0)
        reset_mask[i] = wp.uint8(0)
        out_track_length[i] = 1.0
        out_half_width[i] = 1.0
        out_tangent_yaw[i] = 0.0
        return
    tid = track_id[i]
    offset = offsets[tid]
    count = offsets[tid + 1] - offset
    length = track_length[tid]
    vehicle = load_vehicle(vehicles, i)
    fr = project_window(
        wp.vec2f(vehicle.x, vehicle.y),
        frenet_segment[i],
        point,
        tangent,
        normal,
        segment_length,
        cumulative_length,
        width_left,
        width_right,
        count,
        offset,
    )
    ds = wrapped_delta(fr.s - prev_s[i], length)
    max_ds = 0.05 * length
    if ds > max_ds:
        ds = max_ds
    if ds < -max_ds:
        ds = -max_ds
    prev_s[i] = fr.s
    frenet_s[i] = fr.s
    frenet_ey[i] = fr.ey
    frenet_segment[i] = fr.segment
    progress_s[i] = progress_s[i] + ds
    scale = width_scale[i]
    boundary_distance[i] = wp.min(
        fr.width_left * scale - fr.ey, fr.width_right * scale + fr.ey
    )
    wall = wp.uint8(0)
    if boundary_distance[i] < car_half_width:
        wall = wp.uint8(1)
    wall_contact[i] = wall
    rewards[i] = ds

    # push_command_history
    executed_long_2[i] = executed_long_1[i]
    executed_long_1[i] = executed_long_0[i]
    executed_long_0[i] = applied_effort[i]
    executed_steer_6[i] = executed_steer_5[i]
    executed_steer_5[i] = executed_steer_4[i]
    executed_steer_4[i] = executed_steer_3[i]
    executed_steer_3[i] = executed_steer_2[i]
    executed_steer_2[i] = executed_steer_1[i]
    executed_steer_1[i] = executed_steer_0[i]
    executed_steer_0[i] = steer[i]

    # terminal_mask
    done[i] = wp.uint8(0)
    timeout[i] = wp.uint8(0)
    reset_mask[i] = wp.uint8(0)
    if trainable[i] != 0:
        episode_step[i] = episode_step[i] + 1
        speed = wp.sqrt(vehicle.vx * vehicle.vx + vehicle.vy * vehicle.vy)
        if speed < stall_speed:
            stalled_steps[i] = stalled_steps[i] + 1
        else:
            stalled_steps[i] = 0
        terminal = wp.int32(0)
        if not vehicle_is_finite(vehicle):
            terminal = 1
        if boundary_distance[i] < -car_half_width:
            terminal = 1
        if wall_contact[i] != 0 and speed > catastrophic_speed:
            terminal = 1
        if contact[i] != 0 and contact_closing_speed[i] > catastrophic_speed:
            terminal = 1
        if stalled_steps[i] >= max_stall_steps:
            terminal = 1
        hit_horizon = wp.int32(0)
        if episode_step[i] >= episode_horizon[i]:
            hit_horizon = 1
        if terminal != 0:
            done[i] = wp.uint8(1)
            timeout[i] = wp.uint8(0)
            if async_respawn != 0 and sync_no_respawn == 0:
                reset_mask[i] = wp.uint8(1)
        elif hit_horizon != 0:
            done[i] = wp.uint8(1)
            timeout[i] = wp.uint8(1)
            if async_respawn != 0 and sync_no_respawn == 0:
                reset_mask[i] = wp.uint8(1)

    # slot_reward_geom
    a = offsets[tid]
    b = offsets[tid + 1]
    geom_count = b - a
    seg = frenet_segment[i]
    if seg < 0:
        seg = 0
    if seg > geom_count - 1:
        seg = geom_count - 1
    atlas_seg = a + seg
    txy = tangent[atlas_seg]
    out_track_length[i] = track_length[tid]
    out_half_width[i] = (
        0.5 * (width_right[atlas_seg] + width_left[atlas_seg]) * width_scale[i]
    )
    out_tangent_yaw[i] = wp.atan2(txy[1], txy[0])


@wp.kernel(enable_backward=False)
def pack_compact_state_kernel(
    vehicles: VehicleBuffers,
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    progress_s: wp.array(dtype=wp.float32),
    boundary_distance: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.uint8),
    wall_contact: wp.array(dtype=wp.uint8),
    contact_closing_speed: wp.array(dtype=wp.float32),
    lidar_range_noise_std: wp.array(dtype=wp.float32),
    lidar_dropout_prob: wp.array(dtype=wp.float32),
    lidar_far_dropout_prob: wp.array(dtype=wp.float32),
    lidar_angle_bias: wp.array(dtype=wp.float32),
    lidar_extrinsic_x: wp.array(dtype=wp.float32),
    lidar_extrinsic_y: wp.array(dtype=wp.float32),
    lidar_extrinsic_yaw: wp.array(dtype=wp.float32),
    lidar_sector_start: wp.array(dtype=wp.int32),
    lidar_sector_width: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.uint8),
    stalled_steps: wp.array(dtype=wp.int32),
    executed_long_0: wp.array(dtype=wp.float32),
    executed_long_1: wp.array(dtype=wp.float32),
    executed_long_2: wp.array(dtype=wp.float32),
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
    executed_steer_4: wp.array(dtype=wp.float32),
    executed_steer_5: wp.array(dtype=wp.float32),
    executed_steer_6: wp.array(dtype=wp.float32),
    frenet_segment: wp.array(dtype=wp.int32),
    out_state: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    out_state[i, 0] = vehicles.x[i]
    out_state[i, 1] = vehicles.y[i]
    out_state[i, 2] = vehicles.yaw[i]
    out_state[i, 3] = vehicles.vx[i]
    out_state[i, 4] = vehicles.vy[i]
    out_state[i, 5] = vehicles.yaw_rate[i]
    out_state[i, 6] = vehicles.steer[i]
    out_state[i, 7] = vehicles.applied_effort[i]
    out_state[i, 8] = vehicles.ax[i]
    out_state[i, 9] = vehicles.ay[i]
    out_state[i, 10] = frenet_s[i]
    out_state[i, 11] = frenet_ey[i]
    out_state[i, 12] = progress_s[i]
    out_state[i, 13] = boundary_distance[i]
    out_state[i, 14] = wp.float32(contact[i])
    out_state[i, 15] = wp.float32(wall_contact[i])
    out_state[i, 16] = contact_closing_speed[i]
    out_state[i, 17] = vehicles.mass[i]
    out_state[i, 18] = vehicles.mu[i]
    out_state[i, 19] = vehicles.drive_scale[i]
    out_state[i, 20] = vehicles.steer_scale[i]
    out_state[i, 21] = lidar_range_noise_std[i]
    out_state[i, 22] = lidar_dropout_prob[i]
    out_state[i, 23] = lidar_far_dropout_prob[i]
    out_state[i, 24] = lidar_angle_bias[i]
    out_state[i, 25] = lidar_extrinsic_x[i]
    out_state[i, 26] = lidar_extrinsic_y[i]
    out_state[i, 27] = lidar_extrinsic_yaw[i]
    out_state[i, 28] = wp.float32(lidar_sector_start[i])
    out_state[i, 29] = wp.float32(lidar_sector_width[i])
    out_state[i, 30] = wp.float32(active[i])
    out_state[i, 31] = wp.float32(stalled_steps[i])
    out_state[i, 32] = vehicles.accel_scale[i]
    out_state[i, 33] = vehicles.vmax_scale[i]
    out_state[i, 34] = executed_long_0[i]
    out_state[i, 35] = executed_long_1[i]
    out_state[i, 36] = executed_steer_0[i]
    out_state[i, 37] = executed_steer_1[i]
    out_state[i, 38] = executed_steer_2[i]
    out_state[i, 39] = executed_steer_3[i]
    om = vehicles.omega[i]
    out_state[i, 40] = om[0]
    out_state[i, 41] = om[1]
    out_state[i, 42] = om[2]
    out_state[i, 43] = om[3]
    out_state[i, 44] = wp.float32(frenet_segment[i])
    out_state[i, 45] = vehicles.steer_logical[i]
    out_state[i, 46] = vehicles.i_drive[i]
    out_state[i, 47] = vehicles.i_brake[i]
    out_state[i, 48] = vehicles.i_drive_cmd[i]
    out_state[i, 49] = vehicles.i_brake_cmd[i]
    out_state[i, 50] = vehicles.c_f[i]
    out_state[i, 51] = vehicles.c_r[i]
    out_state[i, 52] = vehicles.izz[i]
    out_state[i, 53] = vehicles.steer_gain[i]
    out_state[i, 54] = vehicles.steer_lag[i]
    out_state[i, 55] = executed_long_2[i]
    out_state[i, 56] = executed_steer_4[i]
    out_state[i, 57] = executed_steer_5[i]
    out_state[i, 58] = executed_steer_6[i]

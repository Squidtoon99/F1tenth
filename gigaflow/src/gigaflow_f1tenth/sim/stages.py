"""Fused physics, progress/wall state, and terminal/reset stage kernels."""

from __future__ import annotations

import warp as wp

from gigaflow_f1tenth.sim.geometry import project_window, wrapped_delta
from gigaflow_f1tenth.sim.vehicle import (
    SimParams,
    VehicleBuffers,
    apply_command,
    clear_vehicle,
    integrate_vehicle_substep,
    load_vehicle,
    store_vehicle,
    vehicle_is_finite,
)


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
    vehicle = apply_command(
        vehicle,
        actions[i],
        vehicles.steer_scale[i],
        vehicles.accel_scale[i],
        params,
    )
    mass = vehicles.mass[i]
    mu = vehicles.mu[i]
    drive_scale = vehicles.drive_scale[i]
    vmax_scale = vehicles.vmax_scale[i]
    # Bound is compile-time; live cadence is control_interval (≤ MAX_PHYSICS_SUBSTEPS).
    for s in range(MAX_PHYSICS_SUBSTEPS):
        if s >= control_interval:
            break
        vehicle = integrate_vehicle_substep(
            vehicle, mass, mu, drive_scale, vmax_scale, params
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
    boundary_distance[i] = fr.boundary_distance
    # Footprint wall contact when boundary clearance < half-width.
    wall = wp.uint8(0)
    if fr.boundary_distance < car_half_width:
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
    executed_steer_0: wp.array(dtype=wp.float32),
    executed_steer_1: wp.array(dtype=wp.float32),
    executed_steer_2: wp.array(dtype=wp.float32),
    executed_steer_3: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if active[i] == 0:
        return
    executed_long_1[i] = executed_long_0[i]
    executed_long_0[i] = applied_effort[i]
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

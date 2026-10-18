"""Device-resident Warp kernels for reward geometry, opponents, and racing terms."""

from __future__ import annotations

import warp as wp

from gigaflow_f1tenth.sim.vehicle import VehicleBuffers

_COLLISION_SPEED_COEF = 0.1
_LANE_CENTER_MAX_ERR = 2.0
_REWARD_TOTAL_CLAMP = 100.0
_PASSING_GATE_AHEAD_M = 40.0
_PASSING_GATE_BEHIND_M = 20.0


@wp.func
def _wrap_pi(angle: float) -> float:
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def _wrap_delta(ds: float, length: float) -> float:
    half = 0.5 * length
    if ds > half:
        ds = ds - length
    if ds < -half:
        ds = ds + length
    return ds


@wp.kernel(enable_backward=False)
def slot_reward_geom_kernel(
    active: wp.array(dtype=wp.uint8),
    track_id: wp.array(dtype=wp.int32),
    frenet_segment: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    track_length_tbl: wp.array(dtype=wp.float32),
    tangent: wp.array(dtype=wp.vec2f),
    width_right: wp.array(dtype=wp.float32),
    width_left: wp.array(dtype=wp.float32),
    out_track_length: wp.array(dtype=wp.float32),
    out_half_width: wp.array(dtype=wp.float32),
    out_tangent_yaw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if active[i] == 0:
        out_track_length[i] = 1.0
        out_half_width[i] = 1.0
        out_tangent_yaw[i] = 0.0
        return
    tid = track_id[i]
    a = offsets[tid]
    b = offsets[tid + 1]
    count = b - a
    # frenet_segment is track-local (same convention as project_window).
    seg = frenet_segment[i]
    if seg < 0:
        seg = 0
    if seg > count - 1:
        seg = count - 1
    atlas_seg = a + seg
    txy = tangent[atlas_seg]
    out_track_length[i] = track_length_tbl[tid]
    out_half_width[i] = 0.5 * (width_right[atlas_seg] + width_left[atlas_seg])
    out_tangent_yaw[i] = wp.atan2(txy[1], txy[0])


@wp.kernel(enable_backward=False)
def pack_opponent_progress_kernel(
    active: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    frenet_s: wp.array(dtype=wp.float32),
    progress_ds: wp.array(dtype=wp.float32),
    max_agents_per_world: int,
    n_others: int,
    out_opp_s: wp.array2d(dtype=wp.float32),
    out_opp_ds: wp.array2d(dtype=wp.float32),
    out_opp_act: wp.array2d(dtype=wp.uint8),
):
    i = wp.tid()
    for k in range(n_others):
        out_opp_s[i, k] = 0.0
        out_opp_ds[i, k] = 0.0
        out_opp_act[i, k] = wp.uint8(0)
    if active[i] == 0:
        return
    wid = world_id[i]
    base = wid * max_agents_per_world
    k = int(0)
    for slot in range(max_agents_per_world):
        j = base + slot
        if j == i:
            continue
        if active[j] == 0:
            continue
        if k >= n_others:
            break
        out_opp_s[i, k] = frenet_s[j]
        out_opp_ds[i, k] = progress_ds[j]
        out_opp_act[i, k] = wp.uint8(1)
        k = k + 1


@wp.kernel(enable_backward=False)
def racing_reward_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    progress_ds: wp.array(dtype=wp.float32),
    wall_contact: wp.array(dtype=wp.uint8),
    contact: wp.array(dtype=wp.uint8),
    frenet_s: wp.array(dtype=wp.float32),
    frenet_ey: wp.array(dtype=wp.float32),
    reset_mask: wp.array(dtype=wp.uint8),
    track_length: wp.array(dtype=wp.float32),
    half_width: wp.array(dtype=wp.float32),
    tangent_yaw: wp.array(dtype=wp.float32),
    alpha_collision: wp.array(dtype=wp.float32),
    alpha_boundary: wp.array(dtype=wp.float32),
    alpha_l_center: wp.array(dtype=wp.float32),
    alpha_center_bias: wp.array(dtype=wp.float32),
    alpha_passing: wp.array(dtype=wp.float32),
    opp_s: wp.array2d(dtype=wp.float32),
    opp_ds: wp.array2d(dtype=wp.float32),
    opp_act: wp.array2d(dtype=wp.uint8),
    prev_gate: wp.array2d(dtype=wp.uint8),
    n_others: int,
    dt: float,
    out_total: wp.array(dtype=wp.float32),
    out_progress: wp.array(dtype=wp.float32),
    out_collision: wp.array(dtype=wp.float32),
    out_boundary: wp.array(dtype=wp.float32),
    out_lane_center: wp.array(dtype=wp.float32),
    out_passing: wp.array(dtype=wp.float32),
    out_gate: wp.array2d(dtype=wp.uint8),
):
    i = wp.tid()
    for k in range(n_others):
        out_gate[i, k] = wp.uint8(0)
    if active[i] == 0:
        out_total[i] = 0.0
        out_progress[i] = 0.0
        out_collision[i] = 0.0
        out_boundary[i] = 0.0
        out_lane_center[i] = 0.0
        out_passing[i] = 0.0
        return

    vx = vehicles.vx[i]
    vy = vehicles.vy[i]
    yaw = vehicles.yaw[i]
    speed = wp.sqrt(vx * vx + vy * vy)
    ty = tangent_yaw[i]
    theta_f = _wrap_pi(yaw - ty)
    hw = half_width[i]
    if hw < 1.0e-3:
        hw = 1.0e-3
    x_f_norm = frenet_ey[i] / hw
    progress = progress_ds[i]
    col_flag = float(contact[i])
    bnd_flag = float(wall_contact[i])

    col = -(alpha_collision[i] + _COLLISION_SPEED_COEF * speed) * col_flag
    bnd = -alpha_boundary[i] * bnd_flag

    cos_t = wp.cos(theta_f)
    active_center = 0.0
    if cos_t > 0.5:
        active_center = 1.0
    err = wp.abs(x_f_norm - alpha_center_bias[i])
    if err > _LANE_CENTER_MAX_ERR:
        err = _LANE_CENTER_MAX_ERR
    center = -alpha_l_center[i] * dt * active_center * err

    length = track_length[i]
    es = frenet_s[i]
    eds = progress
    sum_delta = float(0.0)
    gated_count = float(0.0)
    for k in range(n_others):
        if opp_act[i, k] == 0:
            continue
        gap = _wrap_delta(opp_s[i, k] - es, length)
        in_window = int(0)
        if gap <= _PASSING_GATE_AHEAD_M and gap >= -_PASSING_GATE_BEHIND_M:
            in_window = 1
        cont = int(0)
        if prev_gate[i, k] != 0 and reset_mask[i] == 0:
            cont = 1
        gate = 0
        if in_window != 0 or cont != 0:
            gate = 1
        out_gate[i, k] = wp.uint8(gate)
        if gate != 0:
            sum_delta = sum_delta + (eds - opp_ds[i, k])
            gated_count = gated_count + 1.0
    pas = 0.0
    if gated_count > 0.0:
        pas = alpha_passing[i] * (sum_delta / gated_count)

    total = progress + col + bnd + center + pas
    # Hard sanity bound so a future term bug cannot poison PPO advantages.
    if total > _REWARD_TOTAL_CLAMP:
        total = _REWARD_TOTAL_CLAMP
    if total < -_REWARD_TOTAL_CLAMP:
        total = -_REWARD_TOTAL_CLAMP
    out_progress[i] = progress
    out_collision[i] = col
    out_boundary[i] = bnd
    out_lane_center[i] = center
    out_passing[i] = pas
    out_total[i] = total

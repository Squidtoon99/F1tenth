"""N-car spatial broadphase, swept OBB contact, deterministic Jacobi response."""

from __future__ import annotations

import numpy as np
import warp as wp

from gigaflow_f1tenth.sim.vehicle import (
    VehicleBuffers,
    VehicleLocal,
    body_velocity_world,
    load_vehicle,
    store_vehicle,
)


@wp.struct
class ContactParams:
    car_length: wp.float32
    car_width: wp.float32
    restitution: wp.float32
    max_agents_per_world: wp.int32
    max_pairs: wp.int32


@wp.struct
class ContactResult:
    contact: wp.int32
    normal: wp.vec2f
    depth: wp.float32
    closing_speed: wp.float32


@wp.func
def axis_for_box(yaw: wp.float32, axis_index: wp.int32) -> wp.vec2f:
    if axis_index == 0:
        return wp.vec2f(wp.cos(yaw), wp.sin(yaw))
    return wp.vec2f(-wp.sin(yaw), wp.cos(yaw))


@wp.func
def resolve_pair_contact(
    a: VehicleLocal,
    b: VehicleLocal,
    half_length: wp.float32,
    half_width: wp.float32,
) -> ContactResult:
    result = ContactResult()
    delta = wp.vec2f(a.x - b.x, a.y - b.y)
    a_x = axis_for_box(a.yaw, 0)
    a_y = axis_for_box(a.yaw, 1)
    b_x = axis_for_box(b.yaw, 0)
    b_y = axis_for_box(b.yaw, 1)
    best_depth = wp.float32(1.0e30)
    best_normal = wp.vec2f(0.0)
    separated = wp.int32(0)
    for axis_index in range(4):
        axis = a_x
        if axis_index == 1:
            axis = a_y
        elif axis_index == 2:
            axis = b_x
        elif axis_index == 3:
            axis = b_y
        projection = wp.dot(delta, axis)
        a_radius = (
            half_length * wp.abs(wp.dot(a_x, axis))
            + half_width * wp.abs(wp.dot(a_y, axis))
        )
        b_radius = (
            half_length * wp.abs(wp.dot(b_x, axis))
            + half_width * wp.abs(wp.dot(b_y, axis))
        )
        depth = a_radius + b_radius - wp.abs(projection)
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
        relative_velocity = body_velocity_world(a) - body_velocity_world(b)
        result.closing_speed = wp.max(-wp.dot(relative_velocity, best_normal), 0.0)
    return result


@wp.func
def swept_aabb_overlap(
    ax0: wp.float32,
    ay0: wp.float32,
    ax1: wp.float32,
    ay1: wp.float32,
    bx0: wp.float32,
    by0: wp.float32,
    bx1: wp.float32,
    by1: wp.float32,
    half_diag: wp.float32,
) -> wp.int32:
    amin_x = wp.min(ax0, ax1) - half_diag
    amax_x = wp.max(ax0, ax1) + half_diag
    amin_y = wp.min(ay0, ay1) - half_diag
    amax_y = wp.max(ay0, ay1) + half_diag
    bmin_x = wp.min(bx0, bx1) - half_diag
    bmax_x = wp.max(bx0, bx1) + half_diag
    bmin_y = wp.min(by0, by1) - half_diag
    bmax_y = wp.max(by0, by1) + half_diag
    if amax_x < bmin_x or bmax_x < amin_x or amax_y < bmin_y or bmax_y < amin_y:
        return 0
    return 1


@wp.kernel(enable_backward=False)
def clear_contact_accumulators(
    impulse_x: wp.array(dtype=wp.float32),
    impulse_y: wp.array(dtype=wp.float32),
    correction_x: wp.array(dtype=wp.float32),
    correction_y: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.uint8),
    counterpart: wp.array(dtype=wp.int32),
    closing: wp.array(dtype=wp.float32),
    pair_count: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    impulse_x[tid] = 0.0
    impulse_y[tid] = 0.0
    correction_x[tid] = 0.0
    correction_y[tid] = 0.0
    contact[tid] = wp.uint8(0)
    counterpart[tid] = -1
    closing[tid] = 0.0
    if tid == 0:
        pair_count[0] = 0
        overflow[0] = 0


@wp.kernel(enable_backward=False)
def build_world_pairs_kernel(
    active: wp.array(dtype=wp.uint8),
    world_id: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
    prev_x: wp.array(dtype=wp.float32),
    prev_y: wp.array(dtype=wp.float32),
    params: ContactParams,
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    """Deterministic all-pairs within each world using swept AABB filter."""
    i = wp.tid()
    if active[i] == 0:
        return
    half_diag = 0.5 * wp.sqrt(
        params.car_length * params.car_length + params.car_width * params.car_width
    )
    # Contiguous (world, slot) layout: only scan later slots in this world.
    max_a = params.max_agents_per_world
    wid = world_id[i]
    base = wid * max_a
    local_i = i - base
    for local_j in range(local_i + 1, max_a):
        j = base + local_j
        if active[j] == 0:
            continue
        if (
            swept_aabb_overlap(
                prev_x[i],
                prev_y[i],
                x[i],
                y[i],
                prev_x[j],
                prev_y[j],
                x[j],
                y[j],
                half_diag,
            )
            == 0
        ):
            continue
        # Lexicographic pair order (i < j) is already deterministic.
        idx = wp.atomic_add(pair_count, 0, 1)
        if idx >= params.max_pairs:
            overflow[0] = 1
        else:
            pair_i[idx] = i
            pair_j[idx] = j


@wp.kernel(enable_backward=False)
def resolve_pairs_jacobi_kernel(
    vehicles: VehicleBuffers,
    pair_i: wp.array(dtype=wp.int32),
    pair_j: wp.array(dtype=wp.int32),
    pair_count: wp.array(dtype=wp.int32),
    params: ContactParams,
    impulse_x: wp.array(dtype=wp.float32),
    impulse_y: wp.array(dtype=wp.float32),
    correction_x: wp.array(dtype=wp.float32),
    correction_y: wp.array(dtype=wp.float32),
    contact: wp.array(dtype=wp.uint8),
    counterpart: wp.array(dtype=wp.int32),
    closing: wp.array(dtype=wp.float32),
):
    pid = wp.tid()
    count = pair_count[0]
    if pid >= count:
        return
    i = pair_i[pid]
    j = pair_j[pid]
    a = load_vehicle(vehicles, i)
    b = load_vehicle(vehicles, j)
    half_l = 0.5 * params.car_length
    half_w = 0.5 * params.car_width
    result = resolve_pair_contact(a, b, half_l, half_w)
    if result.contact == 0:
        return
    corr = 0.5 * result.depth * result.normal
    wp.atomic_add(correction_x, i, corr[0])
    wp.atomic_add(correction_y, i, corr[1])
    wp.atomic_add(correction_x, j, -corr[0])
    wp.atomic_add(correction_y, j, -corr[1])
    a_world = body_velocity_world(a)
    b_world = body_velocity_world(b)
    relative = wp.dot(a_world - b_world, result.normal)
    close = wp.min(relative, 0.0)
    impulse = -0.5 * (1.0 + params.restitution) * close
    wp.atomic_add(impulse_x, i, impulse * result.normal[0])
    wp.atomic_add(impulse_y, i, impulse * result.normal[1])
    wp.atomic_add(impulse_x, j, -impulse * result.normal[0])
    wp.atomic_add(impulse_y, j, -impulse * result.normal[1])
    contact[i] = wp.uint8(1)
    contact[j] = wp.uint8(1)
    if counterpart[i] < 0 or j < counterpart[i]:
        counterpart[i] = j
    if counterpart[j] < 0 or i < counterpart[j]:
        counterpart[j] = i
    if result.closing_speed > closing[i]:
        closing[i] = result.closing_speed
    if result.closing_speed > closing[j]:
        closing[j] = result.closing_speed


@wp.kernel(enable_backward=False)
def apply_contact_gather_kernel(
    vehicles: VehicleBuffers,
    active: wp.array(dtype=wp.uint8),
    impulse_x: wp.array(dtype=wp.float32),
    impulse_y: wp.array(dtype=wp.float32),
    correction_x: wp.array(dtype=wp.float32),
    correction_y: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if active[i] == 0:
        return
    vehicle = load_vehicle(vehicles, i)
    vehicle.x = vehicle.x + correction_x[i]
    vehicle.y = vehicle.y + correction_y[i]
    world = body_velocity_world(vehicle)
    world = wp.vec2f(world[0] + impulse_x[i], world[1] + impulse_y[i])
    c = wp.cos(vehicle.yaw)
    s = wp.sin(vehicle.yaw)
    vehicle.vx = c * world[0] + s * world[1]
    vehicle.vy = -s * world[0] + c * world[1]
    store_vehicle(vehicles, i, vehicle)


def resolve_pair_contact_numpy(
    ax: float,
    ay: float,
    ayaw: float,
    bx: float,
    by: float,
    byaw: float,
    car_length: float,
    car_width: float,
    avx: float = 0.0,
    avy: float = 0.0,
    bvx: float = 0.0,
    bvy: float = 0.0,
) -> dict[str, float | int | np.ndarray]:
    """CPU reference SAT for focused tests."""
    half_l = 0.5 * car_length
    half_w = 0.5 * car_width
    delta = np.array([ax - bx, ay - by], dtype=np.float64)

    def axes(yaw: float):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c, s]), np.array([-s, c])

    ax_, ay_ = axes(ayaw)
    bx_, by_ = axes(byaw)
    best_depth = 1.0e30
    best_normal = np.zeros(2)
    separated = False
    for axis in (ax_, ay_, bx_, by_):
        projection = float(np.dot(delta, axis))
        a_radius = half_l * abs(float(np.dot(ax_, axis))) + half_w * abs(
            float(np.dot(ay_, axis))
        )
        b_radius = half_l * abs(float(np.dot(bx_, axis))) + half_w * abs(
            float(np.dot(by_, axis))
        )
        depth = a_radius + b_radius - abs(projection)
        if depth <= 0.0:
            separated = True
        if depth < best_depth:
            best_depth = depth
            best_normal = (1.0 if projection >= 0.0 else -1.0) * axis
    if separated:
        return {"contact": 0, "depth": 0.0, "normal": best_normal, "closing_speed": 0.0}

    def body_to_world(yaw, vx, vy):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * vx - s * vy, s * vx + c * vy])

    rel = body_to_world(ayaw, avx, avy) - body_to_world(byaw, bvx, bvy)
    closing = max(float(-np.dot(rel, best_normal)), 0.0)
    return {
        "contact": 1,
        "depth": max(best_depth, 0.0),
        "normal": best_normal,
        "closing_speed": closing,
    }

"""Offline occupancy-map raycasting for pose-free LiDAR analysis."""

from __future__ import annotations

import numpy as np

from f1tenth_env.utils import CorridorDistanceData

RAY_HIT_WALL = 0
RAY_HIT_UNKNOWN = 1
RAY_HIT_MAX_RANGE = 2


def _sample_distance(
    field: CorridorDistanceData,
    x: np.ndarray,
    y: np.ndarray,
    variant_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= variant_id < field.variant_count:
        raise ValueError("variant_id is outside the distance-field bank")
    gx = np.floor((x - field.origin[0]) / field.resolution).astype(np.int64)
    gy = np.floor((y - field.origin[1]) / field.resolution).astype(np.int64)
    inside = (gx >= 0) & (gy >= 0) & (gx < field.width) & (gy < field.height)
    values = np.full(x.shape, np.nan, dtype=np.float64)
    grid = field.distance.reshape(
        field.variant_count, field.height, field.width
    )[variant_id]
    values[inside] = grid[gy[inside], gx[inside]]
    return values, inside


def raycast_distance_field(
    field: CorridorDistanceData,
    pose: tuple[float, float, float] | np.ndarray,
    angles: np.ndarray,
    range_min: float,
    range_max: float,
    *,
    lidar_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    variant_id: int = 0,
    max_steps: int = 512,
    return_hits: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    x, y, yaw = (float(value) for value in pose)
    offset_x, offset_y, offset_yaw = lidar_offset
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    origin_x = x + cosine * offset_x - sine * offset_y
    origin_y = y + sine * offset_x + cosine * offset_y
    ray_angles = yaw + offset_yaw + np.asarray(angles, dtype=np.float64)
    direction_x = np.cos(ray_angles)
    direction_y = np.sin(ray_angles)
    traveled = np.zeros(ray_angles.shape, dtype=np.float64)
    active = np.ones(ray_angles.shape, dtype=bool)
    kinds = np.full(ray_angles.shape, RAY_HIT_MAX_RANGE, dtype=np.int8)
    hit_epsilon = 0.55 * field.resolution
    minimum_step = 0.5 * field.resolution

    for _ in range(max_steps):
        if not active.any():
            break
        index = np.flatnonzero(active)
        px = origin_x + traveled[index] * direction_x[index]
        py = origin_y + traveled[index] * direction_y[index]
        distance, inside = _sample_distance(field, px, py, variant_id)
        outside_index = index[~inside]
        traveled[outside_index] = range_max
        kinds[outside_index] = RAY_HIT_MAX_RANGE
        active[outside_index] = False
        inside_index = index[inside]
        if inside_index.size == 0:
            continue
        inside_distance = distance[inside]
        unknown = inside_distance < 0.0
        kinds[inside_index[unknown]] = RAY_HIT_UNKNOWN
        traveled[inside_index[unknown]] = range_max
        active[inside_index[unknown]] = False
        remaining = inside_index[~unknown]
        if remaining.size == 0:
            continue
        remaining_distance = inside_distance[~unknown]
        hit = remaining_distance <= hit_epsilon
        kinds[remaining[hit]] = RAY_HIT_WALL
        active[remaining[hit]] = False
        advancing = remaining[~hit]
        traveled[advancing] += np.maximum(remaining_distance[~hit], minimum_step)
        reached_maximum = traveled[advancing] >= range_max
        traveled[advancing[reached_maximum]] = range_max
        kinds[advancing[reached_maximum]] = RAY_HIT_MAX_RANGE
        active[advancing[reached_maximum]] = False

    traveled[active] = range_max
    kinds[active] = RAY_HIT_MAX_RANGE
    ranges = np.clip(traveled, range_min, range_max).astype(np.float32)
    if return_hits:
        return ranges, kinds
    return ranges

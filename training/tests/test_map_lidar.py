from __future__ import annotations

import math

import numpy as np

from analysis.map_lidar import (
    RAY_HIT_MAX_RANGE,
    RAY_HIT_UNKNOWN,
    RAY_HIT_WALL,
    raycast_distance_field,
)
from f1tenth_env.utils import CorridorDistanceData, _euclidean_distance_transform


def test_raycast_distance_field_hits_square_walls():
    occupied = np.zeros((201, 201), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    distance = _euclidean_distance_transform(occupied) * 0.05
    field = CorridorDistanceData(
        distance=distance.astype(np.float32).reshape(-1),
        width=201,
        height=201,
        origin=(0.0, 0.0),
        resolution=0.05,
    )
    angles = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2])

    scan, kinds = raycast_distance_field(
        field, (5.0, 5.0, 0.0), angles, 0.02, 30.0, return_hits=True
    )

    np.testing.assert_allclose(scan, 5.0, atol=field.resolution)
    np.testing.assert_array_equal(kinds, RAY_HIT_WALL)


def test_raycast_unknown_cells_return_max_range_not_a_wall():
    occupied = np.zeros((21, 41), dtype=bool)
    occupied[:, 20] = True
    occupied[10, 20] = False
    unknown = np.zeros_like(occupied)
    unknown[:, 21:] = True
    distance = _euclidean_distance_transform(occupied) * 0.05
    distance[unknown & ~occupied] = -1.0
    field = CorridorDistanceData(
        distance=distance.astype(np.float32).reshape(-1),
        width=41,
        height=21,
        origin=(0.0, 0.0),
        resolution=0.05,
    )

    scan, kinds = raycast_distance_field(
        field, (0.5, 0.525, 0.0), np.array([0.0]), 0.02, 30.0, return_hits=True
    )

    assert float(scan[0]) == 30.0
    assert int(kinds[0]) == RAY_HIT_UNKNOWN
    assert RAY_HIT_MAX_RANGE != RAY_HIT_WALL

"""Tests for frontier detection, DFS stack, and A*."""

from __future__ import annotations

import numpy as np

from f1tenth_mapping.mapping_math import (
    DfsGoalStack,
    MapMeta,
    astar,
    cluster_frontiers,
    coverage_ratio,
    frontier_mask,
    grid_from_occupancy,
)


def _make_corridor_map() -> tuple[MapMeta, np.ndarray]:
    """Simple horizontal corridor with unknown beyond walls."""
    width, height = 40, 20
    grid = np.full((height, width), -1, dtype=np.int16)
    grid[5:15, 5:35] = 0  # free corridor
    grid[4, 5:35] = 100  # top wall
    grid[15, 5:35] = 100  # bottom wall
    meta = MapMeta(width=width, height=height, resolution=0.1, origin_x=0.0, origin_y=0.0)
    return meta, grid


def test_frontier_mask_detects_open_ends():
    meta, grid = _make_corridor_map()
    mask = frontier_mask(grid)
    assert mask[5:15, 5].any()
    assert mask[5:15, 34].any()


def test_cluster_frontiers_finds_two_goals():
    meta, grid = _make_corridor_map()
    clusters = cluster_frontiers(grid, meta, min_cluster_size=1)
    assert len(clusters) >= 2


def test_dfs_stack_pop_order():
    stack = DfsGoalStack()
    ref = np.array([0.0, 0.0])
    clusters = [
        type("C", (), {"centroid_xy": np.array([1.0, 0.0]), "size": 5})(),
        type("C", (), {"centroid_xy": np.array([5.0, 0.0]), "size": 5})(),
    ]
    stack.push_clusters(clusters, ref)
    first = stack.pop()
    assert first is not None
    assert np.linalg.norm(first - np.array([1.0, 0.0])) < 0.01


def test_astar_through_corridor():
    meta, grid = _make_corridor_map()
    start = np.array([1.0, 1.0])
    goal = np.array([3.0, 1.0])
    path = astar(grid, meta, start, goal, inflation_cells=1)
    assert path is not None
    assert path.shape[0] >= 2
    assert np.linalg.norm(path[-1] - goal) < 0.3


def test_coverage_ratio_on_partially_known_roi():
    meta, grid = _make_corridor_map()
    grid[5:15, 30:35] = -1  # unknown patch inside ROI
    roi = (5, 14, 5, 34)
    cov = coverage_ratio(grid, roi)
    assert 0.0 < cov < 1.0


def test_grid_from_occupancy_shape():
    data = np.arange(12, dtype=np.int16)
    grid = grid_from_occupancy(data, width=4, height=3)
    assert grid.shape == (3, 4)

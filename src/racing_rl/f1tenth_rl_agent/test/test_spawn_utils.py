"""Tests for gym-compatible spawn sampling."""

import math
from pathlib import Path

import numpy as np
import pytest

from f1tenth_rl_agent.spawn_utils import (
    MapAssets,
    build_free_spawn_mask,
    centerline_yaw_at,
    grid_to_world,
    is_spawn_cell_free,
    sample_gym_valid_pose,
    sample_reset_pose,
    world_to_grid,
)


def _synthetic_map(size: int = 40) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    occ = np.zeros((size, size), dtype=np.float32)
    occ[5:35, 5:35] = 255.0
    return occ, 0.1, (0.0, 0.0, 0.0)


def test_build_free_spawn_mask_shrinks_near_walls():
    occ, res, _ = _synthetic_map()
    mask = build_free_spawn_mask(occ, res, max_dist=0.5)
    assert mask.any()
    assert mask[10, 10]
    assert not mask[5, 10]


def test_grid_world_roundtrip():
    origin = (1.0, 2.0, 0.0)
    res = 0.5
    row, col = 3, 7
    x, y = grid_to_world(row, col, origin, res)
    assert world_to_grid(x, y, origin, res) == (row, col)


def test_centerline_yaw_along_x_axis():
    centerline = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float32)
    yaw = centerline_yaw_at(5.0, 0.0, centerline)
    assert yaw == pytest.approx(0.0, abs=1e-5)


def test_sample_gym_valid_pose_on_track_corridor():
    occ, res, origin = _synthetic_map()
    mask = build_free_spawn_mask(occ, res, max_dist=0.3)
    centerline = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
    w_left = np.full(3, 1.0, dtype=np.float32)
    w_right = np.full(3, 1.0, dtype=np.float32)
    rng = np.random.default_rng(0)
    x, y, yaw = sample_gym_valid_pose(
        centerline, w_left, w_right, mask, origin, res, rng
    )
    assert is_spawn_cell_free(x, y, mask, origin, res)
    assert -math.pi <= yaw <= math.pi


def test_sample_reset_pose_falls_back_without_map():
    centerline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(1)
    x, y, yaw = sample_reset_pose(centerline, rng, map_assets=None)
    assert math.hypot(x - 0.0, y - 0.0) < 1e-5 or math.hypot(x - 10.0, y - 0.0) < 1e-5
    assert yaw == pytest.approx(0.0, abs=1e-5)


def test_sample_reset_pose_uses_map_when_available():
    occ, res, origin = _synthetic_map()
    assets = MapAssets(occupancy_map=occ, resolution=res, origin=origin)
    centerline = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0]], dtype=np.float32)
    w_left = np.full(3, 1.0, dtype=np.float32)
    w_right = np.full(3, 1.0, dtype=np.float32)
    rng = np.random.default_rng(2)
    x, y, yaw = sample_reset_pose(
        centerline,
        rng,
        w_left=w_left,
        w_right=w_right,
        map_assets=assets,
        spawn_max_dist=0.3,
    )
    mask = build_free_spawn_mask(occ, res, max_dist=0.3)
    assert is_spawn_cell_free(x, y, mask, origin, res)
    assert -math.pi <= yaw <= math.pi


def test_iv2026_static_spawn_yaw():
    csv_path = Path(__file__).resolve().parents[2] / "assets" / "IV_2026_SIM_centerline.csv"
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    cl = data[:, :2] if data.ndim == 2 else data
    yaw = centerline_yaw_at(0.0, 2.0, cl)
    assert yaw == pytest.approx(-1.465477, abs=0.01)

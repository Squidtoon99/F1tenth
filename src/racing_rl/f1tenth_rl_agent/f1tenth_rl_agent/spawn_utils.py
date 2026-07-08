"""Gym-compatible spawn sampling for evaluation resets (no ROS imports).

Samples poses on the track corridor (centerline + widths), then requires the
world position to lie on a free occupancy cell after obstacle dilation, matching
f1tenth_gym ``AllMapResetFn`` collision margins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from f1tenth_rl_agent.eval_logic import sample_centerline_pose


@dataclass(frozen=True)
class MapAssets:
    occupancy_map: np.ndarray
    resolution: float
    origin: tuple[float, float, float]


def _load_grayscale_image(path: Path) -> np.ndarray:
    # Flip top-to-bottom via numpy (image row 0 is the top; occupancy grids index
    # from the bottom). Using np.flipud instead of PIL's Transpose enum keeps this
    # working across Pillow versions and needs no cv2.
    from PIL import Image

    img = Image.open(path).convert("L")
    return np.flipud(np.array(img))


def load_map_assets(map_yaml: str | Path) -> MapAssets:
    """Load occupancy grid + metadata from a ROS-style map YAML (matches f1tenth_gym)."""
    yaml_path = Path(map_yaml)
    with yaml_path.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)

    image_name = meta["image"]
    image_path = yaml_path.parent / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Map image not found: {image_path}")

    raw = _load_grayscale_image(image_path).astype(np.float32)
    occ = np.zeros_like(raw)
    occ[raw > 128] = 255.0

    origin = tuple(float(v) for v in meta["origin"])
    return MapAssets(
        occupancy_map=occ,
        resolution=float(meta["resolution"]),
        origin=origin,
    )


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool, copy=True)
    try:
        import cv2

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)
    except ImportError:
        pad = kernel_size // 2
        padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
        h, w = mask.shape
        out = np.zeros((h, w), dtype=bool)
        for di in range(kernel_size):
            for dj in range(kernel_size):
                out |= padded[di : di + h, dj : dj + w].astype(bool)
        return out


def build_free_spawn_mask(
    occupancy_map: np.ndarray,
    resolution: float,
    max_dist: float = 1.0,
) -> np.ndarray:
    """Free cells after obstacle dilation (same logic as gym ``AllMapResetFn``)."""
    dilation_size = max(1, int(max_dist / resolution))
    inverted = (255.0 - occupancy_map).astype(np.uint8)
    dilated = _dilate_mask(inverted > 0, dilation_size)
    dilated_inverted = np.logical_not(dilated)
    return dilated_inverted


def world_to_grid(
    x: float,
    y: float,
    origin: tuple[float, float, float],
    resolution: float,
) -> tuple[int, int]:
    """Map world (x, y) to occupancy indices (row, col) — matches gym ``xy_2_rc``."""
    ox, oy, otheta = origin
    oc, os = math.cos(otheta), math.sin(otheta)
    x_rot = (x - ox) * oc + (y - oy) * os
    y_rot = -(x - ox) * os + (y - oy) * oc
    col = int(x_rot / resolution)
    row = int(y_rot / resolution)
    return row, col


def grid_to_world(
    row: int,
    col: int,
    origin: tuple[float, float, float],
    resolution: float,
) -> tuple[float, float]:
    ox, oy, otheta = origin
    oc, os = math.cos(otheta), math.sin(otheta)
    x_rot = col * resolution
    y_rot = row * resolution
    x = x_rot * oc - y_rot * os + ox
    y = x_rot * os + y_rot * oc + oy
    return x, y


def is_spawn_cell_free(
    x: float,
    y: float,
    spawn_mask: np.ndarray,
    origin: tuple[float, float, float],
    resolution: float,
) -> bool:
    row, col = world_to_grid(x, y, origin, resolution)
    if row < 0 or col < 0 or row >= spawn_mask.shape[0] or col >= spawn_mask.shape[1]:
        return False
    return bool(spawn_mask[row, col])


def centerline_yaw_at_index(centerline: np.ndarray, idx: int) -> float:
    if len(centerline) < 2:
        raise ValueError("centerline must have at least two points")
    next_idx = (idx + 1) % len(centerline)
    dx = float(centerline[next_idx, 0] - centerline[idx, 0])
    dy = float(centerline[next_idx, 1] - centerline[idx, 1])
    if math.hypot(dx, dy) < 1e-6:
        prev_idx = (idx - 1) % len(centerline)
        dx = float(centerline[idx, 0] - centerline[prev_idx, 0])
        dy = float(centerline[idx, 1] - centerline[prev_idx, 1])
    return math.atan2(dy, dx)


def centerline_yaw_at(x: float, y: float, centerline: np.ndarray) -> float:
    idx = int(np.argmin((centerline[:, 0] - x) ** 2 + (centerline[:, 1] - y) ** 2))
    return centerline_yaw_at_index(centerline, idx)


def sample_gym_valid_pose(
    centerline: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    spawn_mask: np.ndarray,
    origin: tuple[float, float, float],
    resolution: float,
    rng: np.random.Generator,
    lateral_margin: float = 0.15,
    max_attempts: int = 64,
) -> tuple[float, float, float]:
    """Sample an on-track pose whose footprint lies on a dilated-free map cell."""
    if len(centerline) < 2:
        raise ValueError("centerline must have at least two points")
    if len(w_left) != len(centerline) or len(w_right) != len(centerline):
        raise ValueError("track width arrays must match centerline length")

    for _ in range(max_attempts):
        idx = int(rng.integers(0, len(centerline)))
        cx, cy = float(centerline[idx, 0]), float(centerline[idx, 1])
        yaw = centerline_yaw_at_index(centerline, idx)
        max_left = float(w_left[idx]) - lateral_margin
        max_right = float(w_right[idx]) - lateral_margin
        if max_left <= 0.0 or max_right <= 0.0:
            continue
        ey = float(rng.uniform(-max_right, max_left))
        nx = -math.sin(yaw)
        ny = math.cos(yaw)
        x = cx + ey * nx
        y = cy + ey * ny
        if is_spawn_cell_free(x, y, spawn_mask, origin, resolution):
            return x, y, yaw
    raise ValueError("no gym-valid on-track spawn found")


def sample_reset_pose(
    centerline: np.ndarray,
    rng: np.random.Generator,
    w_left: np.ndarray | None = None,
    w_right: np.ndarray | None = None,
    map_assets: MapAssets | None = None,
    spawn_max_dist: float = 1.0,
) -> tuple[float, float, float]:
    """Prefer gym-valid corridor spawns; fall back to centerline sampling."""
    if (
        map_assets is not None
        and w_left is not None
        and w_right is not None
        and len(centerline) >= 2
    ):
        mask = build_free_spawn_mask(
            map_assets.occupancy_map,
            map_assets.resolution,
            max_dist=spawn_max_dist,
        )
        try:
            return sample_gym_valid_pose(
                centerline,
                w_left,
                w_right,
                mask,
                map_assets.origin,
                map_assets.resolution,
                rng,
            )
        except ValueError:
            pass
    return sample_centerline_pose(centerline, rng)

"""Shared track geometry helpers for post-processing and validation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def loop_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)))


def arc_length(points: np.ndarray) -> np.ndarray:
    s = np.zeros(points.shape[0], dtype=np.float64)
    for i in range(1, points.shape[0]):
        s[i] = s[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return s


def track_extent(
    points: list[np.ndarray],
    origin_x: float,
    origin_y: float,
    resolution: float,
    img_shape: tuple[int, int],
    pad_m: float = 5.0,
) -> tuple[float, float, float, float]:
    stacked = np.vstack(points)
    x0 = max(origin_x, float(stacked[:, 0].min()) - pad_m)
    x1 = min(origin_x + img_shape[1] * resolution, float(stacked[:, 0].max()) + pad_m)
    y0 = max(origin_y, float(stacked[:, 1].min()) - pad_m)
    y1 = min(origin_y + img_shape[0] * resolution, float(stacked[:, 1].max()) + pad_m)
    return x0, x1, y0, y1


def resample_to_count(values: np.ndarray, target_count: int) -> np.ndarray:
    if values.size == 0 or target_count <= 0:
        return np.array([], dtype=np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, target_count)
    return np.interp(dst, src, values)


def nearest_distances(source: np.ndarray, target: np.ndarray, step: int = 1) -> np.ndarray:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.array([], dtype=np.float64)
    sampled = source[::step]
    dists, _ = cKDTree(target).query(sampled)
    return np.asarray(dists, dtype=np.float64)


def resample_loop(points: np.ndarray, count: int = 400) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0, append=points[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return points.copy()
    samples = np.linspace(0.0, total, count, endpoint=False)
    out = np.zeros((count, 2), dtype=np.float64)
    j = 0
    for i, s in enumerate(samples):
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        out[i] = points[j] * (1.0 - t) + points[(j + 1) % len(points)] * t
    return out


def best_cyclic_mean(ref: np.ndarray, ext: np.ndarray, count: int = 400) -> float:
    ref_r = resample_loop(ref, count)
    ext_r = resample_loop(ext, count)
    rolled = np.stack([np.roll(ext_r, k, axis=0) for k in range(count)])
    errors = np.mean(np.linalg.norm(ref_r[None, :, :] - rolled, axis=2), axis=1)
    return float(np.min(errors))


def centerline_drivable_fraction(
    centerline: np.ndarray,
    grid: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    step: int = 5,
) -> float:
    """Share of centerline samples on drivable cells (free or unknown)."""
    height, width = grid.shape
    hits = 0
    total = 0
    for pt in centerline[::step]:
        col = int((pt[0] - origin_x) / resolution)
        row = int((pt[1] - origin_y) / resolution)
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        total += 1
        if grid[row, col] != 1:
            hits += 1
    return hits / max(total, 1)


def reference_free_cell_fraction(
    reference: np.ndarray,
    grid: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> float:
    """Share of reference points on strictly free cells (frame sanity check)."""
    height, width = grid.shape
    hits = 0
    for pt in reference:
        col = int((pt[0] - origin_x) / resolution)
        row = int((pt[1] - origin_y) / resolution)
        if 0 <= row < height and 0 <= col < width and grid[row, col] == 0:
            hits += 1
    return 100.0 * hits / max(len(reference), 1)

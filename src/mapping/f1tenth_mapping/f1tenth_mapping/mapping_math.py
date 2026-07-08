"""Occupancy grid utilities, frontier detection, A*, and DFS goal ordering."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

UNKNOWN = -1
FREE_MAX = 99
OCCUPIED_MIN = 100


@dataclass(frozen=True)
class MapMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


@dataclass
class FrontierCluster:
    centroid_xy: np.ndarray
    size: int


def grid_from_occupancy(data: np.ndarray, width: int, height: int) -> np.ndarray:
    """Reshape flat occupancy data to ``(height, width)`` row-major."""
    return np.asarray(data, dtype=np.int16).reshape((height, width))


def world_to_grid(x: float, y: float, meta: MapMeta) -> tuple[int, int]:
    col = int((x - meta.origin_x) / meta.resolution)
    row = int((y - meta.origin_y) / meta.resolution)
    return row, col


def grid_to_world(row: int, col: int, meta: MapMeta) -> tuple[float, float]:
    x = meta.origin_x + (col + 0.5) * meta.resolution
    y = meta.origin_y + (row + 0.5) * meta.resolution
    return x, y


def in_bounds(row: int, col: int, meta: MapMeta) -> bool:
    return 0 <= row < meta.height and 0 <= col < meta.width


def is_free(value: int) -> bool:
    return 0 <= value <= FREE_MAX


def is_unknown(value: int) -> bool:
    return value == UNKNOWN


def is_occupied(value: int) -> bool:
    return value >= OCCUPIED_MIN


def free_bounding_box(free_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(row_min, row_max, col_min, col_max)`` inclusive for free cells."""
    rows, cols = np.where(free_mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def frontier_mask(grid: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Boolean mask of free cells adjacent to unknown cells."""
    free = (grid >= 0) & (grid <= FREE_MAX)
    unknown = grid == UNKNOWN
    height, width = grid.shape
    mask = np.zeros_like(free, dtype=bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_unknown = np.zeros_like(unknown)
        r0 = max(0, dr)
        r1 = height + min(0, dr)
        c0 = max(0, dc)
        c1 = width + min(0, dc)
        sr0 = max(0, -dr)
        sr1 = height - max(0, dr)
        sc0 = max(0, -dc)
        sc1 = width - max(0, dc)
        shifted_unknown[r0:r1, c0:c1] = unknown[sr0:sr1, sc0:sc1]
        mask |= free & shifted_unknown

    if roi is not None:
        r0, r1, c0, c1 = roi
        roi_mask = np.zeros_like(mask)
        roi_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        mask &= roi_mask
    return mask


def cluster_frontiers(
    grid: np.ndarray,
    meta: MapMeta,
    min_cluster_size: int = 3,
    roi: tuple[int, int, int, int] | None = None,
) -> list[FrontierCluster]:
    """Connected-component clustering of frontier cells."""
    fmask = frontier_mask(grid, roi)
    height, width = grid.shape
    visited = np.zeros_like(fmask, dtype=bool)
    clusters: list[FrontierCluster] = []

    for row in range(height):
        for col in range(width):
            if not fmask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            cells: list[tuple[int, int]] = []
            visited[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc, meta) or visited[nr, nc] or not fmask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            if len(cells) < min_cluster_size:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for r, c in cells:
                x, y = grid_to_world(r, c, meta)
                xs.append(x)
                ys.append(y)
            centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
            clusters.append(FrontierCluster(centroid_xy=centroid, size=len(cells)))
    return clusters


def coverage_ratio(grid: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """Fraction of ROI cells that are known (not unknown)."""
    r0, r1, c0, c1 = roi
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    known = patch != UNKNOWN
    return float(np.mean(known))


class DfsGoalStack:
    """Depth-first stack of frontier centroids."""

    def __init__(self) -> None:
        self._stack: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._stack)

    def push_clusters(self, clusters: list[FrontierCluster], reference_xy: np.ndarray) -> None:
        """Push cluster centroids sorted farthest-first so nearest is popped first (DFS)."""
        ordered = sorted(
            clusters,
            key=lambda c: -float(np.linalg.norm(c.centroid_xy - reference_xy)),
        )
        for cluster in ordered:
            self._stack.append(cluster.centroid_xy.copy())

    def pop(self) -> np.ndarray | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def clear(self) -> None:
        self._stack.clear()


def astar(
    grid: np.ndarray,
    meta: MapMeta,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    inflation_cells: int = 2,
) -> np.ndarray | None:
    """Plan a path through known free space; returns ``(N, 2)`` world coordinates."""
    start = world_to_grid(float(start_xy[0]), float(start_xy[1]), meta)
    goal = world_to_grid(float(goal_xy[0]), float(goal_xy[1]), meta)
    if not in_bounds(*start, meta) or not in_bounds(*goal, meta):
        return None

    free = (grid >= 0) & (grid <= FREE_MAX)
    occupied = grid >= OCCUPIED_MIN
    if inflation_cells > 0:
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * inflation_cells + 1, 2 * inflation_cells + 1), dtype=bool)
        blocked = binary_dilation(occupied, structure=structure)
        traversable = free & ~blocked
    else:
        traversable = free

    if not traversable[start]:
        # Allow starting from nearest free cell.
        start = _nearest_traversable(start, traversable, meta)
        if start is None:
            return None
    if not traversable[goal]:
        goal = _nearest_traversable(goal, traversable, meta)
        if goal is None:
            return None

    open_heap: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_heap, (0.0, 0, start[0], start[1]))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    counter = 1

    while open_heap:
        _, _, row, col = heapq.heappop(open_heap)
        if (row, col) == goal:
            return _reconstruct_path(came_from, start, goal, meta)
        for nr, nc, step_cost in _neighbors(row, col, meta):
            if not traversable[nr, nc]:
                continue
            tentative = g_score[(row, col)] + step_cost
            if tentative >= g_score.get((nr, nc), math.inf):
                continue
            came_from[(nr, nc)] = (row, col)
            g_score[(nr, nc)] = tentative
            h = math.hypot(nr - goal[0], nc - goal[1])
            heapq.heappush(open_heap, (tentative + h, counter, nr, nc))
            counter += 1
    return None


def _neighbors(row: int, col: int, meta: MapMeta) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for dr, dc, cost in (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ):
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc, meta):
            out.append((nr, nc, cost))
    return out


def _nearest_traversable(
    cell: tuple[int, int],
    traversable: np.ndarray,
    meta: MapMeta,
    max_radius: int = 20,
) -> tuple[int, int] | None:
    row, col = cell
    for radius in range(max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc, meta) and traversable[nr, nc]:
                    return nr, nc
    return None


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    meta: MapMeta,
) -> np.ndarray:
    cur = goal
    cells = [cur]
    while cur != start:
        cur = came_from[cur]
        cells.append(cur)
    cells.reverse()
    pts = np.zeros((len(cells), 2), dtype=np.float64)
    for i, (row, col) in enumerate(cells):
        x, y = grid_to_world(row, col, meta)
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def distance_to_goal(pose_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    return float(np.linalg.norm(goal_xy - pose_xy))

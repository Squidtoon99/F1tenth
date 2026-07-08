#!/usr/bin/env python3
"""Snap a raceline/centerline onto the drivable corridor of a gym occupancy map.

The IV 2026 assets shipped with a raceline-derived centerline whose (x, y) points
sit ~0.3 m from the PNG walls (Oschersleben is ~1.0 m). That makes the car hit
invisible occupancy boundaries while the policy still thinks it is on track.

This script re-centers each raceline sample perpendicular to the path using the
gym distance transform, and writes measured half-widths from the map.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose

from f1tenth_gym.envs.lidar.laser_models import distance_transform, get_dt


def load_map(map_yaml: Path) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    with map_yaml.open("r", encoding="utf-8") as stream:
        meta = yaml.safe_load(stream)
    image_path = map_yaml.parent / meta["image"]
    img = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
    occ = np.array(img).astype(np.float32)
    occ[occ <= 128] = 0.0
    occ[occ > 128] = 255.0
    origin = tuple(float(v) for v in meta["origin"])
    return occ, float(meta["resolution"]), origin


def load_xy_path(path: Path) -> np.ndarray:
    """Load (x, y) from centerline, smooth raceline, or semicolon raceline CSV."""
    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=np.float64,
            comments="#",
        )
        names = data.dtype.names
        if names and "x_m" in names and "y_m" in names:
            xs = np.atleast_1d(data["x_m"])
            ys = np.atleast_1d(data["y_m"])
            return np.stack([xs, ys], axis=-1).astype(np.float64)
    except (ValueError, OSError):
        pass

    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("x_m") or line.startswith("s_m"):
            continue
        delim = ";" if ";" in line else ","
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) < 2:
            continue
        if delim == ";":
            x, y = float(parts[1]), float(parts[2])
        else:
            x, y = float(parts[0]), float(parts[1])
        rows.append((x, y))
    if len(rows) < 2:
        raise ValueError(f"Need at least two points in {path}")
    return np.asarray(rows, dtype=np.float64)


def _world_to_rc(
    x: float, y: float, origin: tuple[float, float, float], resolution: float
) -> tuple[int, int]:
    ox, oy, otheta = origin
    oc, os = math.cos(otheta), math.sin(otheta)
    x_rot = (x - ox) * oc + (y - oy) * os
    y_rot = -(x - ox) * os + (y - oy) * oc
    return int(y_rot / resolution), int(x_rot / resolution)


def _rc_to_world(
    row: int, col: int, origin: tuple[float, float, float], resolution: float
) -> tuple[float, float]:
    ox, oy, otheta = origin
    oc, os = math.cos(otheta), math.sin(otheta)
    x_rot = col * resolution
    y_rot = row * resolution
    x = x_rot * oc - y_rot * os + ox
    y = x_rot * os + y_rot * oc + oy
    return x, y


def measure_widths_nearest_wall(
    xy: np.ndarray,
    yaws: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: tuple[float, float, float],
    search_radius_m: float = 2.0,
    min_half_width_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure variable half-widths to the nearest wall in each lateral half-plane.

    Thin-line gym maps are mostly free space; marching along the normal often
    misses sparse wall pixels. Instead, scan a local window and take the nearest
  obstacle on the left vs right side of the heading.
    """
    h, w = occ.shape
    w_left = np.zeros(len(xy), dtype=np.float64)
    w_right = np.zeros(len(xy), dtype=np.float64)
    rad_px = int(search_radius_m / resolution) + 2

    for i, ((x, y), yaw) in enumerate(zip(xy, yaws)):
        nx, ny = -math.sin(yaw), math.cos(yaw)
        r0, c0 = _world_to_rc(x, y, origin, resolution)
        best_left = search_radius_m
        best_right = search_radius_m

        for dr in range(-rad_px, rad_px + 1):
            for dc in range(-rad_px, rad_px + 1):
                r, c = r0 + dr, c0 + dc
                if r < 0 or c < 0 or r >= h or c >= w:
                    continue
                if occ[r, c] > 0:
                    continue
                wx, wy = _rc_to_world(r, c, origin, resolution)
                dx, dy = wx - x, wy - y
                dist = math.hypot(dx, dy)
                if dist > search_radius_m or dist < 1e-6:
                    continue
                side = dx * nx + dy * ny
                if side > 0.0:
                    best_left = min(best_left, dist)
                elif side < 0.0:
                    best_right = min(best_right, dist)

        w_left[i] = max(best_left, min_half_width_m)
        w_right[i] = max(best_right, min_half_width_m)

    return w_left, w_right


def measure_widths_along_heading(
    xy: np.ndarray,
    yaws: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: tuple[float, float, float],
    max_half_width_m: float = 1.1,
    search_step_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure left/right half-widths without moving (x, y) — keeps path continuous."""
    h, w = occ.shape
    ox, oy, ot = origin
    oc, os = math.cos(ot), math.sin(ot)
    dt = get_dt(occ, resolution)
    search = np.arange(search_step_m, max_half_width_m + 1e-9, search_step_m)

    w_left = np.zeros(len(xy), dtype=np.float64)
    w_right = np.zeros(len(xy), dtype=np.float64)
    for i, ((x, y), yaw) in enumerate(zip(xy, yaws)):
        nx, ny = -math.sin(yaw), math.cos(yaw)
        left_clear = 0.0
        right_clear = 0.0
        for d in search:
            d_left = float(
                distance_transform(x + d * nx, y + d * ny, ox, oy, oc, os, h, w, resolution, dt)
            )
            if d_left <= 0.05:
                break
            left_clear = d
        for d in search:
            d_right = float(
                distance_transform(x - d * nx, y - d * ny, ox, oy, oc, os, h, w, resolution, dt)
            )
            if d_right <= 0.05:
                break
            right_clear = d
        w_left[i] = max(left_clear, 0.05)
        w_right[i] = max(right_clear, 0.05)
    return w_left, w_right


def load_reference_path(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (xy[N,2], yaws[N] or None) from centerline or smooth raceline CSV."""
    try:
        data = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=np.float64,
            comments="#",
        )
        names = data.dtype.names
        if names and "x_m" in names and "y_m" in names:
            xs = np.atleast_1d(data["x_m"])
            ys = np.atleast_1d(data["y_m"])
            xy = np.stack([xs, ys], axis=-1).astype(np.float64)
            yaws = None
            if "psi_rad" in names:
                yaws = np.atleast_1d(data["psi_rad"]).astype(np.float64)
            return xy, yaws
    except (ValueError, OSError):
        pass

    xy = load_xy_path(path)
    return xy, None


def _yaws(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    yaws = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        dx = xy[j, 0] - xy[i, 0]
        dy = xy[j, 1] - xy[i, 1]
        if math.hypot(dx, dy) < 1e-8:
            k = (i - 1) % n
            dx = xy[i, 0] - xy[k, 0]
            dy = xy[i, 1] - xy[k, 1]
        yaws[i] = math.atan2(dy, dx)
    return yaws


def _max_segment_gap_m(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).max())


def snap_centerline(
    xy: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: tuple[float, float, float],
    search_half_width_m: float = 1.5,
    search_step_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = occ.shape
    ox, oy, ot = origin
    oc, os = math.cos(ot), math.sin(ot)
    dt = get_dt(occ, resolution)
    yaws = _yaws(xy)

    snapped = np.zeros_like(xy)
    w_left = np.zeros(len(xy), dtype=np.float64)
    w_right = np.zeros(len(xy), dtype=np.float64)

    offsets = np.arange(-search_half_width_m, search_half_width_m + 1e-9, search_step_m)

    for i, (cx, cy) in enumerate(xy):
        yaw = yaws[i]
        nx, ny = -math.sin(yaw), math.cos(yaw)
        best_dist = -1.0
        best_x, best_y = cx, cy
        for ey in offsets:
            x = cx + ey * nx
            y = cy + ey * ny
            d = float(
                distance_transform(x, y, ox, oy, oc, os, h, w, resolution, dt)
            )
            if d > best_dist:
                best_dist = d
                best_x, best_y = x, y
        snapped[i, 0] = best_x
        snapped[i, 1] = best_y

        left_clear = 0.0
        right_clear = 0.0
        for ey in offsets:
            if ey >= 0.0:
                x = best_x + ey * nx
                y = best_y + ey * ny
                d = float(
                    distance_transform(x, y, ox, oy, oc, os, h, w, resolution, dt)
                )
                left_clear = max(left_clear, d)
            if ey <= 0.0:
                x = best_x + ey * nx
                y = best_y + ey * ny
                d = float(
                    distance_transform(x, y, ox, oy, oc, os, h, w, resolution, dt)
                )
                right_clear = max(right_clear, d)
        w_left[i] = max(left_clear, 0.05)
        w_right[i] = max(right_clear, 0.05)

    return snapped, w_left, w_right


def write_centerline(
    path: Path,
    xy: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
) -> None:
    lines = ["# x_m, y_m, w_tr_right_m, w_tr_left_m\n"]
    for (x, y), wl, wr in zip(xy, w_left, w_right):
        lines.append(f"{x:.6f}, {y:.6f}, {wr:.6f}, {wl:.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def audit_centerline(
    xy: np.ndarray,
    occ: np.ndarray,
    resolution: float,
    origin: tuple[float, float, float],
) -> dict[str, float]:
    h, w = occ.shape
    ox, oy, ot = origin
    oc, os = math.cos(ot), math.sin(ot)
    dt = get_dt(occ, resolution)
    dists = [
        float(distance_transform(x, y, ox, oy, oc, os, h, w, resolution, dt))
        for x, y in xy
    ]
    arr = np.asarray(dists)
    return {
        "min_m": float(arr.min()),
        "median_m": float(np.median(arr)),
        "max_m": float(arr.max()),
        "frac_lt_0_5": float((arr < 0.5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_yaml", type=Path)
    parser.add_argument("reference_path", type=Path, help="raceline or centerline CSV")
    parser.add_argument("out_csv", type=Path)
    parser.add_argument(
        "--snap",
        action="store_true",
        help="laterally re-center each point (can break continuity on self-crossings)",
    )
    parser.add_argument("--search-half-width", type=float, default=1.5)
    parser.add_argument(
        "--fixed-half-width",
        type=float,
        default=None,
        help="use symmetric half-width (m) instead of measuring from map",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=2.0,
        help="local window (m) for variable width measurement",
    )
    args = parser.parse_args()

    occ, res, origin = load_map(args.map_yaml)
    xy, file_yaws = load_reference_path(args.reference_path)
    yaws = file_yaws if file_yaws is not None else _yaws(xy)
    before = audit_centerline(xy, occ, res, origin)

    if args.snap:
        center_xy, w_left, w_right = snap_centerline(
            xy, occ, res, origin, search_half_width_m=args.search_half_width
        )
        mode = "snap"
    elif args.fixed_half_width is not None:
        center_xy = xy
        w_left = np.full(len(xy), args.fixed_half_width, dtype=np.float64)
        w_right = np.full(len(xy), args.fixed_half_width, dtype=np.float64)
        mode = f"fixed {args.fixed_half_width:.2f}m"
    else:
        center_xy = xy
        w_left, w_right = measure_widths_nearest_wall(
            xy, yaws, occ, res, origin, search_radius_m=args.search_radius
        )
        mode = f"keep-xy + nearest-wall widths (r={args.search_radius:.1f}m)"

    after = audit_centerline(center_xy, occ, res, origin)
    write_centerline(args.out_csv, center_xy, w_left, w_right)

    print(f"Wrote {len(center_xy)} points -> {args.out_csv} ({mode})")
    print(f"max segment gap: {_max_segment_gap_m(center_xy):.3f} m")
    print(
        "wall distance before: "
        f"min={before['min_m']:.3f} median={before['median_m']:.3f} "
        f"frac<0.5m={before['frac_lt_0_5']:.1%}"
    )
    print(
        "wall distance after:  "
        f"min={after['min_m']:.3f} median={after['median_m']:.3f} "
        f"frac<0.5m={after['frac_lt_0_5']:.1%}"
    )
    print(
        f"half-width median: right={np.median(w_right):.3f} left={np.median(w_left):.3f}"
    )


if __name__ == "__main__":
    main()

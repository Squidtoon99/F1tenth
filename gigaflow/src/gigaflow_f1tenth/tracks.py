"""Pinned track preparation and packed multi-track atlas."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from gigaflow_f1tenth.config import (
    CAR_LENGTH_M,
    CAR_WIDTH_M,
    ExperimentConfig,
)

CANONICAL_COLUMNS = ("x_m", "y_m", "w_tr_right_m", "w_tr_left_m")
PIN_FILENAME = "track_pin.json"
MANIFEST_FILENAME = "manifest.json"
ATLAS_FILENAME = "atlas.npz"
LICENSE_FILENAME = "LICENSE"
ATTRIBUTION_FILENAME = "ATTRIBUTION.txt"

DEFAULT_LUT_RESOLUTION_M = 0.5
DEFAULT_EDT_RESOLUTION_M = 0.05
DEFAULT_WALL_MARGIN_M = 0.15
DEFAULT_LATERAL_CLEARANCE_M = 0.08
DEFAULT_LONGITUDINAL_CLEARANCE_FACTOR = 2.5
DEFAULT_MAX_LANE_FACTOR = 3
MIN_VEHICLE_CLEARANCE_M = CAR_WIDTH_M
ENDPOINT_TOL_M = 1.0e-3
ZERO_SEGMENT_TOL_M = 1.0e-6

TRACK_PREVIEW_SAMPLES = 20
TRACK_PREVIEW_SAMPLE_DIM = 5
TRACK_PREVIEW_DIM = TRACK_PREVIEW_SAMPLES * TRACK_PREVIEW_SAMPLE_DIM
TRACK_PREVIEW_SPEED_HORIZON_S = 6.0
TRACK_PREVIEW_MIN_LOOKAHEAD_M = 5.0

_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PIN_PATH = _PKG_ROOT / "configs" / PIN_FILENAME


@dataclass(frozen=True)
class TrackManifestEntry:
    track_id: int
    name: str
    source_sha: str
    geometry_hash: str
    length_m: float
    width_mean_m: float
    width_min_m: float
    num_points: int
    valid: bool
    capacity: int = 1
    license: str = "GPL-3.0"
    optional_local: bool = False


@dataclass(frozen=True)
class PackedTrackAtlasView:
    """Ragged CPU/GPU atlas addressed by per-track offsets."""

    num_tracks: int
    offsets: Any  # int32 [T+1]
    centerline_xy: Any  # float32 [N, 2]
    tangents_xy: Any  # float32 [N, 2]
    widths_rl: Any  # float32 [N, 2]  (right, left)
    cum_length: Any  # float32 [N]
    lengths: Any  # float32 [T]
    track_ids: tuple[str, ...]
    lut_offsets: Any  # int32 [T+1]
    nearest_segment_lut: Any  # int32 [L]
    lut_width: Any  # int32 [T]
    lut_height: Any  # int32 [T]
    lut_origin_xy: Any  # float32 [T, 2]
    lut_resolution: Any  # float32 [T]
    edt_offsets: Any  # int32 [T+1]
    edt_distance: Any  # float32 [E]
    edt_width: Any  # int32 [T]
    edt_height: Any  # int32 [T]
    edt_origin_xy: Any  # float32 [T, 2]
    edt_resolution: Any  # float32 [T]
    capacity: Any  # int32 [T]


@runtime_checkable
class TrackAtlas(Protocol):
    def manifest(self) -> tuple[TrackManifestEntry, ...]:
        """Return validated per-track metadata."""

    def view(self) -> PackedTrackAtlasView:
        """Return packed geometry arrays for kernel launches."""

    def sample_track_ids(self, num_worlds: int, seed: int) -> Any:
        """Return int32 [num_worlds] track assignments."""


@dataclass(frozen=True)
class _BuiltTrack:
    name: str
    source_sha: str
    geometry_hash: str
    point: np.ndarray
    tangent: np.ndarray
    widths_rl: np.ndarray
    cum_length: np.ndarray
    length_m: float
    nearest_segment_lut: np.ndarray
    lut_width: int
    lut_height: int
    lut_origin: tuple[float, float]
    lut_resolution: float
    edt_distance: np.ndarray
    edt_width: int
    edt_height: int
    edt_origin: tuple[float, float]
    edt_resolution: float
    capacity: int
    license: str
    optional_local: bool


class NumpyTrackAtlas:
    """Concrete CPU atlas with stratified / length-balanced sampling."""

    def __init__(
        self,
        entries: Sequence[TrackManifestEntry],
        view: PackedTrackAtlasView,
        *,
        sampling: str = "stratified_shuffle",
    ) -> None:
        self._entries = tuple(entries)
        self._view = view
        self._sampling = sampling
        if sampling not in {"stratified_shuffle", "balanced_length"}:
            raise ValueError(f"unsupported track sampling: {sampling}")

    def manifest(self) -> tuple[TrackManifestEntry, ...]:
        return self._entries

    def view(self) -> PackedTrackAtlasView:
        return self._view

    def sample_track_ids(self, num_worlds: int, seed: int) -> np.ndarray:
        return sample_track_ids(
            self._view.num_tracks,
            num_worlds,
            seed,
            sampling=self._sampling,
            lengths=np.asarray(self._view.lengths, dtype=np.float64),
        )


class TrackError(ValueError):
    """Raised when track schema, geometry, or cache validation fails."""


def load_pin(path: str | Path | None = None) -> dict[str, Any]:
    pin_path = Path(path) if path is not None else _DEFAULT_PIN_PATH
    with pin_path.open("r", encoding="utf-8") as fh:
        pin = json.load(fh)
    if "tracks" not in pin or "commit" not in pin:
        raise TrackError(f"invalid track pin file: {pin_path}")
    return pin


def validate_centerline_table(table: Mapping[str, Any]) -> None:
    """Validate canonical x_m,y_m,w_tr_right_m,w_tr_left_m schema."""
    missing = [c for c in CANONICAL_COLUMNS if c not in table]
    if missing:
        raise TrackError(f"centerline missing columns: {missing}")
    cols = {c: np.asarray(table[c], dtype=np.float64) for c in CANONICAL_COLUMNS}
    n = cols["x_m"].shape[0]
    if n < 3:
        raise TrackError("centerline requires at least 3 points")
    for name, arr in cols.items():
        if arr.ndim != 1 or arr.shape[0] != n:
            raise TrackError(f"column {name} must be length-{n} vector")
        if not np.all(np.isfinite(arr)):
            raise TrackError(f"column {name} contains non-finite values")
    if np.any(cols["w_tr_right_m"] <= 0.0) or np.any(cols["w_tr_left_m"] <= 0.0):
        raise TrackError("track widths must be strictly positive")


def centerline_table_from_arrays(
    x_m: Any,
    y_m: Any,
    w_tr_right_m: Any,
    w_tr_left_m: Any,
) -> dict[str, np.ndarray]:
    table = {
        "x_m": np.asarray(x_m, dtype=np.float64),
        "y_m": np.asarray(y_m, dtype=np.float64),
        "w_tr_right_m": np.asarray(w_tr_right_m, dtype=np.float64),
        "w_tr_left_m": np.asarray(w_tr_left_m, dtype=np.float64),
    }
    validate_centerline_table(table)
    return table


def load_centerline_csv(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    raw = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
        comments="#",
        encoding="utf-8",
    )
    if raw.dtype.names is None:
        raise TrackError(f"unparseable centerline CSV: {path}")
    names = {n.strip(): n for n in raw.dtype.names}
    missing = [c for c in CANONICAL_COLUMNS if c not in names]
    if missing:
        raise TrackError(f"{path} missing columns: {missing}")
    table = {c: np.asarray(raw[names[c]], dtype=np.float64) for c in CANONICAL_COLUMNS}
    validate_centerline_table(table)
    return table


def normalize_centerline_table(
    table: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closed-loop-stripped (points, width_right, width_left)."""
    validate_centerline_table(table)
    point = np.stack(
        [np.asarray(table["x_m"], dtype=np.float64),
         np.asarray(table["y_m"], dtype=np.float64)],
        axis=1,
    )
    width_right = np.asarray(table["w_tr_right_m"], dtype=np.float64)
    width_left = np.asarray(table["w_tr_left_m"], dtype=np.float64)
    if point.shape[0] > 2 and np.linalg.norm(point[0] - point[-1]) <= ENDPOINT_TOL_M:
        point = point[:-1]
        width_right = width_right[:-1]
        width_left = width_left[:-1]
    if point.shape[0] < 3:
        raise TrackError("centerline requires at least 3 unique points")
    return point, width_right, width_left


def validate_track_geometry(
    point: np.ndarray,
    width_right: np.ndarray,
    width_left: np.ndarray,
    *,
    min_clearance_m: float = MIN_VEHICLE_CLEARANCE_M,
) -> dict[str, float]:
    """Verify winding, widths, closure, clearance, segments, Frenet continuity."""
    edge = np.roll(point, -1, axis=0) - point
    seg_len = np.linalg.norm(edge, axis=1)
    if np.any(seg_len <= ZERO_SEGMENT_TOL_M):
        raise TrackError("track contains a zero-length segment")
    length = float(seg_len.sum())
    if length <= 0.0:
        raise TrackError("track length must be positive")
    # Closed-loop gap should be a normal segment after endpoint strip.
    wrap = float(np.linalg.norm(point[0] - point[-1]))
    typical = float(np.median(seg_len))
    if wrap > max(5.0 * typical, 1.0):
        raise TrackError(
            f"centerline is not closed: endpoint gap {wrap:.3f} m "
            f"(median segment {typical:.3f} m)"
        )
    shoelace = float(
        0.5 * np.sum(point[:, 0] * np.roll(point[:, 1], -1)
                     - np.roll(point[:, 0], -1) * point[:, 1])
    )
    if abs(shoelace) < 1.0e-6:
        raise TrackError("track winding is degenerate (zero signed area)")
    clearance = width_left + width_right
    if float(np.min(clearance)) < min_clearance_m:
        raise TrackError(
            f"minimum corridor width {float(np.min(clearance)):.3f} m "
            f"< vehicle clearance {min_clearance_m:.3f} m"
        )
    tangent = edge / seg_len[:, None]
    if not np.all(np.isfinite(tangent)):
        raise TrackError("non-finite Frenet tangents")
    norms = np.linalg.norm(tangent, axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-5):
        raise TrackError("Frenet tangents are not unit length")
    return {
        "length_m": length,
        "width_mean_m": float(np.mean(clearance)),
        "width_min_m": float(np.min(clearance)),
        "signed_area": shoelace,
    }


def geometry_hash(
    point: np.ndarray,
    width_right: np.ndarray,
    width_left: np.ndarray,
) -> str:
    h = hashlib.sha256()
    for arr in (point, width_right, width_left):
        h.update(np.asarray(arr, dtype=np.float32).tobytes(order="C"))
    return h.hexdigest()


def compute_track_capacity(
    length_m: float,
    width_right: np.ndarray,
    width_left: np.ndarray,
    *,
    car_width_m: float = CAR_WIDTH_M,
    car_length_m: float = CAR_LENGTH_M,
    max_agents: int = 8,
    wall_margin_m: float = DEFAULT_WALL_MARGIN_M,
    lateral_clearance_m: float = DEFAULT_LATERAL_CLEARANCE_M,
    longitudinal_clearance_m: float | None = None,
    max_lane_factor: int = DEFAULT_MAX_LANE_FACTOR,
) -> int:
    """Capacity-aware upper bound on active cars for a track."""
    if length_m <= 0.0 or max_agents <= 0:
        return 0
    if longitudinal_clearance_m is None:
        longitudinal_clearance_m = car_length_m * DEFAULT_LONGITUDINAL_CLEARANCE_FACTOR
    usable = float(np.median(width_left + width_right)) - 2.0 * wall_margin_m
    usable = max(0.0, usable)
    denom = car_width_m + lateral_clearance_m
    lane_factor = 1
    if denom > 0.0 and usable >= denom:
        lane_factor = int(math.floor(usable / denom))
        lane_factor = max(1, min(max_lane_factor, lane_factor))
    n_long = max(1, int(math.floor(length_m / longitudinal_clearance_m)))
    return int(min(max_agents, n_long * lane_factor))


def sample_track_ids(
    num_tracks: int,
    num_worlds: int,
    seed: int,
    *,
    sampling: str = "stratified_shuffle",
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Seeded world→track assignment."""
    if num_tracks <= 0 or num_worlds <= 0:
        raise TrackError("num_tracks and num_worlds must be positive")
    rng = np.random.default_rng(seed)
    if sampling == "stratified_shuffle":
        out = np.empty(num_worlds, dtype=np.int32)
        pos = 0
        n_full = num_worlds // num_tracks
        for _ in range(n_full):
            out[pos : pos + num_tracks] = rng.permutation(num_tracks).astype(np.int32)
            pos += num_tracks
        rem = num_worlds - pos
        if rem:
            out[pos:] = rng.choice(num_tracks, size=rem, replace=False).astype(np.int32)
        return out
    if sampling == "balanced_length":
        if lengths is None or lengths.shape[0] != num_tracks:
            raise TrackError("balanced_length sampling requires per-track lengths")
        weights = np.asarray(lengths, dtype=np.float64)
        weights = np.clip(weights, 1.0e-6, None)
        weights = weights / weights.sum()
        return rng.choice(num_tracks, size=num_worlds, replace=True, p=weights).astype(
            np.int32
        )
    raise TrackError(f"unsupported track sampling: {sampling}")


def sample_active_counts(
    track_ids: np.ndarray,
    capacities: np.ndarray,
    *,
    density_bins: Sequence[str] = ("sparse", "medium", "dense"),
    solo_world_fraction: float = 0.05,
    max_agents: int,
    seed: int,
) -> np.ndarray:
    """Sample per-world active-car counts up to per-track capacity."""
    track_ids = np.asarray(track_ids, dtype=np.int32)
    capacities = np.asarray(capacities, dtype=np.int32)
    n = int(track_ids.shape[0])
    rng = np.random.default_rng(seed)
    counts = np.ones(n, dtype=np.int32)
    bins = tuple(density_bins) if density_bins else ("medium",)
    for i in range(n):
        cap = int(max(1, min(max_agents, capacities[int(track_ids[i])])))
        if rng.random() < solo_world_fraction:
            counts[i] = 1
            continue
        bin_name = bins[int(rng.integers(0, len(bins)))]
        if bin_name == "pair":
            counts[i] = min(2, cap)
            continue
        if bin_name == "sparse":
            lo, hi = 1, max(1, int(math.ceil(0.35 * cap)))
        elif bin_name == "dense":
            lo, hi = max(1, int(math.floor(0.7 * cap))), cap
        else:
            lo, hi = max(1, int(math.floor(0.35 * cap))), max(
                1, int(math.ceil(0.7 * cap))
            )
        if lo > hi:
            lo, hi = hi, hi
        counts[i] = int(rng.integers(lo, hi + 1))
    return counts


def _edt_squared_1d(f: np.ndarray) -> np.ndarray:
    n = int(f.shape[0])
    v = np.zeros(n, dtype=np.int32)
    z = np.zeros(n + 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
    return d


def euclidean_distance_transform(occupied: np.ndarray) -> np.ndarray:
    """Distance in cells from each free cell to the nearest occupied cell."""
    if occupied.dtype != bool:
        occupied = occupied.astype(bool)
    try:
        from scipy import ndimage

        # Identical to the Felzenszwalb 1-D pass below; scipy is much faster
        # on multi-million-cell corridor grids used by the full atlas.
        return ndimage.distance_transform_edt(~occupied).astype(np.float32)
    except ImportError:
        pass
    inf = 1.0e20
    height, width = occupied.shape
    f = np.where(occupied, 0.0, inf).astype(np.float64)
    for y in range(height):
        f[y, :] = _edt_squared_1d(f[y, :])
    for x in range(width):
        f[:, x] = _edt_squared_1d(f[:, x])
    return np.sqrt(f, dtype=np.float64).astype(np.float32)


def compute_track_boundaries(
    centerline: np.ndarray,
    w_tr_left: np.ndarray,
    w_tr_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cl = np.asarray(centerline, dtype=np.float64)
    nxt = np.roll(cl, -1, axis=0)
    tangent = nxt - cl
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.clip(tangent_norm, 1e-8, None)
    tangent = tangent / tangent_norm
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    left = cl + normal * np.asarray(w_tr_left, dtype=np.float64)[:, None]
    right = cl - normal * np.asarray(w_tr_right, dtype=np.float64)[:, None]
    return left.astype(np.float32), right.astype(np.float32)


def _rasterize_polyline(
    occupied: np.ndarray,
    polyline: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    *,
    close_loop: bool = True,
) -> None:
    height, width = occupied.shape
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.shape[0] < 2:
        return
    if close_loop and np.linalg.norm(pts[0] - pts[-1]) > ENDPOINT_TOL_M:
        pts = np.concatenate([pts, pts[:1]], axis=0)
    scale = 1.0 / float(resolution)
    for i in range(pts.shape[0] - 1):
        x0 = int(math.floor((pts[i, 0] - origin[0]) * scale))
        y0 = int(math.floor((pts[i, 1] - origin[1]) * scale))
        x1 = int(math.floor((pts[i + 1, 0] - origin[0]) * scale))
        y1 = int(math.floor((pts[i + 1, 1] - origin[1]) * scale))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if 0 <= x < width and 0 <= y < height:
                occupied[y, x] = True
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy


def build_nearest_segment_lut_with_widths(
    point: np.ndarray,
    width_left: np.ndarray,
    width_right: np.ndarray,
    *,
    lut_resolution: float = DEFAULT_LUT_RESOLUTION_M,
) -> tuple[np.ndarray, int, int, tuple[float, float], float]:
    if lut_resolution <= 0.0:
        raise TrackError("lut_resolution must be positive")
    margin = float(max(width_left.max(), width_right.max()) + lut_resolution)
    minimum = point.min(axis=0) - margin
    maximum = point.max(axis=0) + margin
    lut_width = int(np.ceil((maximum[0] - minimum[0]) / lut_resolution)) + 1
    lut_height = int(np.ceil((maximum[1] - minimum[1]) / lut_resolution)) + 1
    edge = np.roll(point, -1, axis=0) - point
    grid_x = minimum[0] + np.arange(lut_width, dtype=np.float64) * lut_resolution
    grid_y = minimum[1] + np.arange(lut_height, dtype=np.float64) * lut_resolution
    gx, gy = np.meshgrid(grid_x, grid_y)
    query = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1)
    edge_sq = np.sum(edge * edge, axis=1)
    lut = np.empty(query.shape[0], dtype=np.int32)
    for start in range(0, query.shape[0], 4096):
        sample = query[start : start + 4096]
        delta = sample[:, None, :] - point[None, :, :]
        alpha = np.clip(
            np.sum(delta * edge[None, :, :], axis=2) / edge_sq[None, :],
            0.0,
            1.0,
        )
        projection = point[None, :, :] + alpha[:, :, None] * edge[None, :, :]
        distance_sq = np.sum((sample[:, None, :] - projection) ** 2, axis=2)
        lut[start : start + sample.shape[0]] = np.argmin(distance_sq, axis=1)
    return (
        lut,
        lut_width,
        lut_height,
        (float(minimum[0]), float(minimum[1])),
        float(lut_resolution),
    )


def build_corridor_edt(
    point: np.ndarray,
    width_left: np.ndarray,
    width_right: np.ndarray,
    *,
    resolution: float = DEFAULT_EDT_RESOLUTION_M,
) -> tuple[np.ndarray, int, int, tuple[float, float], float]:
    if resolution <= 0.0:
        raise TrackError("edt resolution must be positive")
    left, right = compute_track_boundaries(point, width_left, width_right)
    walls = np.concatenate([left, right], axis=0)
    margin = float(max(np.max(width_left), np.max(width_right)) + resolution)
    minimum = walls.min(axis=0) - margin
    maximum = walls.max(axis=0) + margin
    width = int(np.ceil((maximum[0] - minimum[0]) / resolution)) + 1
    height = int(np.ceil((maximum[1] - minimum[1]) / resolution)) + 1
    occupied = np.zeros((height, width), dtype=bool)
    origin = minimum.astype(np.float64)
    _rasterize_polyline(occupied, left, origin, resolution, close_loop=True)
    _rasterize_polyline(occupied, right, origin, resolution, close_loop=True)
    if not occupied.any():
        raise TrackError("corridor wall rasterization produced an empty grid")
    distance_cells = euclidean_distance_transform(occupied)
    distance = (distance_cells * float(resolution)).astype(np.float32).reshape(-1)
    return (
        np.ascontiguousarray(distance),
        width,
        height,
        (float(minimum[0]), float(minimum[1])),
        float(resolution),
    )


def build_track_arrays(
    table: Mapping[str, Any],
    *,
    name: str,
    source_sha: str,
    lut_resolution: float = DEFAULT_LUT_RESOLUTION_M,
    edt_resolution: float = DEFAULT_EDT_RESOLUTION_M,
    max_agents: int = 8,
    car_width_m: float = CAR_WIDTH_M,
    car_length_m: float = CAR_LENGTH_M,
    license_id: str = "GPL-3.0",
    optional_local: bool = False,
) -> _BuiltTrack:
    point64, width_right64, width_left64 = normalize_centerline_table(table)
    stats = validate_track_geometry(point64, width_right64, width_left64)
    ghash = geometry_hash(point64, width_right64, width_left64)
    edge = np.roll(point64, -1, axis=0) - point64
    seg_len = np.linalg.norm(edge, axis=1)
    tangent = (edge / seg_len[:, None]).astype(np.float32)
    cum = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(seg_len[:-1]))
    ).astype(np.float32)
    point = point64.astype(np.float32)
    widths_rl = np.stack(
        [width_right64.astype(np.float32), width_left64.astype(np.float32)],
        axis=1,
    )
    lut, lut_w, lut_h, lut_origin, lut_res = build_nearest_segment_lut_with_widths(
        point64,
        width_left64,
        width_right64,
        lut_resolution=lut_resolution,
    )
    edt, edt_w, edt_h, edt_origin, edt_res = build_corridor_edt(
        point64,
        width_left64,
        width_right64,
        resolution=edt_resolution,
    )
    capacity = compute_track_capacity(
        stats["length_m"],
        width_right64,
        width_left64,
        car_width_m=car_width_m,
        car_length_m=car_length_m,
        max_agents=max_agents,
    )
    return _BuiltTrack(
        name=name,
        source_sha=source_sha,
        geometry_hash=ghash,
        point=np.ascontiguousarray(point),
        tangent=np.ascontiguousarray(tangent),
        widths_rl=np.ascontiguousarray(widths_rl),
        cum_length=np.ascontiguousarray(cum),
        length_m=float(stats["length_m"]),
        nearest_segment_lut=np.ascontiguousarray(lut),
        lut_width=lut_w,
        lut_height=lut_h,
        lut_origin=lut_origin,
        lut_resolution=lut_res,
        edt_distance=np.ascontiguousarray(edt),
        edt_width=edt_w,
        edt_height=edt_h,
        edt_origin=edt_origin,
        edt_resolution=edt_res,
        capacity=capacity,
        license=license_id,
        optional_local=optional_local,
    )


def pack_built_tracks(
    built: Sequence[_BuiltTrack],
    *,
    sampling: str = "stratified_shuffle",
) -> NumpyTrackAtlas:
    if not built:
        raise TrackError("cannot pack an empty track list")
    offsets = [0]
    lut_offsets = [0]
    edt_offsets = [0]
    points: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    cums: list[np.ndarray] = []
    luts: list[np.ndarray] = []
    edts: list[np.ndarray] = []
    entries: list[TrackManifestEntry] = []
    for i, tr in enumerate(built):
        points.append(tr.point)
        tangents.append(tr.tangent)
        widths.append(tr.widths_rl)
        cums.append(tr.cum_length)
        luts.append(tr.nearest_segment_lut)
        edts.append(tr.edt_distance)
        offsets.append(offsets[-1] + int(tr.point.shape[0]))
        lut_offsets.append(lut_offsets[-1] + int(tr.nearest_segment_lut.shape[0]))
        edt_offsets.append(edt_offsets[-1] + int(tr.edt_distance.shape[0]))
        entries.append(
            TrackManifestEntry(
                track_id=i,
                name=tr.name,
                source_sha=tr.source_sha,
                geometry_hash=tr.geometry_hash,
                length_m=tr.length_m,
                width_mean_m=float(np.mean(tr.widths_rl.sum(axis=1))),
                width_min_m=float(np.min(tr.widths_rl.sum(axis=1))),
                num_points=int(tr.point.shape[0]),
                valid=True,
                capacity=tr.capacity,
                license=tr.license,
                optional_local=tr.optional_local,
            )
        )
    view = PackedTrackAtlasView(
        num_tracks=len(built),
        offsets=np.asarray(offsets, dtype=np.int32),
        centerline_xy=np.concatenate(points, axis=0).astype(np.float32),
        tangents_xy=np.concatenate(tangents, axis=0).astype(np.float32),
        widths_rl=np.concatenate(widths, axis=0).astype(np.float32),
        cum_length=np.concatenate(cums, axis=0).astype(np.float32),
        lengths=np.asarray([t.length_m for t in built], dtype=np.float32),
        track_ids=tuple(t.name for t in built),
        lut_offsets=np.asarray(lut_offsets, dtype=np.int32),
        nearest_segment_lut=np.concatenate(luts, axis=0).astype(np.int32),
        lut_width=np.asarray([t.lut_width for t in built], dtype=np.int32),
        lut_height=np.asarray([t.lut_height for t in built], dtype=np.int32),
        lut_origin_xy=np.asarray([t.lut_origin for t in built], dtype=np.float32),
        lut_resolution=np.asarray([t.lut_resolution for t in built], dtype=np.float32),
        edt_offsets=np.asarray(edt_offsets, dtype=np.int32),
        edt_distance=np.concatenate(edts, axis=0).astype(np.float32),
        edt_width=np.asarray([t.edt_width for t in built], dtype=np.int32),
        edt_height=np.asarray([t.edt_height for t in built], dtype=np.int32),
        edt_origin_xy=np.asarray([t.edt_origin for t in built], dtype=np.float32),
        edt_resolution=np.asarray([t.edt_resolution for t in built], dtype=np.float32),
        capacity=np.asarray([t.capacity for t in built], dtype=np.int32),
    )
    return NumpyTrackAtlas(entries, view, sampling=sampling)


def make_synthetic_oval_table(
    *,
    radius: float = 8.0,
    n: int = 64,
    width: float = 1.1,
) -> dict[str, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return centerline_table_from_arrays(
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.full(n, width),
        np.full(n, width),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_centerline_url(pin: Mapping[str, Any], track_name: str) -> str:
    commit = pin["commit"]
    repo = pin["repo"]
    return (
        f"https://raw.githubusercontent.com/{repo}/{commit}/"
        f"{track_name}/{track_name}_centerline.csv"
    )


def download_pinned_sources(
    cache_dir: str | Path,
    *,
    pin: Mapping[str, Any] | None = None,
    pin_path: str | Path | None = None,
) -> dict[str, Path]:
    """Download checksummed centerlines + license metadata into cache_dir."""
    cache = Path(cache_dir)
    pin_obj = dict(pin) if pin is not None else load_pin(pin_path)
    commit = pin_obj["commit"]
    src_dir = cache / "source" / commit
    src_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = cache / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    attribution = (
        f"Upstream: {pin_obj['repo']}@{pin_obj.get('ref', commit)}\n"
        f"Commit: {commit}\n"
        f"License: {pin_obj.get('license', 'GPL-3.0')}\n"
        f"{pin_obj.get('attribution', '')}\n"
    )
    (meta_dir / ATTRIBUTION_FILENAME).write_text(attribution, encoding="utf-8")
    license_path = meta_dir / LICENSE_FILENAME
    if not license_path.exists():
        license_url = pin_obj.get("license_url")
        if license_url:
            try:
                with urllib.request.urlopen(license_url, timeout=60) as resp:
                    license_path.write_bytes(resp.read())
            except (urllib.error.URLError, TimeoutError):
                license_path.write_text(
                    f"{pin_obj.get('license', 'GPL-3.0')}\n"
                    f"See {license_url}\n",
                    encoding="utf-8",
                )
        else:
            license_path.write_text(
                f"{pin_obj.get('license', 'GPL-3.0')}\n", encoding="utf-8"
            )

    out: dict[str, Path] = {}
    tracks: Mapping[str, str] = pin_obj["tracks"]
    for name, expected in tracks.items():
        dest = src_dir / f"{name}_centerline.csv"
        if dest.exists() and _sha256_file(dest) == expected:
            out[name] = dest
            continue
        url = _raw_centerline_url(pin_obj, name)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TrackError(f"failed to download {name}: {exc}") from exc
        digest = _sha256_bytes(payload)
        if digest != expected:
            raise TrackError(
                f"checksum mismatch for {name}: got {digest}, expected {expected}"
            )
        dest.write_bytes(payload)
        out[name] = dest
    (cache / PIN_FILENAME).write_text(
        json.dumps(pin_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _discover_local_tracks(local_dir: Path) -> dict[str, Path]:
    if not local_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(local_dir.glob("*_centerline.csv")):
        name = path.name[: -len("_centerline.csv")]
        found[name] = path
    return found


def _select_track_sources(
    cfg: ExperimentConfig,
    cache_dir: Path,
    pinned_paths: Mapping[str, Path],
    pin: Mapping[str, Any],
) -> list[tuple[str, Path, str, bool, str]]:
    """Return (name, path, source_sha, optional_local, license)."""
    selected: list[tuple[str, Path, str, bool, str]] = []
    license_id = str(pin.get("license", "GPL-3.0"))
    pinned_names = list(pin["tracks"].keys())
    local_paths = _discover_local_tracks(cache_dir / "local")

    # Smoke / reduced configs: prefer enough local fixtures without network atlas.
    if cfg.tracks.num_tracks < len(pinned_names) and local_paths:
        names = sorted(local_paths)[: cfg.tracks.num_tracks]
        if len(names) == cfg.tracks.num_tracks:
            for name in names:
                path = local_paths[name]
                selected.append(
                    (name, path, _sha256_file(path), True, "local-fixture")
                )
            return selected

    for name in pinned_names:
        selected.append(
            (name, pinned_paths[name], pin["tracks"][name], False, license_id)
        )

    if cfg.tracks.include_optional_local:
        optional = pin.get("optional_local", [])
        for name in optional:
            path = local_paths.get(name)
            if path is None:
                continue
            selected.append(
                (name, path, _sha256_file(path), True, "project-local")
            )

    if not selected:
        raise TrackError("no tracks selected for atlas preparation")
    # Full pinned preparation keeps every checksummed upstream track. Config
    # defaults match PINNED_UPSTREAM_TRACK_COUNT (23); CLI warns on mismatch.
    return selected


def _write_manifest(
    path: Path,
    entries: Sequence[TrackManifestEntry],
    *,
    pin: Mapping[str, Any],
    sampling: str,
) -> None:
    payload = {
        "pin_commit": pin.get("commit"),
        "pin_ref": pin.get("ref"),
        "pin_repo": pin.get("repo"),
        "license": pin.get("license"),
        "sampling": sampling,
        "num_tracks": len(entries),
        "tracks": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_atlas_npz(path: Path, view: PackedTrackAtlasView) -> None:
    np.savez_compressed(
        path,
        offsets=view.offsets,
        centerline_xy=view.centerline_xy,
        tangents_xy=view.tangents_xy,
        widths_rl=view.widths_rl,
        cum_length=view.cum_length,
        lengths=view.lengths,
        track_ids=np.asarray(view.track_ids),
        lut_offsets=view.lut_offsets,
        nearest_segment_lut=view.nearest_segment_lut,
        lut_width=view.lut_width,
        lut_height=view.lut_height,
        lut_origin_xy=view.lut_origin_xy,
        lut_resolution=view.lut_resolution,
        edt_offsets=view.edt_offsets,
        edt_distance=view.edt_distance,
        edt_width=view.edt_width,
        edt_height=view.edt_height,
        edt_origin_xy=view.edt_origin_xy,
        edt_resolution=view.edt_resolution,
        capacity=view.capacity,
    )


def _arrays_to_device(view: PackedTrackAtlasView, device: str) -> PackedTrackAtlasView:
    if device == "cpu":
        return view
    try:
        import torch
    except ImportError as exc:
        raise TrackError(
            "loading atlas on a non-cpu device requires torch"
        ) from exc
    torch_device = torch.device(device)

    def _t(arr: Any, dtype: Any = None) -> Any:
        t = torch.as_tensor(np.asarray(arr), device=torch_device)
        return t if dtype is None else t.to(dtype=dtype)

    return PackedTrackAtlasView(
        num_tracks=view.num_tracks,
        offsets=_t(view.offsets, torch.int32),
        centerline_xy=_t(view.centerline_xy, torch.float32),
        tangents_xy=_t(view.tangents_xy, torch.float32),
        widths_rl=_t(view.widths_rl, torch.float32),
        cum_length=_t(view.cum_length, torch.float32),
        lengths=_t(view.lengths, torch.float32),
        track_ids=view.track_ids,
        lut_offsets=_t(view.lut_offsets, torch.int32),
        nearest_segment_lut=_t(view.nearest_segment_lut, torch.int32),
        lut_width=_t(view.lut_width, torch.int32),
        lut_height=_t(view.lut_height, torch.int32),
        lut_origin_xy=_t(view.lut_origin_xy, torch.float32),
        lut_resolution=_t(view.lut_resolution, torch.float32),
        edt_offsets=_t(view.edt_offsets, torch.int32),
        edt_distance=_t(view.edt_distance, torch.float32),
        edt_width=_t(view.edt_width, torch.int32),
        edt_height=_t(view.edt_height, torch.int32),
        edt_origin_xy=_t(view.edt_origin_xy, torch.float32),
        edt_resolution=_t(view.edt_resolution, torch.float32),
        capacity=_t(view.capacity, torch.int32),
    )


def prepare_tracks(
    cfg: ExperimentConfig,
    cache_dir: str,
    *,
    pin_path: str | Path | None = None,
    lut_resolution: float = DEFAULT_LUT_RESOLUTION_M,
    edt_resolution: float = DEFAULT_EDT_RESOLUTION_M,
    skip_download: bool = False,
) -> TrackAtlas:
    """Download/checksum (unless skipped), validate, pack, and cache an atlas."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pin = load_pin(pin_path)
    pinned_count = len(pin["tracks"])
    local_only = (
        cfg.tracks.num_tracks < pinned_count
        and bool(_discover_local_tracks(cache / "local"))
    )
    if skip_download or local_only:
        pinned_paths = {
            name: cache / "source" / pin["commit"] / f"{name}_centerline.csv"
            for name in pin["tracks"]
        }
        if not local_only:
            missing = [n for n, p in pinned_paths.items() if not p.exists()]
            if missing:
                raise TrackError(
                    "skip_download set but cached sources missing: "
                    + ", ".join(missing[:5])
                )
            for name, path in pinned_paths.items():
                digest = _sha256_file(path)
                if digest != pin["tracks"][name]:
                    raise TrackError(f"cached checksum mismatch for {name}")
    else:
        pinned_paths = download_pinned_sources(cache, pin=pin, pin_path=pin_path)

    selected = _select_track_sources(cfg, cache, pinned_paths, pin)
    built: list[_BuiltTrack] = []
    total = len(selected)
    for idx, (name, path, source_sha, optional_local, license_id) in enumerate(
        selected, start=1
    ):
        print(
            f"[prepare-tracks] {idx}/{total} building {name} "
            f"(lut={lut_resolution} edt={edt_resolution})",
            flush=True,
            file=sys.stderr,
        )
        table = load_centerline_csv(path)
        built.append(
            build_track_arrays(
                table,
                name=name,
                source_sha=source_sha,
                lut_resolution=lut_resolution,
                edt_resolution=edt_resolution,
                max_agents=cfg.worlds.max_agents_per_world,
                car_width_m=cfg.agents.car_width_m,
                car_length_m=cfg.agents.car_length_m,
                license_id=license_id,
                optional_local=optional_local,
            )
        )
    print(
        f"[prepare-tracks] packing {len(built)} tracks",
        flush=True,
        file=sys.stderr,
    )
    atlas = pack_built_tracks(built, sampling=cfg.tracks.sampling)
    _write_manifest(cache / MANIFEST_FILENAME, atlas.manifest(), pin=pin,
                    sampling=cfg.tracks.sampling)
    _save_atlas_npz(cache / ATLAS_FILENAME, atlas.view())
    return atlas


def load_atlas(manifest_path: str, device: str = "cpu") -> TrackAtlas:
    manifest_file = Path(manifest_path).expanduser()
    if manifest_file.is_dir():
        cache = manifest_file
        manifest_file = cache / MANIFEST_FILENAME
    else:
        cache = manifest_file.parent
    if not manifest_file.exists():
        raise TrackError(f"manifest not found: {manifest_file}")
    atlas_path = cache / ATLAS_FILENAME
    if not atlas_path.exists():
        raise TrackError(f"atlas archive not found: {atlas_path}")
    with manifest_file.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    entries = tuple(TrackManifestEntry(**row) for row in meta["tracks"])
    data = np.load(atlas_path, allow_pickle=False)
    track_ids = tuple(str(x) for x in data["track_ids"].tolist())
    view = PackedTrackAtlasView(
        num_tracks=int(meta["num_tracks"]),
        offsets=np.asarray(data["offsets"], dtype=np.int32),
        centerline_xy=np.asarray(data["centerline_xy"], dtype=np.float32),
        tangents_xy=np.asarray(data["tangents_xy"], dtype=np.float32),
        widths_rl=np.asarray(data["widths_rl"], dtype=np.float32),
        cum_length=np.asarray(data["cum_length"], dtype=np.float32),
        lengths=np.asarray(data["lengths"], dtype=np.float32),
        track_ids=track_ids,
        lut_offsets=np.asarray(data["lut_offsets"], dtype=np.int32),
        nearest_segment_lut=np.asarray(data["nearest_segment_lut"], dtype=np.int32),
        lut_width=np.asarray(data["lut_width"], dtype=np.int32),
        lut_height=np.asarray(data["lut_height"], dtype=np.int32),
        lut_origin_xy=np.asarray(data["lut_origin_xy"], dtype=np.float32),
        lut_resolution=np.asarray(data["lut_resolution"], dtype=np.float32),
        edt_offsets=np.asarray(data["edt_offsets"], dtype=np.int32),
        edt_distance=np.asarray(data["edt_distance"], dtype=np.float32),
        edt_width=np.asarray(data["edt_width"], dtype=np.int32),
        edt_height=np.asarray(data["edt_height"], dtype=np.int32),
        edt_origin_xy=np.asarray(data["edt_origin_xy"], dtype=np.float32),
        edt_resolution=np.asarray(data["edt_resolution"], dtype=np.float32),
        capacity=np.asarray(data["capacity"], dtype=np.int32),
    )
    view = _arrays_to_device(view, device)
    return NumpyTrackAtlas(
        entries, view, sampling=str(meta.get("sampling", "stratified_shuffle"))
    )


def rebuild_atlas_capacity(
    src_cache: str | Path,
    dst_cache: str | Path,
    *,
    max_agents: int,
    car_width_m: float = CAR_WIDTH_M,
    car_length_m: float = CAR_LENGTH_M,
) -> NumpyTrackAtlas:
    """Copy an atlas and recompute capacity for a new max_agents ceiling.

    Geometry / LUT / EDT are unchanged. Destination must differ from source so
    production caches are never overwritten in place.
    """
    src = Path(src_cache).expanduser().resolve()
    dst = Path(dst_cache).expanduser().resolve()
    if max_agents <= 0:
        raise TrackError("max_agents must be positive")
    if src == dst:
        raise TrackError(
            "rebuild_atlas_capacity refuses in-place overwrite; "
            "choose a distinct destination cache"
        )
    atlas = load_atlas(str(src), device="cpu")
    view = atlas.view()
    entries = atlas.manifest()
    capacities = np.empty(view.num_tracks, dtype=np.int32)
    new_entries: list[TrackManifestEntry] = []
    for i, entry in enumerate(entries):
        a = int(view.offsets[i])
        b = int(view.offsets[i + 1])
        widths = np.asarray(view.widths_rl[a:b], dtype=np.float64)
        cap = compute_track_capacity(
            float(view.lengths[i]),
            widths[:, 0],
            widths[:, 1],
            car_width_m=car_width_m,
            car_length_m=car_length_m,
            max_agents=max_agents,
        )
        capacities[i] = int(cap)
        new_entries.append(
            TrackManifestEntry(
                track_id=entry.track_id,
                name=entry.name,
                source_sha=entry.source_sha,
                geometry_hash=entry.geometry_hash,
                length_m=entry.length_m,
                width_mean_m=entry.width_mean_m,
                width_min_m=entry.width_min_m,
                num_points=entry.num_points,
                valid=entry.valid,
                capacity=int(cap),
                license=entry.license,
                optional_local=entry.optional_local,
            )
        )
    new_view = PackedTrackAtlasView(
        num_tracks=view.num_tracks,
        offsets=np.asarray(view.offsets, dtype=np.int32),
        centerline_xy=np.asarray(view.centerline_xy, dtype=np.float32),
        tangents_xy=np.asarray(view.tangents_xy, dtype=np.float32),
        widths_rl=np.asarray(view.widths_rl, dtype=np.float32),
        cum_length=np.asarray(view.cum_length, dtype=np.float32),
        lengths=np.asarray(view.lengths, dtype=np.float32),
        track_ids=tuple(view.track_ids),
        lut_offsets=np.asarray(view.lut_offsets, dtype=np.int32),
        nearest_segment_lut=np.asarray(view.nearest_segment_lut, dtype=np.int32),
        lut_width=np.asarray(view.lut_width, dtype=np.int32),
        lut_height=np.asarray(view.lut_height, dtype=np.int32),
        lut_origin_xy=np.asarray(view.lut_origin_xy, dtype=np.float32),
        lut_resolution=np.asarray(view.lut_resolution, dtype=np.float32),
        edt_offsets=np.asarray(view.edt_offsets, dtype=np.int32),
        edt_distance=np.asarray(view.edt_distance, dtype=np.float32),
        edt_width=np.asarray(view.edt_width, dtype=np.int32),
        edt_height=np.asarray(view.edt_height, dtype=np.int32),
        edt_origin_xy=np.asarray(view.edt_origin_xy, dtype=np.float32),
        edt_resolution=np.asarray(view.edt_resolution, dtype=np.float32),
        capacity=capacities,
    )
    dst.mkdir(parents=True, exist_ok=True)
    with (src / MANIFEST_FILENAME).open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    pin = {
        "commit": meta.get("pin_commit"),
        "ref": meta.get("pin_ref"),
        "repo": meta.get("pin_repo"),
        "license": meta.get("license", "GPL-3.0"),
    }
    sampling = str(meta.get("sampling", "stratified_shuffle"))
    _write_manifest(dst / MANIFEST_FILENAME, new_entries, pin=pin, sampling=sampling)
    _save_atlas_npz(dst / ATLAS_FILENAME, new_view)
    return NumpyTrackAtlas(new_entries, new_view, sampling=sampling)


def project_to_centerline(
    positions_xy: np.ndarray,
    point: np.ndarray,
    cum_length: np.ndarray,
    length_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU nearest-segment Frenet projection (s, lateral, segment index)."""
    pos = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    edge = np.roll(point, -1, axis=0) - point
    edge_sq = np.sum(edge * edge, axis=1)
    delta = pos[:, None, :] - point[None, :, :]
    alpha = np.clip(
        np.sum(delta * edge[None, :, :], axis=2) / edge_sq[None, :],
        0.0,
        1.0,
    )
    projection = point[None, :, :] + alpha[:, :, None] * edge[None, :, :]
    dist_sq = np.sum((pos[:, None, :] - projection) ** 2, axis=2)
    seg = np.argmin(dist_sq, axis=1)
    ar = np.arange(pos.shape[0])
    a = alpha[ar, seg]
    proj = projection[ar, seg]
    tangent = edge[seg] / np.sqrt(edge_sq[seg])[:, None]
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    lateral = np.sum((pos - proj) * normal, axis=1)
    s = cum_length[seg] + a * np.sqrt(edge_sq[seg])
    s = np.mod(s, length_m)
    return s.astype(np.float32), lateral.astype(np.float32), seg.astype(np.int32)


def sample_track_lookahead(
    view: PackedTrackAtlasView,
    *,
    track_id: Any,
    s: Any,
    x: Any,
    y: Any,
    yaw: Any,
    speed: Any,
    num_samples: int = TRACK_PREVIEW_SAMPLES,
    speed_horizon_s: float = TRACK_PREVIEW_SPEED_HORIZON_S,
    min_lookahead_m: float = TRACK_PREVIEW_MIN_LOOKAHEAD_M,
) -> Any:
    """Ordered ego-frame centerline preview ahead of each agent.

    All six query tensors broadcast against each other to a common ``[..., S]``
    shape; the result is ``[..., S, num_samples * 5]`` float32 on the query
    device, so it can be concatenated straight into an ordered (never pooled)
    critic ego branch.

    Spacing and ordering are fixed: with lookahead
    ``L = max(speed * speed_horizon_s, min_lookahead_m)`` metres, sample ``k``
    (``k = 0 .. num_samples - 1``) sits at arc length ``L * (k + 1) /
    num_samples`` ahead of the agent's Frenet ``s``, so samples are uniformly
    spaced, strictly in front, and ordered nearest-first. Sample ``k``
    contributes ``TRACK_PREVIEW_SAMPLE_DIM`` contiguous features in this order:

    ``0`` ego-frame x of the centerline point (forward positive)
    ``1`` ego-frame y of the centerline point (left positive)
    ``2`` right half-width at the sample, metres
    ``3`` left half-width at the sample, metres
    ``4`` signed centerline curvature at the sample, 1/m (left turn positive)

    Arc length wraps with ``lengths`` per track, so a lookahead crossing the
    start line — or exceeding a lap — keeps sampling forward along the loop.
    Points, widths, and arc length are interpolated linearly inside the packed
    segment containing each sample; ``offsets`` keeps agents on different tracks
    independent, so a mixed-track batch is a single gather.

    Curvature is the turn rate of the packed per-segment tangents: the signed
    angle from the sample segment's tangent to the next segment's tangent
    (``atan2`` of their cross and dot products, exactly zero on a straight)
    divided by the mean of those two segment lengths, which is the arc length
    the vertex turn is spread over. Both lengths are floored at
    ``ZERO_SEGMENT_TOL_M`` so a degenerate segment cannot divide by zero.
    """
    try:
        import torch
    except ImportError as exc:
        raise TrackError("sample_track_lookahead requires torch") from exc
    if num_samples < 1:
        raise TrackError("num_samples must be positive")
    if speed_horizon_s < 0.0:
        raise TrackError("speed_horizon_s must be non-negative")
    if min_lookahead_m <= 0.0:
        raise TrackError("min_lookahead_m must be positive")

    device = torch.device("cpu")
    for query in (s, x, y, yaw, speed, track_id):
        if isinstance(query, torch.Tensor):
            device = query.device
            break

    def _t(array: Any, dtype: Any) -> Any:
        tensor = (
            array
            if isinstance(array, torch.Tensor)
            else torch.as_tensor(np.asarray(array))
        )
        return tensor.to(device=device, dtype=dtype)

    # Arc-length bookkeeping runs in float64: packed cumulative length plus a
    # per-track base spans every track, and float32 cannot resolve segment
    # boundaries at that magnitude. Geometry stays float32.
    track = _t(track_id, torch.int64)
    s_q = _t(s, torch.float64)
    x_q = _t(x, torch.float32)
    y_q = _t(y, torch.float32)
    yaw_q = _t(yaw, torch.float32)
    speed_q = _t(speed, torch.float64)
    track, s_q, x_q, y_q, yaw_q, speed_q = torch.broadcast_tensors(
        track, s_q, x_q, y_q, yaw_q, speed_q
    )

    n_tracks = int(view.num_tracks)
    offsets = _t(view.offsets, torch.int64).reshape(-1)
    if offsets.shape[0] != n_tracks + 1:
        raise TrackError(
            f"offsets length {offsets.shape[0]} != num_tracks + 1 ({n_tracks + 1})"
        )
    point = _t(view.centerline_xy, torch.float32).reshape(-1, 2)
    tangent = _t(view.tangents_xy, torch.float32).reshape(-1, 2)
    widths = _t(view.widths_rl, torch.float32).reshape(-1, 2)
    cum = _t(view.cum_length, torch.float64).reshape(-1)
    lengths = _t(view.lengths, torch.float64).reshape(-1)
    n_points = int(point.shape[0])

    counts = offsets[1:] - offsets[:-1]
    zero = torch.zeros(1, dtype=torch.float64, device=device)
    base = torch.cat([zero, torch.cumsum(lengths, dim=0)])
    point_index = torch.arange(n_points, device=device)
    point_track = torch.searchsorted(offsets, point_index, right=True) - 1
    # Per-track arc length plus track base is strictly increasing over the whole
    # packed array (a track's last point sits below the next track's base), so a
    # single searchsorted locates the segment on whichever track an agent is on.
    global_cum = cum + base[point_track]

    track = track.clamp(0, n_tracks - 1)
    start = offsets[track].unsqueeze(-1)
    count = counts[track].unsqueeze(-1)
    track_len = lengths[track].unsqueeze(-1)

    step = torch.arange(1, num_samples + 1, device=device, dtype=torch.float64)
    lookahead = torch.clamp(
        speed_q * float(speed_horizon_s), min=float(min_lookahead_m)
    )
    ahead_m = lookahead.unsqueeze(-1) * (step / float(num_samples))
    s_sample = torch.remainder(s_q.unsqueeze(-1) + ahead_m, track_len)
    located = torch.searchsorted(
        global_cum, (s_sample + base[track].unsqueeze(-1)).contiguous(), right=True
    )
    index = (located - 1).clamp(min=0, max=n_points - 1)
    local = index - start

    def _segment_length(local_index: Any) -> Any:
        packed = start + local_index
        wraps = ((local_index + 1) % count) == 0
        following = cum[(packed + 1).clamp(max=n_points - 1)]
        end = torch.where(wraps, track_len, following)
        return (end - cum[packed]).clamp(min=ZERO_SEGMENT_TOL_M)

    local_next = (local + 1) % count
    index_next = start + local_next
    segment_m = _segment_length(local)
    alpha = ((s_sample - cum[index]) / segment_m).clamp(0.0, 1.0).to(torch.float32)
    frac = alpha.unsqueeze(-1)

    here = point[index]
    sample_xy = here + frac * (point[index_next] - here)
    width_here = widths[index]
    sample_widths = width_here + frac * (widths[index_next] - width_here)

    tangent_here = tangent[index]
    tangent_next = tangent[index_next]
    cross = (
        tangent_here[..., 0] * tangent_next[..., 1]
        - tangent_here[..., 1] * tangent_next[..., 0]
    )
    dot = (
        tangent_here[..., 0] * tangent_next[..., 0]
        + tangent_here[..., 1] * tangent_next[..., 1]
    )
    turn = torch.atan2(cross, dot).to(torch.float64)
    spread = 0.5 * (segment_m + _segment_length(local_next))
    curvature = (turn / spread).to(torch.float32)

    delta_x = sample_xy[..., 0] - x_q.unsqueeze(-1)
    delta_y = sample_xy[..., 1] - y_q.unsqueeze(-1)
    cos_yaw = torch.cos(yaw_q).unsqueeze(-1)
    sin_yaw = torch.sin(yaw_q).unsqueeze(-1)
    ego_x = cos_yaw * delta_x + sin_yaw * delta_y
    ego_y = -sin_yaw * delta_x + cos_yaw * delta_y
    samples = torch.stack(
        [ego_x, ego_y, sample_widths[..., 0], sample_widths[..., 1], curvature],
        dim=-1,
    )
    return samples.flatten(-2, -1)

"""Extract a race-ready centerline and cleaned map from a SLAM occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.morphology import binary_closing, disk, flood, medial_axis

from track_geometry import loop_length


@dataclass
class MapData:
    grid: np.ndarray  # 0=free, 1=occupied, 2=unknown (normalized)
    resolution: float
    origin_x: float
    origin_y: float


@dataclass
class CenterlineResult:
    centerline: np.ndarray  # (N, 2) world xy
    w_tr_left: np.ndarray
    w_tr_right: np.ndarray


# Bundled gym maps whose authoritative centerline lives in f1tenth_racetracks.
# Keys use rounded origin/resolution so YAML float parsing matches reliably.
_KNOWN_TRACKS: dict[tuple[float, float, float], str] = {
    (-55.07650229, -33.57884064, 0.04295): "Oschersleben",
    (-50.99, -31.58, 0.0625): "IV_2026_SIM",
}


def _map_fingerprint(data: MapData) -> tuple[float, float, float]:
    return (
        round(data.origin_x, 8),
        round(data.origin_y, 8),
        round(data.resolution, 8),
    )


def _occupancy_from_raw(raw: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify pixels using map_server free/occupied thresholds from YAML."""
    raw_f = raw.astype(np.float32)
    if int(meta.get("negate", 0)):
        occ_prob = raw_f / 255.0
    else:
        occ_prob = (255.0 - raw_f) / 255.0
    free = occ_prob < float(meta["free_thresh"])
    occupied = occ_prob > float(meta["occupied_thresh"])
    unknown = ~free & ~occupied
    return free, occupied, unknown


def load_map_yaml(yaml_path: Path) -> MapData:
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)
    image_path = yaml_path.parent / meta["image"]
    if not image_path.is_file():
        image_path = yaml_path.parent / Path(meta["image"]).name
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read map image: {image_path}")

    free, occupied, unknown = _occupancy_from_raw(raw, meta)
    grid = np.zeros_like(raw, dtype=np.uint8)
    grid[free] = 0
    grid[occupied] = 1
    grid[unknown] = 2

    origin = meta["origin"]
    return MapData(
        grid=grid,
        resolution=float(meta["resolution"]),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )


def _known_track_name(data: MapData) -> str | None:
    return _KNOWN_TRACKS.get(_map_fingerprint(data))


def _load_known_centerline(
    track_name: str, spacing_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    repo_root = Path(__file__).resolve().parents[2]
    for directory in (
        repo_root / "ros2_deploy" / "assets",
        repo_root / "ros2_deploy" / "f1tenth_rl_agent" / "assets",
    ):
        csv_path = directory / f"{track_name}_centerline.csv"
        if csv_path.is_file():
            raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
            points = np.stack([raw["x_m"], raw["y_m"]], axis=1)
            w_left = np.asarray(raw["w_tr_left_m"], dtype=np.float64)
            w_right = np.asarray(raw["w_tr_right_m"], dtype=np.float64)
            points, w_left, w_right = _resample_track(points, w_left, w_right, spacing_m)
            points = _ensure_closed_loop(points, spacing_m)
            return points, w_left, w_right
    return None


def world_to_grid(x: float, y: float, data: MapData) -> tuple[int, int]:
    col = int((x - data.origin_x) / data.resolution)
    row = int((y - data.origin_y) / data.resolution)
    return row, col


def _grid_to_world(row: float, col: float, data: MapData) -> tuple[float, float]:
    x = data.origin_x + (col + 0.5) * data.resolution
    y = data.origin_y + (row + 0.5) * data.resolution
    return x, y


def _drivable_mask(grid: np.ndarray) -> np.ndarray:
    return grid != 1


def _occupied_mask(grid: np.ndarray) -> np.ndarray:
    return grid == 1


def _interior_free_mask(free: np.ndarray) -> np.ndarray:
    """Remove free space reachable from the map border (exterior)."""
    height, width = free.shape
    reachable = np.zeros_like(free, dtype=bool)
    stack: list[tuple[int, int]] = []
    for c in range(width):
        if free[0, c]:
            stack.append((0, c))
        if free[height - 1, c]:
            stack.append((height - 1, c))
    for r in range(height):
        if free[r, 0]:
            stack.append((r, 0))
        if free[r, width - 1]:
            stack.append((r, width - 1))
    while stack:
        r, c = stack.pop()
        if reachable[r, c] or not free[r, c]:
            continue
        reachable[r, c] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                stack.append((nr, nc))
    return free & ~reachable


def _component_perimeter_m(mask: np.ndarray, resolution: float) -> float:
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not cnts:
        return 0.0
    contour = max(cnts, key=lambda c: cv2.arcLength(c, True))
    return float(cv2.arcLength(contour, True) * resolution)


def _pick_component_by_perimeter(
    band: np.ndarray,
    resolution: float,
    dist_m: np.ndarray | None = None,
    min_perimeter_m: float = 80.0,
    max_perimeter_m: float = 600.0,
) -> np.ndarray:
    labeled, n_labels = ndimage.label(band)
    if n_labels == 0:
        return band

    best_mask: np.ndarray | None = None
    best_score = -1.0
    target_perimeter = 0.5 * (min_perimeter_m + max_perimeter_m)

    for lab in range(1, n_labels + 1):
        mask = labeled == lab
        perimeter = _component_perimeter_m(mask, resolution)
        if perimeter < min_perimeter_m:
            continue
        if max_perimeter_m > 0 and perimeter > max_perimeter_m:
            continue
        score = perimeter - abs(perimeter - target_perimeter)
        if dist_m is not None:
            score += 0.25 * float(np.max(dist_m[mask]))
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        perimeters = [
            _component_perimeter_m(labeled == lab, resolution)
            for lab in range(1, n_labels + 1)
        ]
        best_lab = int(np.argmax(perimeters)) + 1
        return labeled == best_lab
    return best_mask


def _pick_largest_component(band: np.ndarray) -> np.ndarray:
    labeled, n_labels = ndimage.label(band)
    if n_labels == 0:
        return band
    sizes = ndimage.sum(band, labeled, range(1, n_labels + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _narrow_distance_band(
    drivable: np.ndarray,
    dist_m: np.ndarray,
    half_width_fallback: float,
    use_interior: bool,
    interior: np.ndarray,
) -> np.ndarray:
    near_track = drivable & (dist_m < 4.0)
    if np.any(near_track):
        est_half = float(np.percentile(dist_m[near_track], 75))
    else:
        est_half = half_width_fallback
    est_half = max(est_half, half_width_fallback * 0.5)
    upper = max(2.2 * half_width_fallback, est_half * 1.5)
    band = drivable & (dist_m > 0.2) & (dist_m < upper)
    if use_interior:
        candidate = band & interior
        if np.count_nonzero(candidate) > 100:
            band = candidate
    return band


def _wide_distance_band(
    drivable: np.ndarray,
    dist_m: np.ndarray,
    half_width_fallback: float,
) -> np.ndarray:
    near_walls = drivable & (dist_m < 20.0)
    upper = max(
        2.2 * half_width_fallback,
        float(np.percentile(dist_m[near_walls], 99)) if np.any(near_walls) else 15.0,
    )
    return drivable & (dist_m > 0.05) & (dist_m < upper)


def _map_profile(free: np.ndarray, interior: np.ndarray) -> str:
    interior_frac = float(np.sum(interior)) / max(float(np.sum(free)), 1.0)
    if interior_frac < 0.12:
        return "gym"
    if interior_frac > 0.15:
        return "slam"
    return "mixed"


def _flood_seed_cell(drivable: np.ndarray, data: MapData) -> tuple[int, int]:
    """Pick a seed inside the largest drivable component (robust on partial SLAM maps)."""
    labeled, n_labels = ndimage.label(drivable)
    if n_labels == 0:
        return world_to_grid(0.0, 0.0, data)
    sizes = ndimage.sum(drivable, labeled, range(1, n_labels + 1))
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == best)
    return int(np.mean(ys)), int(np.mean(xs))


def _corridor_candidates(
    drivable: np.ndarray,
    dist_m: np.ndarray,
    half_width_fallback: float,
    resolution: float,
    profile: str,
    data: MapData,
) -> list[np.ndarray]:
    free = data.grid == 0
    interior = _interior_free_mask(free)
    narrow = _narrow_distance_band(
        drivable,
        dist_m,
        half_width_fallback,
        use_interior=(profile != "gym"),
        interior=interior,
    )
    wide = _wide_distance_band(drivable, dist_m, half_width_fallback)

    if profile == "gym":
        bridged = binary_closing(drivable, disk(1))
        seed_row, seed_col = _flood_seed_cell(bridged, data)
        if not (0 <= seed_row < bridged.shape[0] and 0 <= seed_col < bridged.shape[1] and bridged[seed_row, seed_col]):
            ys, xs = np.where(narrow)
            if len(ys) == 0:
                ys, xs = np.where(bridged)
            seed_row, seed_col = int(ys[0]), int(xs[0])
        track = flood(bridged, (seed_row, seed_col)) if bridged[seed_row, seed_col] else bridged
        near_track = track & (dist_m < 20.0)
        upper = max(
            12.0,
            float(np.percentile(dist_m[near_track], 95)) if np.any(near_track) else 12.0,
        )
        gym_band = track & (dist_m > 0.15) & (dist_m < upper)
        candidates = [
            _pick_component_by_perimeter(gym_band, resolution, dist_m),
            _pick_component_by_perimeter(narrow, resolution, dist_m),
        ]
    else:
        candidates = [
            _pick_largest_component(narrow),
            _pick_component_by_perimeter(narrow, resolution, dist_m),
            _pick_component_by_perimeter(wide, resolution, dist_m),
        ]

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in candidates:
        if not np.any(mask):
            continue
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(mask)
    return unique


def _skeleton_neighbors(row: int, col: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((row + dr, col + dc))
    return out


def _order_skeleton_pruned_cycle(
    rows: np.ndarray,
    cols: np.ndarray,
    dist_m: np.ndarray | None = None,
    row_offset: int = 0,
    col_offset: int = 0,
    scale: int = 1,
) -> list[int]:
    """Walk the main loop after pruning degree-1 spurs from the skeleton graph."""
    n = len(rows)
    if n == 0:
        return []
    if n <= 3:
        return list(range(n))

    index: dict[tuple[int, int], int] = {
        (int(rows[i]), int(cols[i])): i for i in range(n)
    }
    neighbors: list[list[int]] = []
    dist_vals: list[float] = []
    for i in range(n):
        nbrs: list[int] = []
        for nr, nc in _skeleton_neighbors(int(rows[i]), int(cols[i])):
            j = index.get((nr, nc))
            if j is not None:
                nbrs.append(j)
        neighbors.append(nbrs)
        if dist_m is not None:
            gr = row_offset + int(rows[i] * scale)
            gc = col_offset + int(cols[i] * scale)
            if 0 <= gr < dist_m.shape[0] and 0 <= gc < dist_m.shape[1]:
                dist_vals.append(float(dist_m[gr, gc]))
            else:
                dist_vals.append(0.0)
        else:
            dist_vals.append(0.0)

    active = set(range(n))
    while True:
        leaves = [
            i
            for i in active
            if len([j for j in neighbors[i] if j in active]) <= 1
        ]
        if not leaves or len(active) - len(leaves) < 10:
            break
        for leaf in leaves:
            active.discard(leaf)

    if not active:
        return list(range(n))

    start = max(active, key=lambda i: dist_vals[i])
    order = [start]
    prev = -1
    cur = start
    while True:
        nbrs = [j for j in neighbors[cur] if j in active and j != prev]
        if not nbrs:
            break
        nxt = max(nbrs, key=lambda j: dist_vals[j]) if len(nbrs) > 1 else nbrs[0]
        if nxt == start and len(order) > 10:
            break
        if nxt in order:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(active) + 1:
            break
    return order


def _centerline_quality_score(
    centerline: np.ndarray,
    data: MapData,
    dist_m: np.ndarray,
    half_width_fallback: float,
) -> float:
    length_m = loop_length(centerline)
    if length_m < 40.0 or length_m > 700.0:
        return -1e9
    if length_m < 180.0:
        return -1e9

    height, width = data.grid.shape
    drivable = _drivable_mask(data.grid)
    free_hits = 0
    total = 0
    dist_samples: list[float] = []
    for pt in centerline[::5]:
        row, col = world_to_grid(float(pt[0]), float(pt[1]), data)
        if row < 0 or col < 0 or row >= height or col >= width:
            continue
        total += 1
        if drivable[row, col]:
            free_hits += 1
            dist_samples.append(float(dist_m[row, col]))
    if total == 0:
        return -1e9
    free_fraction = free_hits / total
    if free_fraction < 0.85:
        return -1e9

    step = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    if step.size and float(np.max(step)) > 8.0:
        return -1e9

    mean_dist = float(np.mean(dist_samples)) if dist_samples else half_width_fallback
    narrow_penalty = max(0.0, half_width_fallback * 0.8 - mean_dist) * 3.0
    length_bonus = min(length_m, 350.0) * 0.06
    return free_fraction * 100.0 + length_bonus - narrow_penalty


def _centerline_from_corridor(
    corridor: np.ndarray,
    data: MapData,
    dist_m: np.ndarray,
    spacing_m: float,
    downsample_factor: int,
) -> np.ndarray:
    rows = np.any(corridor, axis=1)
    cols = np.any(corridor, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    crop = corridor[r0:r1, c0:c1]

    crop_pixels = crop.shape[0] * crop.shape[1]
    factor = 1 if crop_pixels < 2_000_000 else max(1, int(downsample_factor))
    small = crop if factor == 1 else crop[::factor, ::factor]
    skel, _ = medial_axis(small, return_distance=True)
    sk_rows, sk_cols = np.where(skel)
    if sk_rows.size < 10:
        raise ValueError("Medial axis too short; map may be incomplete")

    order = _order_skeleton_pruned_cycle(
        sk_rows,
        sk_cols,
        dist_m=dist_m,
        row_offset=r0,
        col_offset=c0,
        scale=factor,
    )
    if len(order) < 10:
        order = list(range(len(sk_rows)))

    world = np.array(
        [
            _grid_to_world(float(r0 + sk_rows[i] * factor), float(c0 + sk_cols[i] * factor), data)
            for i in order
        ],
        dtype=np.float64,
    )
    world = _smooth_loop(world)
    return _ensure_closed_loop(_resample_polyline(world, spacing_m), spacing_m)


def _smooth_loop(points: np.ndarray, window: int = 11) -> np.ndarray:
    if points.shape[0] < window:
        return points
    window = window if window % 2 == 1 else window + 1
    if points.shape[0] < window:
        return points
    padded = np.vstack([points[-window // 2 :], points, points[: window // 2]])
    sx = savgol_filter(padded[:, 0], window_length=window, polyorder=3, mode="interp")
    sy = savgol_filter(padded[:, 1], window_length=window, polyorder=3, mode="interp")
    return np.column_stack(
        [sx[window // 2 : window // 2 + len(points)], sy[window // 2 : window // 2 + len(points)]]
    )


def extract_centerline(
    data: MapData,
    spacing_m: float = 0.07,
    half_width_fallback: float = 1.1,
    min_width_m: float = 0.5,
    downsample_factor: int = 2,
    prefer_known_track: bool = True,
) -> CenterlineResult:
    if prefer_known_track:
        track_name = _known_track_name(data)
        if track_name is not None:
            loaded = _load_known_centerline(track_name, spacing_m)
            if loaded is not None:
                known, w_left, w_right = loaded
                return CenterlineResult(
                    centerline=known.astype(np.float64),
                    w_tr_left=w_left,
                    w_tr_right=w_right,
                )

    drivable = _drivable_mask(data.grid)
    occupied = _occupied_mask(data.grid)
    if not np.any(drivable):
        raise ValueError("No drivable space found in map")

    free = data.grid == 0
    interior = _interior_free_mask(free)
    profile = _map_profile(free, interior)
    if profile != "gym":
        drivable = binary_closing(drivable, disk(1))
        free = free | (drivable & ~occupied)
    dist_m = ndimage.distance_transform_edt(~occupied) * data.resolution
    corridors = _corridor_candidates(
        drivable, dist_m, half_width_fallback, data.resolution, profile, data
    )
    if not corridors:
        raise ValueError("Could not isolate track corridor from map")

    best_centerline: np.ndarray | None = None
    best_score = -1e18
    for corridor in corridors:
        try:
            candidate = _centerline_from_corridor(
                corridor, data, dist_m, spacing_m, downsample_factor
            )
        except ValueError:
            continue
        score = _centerline_quality_score(candidate, data, dist_m, half_width_fallback)
        if score > best_score:
            best_score = score
            best_centerline = candidate

    if best_centerline is None:
        raise ValueError("Could not extract centerline from map")

    max_width = half_width_fallback * 1.5
    w_left, w_right = _raycast_widths(
        best_centerline, data, half_width_fallback, min_width_m, max_width, dist_m=dist_m
    )
    return CenterlineResult(
        centerline=best_centerline.astype(np.float64),
        w_tr_left=w_left,
        w_tr_right=w_right,
    )


def _resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    resampled, _, _ = _resample_track(
        points,
        np.ones(points.shape[0]),
        np.ones(points.shape[0]),
        spacing_m,
    )
    return resampled


def _resample_track(
    points: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        return points, w_left, w_right
    diffs = np.diff(points, axis=0, append=points[:1])
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < spacing_m:
        return points, w_left, w_right
    samples = np.arange(0.0, total, spacing_m)
    out_pts = np.zeros((len(samples), 2), dtype=np.float64)
    out_left = np.zeros(len(samples), dtype=np.float64)
    out_right = np.zeros(len(samples), dtype=np.float64)
    j = 0
    for i, s in enumerate(samples):
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        nxt = (j + 1) % len(points)
        out_pts[i] = points[j] * (1.0 - t) + points[nxt] * t
        out_left[i] = w_left[j] * (1.0 - t) + w_left[nxt] * t
        out_right[i] = w_right[j] * (1.0 - t) + w_right[nxt] * t
    return out_pts, out_left, out_right


def _smooth_periodic(values: np.ndarray, window: int = 15) -> np.ndarray:
    n = values.shape[0]
    if n < window:
        return values
    window = window if window % 2 == 1 else window + 1
    padded = np.concatenate([values[-window // 2 :], values, values[: window // 2]])
    smoothed = savgol_filter(padded, window_length=window, polyorder=2, mode="interp")
    return smoothed[window // 2 : window // 2 + n]


def _raycast_widths(
    centerline: np.ndarray,
    data: MapData,
    fallback: float,
    min_width_m: float,
    max_width_m: float,
    dist_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = centerline.shape[0]
    w_left = np.full(n, fallback, dtype=np.float64)
    w_right = np.full(n, fallback, dtype=np.float64)
    height, width = data.grid.shape
    occupied = _occupied_mask(data.grid)
    drivable = _drivable_mask(data.grid)
    step_m = max(data.resolution * 0.25, 0.01)

    for i in range(n):
        prev_pt = centerline[i - 1]
        nxt_pt = centerline[(i + 1) % n]
        tangent = nxt_pt - prev_pt
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            continue
        tangent /= norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        row, col = world_to_grid(float(centerline[i, 0]), float(centerline[i, 1]), data)
        local_clear = fallback
        if dist_m is not None and 0 <= row < height and 0 <= col < width:
            local_clear = float(dist_m[row, col])

        for sign, arr in ((1.0, w_left), (-1.0, w_right)):
            dist = 0.0
            limit = min(max_width_m, max(local_clear * 1.2, fallback))
            while dist < limit + step_m:
                dist += step_m
                probe = centerline[i] + sign * normal * dist
                prow, pcol = world_to_grid(float(probe[0]), float(probe[1]), data)
                if prow < 0 or pcol < 0 or prow >= height or pcol >= width:
                    arr[i] = float(np.clip(dist - step_m, min_width_m, max_width_m))
                    break
                if occupied[prow, pcol] or not drivable[prow, pcol]:
                    arr[i] = float(np.clip(dist - step_m, min_width_m, max_width_m))
                    break
            else:
                arr[i] = float(np.clip(min(fallback, local_clear), min_width_m, max_width_m))

    w_left = _smooth_periodic(w_left)
    w_right = _smooth_periodic(w_right)
    return w_left, w_right


def _ensure_closed_loop(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    if np.linalg.norm(points[0] - points[-1]) > spacing_m:
        return np.vstack([points, points[:1]])
    return points


def write_centerline_csv(result: CenterlineResult, out_path: Path) -> None:
    lines = ["# x_m, y_m, w_tr_right_m, w_tr_left_m\n"]
    for i in range(result.centerline.shape[0]):
        x, y = result.centerline[i]
        lines.append(
            f"{x:.6f}, {y:.6f}, {result.w_tr_right[i]:.6f}, {result.w_tr_left[i]:.6f}\n"
        )
    out_path.write_text("".join(lines))


def write_raceline_csv(result: CenterlineResult, out_path: Path) -> None:
    pts = result.centerline
    n = pts.shape[0]
    s = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        s[i] = s[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
    psi = np.zeros(n, dtype=np.float64)
    kappa = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        d1 = p1 - p0
        d2 = p2 - p1
        psi[i] = np.arctan2(d1[1], d1[0])
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        denom = np.linalg.norm(d1) * np.linalg.norm(d2) * np.linalg.norm(p2 - p0)
        kappa[i] = cross / max(denom, 1e-6)
    header = "s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2\n"
    lines = [header]
    for i in range(n):
        lines.append(
            f"{s[i]:.6f}, {pts[i, 0]:.6f}, {pts[i, 1]:.6f}, "
            f"{psi[i]:.6f}, {kappa[i]:.6f}, 0.0, 0.0\n"
        )
    out_path.write_text("".join(lines))


def occupancy_grid_to_gray(grid: np.ndarray) -> np.ndarray:
    img = np.full(grid.shape, 205, dtype=np.uint8)
    img[grid == 0] = 254
    img[grid == 1] = 0
    return img


def export_cleaned_map(
    data: MapData,
    out_dir: Path,
    track_name: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{track_name}.png"
    yaml_path = out_dir / f"{track_name}.yaml"

    cv2.imwrite(str(png_path), occupancy_grid_to_gray(data.grid))

    yaml_data = {
        "image": f"{track_name}.png",
        "resolution": data.resolution,
        "origin": [data.origin_x, data.origin_y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.45,
        "free_thresh": 0.196,
    }
    with yaml_path.open("w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    return png_path, yaml_path

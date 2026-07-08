"""Draw reference vs extracted track overlays on an occupancy map."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from centerline_extractor import MapData, occupancy_grid_to_gray


def _to_px(pts: np.ndarray, data: MapData) -> np.ndarray:
    cols = ((pts[:, 0] - data.origin_x) / data.resolution).astype(np.int32)
    rows = ((pts[:, 1] - data.origin_y) / data.resolution).astype(np.int32)
    return np.stack([cols, rows], axis=1)


def _draw_polyline(
    canvas: np.ndarray,
    pts: np.ndarray,
    data: MapData,
    color: tuple[int, int, int],
    thickness: int = 2,
    closed: bool = True,
) -> None:
    px = _to_px(pts, data)
    if len(px) < 2:
        return
    if closed and np.linalg.norm(pts[0] - pts[-1]) > data.resolution * 2:
        closed = False
    cv2.polylines(
        canvas,
        [px],
        isClosed=closed,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def render_map_base(data: MapData) -> np.ndarray:
    return cv2.cvtColor(occupancy_grid_to_gray(data.grid), cv2.COLOR_GRAY2BGR)


def _crop_to_tracks(
    canvas: np.ndarray,
    data: MapData,
    point_sets: list[np.ndarray],
    pad_px: int = 40,
) -> np.ndarray:
    stacked = np.vstack(point_sets)
    px = _to_px(stacked, data)
    r0 = max(int(px[:, 1].min()) - pad_px, 0)
    r1 = min(int(px[:, 1].max()) + pad_px, canvas.shape[0])
    c0 = max(int(px[:, 0].min()) - pad_px, 0)
    c1 = min(int(px[:, 0].max()) + pad_px, canvas.shape[1])
    return canvas[r0:r1, c0:c1].copy()


def _draw_legend(
    canvas: np.ndarray,
    entries: list[tuple[tuple[int, int, int], str]],
    origin: tuple[int, int] = (10, 10),
    box_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if box_size is None:
        box_size = (360, 30 + 18 * len(entries))
    legend = canvas.copy()
    x0, y0 = origin
    x1 = x0 + box_size[0]
    y1 = y0 + box_size[1]
    cv2.rectangle(legend, (x0, y0), (x1, y1), (240, 240, 240), -1)
    cv2.rectangle(legend, (x0, y0), (x1, y1), (80, 80, 80), 1)
    for i, (color, text) in enumerate(entries):
        y = y0 + 25 + i * 18
        cv2.line(legend, (x0 + 15, y), (x0 + 45, y), color, 2, cv2.LINE_AA)
        cv2.putText(
            legend,
            text,
            (x0 + 55, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return legend


def render_single_track_panel(
    data: MapData,
    centerline: np.ndarray,
    left: np.ndarray | None = None,
    right: np.ndarray | None = None,
    *,
    center_color: tuple[int, int, int] = (0, 200, 0),
    left_color: tuple[int, int, int] = (255, 200, 0),
    right_color: tuple[int, int, int] = (200, 100, 255),
    title: str = "Track",
    zoom_to_tracks: bool = True,
    pad_px: int = 40,
) -> np.ndarray:
    canvas = render_map_base(data)
    if left is not None:
        _draw_polyline(canvas, left, data, left_color, 1)
    if right is not None:
        _draw_polyline(canvas, right, data, right_color, 1)
    _draw_polyline(canvas, centerline, data, center_color, 2)

    point_sets = [centerline]
    if left is not None:
        point_sets.append(left)
    if right is not None:
        point_sets.append(right)
    if zoom_to_tracks:
        canvas = _crop_to_tracks(canvas, data, point_sets, pad_px=pad_px)

    entries = [(center_color, f"{title} centerline")]
    if left is not None:
        entries.append((left_color, f"{title} left boundary"))
    if right is not None:
        entries.append((right_color, f"{title} right boundary"))
    return _draw_legend(canvas, entries)


def render_side_by_side_panels(
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    left_label: str,
    right_label: str,
) -> np.ndarray:
    h = max(left_panel.shape[0], right_panel.shape[0])
    w = left_panel.shape[1] + right_panel.shape[1] + 20
    canvas = np.full((h + 50, w, 3), 245, dtype=np.uint8)
    y0 = 40
    canvas[y0 : y0 + left_panel.shape[0], 0 : left_panel.shape[1]] = left_panel
    x1 = left_panel.shape[1] + 20
    canvas[y0 : y0 + right_panel.shape[0], x1 : x1 + right_panel.shape[1]] = right_panel
    cv2.putText(
        canvas,
        left_label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        right_label,
        (x1, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_track_overlay(
    data: MapData,
    ref_centerline: np.ndarray,
    ext_centerline: np.ndarray,
    ref_left: np.ndarray | None = None,
    ref_right: np.ndarray | None = None,
    ext_left: np.ndarray | None = None,
    ext_right: np.ndarray | None = None,
    zoom_to_tracks: bool = True,
    pad_px: int = 40,
) -> np.ndarray:
    canvas = render_map_base(data)

    if ref_left is not None:
        _draw_polyline(canvas, ref_left, data, (255, 200, 0), 1)
    if ref_right is not None:
        _draw_polyline(canvas, ref_right, data, (200, 100, 255), 1)
    if ext_left is not None:
        _draw_polyline(canvas, ext_left, data, (0, 140, 255), 1)
    if ext_right is not None:
        _draw_polyline(canvas, ext_right, data, (0, 255, 255), 1)

    _draw_polyline(canvas, ref_centerline, data, (0, 200, 0), 2)
    _draw_polyline(canvas, ext_centerline, data, (0, 0, 255), 2)

    if zoom_to_tracks:
        all_pts = [ref_centerline, ext_centerline]
        if ref_left is not None:
            all_pts.append(ref_left)
        if ref_right is not None:
            all_pts.append(ref_right)
        if ext_left is not None:
            all_pts.append(ext_left)
        if ext_right is not None:
            all_pts.append(ext_right)
        canvas = _crop_to_tracks(canvas, data, all_pts, pad_px=pad_px)

    return _draw_legend(
        canvas,
        [
            ((0, 200, 0), "Saved centerline"),
            ((255, 200, 0), "Saved left boundary"),
            ((200, 100, 255), "Saved right boundary"),
            ((0, 0, 255), "Extracted centerline"),
            ((0, 140, 255), "Extracted left boundary"),
            ((0, 255, 255), "Extracted right boundary"),
        ],
        box_size=(430, 150),
    )


def save_track_overlay(
    out_path: Path,
    data: MapData,
    ref_centerline: np.ndarray,
    ext_centerline: np.ndarray,
    ref_left: np.ndarray | None = None,
    ref_right: np.ndarray | None = None,
    ext_left: np.ndarray | None = None,
    ext_right: np.ndarray | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_track_overlay(
        data,
        ref_centerline,
        ext_centerline,
        ref_left=ref_left,
        ref_right=ref_right,
        ext_left=ext_left,
        ext_right=ext_right,
    )
    cv2.imwrite(str(out_path), img)
    return out_path

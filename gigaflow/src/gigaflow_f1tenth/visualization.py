"""Offline GPUDrive-style evaluation rendering."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from gigaflow_f1tenth.sim.geometry import derive_normals
from gigaflow_f1tenth.sim.layout_local import MAX_STEER_RAD
from gigaflow_f1tenth.tracks import PackedTrackAtlasView

# GPUDrive Matplotlib palette (gpudrive/visualize/color.py).
CAR_BLUE = "#4B77BE"
TRACK_BLACK = "#000000"
CENTER_GRAY = "#e6e6e6"
OPP_FILL = "#D8D8D8"

# Fixed half-extent (meters) for legacy close-follow framing tests.
FOLLOW_RADIUS_M = 5.0
# Canonical eval / side-by-side camera (SleepyToaster courtyard SBS).
EVAL_FOLLOW_RADIUS_M = 6.0
EVAL_PANEL_INCHES = 7.0
EVAL_PANEL_DPI = 100
EVAL_PANEL_PX = 700
EVAL_BANNER_PX = 28
EVAL_SBS_WIDTH_PX = 1400
EVAL_SBS_HEIGHT_PX = 728
EVAL_VIEW_ELEV = 30.0
EVAL_VIEW_AZIM = 45.0
EVAL_FPS = 10
# Visual-only cuboid height as a fraction of car length (2x prior 0.22 scale).
CUBOID_HEIGHT_FRAC = 0.44

POSE_X = 0
POSE_Y = 1
POSE_YAW = 2
POSE_ACTIVE = 3
POSE_EFFORT = 4
POSE_STEER = 5
POSE_TRAINABLE = 6
POSE_MIN_DIM = 4
POSE_FULL_DIM = 7

HUD_THROTTLE_RGB = (70, 210, 90)
HUD_BRAKE_RGB = (230, 50, 50)
HUD_PANEL_RGB = (22, 24, 30)
HUD_BAR_BG_RGB = (38, 40, 48)
HUD_MARGIN = 12
HUD_PANEL_W = 168
HUD_PANEL_H = 132
HUD_BAR_W = 16
HUD_BAR_H = 88
HUD_BAR_TOP = 16
HUD_THROTTLE_X = 18
HUD_BRAKE_GAP = 12
HUD_WHEEL_LOCK_DEG = 90.0


def track_lines(
    atlas: PackedTrackAtlasView, track_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, left boundary, and right boundary XY polylines."""
    offsets = np.asarray(atlas.offsets)
    start, stop = int(offsets[track_id]), int(offsets[track_id + 1])
    center = np.asarray(atlas.centerline_xy, dtype=np.float32)[start:stop]
    tangent = np.asarray(atlas.tangents_xy, dtype=np.float32)[start:stop]
    widths = np.asarray(atlas.widths_rl, dtype=np.float32)[start:stop]
    normal = derive_normals(tangent)
    left = center + normal * widths[:, 1:2]
    right = center - normal * widths[:, 0:1]
    return center, left, right


def vehicle_faces(
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    height: float,
) -> list[np.ndarray]:
    """Build the six faces of a low oriented vehicle cuboid."""
    local = np.asarray(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    xy = local @ rotation.T + np.asarray([x, y], dtype=np.float32)
    bottom = np.column_stack((xy, np.zeros(4, dtype=np.float32)))
    top = bottom.copy()
    top[:, 2] = height
    faces = [bottom, top]
    faces.extend(
        np.asarray([bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]])
        for i in range(4)
    )
    return faces


def as_pose_rows(poses: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < POSE_MIN_DIM:
        raise ValueError("poses must have at least [x, y, yaw, active]")
    return np.reshape(arr, (-1, arr.shape[-1]))


def follow_center_xy(poses: np.ndarray) -> tuple[float, float]:
    """Return XY of the first active car, else the first pose slot."""
    arr = as_pose_rows(poses)
    if arr.size == 0:
        raise ValueError("poses must contain at least one car")
    for row in arr:
        if row[POSE_ACTIVE] > 0:
            return float(row[POSE_X]), float(row[POSE_Y])
    return float(arr[0, POSE_X]), float(arr[0, POSE_Y])


def frame_xy_limits(
    boundary_xy: np.ndarray,
    poses: np.ndarray,
    *,
    follow_radius: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return (xlim, ylim, z_top) for full-track or close-follow framing."""
    points = np.asarray(boundary_xy, dtype=np.float32).reshape(-1, 2)
    if follow_radius is not None:
        radius = float(follow_radius)
        if radius <= 0.0:
            raise ValueError("follow_radius must be positive")
        cx, cy = follow_center_xy(poses)
        span = 2.0 * radius
        return (
            (cx - radius, cx + radius),
            (cy - radius, cy + radius),
            max(1.0, 0.08 * span),
        )
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    margin = 0.05 * float(span.max())
    return (
        (float(lo[0] - margin), float(hi[0] + margin)),
        (float(lo[1] - margin), float(hi[1] + margin)),
        max(1.0, 0.08 * float(span.max())),
    )


def _car_style(row: np.ndarray) -> tuple[str, str, float]:
    trainable = 1.0
    if row.shape[0] > POSE_TRAINABLE:
        trainable = float(row[POSE_TRAINABLE])
    if trainable > 0:
        return CAR_BLUE, "black", 0.8
    return OPP_FILL, CAR_BLUE, 1.6


def _hud_follow_row(poses: np.ndarray) -> np.ndarray | None:
    arr = as_pose_rows(poses)
    first_active = None
    for row in arr:
        if row[POSE_ACTIVE] <= 0:
            continue
        if first_active is None:
            first_active = row
        if row.shape[0] <= POSE_TRAINABLE or float(row[POSE_TRAINABLE]) > 0:
            return row
    return first_active


def hud_bar_xy(
    frame_shape: tuple[int, ...], which: str
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of a HUD pedal bar in frame pixels."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    scale = h / float(EVAL_PANEL_PX)
    margin = max(1, int(round(HUD_MARGIN * scale)))
    panel_w = max(8, int(round(HUD_PANEL_W * scale)))
    bar_w = max(2, int(round(HUD_BAR_W * scale)))
    bar_h = max(4, int(round(HUD_BAR_H * scale)))
    bar_top = max(1, int(round(HUD_BAR_TOP * scale)))
    x0 = w - margin - panel_w
    y0 = margin
    throttle_x = x0 + max(1, int(round(HUD_THROTTLE_X * scale)))
    if which == "brake":
        gap = max(1, int(round(HUD_BRAKE_GAP * scale)))
        bx = throttle_x + bar_w + gap
    else:
        bx = throttle_x
    by = y0 + bar_top
    return bx, by, bx + bar_w, by + bar_h


def _steering_wheel_rgba(radius: int, lock_frac: float):
    from PIL import Image, ImageDraw

    pad = 6
    size = 2 * int(radius) + 2 * pad
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = size // 2
    r = int(radius)
    draw.ellipse(
        (c - r, c - r, c + r, c + r),
        outline=(245, 245, 248, 255),
        width=max(2, r // 8),
    )
    draw.ellipse(
        (c - r + 8, c - r + 8, c + r - 8, c + r - 8),
        outline=(70, 72, 80, 255),
        width=2,
    )
    hub = max(4, r // 5)
    draw.ellipse(
        (c - hub, c - hub, c + hub, c + hub),
        fill=(200, 202, 210, 255),
        outline=(245, 245, 248, 255),
        width=2,
    )
    draw.line(
        (c, c - r + 3, c, c - hub),
        fill=(85, 182, 255, 255),
        width=max(3, r // 8),
    )
    angle = float(np.clip(lock_frac, -1.0, 1.0)) * HUD_WHEEL_LOCK_DEG
    return img.rotate(-angle, resample=Image.BILINEAR, center=(c, c))


def overlay_controls_hud(
    frame: np.ndarray,
    *,
    throttle: float,
    brake: float,
    steer_rad: float,
    max_steer_rad: float = MAX_STEER_RAD,
) -> np.ndarray:
    """Draw green throttle, red brake, and a steering wheel in the top-right."""
    out = np.array(frame, copy=True, dtype=np.uint8)
    h, w = out.shape[:2]
    scale = h / float(EVAL_PANEL_PX)
    margin = max(1, int(round(HUD_MARGIN * scale)))
    panel_w = max(8, int(round(HUD_PANEL_W * scale)))
    panel_h = max(8, int(round(HUD_PANEL_H * scale)))
    x0 = w - margin - panel_w
    y0 = margin
    x1 = min(w - 1, x0 + panel_w)
    y1 = min(h - 1, y0 + panel_h)
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = np.asarray(HUD_PANEL_RGB, dtype=np.uint8)

    def _fill_bar(which: str, frac: float, color: tuple[int, int, int]) -> None:
        bx0, by0, bx1, by1 = hud_bar_xy(out.shape, which)
        bx0, by0 = max(x0, bx0), max(y0, by0)
        bx1, by1 = min(x1, bx1), min(y1, by1)
        if bx1 <= bx0 or by1 <= by0:
            return
        out[by0:by1, bx0:bx1] = np.asarray(HUD_BAR_BG_RGB, dtype=np.uint8)
        fill = float(np.clip(frac, 0.0, 1.0))
        fh = int(round((by1 - by0) * fill))
        if fh > 0:
            out[by1 - fh : by1, bx0:bx1] = np.asarray(color, dtype=np.uint8)

    _fill_bar("throttle", throttle, HUD_THROTTLE_RGB)
    _fill_bar("brake", brake, HUD_BRAKE_RGB)

    wheel_r = max(8, int(round(36 * scale)))
    lock = float(steer_rad) / max(float(max_steer_rad), 1e-6)
    wheel = _steering_wheel_rgba(wheel_r, lock)
    wx = x0 + max(1, int(round(88 * scale)))
    wy = y0 + max(1, int(round(14 * scale)))
    overlay = np.asarray(wheel)
    y_end = min(out.shape[0], wy + overlay.shape[0])
    x_end = min(out.shape[1], wx + overlay.shape[1])
    patch = overlay[: y_end - wy, : x_end - wx]
    if patch.size:
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        roi = out[wy:y_end, wx:x_end].astype(np.float32)
        out[wy:y_end, wx:x_end] = np.clip(
            alpha * patch[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi,
            0,
            255,
        ).astype(np.uint8)
    return out


def append_banner(frame: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((EVAL_BANNER_PX, frame.shape[1], 3), 20, dtype=np.uint8)
    out = np.concatenate([bar, frame], axis=0)
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(out)
        ImageDraw.Draw(img).text((8, 6), text, fill=(255, 255, 255))
        return np.asarray(img)
    except Exception:
        return out


def render_frame(
    atlas: PackedTrackAtlasView,
    track_id: int,
    poses: np.ndarray,
    *,
    car_length: float,
    car_width: float,
    dpi: int = EVAL_PANEL_DPI,
    follow_radius: float | None = None,
    hud: bool = True,
) -> np.ndarray:
    """Render one RGB frame with the GPUDrive demo aesthetic."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    center, left, right = track_lines(atlas, track_id)
    fig = Figure(
        figsize=(EVAL_PANEL_INCHES, EVAL_PANEL_INCHES), dpi=dpi, facecolor="white"
    )
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.plot(left[:, 0], left[:, 1], 0.0, color=TRACK_BLACK, linewidth=1.8)
    ax.plot(right[:, 0], right[:, 1], 0.0, color=TRACK_BLACK, linewidth=1.8)
    ax.plot(center[:, 0], center[:, 1], 0.0, color=CENTER_GRAY, linewidth=1.0)

    height = CUBOID_HEIGHT_FRAC * car_length
    rows = as_pose_rows(poses)
    for row in rows:
        if row[POSE_ACTIVE] <= 0:
            continue
        x, y, yaw = float(row[POSE_X]), float(row[POSE_Y]), float(row[POSE_YAW])
        face, edge, lw = _car_style(row)
        box = Poly3DCollection(
            vehicle_faces(x, y, yaw, car_length, car_width, height),
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.7,
        )
        ax.add_collection3d(box)
        ax.quiver(
            x,
            y,
            height,
            0.8 * car_length * math.cos(yaw),
            0.8 * car_length * math.sin(yaw),
            0.0,
            color="black",
            linewidth=1.0,
            arrow_length_ratio=0.25,
        )

    boundary = np.concatenate((left, right), axis=0)
    xlim, ylim, z_top = frame_xy_limits(
        boundary, poses, follow_radius=follow_radius
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0.0, z_top)
    ax.set_box_aspect(
        (float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]), z_top)
    )
    ax.view_init(elev=EVAL_VIEW_ELEV, azim=EVAL_VIEW_AZIM)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    fig.clear()
    if hud:
        follow = _hud_follow_row(poses)
        effort = 0.0
        steer = 0.0
        if follow is not None:
            if follow.shape[0] > POSE_EFFORT:
                effort = float(follow[POSE_EFFORT])
            if follow.shape[0] > POSE_STEER:
                steer = float(follow[POSE_STEER])
        rgb = overlay_controls_hud(
            rgb,
            throttle=max(effort, 0.0),
            brake=max(-effort, 0.0),
            steer_rad=steer,
        )
    return rgb


def render_episode(
    atlas: PackedTrackAtlasView,
    track_id: int,
    frames: list[np.ndarray],
    output_stem: str | Path,
    *,
    car_length: float,
    car_width: float,
    fps: int = EVAL_FPS,
    follow_radius: float | None = EVAL_FOLLOW_RADIUS_M,
    hud: bool = True,
    banner: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write a final PNG plus streaming MP4 and GIF animations."""
    import imageio.v2 as imageio

    if not frames:
        raise ValueError("cannot render an empty episode")

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_name(f"{stem.name}_frame").with_suffix(".png")
    mp4_path = stem.with_suffix(".mp4")
    gif_path = stem.with_suffix(".gif")
    with imageio.get_writer(
        mp4_path, fps=fps, codec="libx264", macro_block_size=1
    ) as mp4, imageio.get_writer(
        gif_path, mode="I", duration=1000.0 / float(fps), loop=0
    ) as gif:
        last = None
        for poses in frames:
            last = render_frame(
                atlas,
                track_id,
                poses,
                car_length=car_length,
                car_width=car_width,
                follow_radius=follow_radius,
                hud=hud,
            )
            if banner:
                last = append_banner(last, banner)
            mp4.append_data(last)
            gif.append_data(last)
    imageio.imwrite(png_path, last)
    return png_path, mp4_path, gif_path


def render_side_by_side(
    atlas: PackedTrackAtlasView,
    left_poses: Sequence[np.ndarray],
    right_poses: Sequence[np.ndarray],
    output: Path,
    *,
    car_length: float,
    car_width: float,
    fps: int = EVAL_FPS,
    track_id: int = 0,
    left_label: str = "Left",
    right_label: str = "Right",
    follow_radius: float = EVAL_FOLLOW_RADIUS_M,
    hud: bool = True,
) -> Path:
    """Write the canonical 1400x728 SleepyToaster side-by-side eval video."""
    import imageio.v2 as imageio

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(left_poses), len(right_poses))
    empty = np.zeros((1, POSE_FULL_DIM), dtype=np.float32)
    last_l = left_poses[-1] if left_poses else empty
    last_r = right_poses[-1] if right_poses else empty
    with imageio.get_writer(
        output, fps=fps, codec="libx264", macro_block_size=1
    ) as mp4:
        for i in range(n):
            lp = left_poses[i] if i < len(left_poses) else last_l
            rp = right_poses[i] if i < len(right_poses) else last_r
            left = render_frame(
                atlas,
                track_id,
                lp,
                car_length=car_length,
                car_width=car_width,
                follow_radius=follow_radius,
                hud=hud,
            )
            right = render_frame(
                atlas,
                track_id,
                rp,
                car_length=car_length,
                car_width=car_width,
                follow_radius=follow_radius,
                hud=hud,
            )
            left = append_banner(left, left_label)
            right = append_banner(right, right_label)
            h = min(left.shape[0], right.shape[0])
            w = min(left.shape[1], right.shape[1])
            mp4.append_data(np.concatenate([left[:h, :w], right[:h, :w]], axis=1))
    return output

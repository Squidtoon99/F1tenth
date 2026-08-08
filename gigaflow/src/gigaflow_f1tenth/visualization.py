"""Offline GPUDrive-style evaluation rendering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gigaflow_f1tenth.sim.geometry import derive_normals
from gigaflow_f1tenth.tracks import PackedTrackAtlasView

# GPUDrive Matplotlib palette (gpudrive/visualize/color.py).
CAR_BLUE = "#4B77BE"
TRACK_BLACK = "#000000"
CENTER_GRAY = "#e6e6e6"

# Fixed half-extent (meters) for solo close-follow framing.
FOLLOW_RADIUS_M = 5.0
# Visual-only cuboid height as a fraction of car length (2x prior 0.22 scale).
CUBOID_HEIGHT_FRAC = 0.44


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


def follow_center_xy(poses: np.ndarray) -> tuple[float, float]:
    """Return XY of the first active car, else the first pose slot."""
    arr = np.asarray(poses, dtype=np.float32).reshape(-1, 4)
    if arr.size == 0:
        raise ValueError("poses must contain at least one car")
    for x, y, _yaw, active in arr:
        if active > 0:
            return float(x), float(y)
    return float(arr[0, 0]), float(arr[0, 1])


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


def render_frame(
    atlas: PackedTrackAtlasView,
    track_id: int,
    poses: np.ndarray,
    *,
    car_length: float,
    car_width: float,
    dpi: int = 100,
    follow_radius: float | None = None,
) -> np.ndarray:
    """Render one RGB frame with the GPUDrive demo aesthetic."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    center, left, right = track_lines(atlas, track_id)
    fig = Figure(figsize=(7, 7), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.plot(left[:, 0], left[:, 1], 0.0, color=TRACK_BLACK, linewidth=1.8)
    ax.plot(right[:, 0], right[:, 1], 0.0, color=TRACK_BLACK, linewidth=1.8)
    ax.plot(center[:, 0], center[:, 1], 0.0, color=CENTER_GRAY, linewidth=1.0)

    height = CUBOID_HEIGHT_FRAC * car_length
    for x, y, yaw, active in np.asarray(poses).reshape(-1, 4):
        if active <= 0:
            continue
        box = Poly3DCollection(
            vehicle_faces(x, y, yaw, car_length, car_width, height),
            facecolor=CAR_BLUE,
            edgecolor="black",
            linewidth=0.8,
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
    ax.view_init(elev=30.0, azim=45.0)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    fig.clear()
    return rgb


def render_episode(
    atlas: PackedTrackAtlasView,
    track_id: int,
    frames: list[np.ndarray],
    output_stem: str | Path,
    *,
    car_length: float,
    car_width: float,
    fps: int = 10,
    follow_radius: float | None = None,
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
            )
            mp4.append_data(last)
            gif.append_data(last)
    imageio.imwrite(png_path, last)
    return png_path, mp4_path, gif_path

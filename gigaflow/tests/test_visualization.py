from __future__ import annotations

import numpy as np

from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.visualization import (
    CUBOID_HEIGHT_FRAC,
    FOLLOW_RADIUS_M,
    frame_xy_limits,
    render_episode,
    render_frame,
    track_lines,
    vehicle_faces,
)


def test_track_lines_follow_atlas_widths():
    atlas = make_synthetic_oval_atlas(radius=8.0, half_width=1.1)
    center, left, right = track_lines(atlas, 0)
    assert np.allclose(np.linalg.norm(left - center, axis=1), 1.1, atol=1e-4)
    assert np.allclose(np.linalg.norm(right - center, axis=1), 1.1, atol=1e-4)


def test_vehicle_cuboid_and_frame_render():
    # Render scale is visual-only: 2x the prior 0.22 * length fraction.
    assert CUBOID_HEIGHT_FRAC == 0.44
    height = CUBOID_HEIGHT_FRAC * 0.568
    faces = vehicle_faces(0.0, 0.0, 0.0, 0.568, 0.296, height)
    assert len(faces) == 6
    assert all(face.shape == (4, 3) for face in faces)
    assert np.allclose(faces[1][:, 2], height)

    atlas = make_synthetic_oval_atlas()
    image = render_frame(
        atlas,
        0,
        np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32),
        car_length=0.568,
        car_width=0.296,
        dpi=40,
    )
    assert image.shape == (280, 280, 3)
    assert image.dtype == np.uint8
    assert image.min() < 100


def test_inactive_cars_are_skipped():
    atlas = make_synthetic_oval_atlas()
    blank = render_frame(
        atlas,
        0,
        np.asarray([[8.0, 0.0, 1.57, 0.0]], dtype=np.float32),
        car_length=0.568,
        car_width=0.296,
        dpi=40,
    )
    active = render_frame(
        atlas,
        0,
        np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32),
        car_length=0.568,
        car_width=0.296,
        dpi=40,
    )
    assert not np.array_equal(blank, active)


def test_full_track_framing_covers_boundaries():
    atlas = make_synthetic_oval_atlas(radius=8.0, half_width=1.1)
    _center, left, right = track_lines(atlas, 0)
    boundary = np.concatenate((left, right), axis=0)
    poses = np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32)
    xlim, ylim, z_top = frame_xy_limits(boundary, poses, follow_radius=None)
    lo, hi = boundary.min(axis=0), boundary.max(axis=0)
    assert xlim[0] <= lo[0] and xlim[1] >= hi[0]
    assert ylim[0] <= lo[1] and ylim[1] >= hi[1]
    assert z_top >= 1.0
    # Whole-track window is larger than the solo follow box.
    assert (xlim[1] - xlim[0]) > 2.0 * FOLLOW_RADIUS_M * 0.9


def test_follow_framing_centers_car_with_stable_scale():
    atlas = make_synthetic_oval_atlas(radius=8.0, half_width=1.1)
    _center, left, right = track_lines(atlas, 0)
    boundary = np.concatenate((left, right), axis=0)
    radius = FOLLOW_RADIUS_M
    poses_a = np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32)
    poses_b = np.asarray([[0.0, 8.0, 0.0, 1.0]], dtype=np.float32)
    xlim_a, ylim_a, z_a = frame_xy_limits(
        boundary, poses_a, follow_radius=radius
    )
    xlim_b, ylim_b, z_b = frame_xy_limits(
        boundary, poses_b, follow_radius=radius
    )
    assert abs((xlim_a[0] + xlim_a[1]) * 0.5 - 8.0) < 1e-5
    assert abs((ylim_a[0] + ylim_a[1]) * 0.5 - 0.0) < 1e-5
    assert abs((xlim_b[0] + xlim_b[1]) * 0.5 - 0.0) < 1e-5
    assert abs((ylim_b[0] + ylim_b[1]) * 0.5 - 8.0) < 1e-5
    assert abs((xlim_a[1] - xlim_a[0]) - 2.0 * radius) < 1e-5
    assert abs((ylim_a[1] - ylim_a[0]) - 2.0 * radius) < 1e-5
    assert abs((xlim_b[1] - xlim_b[0]) - (xlim_a[1] - xlim_a[0])) < 1e-5
    assert abs((ylim_b[1] - ylim_b[0]) - (ylim_a[1] - ylim_a[0])) < 1e-5
    assert abs(z_a - z_b) < 1e-5

    # Inactive cars still keep close-follow scale (no full-track fallback).
    poses_dead = np.asarray([[3.0, -2.0, 0.4, 0.0]], dtype=np.float32)
    xlim_d, ylim_d, _z_d = frame_xy_limits(
        boundary, poses_dead, follow_radius=radius
    )
    assert abs((xlim_d[0] + xlim_d[1]) * 0.5 - 3.0) < 1e-5
    assert abs((ylim_d[0] + ylim_d[1]) * 0.5 - (-2.0)) < 1e-5
    assert abs((xlim_d[1] - xlim_d[0]) - 2.0 * radius) < 1e-5


def test_follow_frame_render_differs_from_full_track():
    atlas = make_synthetic_oval_atlas()
    poses = np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32)
    full = render_frame(
        atlas,
        0,
        poses,
        car_length=0.568,
        car_width=0.296,
        dpi=40,
        follow_radius=None,
    )
    follow = render_frame(
        atlas,
        0,
        poses,
        car_length=0.568,
        car_width=0.296,
        dpi=40,
        follow_radius=FOLLOW_RADIUS_M,
    )
    assert full.shape == follow.shape == (280, 280, 3)
    assert not np.array_equal(full, follow)


def test_render_episode_writes_png_gif_mp4(tmp_path):
    atlas = make_synthetic_oval_atlas()
    frames = [
        np.asarray([[8.0, 0.0, 1.57, 1.0]], dtype=np.float32),
        np.asarray([[8.2, 0.1, 1.40, 1.0]], dtype=np.float32),
    ]
    png, mp4, gif = render_episode(
        atlas,
        0,
        frames,
        tmp_path / "solo_seed0",
        car_length=0.568,
        car_width=0.296,
        fps=5,
        follow_radius=FOLLOW_RADIUS_M,
    )
    assert png.name == "solo_seed0_frame.png"
    assert png.is_file() and png.stat().st_size > 0
    assert mp4.is_file() and mp4.stat().st_size > 0
    assert gif.is_file() and gif.stat().st_size > 0

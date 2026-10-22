from __future__ import annotations

import numpy as np

from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.visualization import (
    CUBOID_HEIGHT_FRAC,
    EVAL_FOLLOW_RADIUS_M,
    EVAL_SBS_HEIGHT_PX,
    EVAL_SBS_WIDTH_PX,
    FOLLOW_RADIUS_M,
    HUD_BRAKE_RGB,
    HUD_THROTTLE_RGB,
    frame_xy_limits,
    hud_bar_xy,
    render_episode,
    render_frame,
    render_side_by_side,
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


def _mean_rgb(image, which):
    x0, y0, x1, y1 = hud_bar_xy(image.shape, which)
    mid_x = (x0 + x1) // 2
    fill_y = y1 - max(2, (y1 - y0) // 8)
    return image[fill_y, mid_x].astype(np.float32)


def test_hud_throttle_is_green_and_brake_is_red():
    atlas = make_synthetic_oval_atlas()
    throttle = render_frame(
        atlas,
        0,
        np.asarray([[8.0, 0.0, 1.57, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32),
        car_length=0.568,
        car_width=0.296,
        dpi=100,
        follow_radius=EVAL_FOLLOW_RADIUS_M,
    )
    brake = render_frame(
        atlas,
        0,
        np.asarray([[8.0, 0.0, 1.57, 1.0, -1.0, 0.0, 1.0]], dtype=np.float32),
        car_length=0.568,
        car_width=0.296,
        dpi=100,
        follow_radius=EVAL_FOLLOW_RADIUS_M,
    )
    assert throttle.shape == (700, 700, 3)
    t_rgb = _mean_rgb(throttle, "throttle")
    b_rgb = _mean_rgb(brake, "brake")
    assert t_rgb[1] > t_rgb[0] and t_rgb[1] > t_rgb[2]
    assert abs(t_rgb[0] - HUD_THROTTLE_RGB[0]) < 5
    assert abs(t_rgb[1] - HUD_THROTTLE_RGB[1]) < 5
    assert b_rgb[0] > b_rgb[1] and b_rgb[0] > b_rgb[2]
    assert b_rgb[0] > 180
    assert abs(b_rgb[0] - HUD_BRAKE_RGB[0]) < 8
    assert b_rgb[1] < 80 and b_rgb[2] < 80


def test_render_side_by_side_is_1400x728(tmp_path):
    import imageio.v2 as imageio

    atlas = make_synthetic_oval_atlas()
    left = [np.asarray([[8.0, 0.0, 1.57, 1.0, 0.8, 0.1, 1.0]], dtype=np.float32)]
    right = [
        np.asarray(
            [
                [8.0, 0.0, 1.57, 1.0, 0.4, -0.1, 1.0],
                [7.2, 0.4, 0.2, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    out = tmp_path / "side_by_side.mp4"
    path = render_side_by_side(
        atlas,
        left,
        right,
        out,
        car_length=0.568,
        car_width=0.296,
        left_label="Left: Base actor_005700",
        right_label="Right: Courtyard Fine-Tuned",
    )
    assert path.is_file() and path.stat().st_size > 0
    reader = imageio.get_reader(path)
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    assert frame.shape[:2] == (EVAL_SBS_HEIGHT_PX, EVAL_SBS_WIDTH_PX)
    assert frame.shape[-1] >= 3

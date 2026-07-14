"""Analytic known-answer tests for the geometry observation components.

Uses synthetic straight/circle tracks where every value has a closed form, so a
failure pins the exact component and convention that is wrong. Genesis is only
needed for `quat_to_xyz` (planar yaw) and `tc_float`; no sim is started.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import (
    build_track_state,
    make_circle_track,
    make_straight_track,
    yaw_quat_wxyz,
)

DEVICE = torch.device("cpu")


def _step_state(real_modules, track_state, pos_xy):
    utils = real_modules.utils
    pos = np.asarray(pos_xy, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[None, :]
    base_pos = torch.tensor(
        np.concatenate([pos, np.zeros((pos.shape[0], 1), np.float32)], axis=-1),
        dtype=torch.float32,
    )
    return base_pos, utils.build_step_state(
        base_pos=base_pos,
        track_state=track_state,
        device=DEVICE,
        cache_id="geom",
    )


# --- centerline distance (ey) -------------------------------------------------
@pytest.mark.parametrize("y0", [0.0, 0.7, -0.5, 1.2])
def test_centerline_distance_straight_sign_and_magnitude(real_modules, y0):
    """Straight track along +x: ey must equal the signed lateral offset (left +)."""
    cl, wl, wr = make_straight_track()
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    _, ss = _step_state(real_modules, ts, [10.0, y0])
    ey = real_modules.observations.obs_centerline_distance(ss)[0, 0].item()
    assert ey == pytest.approx(y0, abs=2e-3), f"expected ey={y0}, got {ey}"


@pytest.mark.parametrize("r", [22.0, 25.0, 27.0])
def test_centerline_distance_circle(real_modules, r):
    """CCW circle radius R: a car at radius r has ey == R - r (inside is left/+)."""
    R = 25.0
    cl, wl, wr = make_circle_track(radius=R, n=720)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    phi = 0.6
    pos = [r * math.cos(phi), r * math.sin(phi)]
    _, ss = _step_state(real_modules, ts, pos)
    ey = real_modules.observations.obs_centerline_distance(ss)[0, 0].item()
    assert ey == pytest.approx(R - r, abs=2e-2), f"expected ey={R - r}, got {ey}"


# --- centerline heading error -------------------------------------------------
@pytest.mark.parametrize("yaw", [0.0, 0.3, -0.5, 1.2, math.pi - 0.1])
def test_centerline_angle_straight(real_modules, yaw):
    """Straight track (tangent angle 0): heading error == car yaw."""
    cl, wl, wr = make_straight_track()
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    _, ss = _step_state(real_modules, ts, [10.0, 0.0])
    quat = yaw_quat_wxyz(yaw)
    err = real_modules.observations.obs_centerline_angle(ss, quat)[0, 0].item()
    assert err == pytest.approx(yaw, abs=2e-3), f"expected err={yaw}, got {err}"


def test_centerline_angle_circle_aligned_is_zero(real_modules):
    """On a CCW circle, a car aligned with the tangent (yaw=phi+pi/2) has err ~ 0."""
    R = 25.0
    cl, wl, wr = make_circle_track(radius=R, n=720)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    phi = 0.6
    pos = [R * math.cos(phi), R * math.sin(phi)]
    _, ss = _step_state(real_modules, ts, pos)
    quat = yaw_quat_wxyz(phi + math.pi / 2.0)
    err = real_modules.observations.obs_centerline_angle(ss, quat)[0, 0].item()
    assert err == pytest.approx(0.0, abs=1e-2), f"expected err~0, got {err}"


# --- track progress -----------------------------------------------------------
@pytest.mark.parametrize("phi", [0.0, 1.0, 2.5, 4.0, 5.8])
def test_track_progress_circle(real_modules, phi):
    """On a circle starting at angle 0, progress (cos,sin) tracks the angle phi."""
    R = 25.0
    cl, wl, wr = make_circle_track(radius=R, n=720)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    pos = [R * math.cos(phi), R * math.sin(phi)]
    _, ss = _step_state(real_modules, ts, pos)
    prog = real_modules.observations.obs_track_progress(ss, DEVICE)[0]
    assert prog[0].item() == pytest.approx(math.cos(phi), abs=2e-2)
    assert prog[1].item() == pytest.approx(math.sin(phi), abs=2e-2)


# --- contact flag -------------------------------------------------------------
def test_contact_flag_inside_and_near_boundary(real_modules, obs_cfg):
    cl, wl, wr = make_straight_track(w_left=1.5, w_right=1.5)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    # centered: far from both boundaries -> no contact
    _, ss = _step_state(real_modules, ts, [10.0, 0.0])
    flag = real_modules.observations.obs_contact_flag(ss, obs_cfg)[0, 0].item()
    assert flag == 0.0
    # 0.05 m inside the left boundary (margin 0.08) -> contact
    _, ss = _step_state(real_modules, ts, [10.0, 1.45])
    flag = real_modules.observations.obs_contact_flag(ss, obs_cfg)[0, 0].item()
    assert flag == 1.0


# --- future track points ------------------------------------------------------
def test_future_points_straight_layout(real_modules, obs_cfg):
    """Straight track, yaw 0, forward speed: center points march +x, left/right at
    +/- half width; verifies ordering [center, left, right] and ego transform."""
    w_left, w_right = 1.2, 0.9
    cl, wl, wr = make_straight_track(length=60.0, n=240, w_left=w_left, w_right=w_right)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _step_state(real_modules, ts, [10.0, 0.0])
    quat = yaw_quat_wxyz(0.0)
    vel = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)  # 5 m/s forward

    out = real_modules.observations.obs_future_track_points(
        base_pos, quat, vel, obs_cfg, DEVICE, ss
    )
    n = int(obs_cfg["future_track_num_points"])
    flat = out[0]
    center = flat[: 2 * n].view(n, 2)
    left = flat[2 * n : 4 * n].view(n, 2)
    right = flat[4 * n :].view(n, 2)

    horizon = float(obs_cfg["future_track_horizon_s"])
    spacing = (5.0 * horizon) / n  # lookahead / N along arc length

    # center marches forward along ego +x at uniform spacing, ~0 lateral
    assert torch.allclose(center[:, 1], torch.zeros(n), atol=1e-2)
    assert center[0, 0].item() == pytest.approx(spacing, abs=2e-2)
    assert center[-1, 0].item() == pytest.approx(n * spacing, abs=5e-2)
    # left/right offset by per-vertex CSV half-widths in ego y
    assert torch.allclose(left[:, 1], torch.full((n,), w_left), atol=2e-2)
    assert torch.allclose(right[:, 1], torch.full((n,), -w_right), atol=2e-2)


def test_future_points_speed_zero_uses_min_lookahead(real_modules, obs_cfg):
    """FIX: at speed 0 the lookahead is floored at future_track_min_lookahead_m, so
    the policy still gets a track preview spanning that distance instead of all 60
    samples collapsing onto the current point."""
    w_left, w_right = 1.2, 0.9
    cl, wl, wr = make_straight_track(length=60.0, n=240, w_left=w_left, w_right=w_right)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _step_state(real_modules, ts, [10.0, 0.0])
    quat = yaw_quat_wxyz(0.0)
    vel = torch.zeros((1, 3), dtype=torch.float32)

    out = real_modules.observations.obs_future_track_points(
        base_pos, quat, vel, obs_cfg, DEVICE, ss
    )
    n = int(obs_cfg["future_track_num_points"])
    min_la = float(obs_cfg["future_track_min_lookahead_m"])
    center = out[0][: 2 * n].view(n, 2)

    # samples no longer collapse: they span the min-lookahead distance along +x
    spread = (center - center[0:1]).abs().max().item()
    assert spread > 0.5 * min_la, f"expected preview, got spread={spread}"
    spacing = min_la / n
    assert center[0, 0].item() == pytest.approx(spacing, abs=2e-2)
    assert center[-1, 0].item() == pytest.approx(min_la, abs=5e-2)
    assert torch.allclose(center[:, 1], torch.zeros(n), atol=1e-2)

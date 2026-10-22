"""Courtyard path/teacher helpers (no actor geometry inputs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gigaflow_f1tenth.courtyard_adapt import (
    COURTYARD_BAG_PREFIX,
    SPAWN_S,
    WHEELBASE,
    CenterlinePath,
    is_courtyard_bag,
    spawn_menu,
    teacher_action,
    wrap_s,
)
from gigaflow_f1tenth.sim.layout_local import MAX_STEER_RAD, STEERING_DELTA_MAX_RAD
from gigaflow_f1tenth.teleop_dataset import SENSOR_OBS_DIM, teleop_to_action


def _circle(radius: float = 2.0, n: int = 180) -> CenterlinePath:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    widths = np.full((n, 2), 0.8, dtype=np.float64)
    return CenterlinePath(xy, widths, name="circle")


def test_circle_curvature_and_length():
    radius = 2.0
    path = _circle(radius=radius)
    assert abs(path.length - 2.0 * np.pi * radius) < 0.08
    kappa = np.mean(path.kappa)
    assert abs(kappa - 1.0 / radius) < 0.05


def test_teacher_left_circle_commands_positive_steer():
    path = _circle(radius=1.5)
    pose = path.pose_at(0.0, ey=0.0)
    act = teacher_action(
        path,
        x=pose["x"],
        y=pose["y"],
        yaw=pose["yaw"],
        s=pose["s"],
        ey=0.0,
        vx=1.2,
        steer_now=0.0,
    )
    assert act.shape == (2,)
    assert act[1] > 0.2
    req = path.required_steer(0.0, lookahead_m=0.0)
    assert 0.1 < req <= MAX_STEER_RAD
    assert np.arctan(WHEELBASE / 1.5) > 0.1


def test_courtyard_bag_filter_excludes_skidpad_and_arvin():
    assert is_courtyard_bag(Path("/data/boring_track_data_12"))
    assert not is_courtyard_bag(Path("/data/manual_control_arvin_1"))
    assert not is_courtyard_bag(Path("/data/skidpad_lap"))
    assert COURTYARD_BAG_PREFIX == "boring_track_data_"


def test_spawn_menu_includes_from_rest_and_eight_stations():
    path = _circle(radius=2.8)
    menu = spawn_menu(path)
    assert len(menu) == 9
    assert menu[0] == (SPAWN_S, 0.0)
    stations = [row[0] for row in menu[1:]]
    assert abs(stations[0]) < 1e-9
    step = path.length / 8.0
    assert abs(stations[3] - 3.0 * step) < 1e-9


def test_steer_delta_needs_many_steps_to_reach_max():
    steps = int(np.ceil(MAX_STEER_RAD / STEERING_DELTA_MAX_RAD))
    assert steps >= 12
    action, steer = teleop_to_action(0.4, MAX_STEER_RAD, 0.0)
    assert action.shape == (2,)
    assert action[1] == 1.0
    assert abs(steer - MAX_STEER_RAD) < 1e-6


def test_student_observation_has_no_track_geometry_slots():
    assert SENSOR_OBS_DIM == 1097
    path = _circle()
    assert wrap_s(path.length + 0.3, path.length) < 0.31


def test_centerline_interp_wraps_at_full_length():
    path = _circle(radius=2.0, n=64)
    a = path.xy_at(0.0)
    b = path.xy_at(path.length)
    c = path.xy_at(path.length - 1e-6)
    d = path.yaw_at(path.length + 0.4)
    assert np.allclose(a, b, atol=1e-5)
    assert np.linalg.norm(c - a) < 0.05
    assert np.isfinite(d)
    teacher_action(
        path,
        x=float(a[0]),
        y=float(a[1]),
        yaw=path.yaw_at(0.0),
        s=path.length - 1e-9,
        ey=0.0,
        vx=1.0,
        steer_now=0.0,
    )

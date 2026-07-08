"""Tests for pure pursuit steering."""

from __future__ import annotations

import math

import numpy as np

from f1tenth_mapping.pure_pursuit import (
    compute_steering,
    lookahead_point,
    nearest_path_index,
)


def test_nearest_path_index():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert nearest_path_index(path, np.array([1.1, 0.0])) == 1


def test_lookahead_point_returns_far_point():
    path = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    pt = lookahead_point(path, np.array([0.0, 0.0]), 1.0)
    assert pt is not None
    assert pt[0] == 2.0


def test_compute_steering_straight_path():
    path = np.array([[0.0, 0.0], [5.0, 0.0]])
    steer, _ = compute_steering(
        np.array([0.0, 0.0]),
        0.0,
        path,
        wheelbase_m=0.33,
        lookahead_m=1.0,
        max_steer_rad=0.44,
    )
    assert abs(steer) < 0.05


def test_compute_steering_left_turn():
    path = np.array([[0.0, 0.0], [0.0, 3.0]])
    steer, _ = compute_steering(
        np.array([0.0, 0.0]),
        0.0,
        path,
        wheelbase_m=0.33,
        lookahead_m=0.5,
        max_steer_rad=0.44,
    )
    assert steer > 0.1


def test_speed_pid_limits_output():
    from f1tenth_mapping.pure_pursuit import SpeedPID

    pid = SpeedPID(kp=10.0, ki=0.0, kd=0.0, limit=0.5)
    out = pid.compute(target_speed=5.0, current_speed=0.0, dt=0.1)
    assert out == 0.5

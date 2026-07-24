"""Tests for normalized-action to physical actuator mapping."""

from __future__ import annotations

import math

import pytest

from f1tenth_rl_agent.sensor_action import (
    integrate_steering_delta,
    normalized_action_to_physical,
)


def test_drive_maps_to_positive_current_only():
    drive, brake, servo, long_norm, steer_norm = normalized_action_to_physical(
        0.5,
        0.0,
        i_drive_max_a=10.0,
        i_brake_max_a=8.0,
        max_steer=0.33,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive == pytest.approx(5.0)
    assert brake == pytest.approx(0.0)
    assert long_norm == pytest.approx(0.5)
    assert steer_norm == pytest.approx(0.0)
    assert servo == pytest.approx(0.4495)


def test_brake_maps_to_positive_brake_current_only():
    drive, brake, _, long_norm, _ = normalized_action_to_physical(
        -0.25,
        0.0,
        i_drive_max_a=10.0,
        i_brake_max_a=8.0,
        max_steer=0.33,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive == pytest.approx(0.0)
    assert brake == pytest.approx(2.0)
    assert long_norm == pytest.approx(-0.25)


def test_steering_maps_to_servo_with_calibration():
    _, _, servo, _, steer_norm = normalized_action_to_physical(
        0.0,
        1.0,
        i_drive_max_a=10.0,
        i_brake_max_a=8.0,
        max_steer=0.33,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert steer_norm == pytest.approx(1.0)
    assert servo == pytest.approx(-1.2135 * 0.33 + 0.4495)


def test_nonfinite_input_coasts():
    drive, brake, servo, long_norm, steer_norm = normalized_action_to_physical(
        float("nan"),
        float("inf"),
        i_drive_max_a=10.0,
        i_brake_max_a=8.0,
        max_steer=0.33,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive == pytest.approx(0.0)
    assert brake == pytest.approx(0.0)
    assert servo == pytest.approx(0.4495)
    assert long_norm == pytest.approx(0.0)
    assert steer_norm == pytest.approx(0.0)


def test_delta_steering_integrates_three_degrees_and_saturates():
    steer = 0.0
    step = math.pi / 60.0
    steer = integrate_steering_delta(
        steer, 1.0, delta_max_rad=step, max_steer=0.33
    )
    assert steer == pytest.approx(step)
    steer = integrate_steering_delta(
        steer, -1.0, delta_max_rad=step, max_steer=0.33
    )
    assert steer == pytest.approx(0.0)
    for _ in range(20):
        steer = integrate_steering_delta(
            steer, 1.0, delta_max_rad=step, max_steer=0.33
        )
    assert steer == pytest.approx(0.33)

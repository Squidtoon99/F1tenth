"""Pure tests for the action -> Ackermann mapping."""

import math

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_control.drive_math import lag_alpha, map_action_to_drive, step_first_order_lag


def test_full_throttle_zero_steer():
    speed, steer = map_action_to_drive(1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, ifc.MAX_SPEED)
    assert math.isclose(steer, 0.0)


def test_brake_stops_by_default():
    speed, steer = map_action_to_drive(-1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, 0.0)


def test_reverse_behavior():
    speed, _ = map_action_to_drive(
        -1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER, brake_behavior="reverse"
    )
    assert math.isclose(speed, -ifc.MAX_SPEED)


def test_full_left_steer():
    _, steer = map_action_to_drive(0.0, 1.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(steer, ifc.MAX_STEER)


def test_clipping():
    speed, steer = map_action_to_drive(5.0, -5.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, ifc.MAX_SPEED)
    assert math.isclose(steer, -ifc.MAX_STEER)


def test_lag_alpha_training_defaults():
    assert math.isclose(lag_alpha(0.1, 0.1), 0.5)


def test_step_first_order_lag():
    assert math.isclose(step_first_order_lag(0.0, 1.0, 0.5), 0.5)
    assert math.isclose(step_first_order_lag(0.5, 1.0, 0.5), 0.75)

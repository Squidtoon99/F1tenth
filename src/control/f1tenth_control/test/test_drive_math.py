"""Pure tests for the action -> Ackermann force/brake mapping."""

import math

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_control.drive_math import (
    force_to_motor_currents,
    lag_alpha,
    map_action_to_force,
    step_first_order_lag,
)


def test_full_drive_zero_steer():
    long_cmd, steer = map_action_to_force(1.0, 0.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, 1.0)
    assert math.isclose(steer, 0.0)


def test_full_brake():
    long_cmd, steer = map_action_to_force(-1.0, 0.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, -1.0)
    assert math.isclose(steer, 0.0)


def test_coast():
    long_cmd, _ = map_action_to_force(0.0, 0.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, 0.0)


def test_full_left_steer():
    _, steer = map_action_to_force(0.0, 1.0, ifc.MAX_STEER)
    assert math.isclose(steer, ifc.MAX_STEER)


def test_clipping():
    long_cmd, steer = map_action_to_force(5.0, -5.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, 1.0)
    assert math.isclose(steer, -ifc.MAX_STEER)


def test_map_action_to_force_partial():
    long_cmd, steer = map_action_to_force(0.5, -1.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, 0.5)
    assert math.isclose(steer, -ifc.MAX_STEER)
    long_cmd, _ = map_action_to_force(-0.8, 0.0, ifc.MAX_STEER)
    assert math.isclose(long_cmd, -0.8)


def test_force_to_motor_currents_mutex():
    i_d, i_b = force_to_motor_currents(0.5, 20.0, 30.0)
    assert math.isclose(i_d, 10.0)
    assert math.isclose(i_b, 0.0)
    i_d, i_b = force_to_motor_currents(-0.5, 20.0, 30.0)
    assert math.isclose(i_d, 0.0)
    assert math.isclose(i_b, 15.0)
    i_d, i_b = force_to_motor_currents(0.0, 20.0, 30.0)
    assert math.isclose(i_d, 0.0) and math.isclose(i_b, 0.0)
    i_d, i_b = force_to_motor_currents(float("nan"), 20.0, 30.0)
    assert math.isclose(i_d, 0.0) and math.isclose(i_b, 0.0)


def test_lag_alpha_training_defaults():
    # 20 Hz control with t_delta=0.1 -> alpha = dt/(dt+t_delta) = 0.05/0.15
    assert math.isclose(lag_alpha(0.05, 0.1), 0.05 / 0.15)
    assert math.isclose(lag_alpha(0.1, 0.1), 0.5)


def test_step_first_order_lag():
    assert math.isclose(step_first_order_lag(0.0, 1.0, 0.5), 0.5)
    assert math.isclose(step_first_order_lag(0.5, 1.0, 0.5), 0.75)

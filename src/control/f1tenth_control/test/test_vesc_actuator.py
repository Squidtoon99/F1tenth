"""Pure tests for force->current mapping used by vesc_actuator."""

import math

from f1tenth_control.drive_math import force_to_motor_currents, map_action_to_force


def test_full_scale_drive_and_brake():
    i_d, i_b = force_to_motor_currents(1.0, 12.0, 8.0)
    assert math.isclose(i_d, 12.0)
    assert math.isclose(i_b, 0.0)
    i_d, i_b = force_to_motor_currents(-1.0, 12.0, 8.0)
    assert math.isclose(i_d, 0.0)
    assert math.isclose(i_b, 8.0)


def test_clip_above_one():
    # Caller should clip, but mapping also saturates via min(|cmd|, 1).
    i_d, i_b = force_to_motor_currents(2.0, 10.0, 10.0)
    assert math.isclose(i_d, 10.0)
    assert math.isclose(i_b, 0.0)


def test_force_action_roundtrip():
    long_cmd, steer = map_action_to_force(0.25, 0.5, 0.33)
    i_d, i_b = force_to_motor_currents(long_cmd, 40.0, 40.0)
    assert math.isclose(i_d, 10.0)
    assert math.isclose(i_b, 0.0)
    assert math.isclose(steer, 0.165)

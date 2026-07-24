"""Fence sensor_racer action mapping against f1tenth_control.drive_math."""

from __future__ import annotations

import math

from f1tenth_control.drive_math import force_to_motor_currents, map_action_to_force
from f1tenth_rl_agent.sensor_action import normalized_action_to_physical


def test_normalized_action_matches_drive_math():
    cases = [
        (0.5, 0.0, 10.0, 8.0, 0.33),
        (-0.25, 0.0, 10.0, 8.0, 0.33),
        (0.0, 1.0, 10.0, 8.0, 0.33),
        (1.0, -1.0, 20.0, 15.0, 0.33),
    ]
    gain = -1.2135
    offset = 0.4495
    for long_in, steer_in, i_drive, i_brake, max_steer in cases:
        long_norm, steer_rad = map_action_to_force(long_in, steer_in, max_steer)
        exp_drive, exp_brake = force_to_motor_currents(long_norm, i_drive, i_brake)
        exp_servo = gain * steer_rad + offset
        exp_steer_norm = steer_rad / max_steer if max_steer > 0.0 else 0.0

        drive, brake, servo, got_long, got_steer = normalized_action_to_physical(
            long_in,
            steer_in,
            i_drive_max_a=i_drive,
            i_brake_max_a=i_brake,
            max_steer=max_steer,
            steering_angle_to_servo_gain=gain,
            steering_angle_to_servo_offset=offset,
        )
        assert math.isclose(drive, exp_drive)
        assert math.isclose(brake, exp_brake)
        assert math.isclose(servo, exp_servo)
        assert math.isclose(got_long, long_norm)
        assert math.isclose(got_steer, exp_steer_norm)


def test_nonfinite_action_coasts():
    drive, brake, servo, long_norm, steer_norm = normalized_action_to_physical(
        float("nan"),
        float("inf"),
        i_drive_max_a=10.0,
        i_brake_max_a=8.0,
        max_steer=0.33,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
    )
    assert drive == 0.0 and brake == 0.0
    assert long_norm == 0.0 and steer_norm == 0.0
    assert servo == 0.4495

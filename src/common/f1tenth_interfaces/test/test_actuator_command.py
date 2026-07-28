"""Serialization and source-enum parity for ActuatorCommand."""

import pytest
from builtin_interfaces.msg import Time
from f1tenth_interfaces.msg import ActuatorCommand
from rclpy.serialization import deserialize_message, serialize_message
from std_msgs.msg import Header


def test_actuator_command_serialization_round_trip():
    msg = ActuatorCommand()
    msg.header = Header(stamp=Time(sec=10, nanosec=500), frame_id="base_link")
    msg.generation = 99
    msg.observation_stamp = Time(sec=9, nanosec=250)
    msg.drive_current_a = 0.0
    msg.brake_current_a = 5.0
    msg.servo_position = 0.12
    msg.longitudinal = -0.5
    msg.steering = 0.33
    msg.source = ActuatorCommand.SOURCE_SAFETY

    restored = deserialize_message(serialize_message(msg), ActuatorCommand)

    assert restored.header.stamp.sec == 10
    assert restored.header.stamp.nanosec == 500
    assert restored.header.frame_id == "base_link"
    assert restored.generation == 99
    assert restored.observation_stamp.sec == 9
    assert restored.observation_stamp.nanosec == 250
    assert restored.drive_current_a == 0.0
    assert restored.brake_current_a == 5.0
    assert restored.servo_position == pytest.approx(0.12)
    assert restored.longitudinal == pytest.approx(-0.5)
    assert restored.steering == pytest.approx(0.33)
    assert restored.source == ActuatorCommand.SOURCE_SAFETY


def test_actuator_command_source_constants():
    assert ActuatorCommand.SOURCE_SAFE == 0
    assert ActuatorCommand.SOURCE_RL == 1
    assert ActuatorCommand.SOURCE_TELEOP == 2
    assert ActuatorCommand.SOURCE_SAFETY == 3

"""rclpy integration tests for direct-policy vesc_actuator mode."""

from __future__ import annotations

import math

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Joy  # noqa: E402
from std_msgs.msg import Float32MultiArray, Float64  # noqa: E402

from f1tenth_control.vesc_actuator_node import VescActuatorNode  # noqa: E402


def _spin_until(nodes, predicate, timeout_s=2.0):
    end = nodes[0].get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while nodes[0].get_clock().now().nanoseconds < end:
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=0.02)
        if predicate():
            return True
    return False


def _make_direct_actuator(**overrides):
    params = [
        Parameter("mode", Parameter.Type.STRING, "direct_policy"),
        Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 10.0),
        Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 10.0),
        Parameter("i_brake_safe_a", Parameter.Type.DOUBLE, 4.0),
        Parameter("i_slew_a_per_s", Parameter.Type.DOUBLE, 1e6),
        Parameter("watchdog_timeout_s", Parameter.Type.DOUBLE, 0.15),
        Parameter("policy_action_timeout_s", Parameter.Type.DOUBLE, 0.5),
        Parameter("joy_timeout_s", Parameter.Type.DOUBLE, 0.5),
        Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 50.0),
        Parameter("publish_servo", Parameter.Type.BOOL, False),
        Parameter("autonomous_button", Parameter.Type.INTEGER, 10),
    ]
    for key, value in overrides.items():
        if isinstance(value, bool):
            ptype = Parameter.Type.BOOL
        elif isinstance(value, int):
            ptype = Parameter.Type.INTEGER
        else:
            ptype = Parameter.Type.DOUBLE
        params.append(Parameter(key, ptype, value))
    return VescActuatorNode(parameter_overrides=params)


def test_direct_policy_requires_r1_for_drive():
    rclpy.init()
    node = helper = None
    try:
        node = _make_direct_actuator()
        helper = rclpy.create_node("direct_policy_helper")
        brakes = []
        currents = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        action_pub = helper.create_publisher(
            Float32MultiArray, "/sensor_policy/desired_action", 10
        )
        joy_pub = helper.create_publisher(Joy, "/joy", 10)

        action = Float32MultiArray()
        action.data = [0.8, 0.2]

        def publish_action_only():
            action_pub.publish(action)
            return bool(brakes)

        assert _spin_until([node, helper], publish_action_only)
        assert any(math.isclose(b, 4.0, abs_tol=0.05) for b in brakes)
        assert not currents

        brakes.clear()
        joy = Joy()
        joy.buttons = [0] * 11
        joy.buttons[10] = 1

        def publish_with_r1():
            joy_pub.publish(joy)
            action_pub.publish(action)
            return bool(currents)

        assert _spin_until([node, helper], publish_with_r1)
        assert currents[-1] == pytest.approx(8.0, abs=0.05)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_r1_release_interleaved_after_action_never_publishes_drive():
    rclpy.init()
    node = helper = None
    try:
        node = _make_direct_actuator()
        helper = rclpy.create_node("direct_policy_release_interleaving")
        brakes = []
        currents = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )

        held = Joy()
        held.buttons = [0] * 11
        held.buttons[10] = 1
        released = Joy()
        released.buttons = [0] * 11
        action = Float32MultiArray()
        action.data = [0.8, 0.0]

        node._on_joy(held)
        node._on_desired_action(action)
        node._on_timer()
        node._on_joy(released)
        node._on_timer()

        assert _spin_until([helper], lambda: bool(brakes))
        assert any(math.isclose(b, 4.0, abs_tol=0.05) for b in brakes)
        assert not any(current > 0.0 for current in currents)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_direct_policy_publishes_applied_action_feedback():
    rclpy.init()
    node = helper = None
    try:
        node = _make_direct_actuator()
        helper = rclpy.create_node("direct_policy_applied_helper")
        applied = []
        helper.create_subscription(
            Float32MultiArray,
            "/sensor_policy/applied_action",
            lambda m: applied.append(list(m.data)),
            10,
        )
        action_pub = helper.create_publisher(
            Float32MultiArray, "/sensor_policy/desired_action", 10
        )
        joy_pub = helper.create_publisher(Joy, "/joy", 10)

        joy = Joy()
        joy.buttons = [0] * 11
        joy.buttons[10] = 1
        action = Float32MultiArray()
        action.data = [0.5, -0.5]

        def saw_applied():
            joy_pub.publish(joy)
            action_pub.publish(action)
            return bool(applied) and applied[-1][0] == pytest.approx(0.5, abs=0.05)

        assert _spin_until([node, helper], saw_applied)
        assert applied[-1][0] == pytest.approx(0.5, abs=0.05)
        assert applied[-1][1] == pytest.approx(-0.5, abs=0.05)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

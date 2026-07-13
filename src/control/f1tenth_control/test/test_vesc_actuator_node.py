"""rclpy integration tests for vesc_actuator current/brake ownership."""

import math

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter  # noqa: E402

from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

from f1tenth_control.vesc_actuator_node import VescActuatorNode  # noqa: E402


def _spin_until(nodes, predicate, timeout_s=2.0):
    end = nodes[0].get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while nodes[0].get_clock().now().nanoseconds < end:
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=0.02)
        if predicate():
            return True
    return False


def _make_actuator(overrides):
    return VescActuatorNode(parameter_overrides=overrides)


def test_vesc_actuator_positive_current_mutex():
    rclpy.init()
    node = helper = None
    try:
        node = _make_actuator([
            Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 20.0),
            Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 30.0),
            Parameter("i_slew_a_per_s", Parameter.Type.DOUBLE, 1e6),
            Parameter("watchdog_timeout_s", Parameter.Type.DOUBLE, 1.0),
            Parameter("zero_erpm", Parameter.Type.BOOL, False),
            Parameter("publish_servo", Parameter.Type.BOOL, False),
        ])
        helper = rclpy.create_node("vesc_act_helper")
        currents = []
        brakes = []
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        pub = helper.create_publisher(AckermannDriveStamped, "/ackermann_cmd", 10)

        msg = AckermannDriveStamped()
        msg.drive.acceleration = 0.5
        msg.drive.steering_angle = 0.0

        def saw():
            pub.publish(msg)
            return any(c > 9.0 for c in currents) and any(
                math.isclose(b, 0.0, abs_tol=1e-6) for b in brakes
            )

        assert _spin_until([node, helper], saw), "no positive current command"
        assert currents[-1] == pytest.approx(10.0, abs=0.05)
        assert brakes[-1] == pytest.approx(0.0, abs=1e-6)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_vesc_actuator_negative_brake_mutex():
    rclpy.init()
    node = helper = None
    try:
        node = _make_actuator([
            Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 20.0),
            Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 30.0),
            Parameter("i_slew_a_per_s", Parameter.Type.DOUBLE, 1e6),
            Parameter("watchdog_timeout_s", Parameter.Type.DOUBLE, 1.0),
            Parameter("zero_erpm", Parameter.Type.BOOL, False),
            Parameter("publish_servo", Parameter.Type.BOOL, False),
        ])
        helper = rclpy.create_node("vesc_act_helper_br")
        currents = []
        brakes = []
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        pub = helper.create_publisher(AckermannDriveStamped, "/ackermann_cmd", 10)

        msg = AckermannDriveStamped()
        msg.drive.acceleration = -0.5
        msg.drive.steering_angle = 0.1

        def saw():
            pub.publish(msg)
            return any(b > 14.0 for b in brakes)

        assert _spin_until([node, helper], saw), "no brake current command"
        assert currents[-1] == pytest.approx(0.0, abs=1e-6)
        assert brakes[-1] == pytest.approx(15.0, abs=0.05)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_vesc_actuator_watchdog_safe_brake():
    rclpy.init()
    node = helper = None
    try:
        node = _make_actuator([
            Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 10.0),
            Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 10.0),
            Parameter("i_brake_safe_a", Parameter.Type.DOUBLE, 4.0),
            Parameter("i_slew_a_per_s", Parameter.Type.DOUBLE, 1e6),
            Parameter("watchdog_timeout_s", Parameter.Type.DOUBLE, 0.15),
            Parameter("publish_rate_hz", Parameter.Type.DOUBLE, 50.0),
            Parameter("zero_erpm", Parameter.Type.BOOL, False),
            Parameter("publish_servo", Parameter.Type.BOOL, False),
        ])
        helper = rclpy.create_node("vesc_act_helper_wd")
        brakes = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        pub = helper.create_publisher(AckermannDriveStamped, "/ackermann_cmd", 10)

        msg = AckermannDriveStamped()
        msg.drive.acceleration = 1.0
        end = node.get_clock().now().nanoseconds + int(0.3 * 1e9)
        while node.get_clock().now().nanoseconds < end:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(helper, timeout_sec=0.02)

        brakes.clear()
        end = node.get_clock().now().nanoseconds + int(1.0 * 1e9)
        saw_safe = False
        while node.get_clock().now().nanoseconds < end and not saw_safe:
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(helper, timeout_sec=0.02)
            saw_safe = any(math.isclose(b, 4.0, abs_tol=0.05) for b in brakes)
        assert saw_safe, "watchdog did not publish safe brake current"
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

"""rclpy integration test for drive_command_node (runs in the container)."""

import math

import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter  # noqa: E402

from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_control.drive_command_node import DriveCommandNode  # noqa: E402


def _collect_drive(node, pub, action, timeout_s=3.0):
    received = []
    pub_node = rclpy.create_node("act_pub")
    try:
        act_pub = pub_node.create_publisher(Float32MultiArray, ifc.TOPIC_ACTION, 10)
        node.create_subscription(
            AckermannDriveStamped, ifc.TOPIC_DRIVE,
            lambda m: received.append(m), 10)
        msg = Float32MultiArray()
        msg.data = action
        end = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while node.get_clock().now().nanoseconds < end and not received:
            act_pub.publish(msg)
            rclpy.spin_once(pub_node, timeout_sec=0.02)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        pub_node.destroy_node()
    return received


def test_drive_command_maps_action():
    rclpy.init()
    node = None
    try:
        node = DriveCommandNode()
        node.set_parameters([
            Parameter("enable_output_filter", Parameter.Type.BOOL, False),
            Parameter("speed_limit_mps", Parameter.Type.DOUBLE, 15.0),
        ])
        received = _collect_drive(node, None, [1.0, 0.0])
        assert received
        assert math.isclose(received[-1].drive.speed, ifc.MAX_SPEED, rel_tol=1e-4)
        assert math.isclose(received[-1].drive.steering_angle, 0.0, abs_tol=1e-5)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

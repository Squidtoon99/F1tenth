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
    # Subscribe to /drive on a SEPARATE node (not on `node` itself): with
    # single-threaded spin_once, subscribing on the same node that also has a busy
    # /rl/action subscription starves the drive callback. Also throttle the action
    # publish so the node's action callback does not monopolize every spin.
    pub_node = rclpy.create_node("act_pub")
    try:
        act_pub = pub_node.create_publisher(Float32MultiArray, ifc.TOPIC_ACTION, 10)
        pub_node.create_subscription(
            AckermannDriveStamped, ifc.TOPIC_DRIVE,
            lambda m: received.append(m), 10)
        msg = Float32MultiArray()
        msg.data = action
        end = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        count = 0
        while node.get_clock().now().nanoseconds < end and not received:
            if count % 5 == 0:
                act_pub.publish(msg)
            count += 1
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(pub_node, timeout_sec=0.02)
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


def test_drive_command_non_finite_action_is_safe_stop():
    """A NaN/inf action must never reach the actuator: command speed/steer 0."""
    rclpy.init()
    node = None
    try:
        node = DriveCommandNode()
        node.set_parameters([
            Parameter("enable_output_filter", Parameter.Type.BOOL, False),
            Parameter("speed_limit_mps", Parameter.Type.DOUBLE, 15.0),
        ])
        received = _collect_drive(node, None, [float("nan"), float("inf")])
        assert received
        assert math.isclose(received[-1].drive.speed, 0.0, abs_tol=1e-6)
        assert math.isclose(received[-1].drive.steering_angle, 0.0, abs_tol=1e-6)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_drive_command_watchdog_stops_on_action_loss():
    """After actions stop for > watchdog_timeout_s, /drive is held at a safe stop."""
    rclpy.init()
    node = None
    pub_node = None
    try:
        node = DriveCommandNode()
        node.set_parameters([
            Parameter("enable_output_filter", Parameter.Type.BOOL, False),
            Parameter("speed_limit_mps", Parameter.Type.DOUBLE, 15.0),
            Parameter("watchdog_timeout_s", Parameter.Type.DOUBLE, 0.2),
        ])
        pub_node = rclpy.create_node("act_pub_wd")
        act_pub = pub_node.create_publisher(Float32MultiArray, ifc.TOPIC_ACTION, 10)
        received = []
        pub_node.create_subscription(
            AckermannDriveStamped, ifc.TOPIC_DRIVE, lambda m: received.append(m), 10)

        # Phase 1: drive the car with a steady throttle so speed goes non-zero.
        msg = Float32MultiArray()
        msg.data = [1.0, 0.0]
        end = node.get_clock().now().nanoseconds + int(1.0 * 1e9)
        moving = False
        while node.get_clock().now().nanoseconds < end and not moving:
            act_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(pub_node, timeout_sec=0.02)
            moving = any(m.drive.speed > 0.1 for m in received)
        assert moving, "car never started moving under throttle"

        # Phase 2: stop publishing actions; the watchdog must force a safe stop.
        received.clear()
        end = node.get_clock().now().nanoseconds + int(2.0 * 1e9)
        stopped = False
        while node.get_clock().now().nanoseconds < end and not stopped:
            rclpy.spin_once(node, timeout_sec=0.05)
            rclpy.spin_once(pub_node, timeout_sec=0.02)
            stopped = any(
                m.drive.speed == 0.0 and m.drive.steering_angle == 0.0
                for m in received
            )
        assert stopped, "watchdog did not command a safe stop after action loss"
    finally:
        if pub_node is not None:
            pub_node.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

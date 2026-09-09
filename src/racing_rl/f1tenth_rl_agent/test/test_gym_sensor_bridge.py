"""rclpy tests for gym_sensor_bridge_node."""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from f1tenth_interfaces.msg import ActuatorCommand  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.gym_sensor_bridge_node import GymSensorBridgeNode  # noqa: E402


def _spin_until(nodes, predicate, timeout_s=3.0):
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        end = nodes[0].get_clock().now().nanoseconds + int(timeout_s * 1e9)
        while nodes[0].get_clock().now().nanoseconds < end:
            executor.spin_once(timeout_sec=0.05)
            if predicate():
                return True
        return False
    finally:
        for node in reversed(nodes):
            executor.remove_node(node)


def test_gym_sensor_bridge_seeds_safe_zero_applied():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 2),
                Parameter("seed_applied_hz", Parameter.Type.DOUBLE, 50.0),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_seed_helper")
        applied = []
        drives = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_APPLIED_ACTUATOR,
            lambda m: applied.append(m),
            10,
        )
        helper.create_subscription(
            AckermannDriveStamped,
            ifc.TOPIC_DRIVE,
            lambda m: drives.append(m),
            10,
        )

        def saw_seed():
            return bool(applied) and bool(drives)

        assert _spin_until([node, helper], saw_seed)
        assert applied[0].source == ActuatorCommand.SOURCE_SAFE
        assert applied[0].longitudinal == 0.0
        assert applied[0].steering == 0.0
        assert applied[0].drive_current_a == 0.0
        assert applied[0].brake_current_a == 0.0
        assert drives[0].drive.acceleration == 0.0
        assert drives[0].drive.steering_angle == 0.0
        assert drives[0].drive.speed == 0.0
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gym_sensor_bridge_publishes_si_imu_from_odom():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 0),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_imu_helper")
        imus = []
        helper.create_subscription(
            Imu, si.TOPIC_IMU, lambda m: imus.append(m), qos_profile_sensor_data
        )
        odom_pub = helper.create_publisher(Odometry, ifc.TOPIC_ODOM, 10)

        def publish_pair():
            first = Odometry()
            first.header.stamp.sec = 1
            first.header.stamp.nanosec = 0
            first.twist.twist.linear.x = 1.0
            first.twist.twist.angular.z = 0.2
            second = Odometry()
            second.header.stamp.sec = 1
            second.header.stamp.nanosec = 100_000_000
            second.twist.twist.linear.x = 1.5
            second.twist.twist.angular.z = -0.4
            odom_pub.publish(first)
            odom_pub.publish(second)
            return any(
                abs(m.linear_acceleration.x - 5.0) < 1e-6
                and abs(m.angular_velocity.z + 0.4) < 1e-6
                for m in imus
            )

        assert _spin_until([node, helper], publish_pair)
        matched = next(
            m
            for m in imus
            if abs(m.linear_acceleration.x - 5.0) < 1e-6
        )
        assert matched.linear_acceleration.z == pytest.approx(si.GRAVITY_MS2)
        assert matched.angular_velocity.z == pytest.approx(-0.4)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gym_sensor_bridge_maps_applied_to_drive():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 0),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_drive_helper")
        drives = []
        helper.create_subscription(
            AckermannDriveStamped,
            ifc.TOPIC_DRIVE,
            lambda m: drives.append(m),
            10,
        )
        applied_pub = helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        )

        def publish_applied():
            msg = ActuatorCommand()
            msg.header.stamp = helper.get_clock().now().to_msg()
            msg.longitudinal = 0.4
            msg.steering = -0.2
            msg.drive_current_a = 26.0
            msg.source = ActuatorCommand.SOURCE_RL
            applied_pub.publish(msg)
            return any(abs(d.drive.acceleration - 0.4) < 1e-6 for d in drives)

        assert _spin_until([node, helper], publish_applied)
        drive = next(d.drive for d in drives if abs(d.drive.acceleration - 0.4) < 1e-6)
        assert drive.acceleration == pytest.approx(0.4)
        assert drive.steering_angle == pytest.approx(-0.2)
        assert drive.speed == pytest.approx(0.4)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

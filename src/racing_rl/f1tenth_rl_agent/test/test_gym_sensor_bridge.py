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
from std_msgs.msg import Bool  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.gym_certify import WarpLongitudinalPlant  # noqa: E402
from f1tenth_rl_agent.gym_sensor_bridge_node import GymSensorBridgeNode  # noqa: E402
from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
)


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
                abs(m.linear_acceleration.y + 5.0) < 1e-6
                and abs(m.linear_acceleration.x) < 1e-6
                and abs(m.angular_velocity.z + 0.4) < 1e-6
                for m in imus
            )

        assert _spin_until([node, helper], publish_pair)
        matched = next(
            m
            for m in imus
            if abs(m.linear_acceleration.y + 5.0) < 1e-6
        )
        assert matched.linear_acceleration.x == pytest.approx(0.0)
        assert matched.linear_acceleration.z == pytest.approx(si.GRAVITY_MS2)
        assert matched.angular_velocity.z == pytest.approx(-0.4)
        actor_imu, _ = actor_imu_from_interval(
            [
                RawImuSample(
                    stamp_s=0.0,
                    ax=matched.linear_acceleration.x,
                    ay=matched.linear_acceleration.y,
                    az=matched.linear_acceleration.z,
                    gx=matched.angular_velocity.x,
                    gy=matched.angular_velocity.y,
                    gz=matched.angular_velocity.z,
                )
            ],
            ImuCalibration(accel_to_ms2=1.0, gyro_to_rads=1.0),
        )
        assert actor_imu[0] == pytest.approx(5.0)
        assert actor_imu[1] == pytest.approx(0.0)
        assert actor_imu[5] == pytest.approx(-0.4)
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
        assert drive.steering_angle == pytest.approx(-0.2 * si.MAX_STEER_RAD)
        assert drive.speed == pytest.approx(WarpLongitudinalPlant().step(0.0, 26.0, 0.0))
        assert drive.speed < 0.05
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gym_sensor_bridge_one_second_full_throttle_matches_plant():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 0),
                Parameter("odom_topic", Parameter.Type.STRING, "/parity_test/odom"),
                Parameter("drive_topic", Parameter.Type.STRING, "/parity_test/drive"),
                Parameter("applied_topic", Parameter.Type.STRING, "/parity_test/applied"),
                Parameter("imu_topic", Parameter.Type.STRING, "/parity_test/imu"),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_throttle_helper")
        drives = []
        helper.create_subscription(
            AckermannDriveStamped,
            "/parity_test/drive",
            lambda m: drives.append(m),
            10,
        )
        odom_pub = helper.create_publisher(Odometry, "/parity_test/odom", 10)
        applied_pub = helper.create_publisher(
            ActuatorCommand, "/parity_test/applied", 10
        )
        plant = WarpLongitudinalPlant()
        vx = 0.0
        expected = []
        from rclpy.executors import SingleThreadedExecutor

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(helper)
        try:
            for step in range(20):
                nsec = (step + 1) * 50_000_000
                before = len(drives)
                odom = Odometry()
                odom.header.stamp.sec = nsec // 1_000_000_000
                odom.header.stamp.nanosec = nsec % 1_000_000_000
                odom.twist.twist.linear.x = vx
                odom_pub.publish(odom)
                msg = ActuatorCommand()
                msg.header.stamp = odom.header.stamp
                msg.longitudinal = 1.0
                msg.steering = 0.0
                msg.drive_current_a = 80.0
                msg.source = ActuatorCommand.SOURCE_RL
                applied_pub.publish(msg)
                end = node.get_clock().now().nanoseconds + int(2.0 * 1e9)
                while len(drives) <= before:
                    if node.get_clock().now().nanoseconds >= end:
                        raise AssertionError("drive command did not arrive")
                    executor.spin_once(timeout_sec=0.05)
                vx = plant.step(vx, 80.0, 0.0)
                expected.append(vx)
        finally:
            executor.remove_node(helper)
            executor.remove_node(node)
        speeds = [float(d.drive.speed) for d in drives[-20:]]
        steers = [float(d.drive.steering_angle) for d in drives[-20:]]
        assert abs(speeds[-1] - expected[-1]) < 1.0
        assert 4.0 < speeds[-1] < 8.0
        assert max(abs(s) for s in steers) < 1e-9
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gym_sensor_bridge_drive_stamps_are_monotonic():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 0),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_stamp_helper")
        stamps = []
        helper.create_subscription(
            AckermannDriveStamped,
            ifc.TOPIC_DRIVE,
            lambda m: stamps.append(
                int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
            ),
            10,
        )
        applied_pub = helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        )

        def publish_same_stamp():
            msg = ActuatorCommand()
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.longitudinal = 0.1
            applied_pub.publish(msg)
            return len(stamps) >= 2

        assert _spin_until([node, helper], publish_same_stamp)
        assert stamps[0] < stamps[1]
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gym_sensor_bridge_unpause_resets_plant_and_speed():
    rclpy.init()
    node = helper = None
    try:
        node = GymSensorBridgeNode(
            parameter_overrides=[
                Parameter("seed_applied_count", Parameter.Type.INTEGER, 0),
            ]
        )
        helper = rclpy.create_node("gym_sensor_bridge_unpause_helper")
        pause_pub = helper.create_publisher(Bool, "/pause_sim", 10)
        node._plant.step(0.0, 80.0, 0.0)
        node._vx = 3.0
        node._prev_vx = 2.5
        node._drive_seq = 9
        assert node._plant.i_drive > 0.0

        def unpause():
            msg = Bool()
            msg.data = False
            pause_pub.publish(msg)
            return node._plant.i_drive == 0.0 and node._vx == 0.0

        assert _spin_until([node, helper], unpause)
        assert node._plant.i_brake == 0.0
        assert node._prev_vx == 0.0
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

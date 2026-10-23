"""rclpy integration tests for gigaflow_racer_node."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

rclpy = pytest.importorskip("rclpy")

_REPO_SRC = None
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "gigaflow" / "src"
    if (_candidate / "gigaflow_f1tenth").is_dir():
        _REPO_SRC = _candidate
        sys.path.insert(0, str(_candidate))
        break

pytest.importorskip("gigaflow_f1tenth")

import numpy as np  # noqa: E402
from f1tenth_interfaces.msg import ActuatorCommand  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.gigaflow_racer_node import GigaflowRacerNode  # noqa: E402
from gigaflow_f1tenth.artifacts import build_actor_artifact_payload  # noqa: E402
from gigaflow_f1tenth.config import load_config  # noqa: E402
from gigaflow_f1tenth.model import build_actor  # noqa: E402


def _write_gigaflow_checkpoint(path: str) -> None:
    assert _REPO_SRC is not None
    cfg = load_config(_REPO_SRC.parent / "configs" / "smoke.yaml")
    actor = build_actor(cfg)
    torch.save(build_actor_artifact_payload(cfg, actor), path)


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


def _applied_rl(longitudinal: float = 0.2, steering: float = 0.0) -> ActuatorCommand:
    msg = ActuatorCommand()
    msg.longitudinal = float(longitudinal)
    msg.steering = float(steering)
    msg.drive_current_a = 5.0 * max(float(longitudinal), 0.0)
    msg.brake_current_a = 5.0 * max(-float(longitudinal), 0.0)
    msg.servo_position = 0.4495
    msg.source = ActuatorCommand.SOURCE_RL
    return msg


def test_gigaflow_racer_publishes_actuator_command_with_fresh_sensors():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "gigaflow_policy.pt")
            _write_gigaflow_checkpoint(ckpt)
            node = GigaflowRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter("control_hz", Parameter.Type.DOUBLE, 20.0),
                ]
            )
        helper = rclpy.create_node("gigaflow_racer_helper")
        desired = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.append(m),
            10,
        )
        scan = LaserScan()
        scan.angle_min = si.LIDAR_ANGLE_MIN
        scan.angle_increment = si.LIDAR_ANGLE_INCREMENT
        scan.range_min = si.LIDAR_RANGE_MIN
        scan.range_max = si.LIDAR_RANGE_MAX
        scan.ranges = [1.0] * si.LIDAR_DIM
        odom = Odometry()
        odom.twist.twist.linear.x = -1.0
        imu = Imu()
        helper.create_publisher(LaserScan, si.TOPIC_SCAN, 10).publish(scan)
        helper.create_publisher(Odometry, si.TOPIC_ODOM, 10).publish(odom)
        helper.create_publisher(Imu, si.TOPIC_IMU, 10).publish(imu)
        helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        ).publish(_applied_rl())

        assert _spin_until([helper, node], lambda: len(desired) >= 1)
        cmd = desired[-1]
        assert cmd.source == ActuatorCommand.SOURCE_RL
        assert abs(cmd.longitudinal) <= 1.0 + 1e-5
        assert abs(cmd.steering) <= 1.0 + 1e-5
        assert cmd.drive_current_a >= 0.0
        assert cmd.brake_current_a >= 0.0
        assert not (cmd.drive_current_a > 0.0 and cmd.brake_current_a > 0.0)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _fresh_scan() -> LaserScan:
    scan = LaserScan()
    scan.angle_min = si.LIDAR_ANGLE_MIN
    scan.angle_increment = si.LIDAR_ANGLE_INCREMENT
    scan.range_min = si.LIDAR_RANGE_MIN
    scan.range_max = si.LIDAR_RANGE_MAX
    scan.ranges = [1.0] * si.LIDAR_DIM
    return scan


def _applied(
    source: int,
    longitudinal: float,
    steering: float = 0.0,
    *,
    i_drive_max_a: float = 60.0,
    i_brake_max_a: float = 20.0,
) -> ActuatorCommand:
    msg = ActuatorCommand()
    msg.longitudinal = float(longitudinal)
    msg.steering = float(steering)
    if longitudinal >= 0.0:
        msg.drive_current_a = float(i_drive_max_a) * float(longitudinal)
        msg.brake_current_a = 0.0
    else:
        msg.drive_current_a = 0.0
        msg.brake_current_a = float(i_brake_max_a) * (-float(longitudinal))
    msg.servo_position = 0.4495
    msg.source = int(source)
    return msg


def test_gigaflow_racer_teleop_resets_before_pack_and_zeros_history():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "gigaflow_policy.pt")
            _write_gigaflow_checkpoint(ckpt)
            node = GigaflowRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter("control_hz", Parameter.Type.DOUBLE, 20.0),
                    Parameter("publish_raw_observation", Parameter.Type.BOOL, True),
                    Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 60.0),
                    Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 20.0),
                ]
            )
        helper = rclpy.create_node("gigaflow_racer_teleop_helper")
        desired = []
        raw_obs = []
        diagnostics = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.append(m),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            si.TOPIC_OBSERVATION,
            lambda m: raw_obs.append(list(m.data)),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            si.TOPIC_DIAGNOSTICS,
            lambda m: diagnostics.append(list(m.data)),
            10,
        )
        scan_pub = helper.create_publisher(LaserScan, si.TOPIC_SCAN, 10)
        imu_pub = helper.create_publisher(Imu, si.TOPIC_IMU, 10)
        odom_pub = helper.create_publisher(Odometry, si.TOPIC_ODOM, 10)
        applied_pub = helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        )
        scan = _fresh_scan()
        odom = Odometry()
        imu = Imu()
        applied = _applied(ActuatorCommand.SOURCE_TELEOP, -1.0)

        def publish_all():
            now = helper.get_clock().now().to_msg()
            scan.header.stamp = now
            imu.header.stamp = now
            odom.header.stamp = now
            applied.header.stamp = now
            scan_pub.publish(scan)
            imu_pub.publish(imu)
            odom_pub.publish(odom)
            applied_pub.publish(applied)

        def saw_teleop_hold():
            publish_all()
            if not diagnostics:
                return False
            diag = diagnostics[-1]
            return (
                diag[si.DIAG_APPLIED_SOURCE]
                == float(ActuatorCommand.SOURCE_TELEOP)
                and diag[si.DIAG_STALE_SENSORS] == pytest.approx(0.0)
                and diag[si.DIAG_VALID_TICK] == pytest.approx(0.0)
            )

        assert _spin_until([helper, node], saw_teleop_hold)
        assert desired == []
        assert raw_obs == []
        assert np.allclose(node._runtime._long_ring, 0.0)
        assert np.allclose(node._runtime._steer_ring, 0.0)
        assert float(node._runtime._hidden.abs().max()) == pytest.approx(0.0)

        applied = _applied(ActuatorCommand.SOURCE_RL, 0.2, 0.1)
        raw_obs.clear()

        def saw_rl_obs():
            publish_all()
            return len(raw_obs) >= 1 and len(desired) >= 1

        assert _spin_until([helper, node], saw_rl_obs)
        obs = np.asarray(raw_obs[-1], dtype=np.float32)
        assert obs[si.THROTTLE_CURRENT] == pytest.approx(0.2, abs=1e-5)
        assert obs[si.THROTTLE_PRED] == pytest.approx(0.0, abs=1e-5)
        expected_vesc = si.VESC_CURRENT_SCALE_A * si.applied_current_fraction(
            60.0 * 0.2, 60.0, 20.0
        )
        assert obs[si.VESC_CURRENT] == pytest.approx(expected_vesc, abs=1e-5)
        assert expected_vesc == pytest.approx(1.0, abs=1e-5)
        assert obs[si.STEER_T] == pytest.approx(0.1 * 0.33, abs=1e-5)
        assert obs[si.STEER_T1] == pytest.approx(0.0, abs=1e-5)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_gigaflow_racer_safe_applied_is_zeroed_in_observation():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "gigaflow_policy.pt")
            _write_gigaflow_checkpoint(ckpt)
            node = GigaflowRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter("control_hz", Parameter.Type.DOUBLE, 20.0),
                    Parameter("publish_raw_observation", Parameter.Type.BOOL, True),
                    Parameter("i_drive_max_a", Parameter.Type.DOUBLE, 60.0),
                    Parameter("i_brake_max_a", Parameter.Type.DOUBLE, 20.0),
                ]
            )
        helper = rclpy.create_node("gigaflow_racer_safe_helper")
        desired = []
        raw_obs = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.append(m),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            si.TOPIC_OBSERVATION,
            lambda m: raw_obs.append(list(m.data)),
            10,
        )
        scan_pub = helper.create_publisher(LaserScan, si.TOPIC_SCAN, 10)
        imu_pub = helper.create_publisher(Imu, si.TOPIC_IMU, 10)
        odom_pub = helper.create_publisher(Odometry, si.TOPIC_ODOM, 10)
        applied_pub = helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        )
        scan = _fresh_scan()
        odom = Odometry()
        imu = Imu()
        applied = _applied(ActuatorCommand.SOURCE_SAFE, -0.25)

        def publish_all():
            now = helper.get_clock().now().to_msg()
            scan.header.stamp = now
            imu.header.stamp = now
            odom.header.stamp = now
            applied.header.stamp = now
            scan_pub.publish(scan)
            imu_pub.publish(imu)
            odom_pub.publish(odom)
            applied_pub.publish(applied)
            return len(raw_obs) >= 1 and len(desired) >= 1

        assert _spin_until([helper, node], publish_all)
        obs = np.asarray(raw_obs[-1], dtype=np.float32)
        assert obs[si.THROTTLE_CURRENT] == pytest.approx(0.0, abs=1e-5)
        assert obs[si.THROTTLE_PRED] == pytest.approx(0.0, abs=1e-5)
        assert obs[si.VESC_CURRENT] == pytest.approx(0.0, abs=1e-5)
        assert obs[si.STEER_T] == pytest.approx(0.0, abs=1e-5)
        assert desired[-1].source == ActuatorCommand.SOURCE_RL
        assert np.allclose(node._runtime._long_ring, 0.0)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

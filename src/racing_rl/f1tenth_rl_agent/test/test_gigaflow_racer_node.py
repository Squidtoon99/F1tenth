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

from f1tenth_interfaces.msg import ActuatorCommand  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402

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

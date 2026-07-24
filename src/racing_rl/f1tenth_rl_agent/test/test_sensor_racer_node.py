"""rclpy integration tests for sensor_racer_node."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

rclpy = pytest.importorskip("rclpy")
from f1tenth_interfaces.msg import ActuatorCommand  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.policy_model import SquashedGaussianLidarGRUActor  # noqa: E402
from f1tenth_rl_agent.sensor_racer_node import SensorRacerNode  # noqa: E402


def _write_sensor_checkpoint(
    path: str,
    *,
    steering_action_mode: str = "delta",
    steering_mu: float | None = None,
) -> None:
    actor = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    if steering_mu is not None:
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.mu_layer.bias[1] = float(steering_mu)
    payload = {
        "actor": actor.state_dict(),
        "obs_norm": {
            "mean": torch.zeros(si.NUM_OBS),
            "var": torch.ones(si.NUM_OBS),
            "count": torch.tensor(1.0),
        },
        "obs_dim": si.NUM_OBS,
        "actor_obs_dim": si.NUM_OBS,
        "critic_obs_dim": 392,
        "actor_layout_version": si.ACTOR_LAYOUT_VERSION,
        "action_dim": si.NUM_ACTIONS,
        "policy_format_version": si.POLICY_FORMAT_VERSION,
        "observation_preprocessing_version": si.OBS_PREPROCESSING_VERSION,
        "artifact_scope": si.ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": actor.actor_architecture,
        "longitudinal_mode": "force",
        "steering_action_mode": steering_action_mode,
        "steering_delta_max_rad": 0.05235987755982988,
        "control_hz": si.CONTROL_HZ,
    }
    torch.save(payload, path)


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
    msg.drive_current_a = si.VESC_CURRENT_SCALE_A * max(float(longitudinal), 0.0)
    msg.brake_current_a = si.VESC_CURRENT_SCALE_A * max(-float(longitudinal), 0.0)
    msg.servo_position = 0.4495
    msg.source = ActuatorCommand.SOURCE_RL
    return msg


def test_sensor_racer_publishes_actuator_command_with_fresh_sensors():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "policy.pt")
            _write_sensor_checkpoint(ckpt)
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter("recording_mode", Parameter.Type.BOOL, True),
                    Parameter("publish_raw_observation", Parameter.Type.BOOL, True),
                ]
            )
        helper = rclpy.create_node("sensor_racer_helper")
        desired = []
        diagnostics = []
        raw_obs = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.append(m),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            si.TOPIC_DIAGNOSTICS,
            lambda m: diagnostics.append(list(m.data)),
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

        scan = LaserScan()
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_increment = float(si.LIDAR_ANGLE_INCREMENT)
        scan.range_min = 0.05
        scan.ranges = [1.0] * si.LIDAR_DIM
        odom = Odometry()
        odom.twist.twist.linear.x = 2.0
        imu = Imu()
        applied = _applied_rl(0.2, 0.0)

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
            return bool(desired and raw_obs and diagnostics)

        assert _spin_until([node, helper], publish_all)
        cmd = desired[-1]
        assert cmd.source == ActuatorCommand.SOURCE_RL
        assert cmd.generation >= 1
        assert abs(cmd.longitudinal) <= si.CLIP_ACTIONS + 1e-5
        assert abs(cmd.steering) <= si.CLIP_ACTIONS + 1e-5
        assert not (cmd.drive_current_a > 0.0 and cmd.brake_current_a > 0.0)
        assert raw_obs[-1][si.VESC_SPEED] == pytest.approx(-2.0, abs=0.05)
        assert raw_obs[-1][si.VESC_CURRENT] == pytest.approx(
            si.VESC_CURRENT_SCALE_A * 0.2, abs=0.05
        )
        assert diagnostics[-1][si.DIAG_VALID_TICK] == pytest.approx(1.0)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sensor_racer_fail_closed_without_sensors():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "policy.pt")
            _write_sensor_checkpoint(ckpt)
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                ]
            )
        helper = rclpy.create_node("sensor_racer_stale_helper")
        desired = {"count": 0}
        diagnostics = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.__setitem__("count", desired["count"] + 1),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            si.TOPIC_DIAGNOSTICS,
            lambda m: diagnostics.append(list(m.data)),
            10,
        )

        end = node.get_clock().now().nanoseconds + int(0.5 * 1e9)
        while node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(helper, timeout_sec=0.02)

        assert desired["count"] == 0
        assert diagnostics
        assert diagnostics[-1][si.DIAG_STALE_SENSORS] == pytest.approx(1.0)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sensor_racer_resets_when_applied_source_not_rl():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "policy.pt")
            _write_sensor_checkpoint(ckpt)
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                ]
            )
        helper = rclpy.create_node("sensor_racer_applied_source_helper")
        desired = {"count": 0}
        diagnostics = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.__setitem__("count", desired["count"] + 1),
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

        scan = LaserScan()
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_increment = float(si.LIDAR_ANGLE_INCREMENT)
        scan.range_min = 0.05
        scan.ranges = [1.0] * si.LIDAR_DIM
        odom = Odometry()
        imu = Imu()
        applied = _applied_rl()
        applied.source = ActuatorCommand.SOURCE_TELEOP

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

        def saw_teleop_reset():
            publish_all()
            if not diagnostics:
                return False
            d = diagnostics[-1]
            return (
                d[si.DIAG_GRU_RESET] == 1.0
                and d[si.DIAG_STALE_SENSORS] == 0.0
                and d[si.DIAG_VALID_TICK] == 0.0
                and d[si.DIAG_APPLIED_SOURCE]
                == float(ActuatorCommand.SOURCE_TELEOP)
            )

        assert _spin_until([node, helper], saw_teleop_reset)
        assert desired["count"] == 0
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sensor_racer_publishes_through_safe_applied_bootstrap():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "policy.pt")
            _write_sensor_checkpoint(ckpt)
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                ]
            )
        helper = rclpy.create_node("sensor_racer_safe_bootstrap_helper")
        desired = []
        diagnostics = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda m: desired.append(m),
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

        scan = LaserScan()
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_increment = float(si.LIDAR_ANGLE_INCREMENT)
        scan.range_min = 0.05
        scan.ranges = [1.0] * si.LIDAR_DIM
        odom = Odometry()
        imu = Imu()
        applied = _applied_rl()
        applied.source = ActuatorCommand.SOURCE_SAFE
        applied.longitudinal = -0.5
        applied.steering = 0.0

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
            return (
                bool(desired)
                and len(diagnostics) >= 3
                and diagnostics[-1][si.DIAG_CONSECUTIVE_SAFE] >= 2.0
            )

        assert _spin_until([node, helper], publish_all)
        assert desired[-1].source == ActuatorCommand.SOURCE_RL
        assert any(
            diag[si.DIAG_GRU_RESET_REASON] == float(si.GRU_RESET_APPLIED_SAFE)
            for diag in diagnostics
        )
        safe_reset_total = diagnostics[-1][si.DIAG_GRU_RESETS]
        assert diagnostics[-2][si.DIAG_GRU_RESETS] == safe_reset_total

        applied.source = ActuatorCommand.SOURCE_RL

        def saw_rl_ownership():
            publish_all()
            return (
                diagnostics[-1][si.DIAG_APPLIED_SOURCE]
                == float(ActuatorCommand.SOURCE_RL)
                and diagnostics[-1][si.DIAG_CONSECUTIVE_SAFE] == 0.0
            )

        assert _spin_until([node, helper], saw_rl_ownership)
        assert diagnostics[-1][si.DIAG_GRU_RESETS] == safe_reset_total
        assert diagnostics[-1][si.DIAG_GRU_RESET_REASON] == float(
            si.GRU_RESET_NONE
        )
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sensor_racer_post_gate_history_in_observation():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "policy.pt")
            _write_sensor_checkpoint(ckpt)
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter("publish_raw_observation", Parameter.Type.BOOL, True),
                ]
            )
        helper = rclpy.create_node("sensor_racer_history_helper")
        raw_obs = []
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

        scan = LaserScan()
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_increment = float(si.LIDAR_ANGLE_INCREMENT)
        scan.range_min = 0.05
        scan.ranges = [1.0] * si.LIDAR_DIM
        odom = Odometry()
        imu = Imu()

        def publish_applied(longitudinal: float, steering: float):
            now = helper.get_clock().now().to_msg()
            scan.header.stamp = now
            imu.header.stamp = now
            odom.header.stamp = now
            applied = _applied_rl(longitudinal, steering)
            applied.header.stamp = now
            scan_pub.publish(scan)
            imu_pub.publish(imu)
            odom_pub.publish(odom)
            applied_pub.publish(applied)

        publish_applied(0.4, 0.1)
        assert _spin_until([node, helper], lambda: len(raw_obs) >= 1)
        publish_applied(-0.3, -0.2)
        assert _spin_until([node, helper], lambda: len(raw_obs) >= 2)

        obs = np.asarray(raw_obs[-1], dtype=np.float32)
        assert obs[si.THROTTLE_CURRENT] == pytest.approx(-0.3, abs=1e-5)
        assert obs[si.THROTTLE_PRED] == pytest.approx(0.4, abs=1e-5)
        assert obs[si.VESC_CURRENT] == pytest.approx(
            si.VESC_CURRENT_SCALE_A * -0.3, abs=1e-5
        )
        assert obs[si.STEER_T] == pytest.approx(-0.2 * 0.33, abs=1e-5)
        assert obs[si.STEER_T1] == pytest.approx(0.1 * 0.33, abs=1e-5)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_sensor_racer_integrates_delta_policy_and_resets_accumulator():
    rclpy.init()
    node = helper = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "delta_policy.pt")
            _write_sensor_checkpoint(
                ckpt,
                steering_action_mode="delta",
                steering_mu=1.0,
            )
            node = SensorRacerNode(
                parameter_overrides=[
                    Parameter("checkpoint_path", Parameter.Type.STRING, ckpt),
                    Parameter("require_checkpoint", Parameter.Type.BOOL, True),
                    Parameter("sensor_max_age_s", Parameter.Type.DOUBLE, 1.0),
                    Parameter(
                        "steering_action_mode", Parameter.Type.STRING, "delta"
                    ),
                    Parameter(
                        "steering_delta_max_rad",
                        Parameter.Type.DOUBLE,
                        np.pi / 60.0,
                    ),
                ]
            )
        helper = rclpy.create_node("sensor_racer_delta_helper")
        desired = []
        helper.create_subscription(
            ActuatorCommand,
            si.TOPIC_DESIRED_ACTUATOR,
            lambda msg: desired.append(msg),
            10,
        )
        scan_pub = helper.create_publisher(LaserScan, si.TOPIC_SCAN, 10)
        imu_pub = helper.create_publisher(Imu, si.TOPIC_IMU, 10)
        odom_pub = helper.create_publisher(Odometry, si.TOPIC_ODOM, 10)
        applied_pub = helper.create_publisher(
            ActuatorCommand, si.TOPIC_APPLIED_ACTUATOR, 10
        )
        scan = LaserScan()
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_increment = float(si.LIDAR_ANGLE_INCREMENT)
        scan.range_min = 0.05
        scan.ranges = [1.0] * si.LIDAR_DIM
        imu = Imu()
        odom = Odometry()
        applied = _applied_rl()

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
            return len(desired) >= 3

        assert _spin_until([node, helper], publish_all)
        realized = np.asarray([msg.steering * 0.33 for msg in desired[:3]])
        expected_step = np.tanh(1.0) * np.pi / 60.0
        assert np.diff(realized) == pytest.approx(
            [expected_step, expected_step], abs=1e-5
        )
        node._reset_runtime(si.GRU_RESET_APPLIED_SAFETY)
        assert node._realized_steer_rad == pytest.approx(0.0)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

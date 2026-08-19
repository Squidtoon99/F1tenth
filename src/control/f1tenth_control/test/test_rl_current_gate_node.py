"""rclpy integration tests for rl_current_gate_node."""

from __future__ import annotations

import math

import pytest

rclpy = pytest.importorskip("rclpy")
from builtin_interfaces.msg import Time  # noqa: E402
from f1tenth_interfaces.msg import ActuatorCommand  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from std_msgs.msg import Float32MultiArray, Float64  # noqa: E402

from f1tenth_control import current_gate as cg  # noqa: E402
from f1tenth_control.current_gate import (  # noqa: E402
    SOURCE_RL,
    SOURCE_SAFE,
    SOURCE_SAFETY,
    SOURCE_TELEOP,
)
from f1tenth_control.rl_current_gate_node import RlCurrentGateNode  # noqa: E402


def _spin_until(nodes, predicate, timeout_s=2.0):
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


def _make_gate(**overrides):
    values = {
        "i_drive_max_a": 10.0,
        "i_brake_max_a": 10.0,
        "i_brake_safe_a": 4.0,
        "i_slew_a_per_s": 1e6,
        "rl_command_timeout_s": 0.5,
        "teleop_timeout_s": 0.5,
        "safety_timeout_s": 0.5,
        "publish_rate_hz": 50.0,
        "publish_servo": False,
    }
    values.update(overrides)
    params = []
    for key, value in values.items():
        if isinstance(value, bool):
            ptype = Parameter.Type.BOOL
        elif isinstance(value, int):
            ptype = Parameter.Type.INTEGER
        else:
            ptype = Parameter.Type.DOUBLE
        params.append(Parameter(key, ptype, value))
    return RlCurrentGateNode(parameter_overrides=params)


def _rl_msg(**overrides):
    msg = ActuatorCommand()
    msg.generation = 1
    msg.drive_current_a = 8.0
    msg.brake_current_a = 0.0
    msg.servo_position = 0.45
    msg.longitudinal = 0.8
    msg.steering = 0.1
    msg.source = ActuatorCommand.SOURCE_RL
    msg.observation_stamp = Time(sec=1, nanosec=2)
    for key, value in overrides.items():
        setattr(msg, key, value)
    return msg


def _ack(acceleration, steering=0.0):
    msg = AckermannDriveStamped()
    msg.drive.acceleration = acceleration
    msg.drive.steering_angle = steering
    return msg


def test_gate_node_defaults_match_sensor_policy_artifact_envelope():
    rclpy.init()
    node = None
    try:
        node = RlCurrentGateNode()
        assert node._cfg.i_drive_max_a == pytest.approx(80.0)
        assert node._cfg.i_brake_max_a == pytest.approx(20.0)
        assert node._cfg.i_brake_safe_a == pytest.approx(5.0)
        assert node._cfg.i_slew_a_per_s == pytest.approx(200.0)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("i_drive_max_a", float("nan")),
        ("i_drive_max_a", float("inf")),
        ("i_brake_max_a", float("nan")),
        ("i_brake_max_a", float("inf")),
    ],
)
def test_gate_node_refuses_nonfinite_current_limits(field, invalid):
    rclpy.init()
    node = None
    try:
        with pytest.raises(RuntimeError, match=field):
            node = _make_gate(**{field: invalid})
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_publishes_rl_drive_when_teleop_stale():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_rl")
        currents = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )

        def saw_rl():
            rl_pub.publish(_rl_msg())
            return (
                bool(applied)
                and applied[-1].source == SOURCE_RL
                and applied[-1].drive_current_a > 0.0
            )

        assert _spin_until([node, helper], saw_rl)
        assert currents[-1] == pytest.approx(8.0, abs=0.05)
        assert applied[-1].source == SOURCE_RL
        assert applied[-1].drive_current_a == pytest.approx(8.0, abs=0.05)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_teleop_overrides_rl():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_teleop")
        currents = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )
        teleop_pub = helper.create_publisher(
            AckermannDriveStamped, "/teleop", 10
        )

        def saw_teleop():
            rl_pub.publish(_rl_msg(drive_current_a=8.0))
            teleop_pub.publish(_ack(0.5))
            return bool(applied) and applied[-1].source == SOURCE_TELEOP

        assert _spin_until([node, helper], saw_teleop)
        assert currents[-1] == pytest.approx(5.0, abs=0.05)
        assert applied[-1].source == SOURCE_TELEOP
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_safety_overrides_teleop_and_rl():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_safety")
        brakes = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )
        teleop_pub = helper.create_publisher(
            AckermannDriveStamped, "/teleop", 10
        )
        brake_pub = helper.create_publisher(AckermannDriveStamped, "/brake", 10)

        def saw_safety():
            rl_pub.publish(_rl_msg())
            teleop_pub.publish(_ack(0.5))
            brake_pub.publish(_ack(-1.0))
            return bool(applied) and applied[-1].source == SOURCE_SAFETY

        assert _spin_until([node, helper], saw_safety)
        assert math.isclose(brakes[-1], 10.0, abs_tol=0.05)
        assert applied[-1].source == SOURCE_SAFETY
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_deadman_block_teleop_brake_overrides_rl():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_deadman")
        brakes = []
        currents = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )
        teleop_pub = helper.create_publisher(
            AckermannDriveStamped, "/teleop", 10
        )

        def saw_deadman_brake():
            rl_pub.publish(_rl_msg())
            teleop_pub.publish(_ack(-1.0))
            return (
                bool(applied)
                and applied[-1].source == SOURCE_TELEOP
                and math.isclose(applied[-1].brake_current_a, 10.0, abs_tol=0.05)
            )

        assert _spin_until([node, helper], saw_deadman_brake)
        assert math.isclose(brakes[-1], 10.0, abs_tol=0.05)
        assert not any(c > 0.0 for c in currents)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_watchdog_safe_brake_without_inputs():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate(rl_command_timeout_s=0.05, teleop_timeout_s=0.05)
        helper = rclpy.create_node("rl_gate_helper_watchdog")
        brakes = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )

        assert _spin_until(
            [node, helper],
            lambda: applied and applied[-1].source == SOURCE_SAFE,
        )
        assert math.isclose(brakes[-1], 4.0, abs_tol=0.05)
        assert applied[-1].source == SOURCE_SAFE
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_rejects_invalid_rl_and_falls_back():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_invalid_rl")
        brakes = []
        applied = []
        helper.create_subscription(
            Float64, "commands/motor/brake", lambda m: brakes.append(m.data), 10
        )
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )

        def saw_safe():
            rl_pub.publish(
                _rl_msg(
                    drive_current_a=5.0,
                    brake_current_a=3.0,
                    generation=1,
                )
            )
            return bool(brakes) and applied and applied[-1].source == SOURCE_SAFE

        assert _spin_until([node, helper], saw_safe)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_reports_limit_rejection_then_5a_rl_ownership():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate(i_drive_max_a=5.0, i_brake_max_a=5.0)
        helper = rclpy.create_node("rl_gate_helper_limit_diagnostics")
        applied = []
        diagnostics = []
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        helper.create_subscription(
            Float32MultiArray,
            "/rl_current_gate/diagnostics",
            lambda m: diagnostics.append(list(m.data)),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )

        def saw_limit_rejection():
            rl_pub.publish(_rl_msg(generation=1, drive_current_a=10.0))
            return (
                bool(applied and diagnostics)
                and applied[-1].source == SOURCE_SAFE
                and diagnostics[-1][cg.GATE_DIAG_LAST_REJECT_REASON]
                == float(cg.REJECT_DRIVE_LIMIT)
            )

        assert _spin_until([node, helper], saw_limit_rejection)
        assert diagnostics[-1][cg.GATE_DIAG_RL_REJECT_TOTAL] >= 1.0
        assert (
            diagnostics[-1][
                cg.GATE_DIAG_REJECT_COUNTS_START + cg.REJECT_DRIVE_LIMIT
            ]
            >= 1.0
        )

        def saw_rl_ownership():
            rl_pub.publish(_rl_msg(generation=2, drive_current_a=5.0))
            return (
                bool(applied and diagnostics)
                and applied[-1].source == SOURCE_RL
                and diagnostics[-1][cg.GATE_DIAG_APPLIED_SOURCE]
                == float(SOURCE_RL)
            )

        assert _spin_until([node, helper], saw_rl_ownership)
        assert applied[-1].drive_current_a == pytest.approx(5.0, abs=0.05)
        assert diagnostics[-1][cg.GATE_DIAG_CONSECUTIVE_SAFE] == 0.0
        assert diagnostics[-1][cg.GATE_DIAG_LAST_REJECT_REASON] == float(
            cg.REJECT_NONE
        )
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_stale_rl_generation_rejected():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_gen")
        applied = []
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )

        rl_pub.publish(_rl_msg(generation=1))
        assert _spin_until(
            [node, helper],
            lambda: applied and applied[-1].source == SOURCE_RL,
        )

        applied.clear()
        rl_pub.publish(_rl_msg(generation=1, drive_current_a=9.0))
        end = node.get_clock().now().nanoseconds + int(0.5 * 1e9)
        while node.get_clock().now().nanoseconds < end:
            rl_pub.publish(_rl_msg(generation=1, drive_current_a=9.0))
            rclpy.spin_once(node, timeout_sec=0.02)
            rclpy.spin_once(helper, timeout_sec=0.02)
        assert not any(a.source == SOURCE_RL and a.drive_current_a > 8.5 for a in applied)
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_applied_feedback_every_heartbeat():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate()
        helper = rclpy.create_node("rl_gate_helper_applied")
        applied = []
        helper.create_subscription(
            ActuatorCommand,
            "/rl/actuator/applied",
            lambda m: applied.append(m),
            10,
        )

        def saw_two_heartbeats():
            return (
                len(applied) >= 2
                and applied[-1].generation > applied[0].generation
                and applied[-1].source == SOURCE_SAFE
            )

        assert _spin_until(
            [node, helper],
            saw_two_heartbeats,
            timeout_s=3.0,
        )
        assert applied[-1].generation > applied[0].generation
        assert applied[-1].source == SOURCE_SAFE
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_gate_slew_limits_rl_current_ramp():
    rclpy.init()
    node = helper = None
    try:
        node = _make_gate(i_slew_a_per_s=100.0)
        helper = rclpy.create_node("rl_gate_helper_slew")
        currents = []
        helper.create_subscription(
            Float64, "commands/motor/current", lambda m: currents.append(m.data), 10
        )
        rl_pub = helper.create_publisher(
            ActuatorCommand, "/rl/actuator/desired", 10
        )

        rl_pub.publish(_rl_msg(drive_current_a=10.0, generation=1))

        def ramped():
            rl_pub.publish(_rl_msg(drive_current_a=10.0, generation=1))
            return bool(currents) and 0.0 < max(currents) < 10.0

        assert _spin_until([node, helper], ramped, timeout_s=1.0)
        assert max(currents) < 10.0
    finally:
        if helper is not None:
            helper.destroy_node()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

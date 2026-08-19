"""External current-level gate: sole VESC command owner for the RL graph."""

from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_interfaces.msg import ActuatorCommand
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

from f1tenth_control import current_gate as cg
from f1tenth_control.current_gate import (
    AckermannInput,
    GateConfig,
    GateState,
    RlInput,
    apply_slew,
    arbitrate,
    next_applied_generation,
    validate_gate_config,
)


class RlCurrentGateNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("rl_current_gate", **kwargs)

        self._cfg = GateConfig(
            i_drive_max_a=float(self.declare_parameter("i_drive_max_a", 80.0).value),
            i_brake_max_a=float(self.declare_parameter("i_brake_max_a", 20.0).value),
            i_brake_safe_a=float(self.declare_parameter("i_brake_safe_a", 5.0).value),
            i_slew_a_per_s=float(
                self.declare_parameter("i_slew_a_per_s", 200.0).value
            ),
            rl_command_timeout_s=float(
                self.declare_parameter("rl_command_timeout_s", 0.15).value
            ),
            teleop_timeout_s=float(
                self.declare_parameter("teleop_timeout_s", 0.2).value
            ),
            safety_timeout_s=float(
                self.declare_parameter("safety_timeout_s", 0.1).value
            ),
            steering_angle_to_servo_gain=float(
                self.declare_parameter("steering_angle_to_servo_gain", -1.2135).value
            ),
            steering_angle_to_servo_offset=float(
                self.declare_parameter("steering_angle_to_servo_offset", 0.4495).value
            ),
            max_steer=float(self.declare_parameter("max_steer", 0.33).value),
        )
        self._publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 50.0).value
        )
        self._publish_servo = bool(
            self.declare_parameter("publish_servo", True).value
        )
        desired_topic = str(
            self.declare_parameter("desired_topic", "/rl/actuator/desired").value
        )
        applied_topic = str(
            self.declare_parameter("applied_topic", "/rl/actuator/applied").value
        )
        teleop_topic = str(self.declare_parameter("teleop_topic", "/teleop").value)
        brake_topic = str(self.declare_parameter("brake_topic", "/brake").value)
        diagnostics_topic = str(
            self.declare_parameter(
                "diagnostics_topic", "/rl_current_gate/diagnostics"
            ).value
        )

        try:
            validate_gate_config(self._cfg)
        except ValueError as exc:
            raise RuntimeError(f"rl_current_gate: {exc}") from exc

        self._state = GateState(last_pub_s=self._now_s())
        self._rl: RlInput | None = None
        self._teleop: AckermannInput | None = None
        self._safety: AckermannInput | None = None

        self._current_pub = self.create_publisher(Float64, "commands/motor/current", 10)
        self._brake_pub = self.create_publisher(Float64, "commands/motor/brake", 10)
        self._servo_pub = self.create_publisher(Float64, "commands/servo/position", 10)
        self._applied_pub = self.create_publisher(ActuatorCommand, applied_topic, 10)
        self._diagnostics_pub = self.create_publisher(
            Float32MultiArray, diagnostics_topic, 10
        )
        self._diagnostics_msg = Float32MultiArray()

        self.create_subscription(
            ActuatorCommand, desired_topic, self._on_desired, 10
        )
        self.create_subscription(
            AckermannDriveStamped, teleop_topic, self._on_teleop, 10
        )
        self.create_subscription(
            AckermannDriveStamped, brake_topic, self._on_brake, 10
        )

        period = 1.0 / max(self._publish_rate_hz, 1.0)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "rl_current_gate: sole VESC owner "
            f"i_drive_max={self._cfg.i_drive_max_a:.1f}A "
            f"i_brake_max={self._cfg.i_brake_max_a:.1f}A "
            f"i_brake_safe={self._cfg.i_brake_safe_a:.1f}A "
            f"rl_timeout={self._cfg.rl_command_timeout_s:.2f}s "
            f"teleop_timeout={self._cfg.teleop_timeout_s:.2f}s"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_desired(self, msg: ActuatorCommand) -> None:
        self._rl = RlInput(
            generation=int(msg.generation),
            received_s=self._now_s(),
            drive_current_a=float(msg.drive_current_a),
            brake_current_a=float(msg.brake_current_a),
            servo_position=float(msg.servo_position),
            longitudinal=float(msg.longitudinal),
            steering=float(msg.steering),
            source=int(msg.source),
            observation_stamp_sec=int(msg.observation_stamp.sec),
            observation_stamp_nanosec=int(msg.observation_stamp.nanosec),
        )

    def _on_teleop(self, msg: AckermannDriveStamped) -> None:
        self._teleop = AckermannInput(
            acceleration=float(msg.drive.acceleration),
            steering_angle=float(msg.drive.steering_angle),
            received_s=self._now_s(),
        )

    def _on_brake(self, msg: AckermannDriveStamped) -> None:
        self._safety = AckermannInput(
            acceleration=float(msg.drive.acceleration),
            steering_angle=float(msg.drive.steering_angle),
            received_s=self._now_s(),
        )

    def _publish_vesc(self, cmd: cg.PhysicalCommand) -> None:
        if cmd.drive_current_a > 0.0:
            self._current_pub.publish(Float64(data=float(cmd.drive_current_a)))
        elif cmd.brake_current_a > 0.0:
            self._brake_pub.publish(Float64(data=float(cmd.brake_current_a)))
        else:
            self._current_pub.publish(Float64(data=0.0))

        if self._publish_servo:
            self._servo_pub.publish(Float64(data=float(cmd.servo_position)))

    def _publish_applied(self, cmd: cg.PhysicalCommand) -> None:
        msg = ActuatorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.generation = next_applied_generation(self._state)
        msg.observation_stamp.sec = cmd.observation_stamp_sec
        msg.observation_stamp.nanosec = cmd.observation_stamp_nanosec
        msg.drive_current_a = cmd.drive_current_a
        msg.brake_current_a = cmd.brake_current_a
        msg.servo_position = cmd.servo_position
        msg.longitudinal = float(cmd.longitudinal)
        msg.steering = float(cmd.steering)
        msg.source = int(cmd.source)
        self._applied_pub.publish(msg)

    def _publish_diagnostics(self, source: int) -> None:
        diag = [0.0] * cg.GATE_DIAG_LEN
        diag[cg.GATE_DIAG_APPLIED_SOURCE] = float(source)
        diag[cg.GATE_DIAG_CONSECUTIVE_SAFE] = float(
            self._state.consecutive_safe_ticks
        )
        diag[cg.GATE_DIAG_RL_REJECT_TOTAL] = float(self._state.rl_reject_total)
        diag[cg.GATE_DIAG_LAST_REJECT_REASON] = float(
            self._state.last_reject_reason
        )
        for reason, count in enumerate(self._state.rejection_counts):
            diag[cg.GATE_DIAG_REJECT_COUNTS_START + reason] = float(count)
        self._diagnostics_msg.data = diag
        self._diagnostics_pub.publish(self._diagnostics_msg)

    def _on_timer(self) -> None:
        now_s = self._now_s()
        selected = arbitrate(
            self._cfg,
            self._state,
            now_s,
            self._rl,
            self._teleop,
            self._safety,
        )
        if selected.source == cg.SOURCE_SAFE:
            self._state.consecutive_safe_ticks += 1
        else:
            self._state.consecutive_safe_ticks = 0
        slewed = apply_slew(selected, self._state, self._cfg, now_s)
        self._publish_vesc(slewed)
        self._publish_applied(slewed)
        self._publish_diagnostics(slewed.source)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RlCurrentGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

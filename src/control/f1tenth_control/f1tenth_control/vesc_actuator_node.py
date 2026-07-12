"""Force-mode VESC actuator: Ackermann acceleration -> current / brake / servo.

Disabled by default. When ``enabled`` is true this node exclusively owns motor
current, brake current, and (optionally) servo commands. ERPM publication from
``ackermann_to_vesc`` must be remapped away in launch when this node is enabled.
"""

from __future__ import annotations

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float64

from f1tenth_control.drive_math import force_to_motor_currents


class VescActuatorNode(Node):
    def __init__(self) -> None:
        super().__init__("vesc_actuator")

        self.enabled = self.declare_parameter("enabled", False).value
        self.i_drive_max_a = float(
            self.declare_parameter("i_drive_max_a", 10.0).value
        )
        self.i_brake_max_a = float(
            self.declare_parameter("i_brake_max_a", 10.0).value
        )
        self.i_brake_safe_a = float(
            self.declare_parameter("i_brake_safe_a", 5.0).value
        )
        self.watchdog_timeout_s = float(
            self.declare_parameter("watchdog_timeout_s", 0.5).value
        )
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 50.0).value
        )
        self.publish_servo = bool(
            self.declare_parameter("publish_servo", True).value
        )
        self.zero_erpm = bool(self.declare_parameter("zero_erpm", True).value)
        self.steering_angle_to_servo_gain = float(
            self.declare_parameter("steering_angle_to_servo_gain", -1.2135).value
        )
        self.steering_angle_to_servo_offset = float(
            self.declare_parameter("steering_angle_to_servo_offset", 0.4495).value
        )
        ackermann_topic = self.declare_parameter(
            "ackermann_topic", "/ackermann_cmd"
        ).value

        self._longitudinal = 0.0
        self._steering = 0.0
        self._have_cmd = False
        self._last_cmd_time = self.get_clock().now()
        self._fault = False

        self._current_pub = self.create_publisher(Float64, "commands/motor/current", 10)
        self._brake_pub = self.create_publisher(Float64, "commands/motor/brake", 10)
        self._speed_pub = self.create_publisher(Float64, "commands/motor/speed", 10)
        self._servo_pub = self.create_publisher(Float64, "commands/servo/position", 10)

        self.create_subscription(
            AckermannDriveStamped, ackermann_topic, self._on_ackermann, 10
        )
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"vesc_actuator: enabled={self.enabled} "
            f"i_drive_max={self.i_drive_max_a:.1f}A "
            f"i_brake_max={self.i_brake_max_a:.1f}A "
            f"i_brake_safe={self.i_brake_safe_a:.1f}A "
            f"watchdog={self.watchdog_timeout_s:.2f}s"
        )

    def _on_ackermann(self, msg: AckermannDriveStamped) -> None:
        accel = float(msg.drive.acceleration)
        steer = float(msg.drive.steering_angle)
        if not math.isfinite(accel) or not math.isfinite(steer):
            self._fault = True
            return
        self._longitudinal = max(-1.0, min(1.0, accel))
        self._steering = steer
        self._have_cmd = True
        self._fault = False
        self._last_cmd_time = self.get_clock().now()

    def _safe_brake(self) -> bool:
        if self._fault:
            return True
        if not self._have_cmd:
            return True
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        return elapsed > self.watchdog_timeout_s

    def _publish(self, i_drive: float, i_brake: float, steering: float) -> None:
        # Mutual exclusion enforced by force_to_motor_currents / callers.
        cur = Float64()
        brk = Float64()
        cur.data = float(i_drive)
        brk.data = float(i_brake)
        self._current_pub.publish(cur)
        self._brake_pub.publish(brk)

        if self.zero_erpm:
            spd = Float64()
            spd.data = 0.0
            self._speed_pub.publish(spd)

        if self.publish_servo:
            servo = Float64()
            servo.data = (
                self.steering_angle_to_servo_gain * steering
                + self.steering_angle_to_servo_offset
            )
            self._servo_pub.publish(servo)

    def _on_timer(self) -> None:
        if not self.enabled:
            return

        if self._safe_brake():
            self._publish(0.0, self.i_brake_safe_a, 0.0)
            return

        i_drive, i_brake = force_to_motor_currents(
            self._longitudinal, self.i_drive_max_a, self.i_brake_max_a
        )
        self._publish(i_drive, i_brake, self._steering)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VescActuatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

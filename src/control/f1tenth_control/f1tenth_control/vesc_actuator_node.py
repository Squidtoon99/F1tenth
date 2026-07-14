"""Force-mode VESC actuator: Ackermann acceleration -> current / brake / servo.

Exclusive owner of motor current, brake current, and (optionally) servo commands.
Vendored ``ackermann_to_vesc`` motor/servo outputs are remapped away in
``car.launch.py``. New commands publish immediately; the timer only enforces the
watchdog heartbeat.
"""

from __future__ import annotations

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from std_msgs.msg import Float64

from f1tenth_control.drive_math import force_to_motor_currents


class VescActuatorNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("vesc_actuator", **kwargs)

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
            self.declare_parameter("watchdog_timeout_s", 0.15).value
        )
        self.publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 50.0).value
        )
        self.publish_servo = bool(
            self.declare_parameter("publish_servo", True).value
        )
        self.i_slew_a_per_s = float(
            self.declare_parameter("i_slew_a_per_s", 200.0).value
        )
        self.steering_angle_to_servo_gain = float(
            self.declare_parameter("steering_angle_to_servo_gain", -1.2135).value
        )
        self.steering_angle_to_servo_offset = float(
            self.declare_parameter("steering_angle_to_servo_offset", 0.4495).value
        )
        ackermann_topic = self.declare_parameter(
            "ackermann_topic", "/ackermann_cmd"
        ).value

        if self.i_drive_max_a <= 0.0 or self.i_brake_max_a <= 0.0:
            raise RuntimeError(
                "vesc_actuator: i_drive_max_a and i_brake_max_a must be > 0"
            )

        self._longitudinal = 0.0
        self._steering = 0.0
        self._have_cmd = False
        self._last_cmd_time = self.get_clock().now()
        self._fault = False
        self._last_i_drive = 0.0
        self._last_i_brake = 0.0
        self._last_pub_time = self.get_clock().now()

        self._current_pub = self.create_publisher(Float64, "commands/motor/current", 10)
        self._brake_pub = self.create_publisher(Float64, "commands/motor/brake", 10)
        self._servo_pub = self.create_publisher(Float64, "commands/servo/position", 10)

        self.create_subscription(
            AckermannDriveStamped, ackermann_topic, self._on_ackermann, 10
        )
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"vesc_actuator: i_drive_max={self.i_drive_max_a:.1f}A "
            f"i_brake_max={self.i_brake_max_a:.1f}A "
            f"i_brake_safe={self.i_brake_safe_a:.1f}A "
            f"watchdog={self.watchdog_timeout_s:.2f}s "
            f"slew={self.i_slew_a_per_s:.1f}A/s"
        )

    def _on_ackermann(self, msg: AckermannDriveStamped) -> None:
        accel = float(msg.drive.acceleration)
        steer = float(msg.drive.steering_angle)
        if not math.isfinite(accel) or not math.isfinite(steer):
            self._fault = True
            self._publish_safe()
            return
        self._longitudinal = max(-1.0, min(1.0, accel))
        self._steering = steer
        self._have_cmd = True
        self._fault = False
        self._last_cmd_time = self.get_clock().now()
        self._publish_cmd()

    def _safe_brake(self) -> bool:
        if self._fault:
            return True
        if not self._have_cmd:
            return True
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        return elapsed > self.watchdog_timeout_s

    def _slew(self, target_drive: float, target_brake: float) -> tuple[float, float]:
        now = self.get_clock().now()
        dt = (now - self._last_pub_time).nanoseconds * 1e-9
        self._last_pub_time = now
        if dt <= 0.0 or not math.isfinite(dt):
            return target_drive, target_brake
        max_step = abs(self.i_slew_a_per_s) * dt
        if max_step <= 0.0:
            return target_drive, target_brake

        def step(prev: float, tgt: float) -> float:
            delta = tgt - prev
            if abs(delta) <= max_step:
                return tgt
            return prev + math.copysign(max_step, delta)

        # Apply slew independently; mutual exclusion restored after.
        i_drive = step(self._last_i_drive, target_drive)
        i_brake = step(self._last_i_brake, target_brake)
        if i_drive > 0.0 and i_brake > 0.0:
            if target_drive >= target_brake:
                i_brake = 0.0
            else:
                i_drive = 0.0
        return i_drive, i_brake

    def _publish(self, i_drive: float, i_brake: float, steering: float) -> None:
        i_drive, i_brake = self._slew(i_drive, i_brake)
        self._last_i_drive = i_drive
        self._last_i_brake = i_brake

        if i_drive > 0.0:
            self._current_pub.publish(Float64(data=float(i_drive)))
        elif i_brake > 0.0:
            self._brake_pub.publish(Float64(data=float(i_brake)))
        else:
            self._current_pub.publish(Float64(data=0.0))

        if self.publish_servo:
            servo = Float64()
            servo.data = (
                self.steering_angle_to_servo_gain * steering
                + self.steering_angle_to_servo_offset
            )
            self._servo_pub.publish(servo)

    def _publish_safe(self) -> None:
        self._publish(0.0, self.i_brake_safe_a, 0.0)

    def _publish_cmd(self) -> None:
        i_drive, i_brake = force_to_motor_currents(
            self._longitudinal, self.i_drive_max_a, self.i_brake_max_a
        )
        self._publish(i_drive, i_brake, self._steering)

    def _on_timer(self) -> None:
        if self._safe_brake():
            self._publish_safe()
            return
        # Heartbeat: re-publish last command so the VESC does not time out.
        self._publish_cmd()


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

"""Force-mode VESC actuator: Ackermann acceleration -> current / brake / servo.

Exclusive owner of motor current, brake current, and (optionally) servo commands.
Vendored ``ackermann_to_vesc`` motor/servo outputs are remapped away in
``car.launch.py``. New commands publish immediately; the timer only enforces the
watchdog heartbeat.

In ``direct_policy`` mode the node gates normalized policy actions behind the R1
joystick deadman, publishes applied-action feedback, and fails closed to safe brake
when inputs are stale or invalid.
"""

from __future__ import annotations

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, Float64

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
        self.mode = str(self.declare_parameter("mode", "ackermann").value)
        ackermann_topic = self.declare_parameter(
            "ackermann_topic", "/ackermann_cmd"
        ).value
        self.desired_action_topic = str(
            self.declare_parameter(
                "desired_action_topic", "/sensor_policy/desired_action"
            ).value
        )
        self.applied_action_topic = str(
            self.declare_parameter(
                "applied_action_topic", "/sensor_policy/applied_action"
            ).value
        )
        self.joy_topic = str(self.declare_parameter("joy_topic", "/joy").value)
        self.autonomous_button = int(
            self.declare_parameter("autonomous_button", 10).value
        )
        self.joy_timeout_s = float(
            self.declare_parameter("joy_timeout_s", 0.25).value
        )
        self.policy_action_timeout_s = float(
            self.declare_parameter("policy_action_timeout_s", 0.15).value
        )
        self.max_steer = float(self.declare_parameter("max_steer", 0.33).value)

        if self.i_drive_max_a <= 0.0 or self.i_brake_max_a <= 0.0:
            raise RuntimeError(
                "vesc_actuator: i_drive_max_a and i_brake_max_a must be > 0"
            )
        if self.mode not in ("ackermann", "direct_policy"):
            raise RuntimeError(
                f"vesc_actuator: unsupported mode={self.mode!r}; "
                "expected 'ackermann' or 'direct_policy'"
            )

        self._longitudinal = 0.0
        self._steering = 0.0
        self._have_cmd = False
        self._last_cmd_time = self.get_clock().now()
        self._fault = False
        self._last_i_drive = 0.0
        self._last_i_brake = 0.0
        self._last_pub_time = self.get_clock().now()
        self._r1_held = False
        self._last_joy_time = None
        self._joy_sequence = 0
        self._action_joy_sequence = 0

        self._current_pub = self.create_publisher(Float64, "commands/motor/current", 10)
        self._brake_pub = self.create_publisher(Float64, "commands/motor/brake", 10)
        self._servo_pub = self.create_publisher(Float64, "commands/servo/position", 10)
        self._applied_pub = self.create_publisher(
            Float32MultiArray, self.applied_action_topic, 10
        )

        if self.mode == "direct_policy":
            self.create_subscription(
                Float32MultiArray,
                self.desired_action_topic,
                self._on_desired_action,
                10,
            )
            self.create_subscription(Joy, self.joy_topic, self._on_joy, 10)
        else:
            self.create_subscription(
                AckermannDriveStamped, ackermann_topic, self._on_ackermann, 10
            )

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"vesc_actuator mode={self.mode}: i_drive_max={self.i_drive_max_a:.1f}A "
            f"i_brake_max={self.i_brake_max_a:.1f}A "
            f"i_brake_safe={self.i_brake_safe_a:.1f}A "
            f"watchdog={self.watchdog_timeout_s:.2f}s "
            f"slew={self.i_slew_a_per_s:.1f}A/s"
        )

    def _on_joy(self, msg: Joy) -> None:
        buttons = msg.buttons
        self._r1_held = (
            self.autonomous_button < len(buttons)
            and buttons[self.autonomous_button] == 1
        )
        self._joy_sequence += 1
        self._last_joy_time = self.get_clock().now()
        if not self._r1_held:
            self._have_cmd = False
            self._publish_safe()

    def _on_desired_action(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self._fault = True
            self._have_cmd = False
            self._publish_safe()
            return
        force = float(msg.data[0])
        steer_norm = float(msg.data[1])
        if not math.isfinite(force) or not math.isfinite(steer_norm):
            self._fault = True
            self._have_cmd = False
            self._publish_safe()
            return
        if not self._r1_held or self._joy_stale():
            self._have_cmd = False
            self._publish_safe()
            return
        self._longitudinal = max(-1.0, min(1.0, force))
        self._steering = max(-1.0, min(1.0, steer_norm)) * self.max_steer
        self._have_cmd = True
        self._fault = False
        self._last_cmd_time = self.get_clock().now()
        self._action_joy_sequence = self._joy_sequence

    def _joy_stale(self) -> bool:
        if self._last_joy_time is None:
            return True
        elapsed = (self.get_clock().now() - self._last_joy_time).nanoseconds * 1e-9
        return elapsed > self.joy_timeout_s

    def _direct_policy_allowed(self) -> bool:
        if not self._r1_held or self._joy_stale():
            return False
        if self._fault or not self._have_cmd:
            return False
        if self._joy_sequence <= self._action_joy_sequence:
            return False
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        if elapsed > self.policy_action_timeout_s:
            return False
        if not math.isfinite(self._longitudinal) or not math.isfinite(self._steering):
            return False
        if abs(self._longitudinal) > 1.0 or abs(self._steering) > self.max_steer + 1e-6:
            return False
        return True

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
        timeout = (
            self.policy_action_timeout_s
            if self.mode == "direct_policy"
            else self.watchdog_timeout_s
        )
        if elapsed > timeout:
            return True
        if self.mode == "direct_policy" and not self._direct_policy_allowed():
            return True
        return False

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

        i_drive = step(self._last_i_drive, target_drive)
        i_brake = step(self._last_i_brake, target_brake)
        if i_drive > 0.0 and i_brake > 0.0:
            if target_drive >= target_brake:
                i_brake = 0.0
            else:
                i_drive = 0.0
        return i_drive, i_brake

    def _publish_applied_feedback(
        self, i_drive: float, i_brake: float, steering: float
    ) -> None:
        if self.mode != "direct_policy":
            return
        if i_drive > 0.0 and self.i_drive_max_a > 0.0:
            longitudinal = i_drive / self.i_drive_max_a
        elif i_brake > 0.0 and self.i_brake_max_a > 0.0:
            longitudinal = -i_brake / self.i_brake_max_a
        else:
            longitudinal = 0.0
        msg = Float32MultiArray()
        msg.data = [
            float(longitudinal),
            float(steering / self.max_steer if self.max_steer > 0.0 else 0.0),
        ]
        self._applied_pub.publish(msg)

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

        self._publish_applied_feedback(i_drive, i_brake, steering)

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

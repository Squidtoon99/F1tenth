"""drive_command_node: convert policy actions into Ackermann drive commands.

Subscribes to ``/rl/action`` and publishes ``/drive`` (AckermannDriveStamped).
Includes a watchdog that commands a safe stop if no action arrives within
``watchdog_timeout_s``.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_control.drive_math import (
    lag_alpha,
    map_action_to_drive,
    step_first_order_lag,
)


class DriveCommandNode(Node):
    def __init__(self, **kwargs):
        super().__init__("drive_command", **kwargs)
        self.declare_parameter("max_speed", ifc.MAX_SPEED)
        self.declare_parameter("max_steer", ifc.MAX_STEER)
        self.declare_parameter("clip_actions", ifc.CLIP_ACTIONS)
        self.declare_parameter("speed_limit_mps", 15.0)
        self.declare_parameter("watchdog_timeout_s", 0.5)
        self.declare_parameter("brake_behavior", "stop")
        # Match Genesis F1tenthEnv steer lag (config.py t_delta + 10 Hz control).
        self.declare_parameter("enable_output_filter", True)
        self.declare_parameter("t_delta", 0.1)
        self.declare_parameter("control_dt", 0.1)

        gp = self.get_parameter
        self.max_speed = gp("max_speed").get_parameter_value().double_value
        self.max_steer = gp("max_steer").get_parameter_value().double_value
        self.clip_actions = gp("clip_actions").get_parameter_value().double_value
        self.speed_limit_mps = (
            gp("speed_limit_mps").get_parameter_value().double_value
        )
        self.watchdog_timeout = (
            gp("watchdog_timeout_s").get_parameter_value().double_value
        )
        self.brake_behavior = gp("brake_behavior").get_parameter_value().string_value
        self.enable_output_filter = (
            gp("enable_output_filter").get_parameter_value().bool_value
        )
        self.steer_lag_alpha = lag_alpha(
            gp("control_dt").get_parameter_value().double_value,
            gp("t_delta").get_parameter_value().double_value,
        )
        self._filtered_steer = 0.0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, ifc.TOPIC_DRIVE, 10
        )
        self.create_subscription(Float32MultiArray, ifc.TOPIC_ACTION, self._on_action, 10)

        self._last_action_time = None
        self.watchdog = self.create_timer(0.1, self._watchdog_check)

    def _publish_drive(self, speed: float, steering_angle: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ifc.FRAME_BASE_LINK
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(msg)

    def _on_action(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("action message has < 2 elements; ignoring")
            return
        speed, steering_angle = map_action_to_drive(
            throttle=msg.data[0],
            steering=msg.data[1],
            max_speed=self.max_speed,
            max_steer=self.max_steer,
            clip_actions=self.clip_actions,
            brake_behavior=self.brake_behavior,
        )
        if self.enable_output_filter:
            self._filtered_steer = step_first_order_lag(
                self._filtered_steer, steering_angle, self.steer_lag_alpha
            )
            steering_angle = self._filtered_steer
        if speed > self.speed_limit_mps:
            speed = self.speed_limit_mps
        self._publish_drive(speed, steering_angle)
        self._last_action_time = self.get_clock().now()

    def _watchdog_check(self):
        if self._last_action_time is None:
            return
        elapsed = (self.get_clock().now() - self._last_action_time).nanoseconds * 1e-9
        if elapsed > self.watchdog_timeout:
            self._filtered_steer = 0.0
            self._publish_drive(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = DriveCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

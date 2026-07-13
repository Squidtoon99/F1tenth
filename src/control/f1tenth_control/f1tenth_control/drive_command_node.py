"""drive_command_node: convert policy actions into Ackermann drive commands.

Subscribes to ``/rl/action`` and publishes ``/drive`` (AckermannDriveStamped)
with force/brake effort on ``acceleration`` (ADR 0006). Includes a watchdog that
commands full brake if no action arrives within ``watchdog_timeout_s``.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_control.drive_math import (
    lag_alpha,
    map_action_to_force,
    step_first_order_lag,
)


class DriveCommandNode(Node):
    def __init__(self, **kwargs):
        super().__init__("drive_command", **kwargs)
        self.declare_parameter("max_steer", ifc.MAX_STEER)
        self.declare_parameter("clip_actions", ifc.CLIP_ACTIONS)
        self.declare_parameter("watchdog_timeout_s", 0.15)
        self.declare_parameter("enable_output_filter", True)
        self.declare_parameter("t_delta", 0.1)
        self.declare_parameter("control_dt", 1.0 / ifc.CONTROL_HZ)

        gp = self.get_parameter
        self.max_steer = gp("max_steer").get_parameter_value().double_value
        self.clip_actions = gp("clip_actions").get_parameter_value().double_value
        self.watchdog_timeout = (
            gp("watchdog_timeout_s").get_parameter_value().double_value
        )
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
        self.watchdog = self.create_timer(
            1.0 / ifc.CONTROL_HZ, self._watchdog_check
        )

    def _publish_drive(self, acceleration: float, steering_angle: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ifc.FRAME_BASE_LINK
        msg.drive.speed = 0.0
        msg.drive.steering_angle = float(steering_angle)
        msg.drive.acceleration = float(acceleration)
        self.drive_pub.publish(msg)

    def _on_action(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("action message has < 2 elements; ignoring")
            return
        if not (math.isfinite(msg.data[0]) and math.isfinite(msg.data[1])):
            self.get_logger().warn("non-finite action; commanding safe brake")
            self._filtered_steer = 0.0
            self._publish_drive(-1.0, 0.0)
            self._last_action_time = self.get_clock().now()
            return
        acceleration, steering_angle = map_action_to_force(
            throttle=msg.data[0],
            steering=msg.data[1],
            max_steer=self.max_steer,
            clip_actions=self.clip_actions,
        )
        if self.enable_output_filter:
            self._filtered_steer = step_first_order_lag(
                self._filtered_steer, steering_angle, self.steer_lag_alpha
            )
            steering_angle = self._filtered_steer
        self._publish_drive(acceleration, steering_angle)
        self._last_action_time = self.get_clock().now()

    def _watchdog_check(self):
        if self._last_action_time is None:
            return
        elapsed = (self.get_clock().now() - self._last_action_time).nanoseconds * 1e-9
        if elapsed > self.watchdog_timeout:
            self._filtered_steer = 0.0
            self._publish_drive(-1.0, 0.0)


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

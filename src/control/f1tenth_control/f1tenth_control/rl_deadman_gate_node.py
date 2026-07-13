"""Block RL /drive on the mux unless R1 (autonomous) or L1 (manual) is held."""

from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from sensor_msgs.msg import Joy


class RlDeadmanGateNode(Node):
    """Publish zero-speed teleop while neither manual nor autonomous deadman is held."""

    def __init__(self) -> None:
        super().__init__('rl_deadman_gate')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('teleop_topic', '/teleop')
        self.declare_parameter('manual_button', 4)
        self.declare_parameter('autonomous_button', 5)
        self.declare_parameter('publish_rate_hz', 20.0)

        joy_topic = self.get_parameter('joy_topic').value
        teleop_topic = self.get_parameter('teleop_topic').value
        self.manual_button = int(self.get_parameter('manual_button').value)
        self.autonomous_button = int(self.get_parameter('autonomous_button').value)
        rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self._manual_held = False
        self._autonomous_held = False

        self.create_subscription(Joy, joy_topic, self._joy_cb, 10)
        self._teleop_pub = self.create_publisher(AckermannDriveStamped, teleop_topic, 10)
        self.create_timer(1.0 / rate_hz, self._publish_block)

        self.get_logger().info(
            f'RL deadman gate: block /drive unless L1 (button {self.manual_button}) '
            f'or R1 (button {self.autonomous_button}) held'
        )

    def _joy_cb(self, msg: Joy) -> None:
        buttons = msg.buttons
        self._manual_held = (
            self.manual_button < len(buttons) and buttons[self.manual_button] == 1
        )
        self._autonomous_held = (
            self.autonomous_button < len(buttons) and buttons[self.autonomous_button] == 1
        )

    def _deadman_held(self) -> bool:
        return self._manual_held or self._autonomous_held

    def _publish_block(self) -> None:
        if self._deadman_held():
            return

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        # Request safe brake while blocking so the car does not coast under force mode.
        msg.drive.acceleration = -1.0
        self._teleop_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RlDeadmanGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

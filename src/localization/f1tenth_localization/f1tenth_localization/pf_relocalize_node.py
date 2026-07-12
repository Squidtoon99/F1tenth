"""Relocalize the particle filter from PS4 controller button presses."""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty


class PfRelocalizeNode(Node):
    """Publish PF reset topics when X (local) or Square (track spread) is pressed on /joy."""

    def __init__(self) -> None:
        super().__init__('pf_relocalize')

        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('track_relocalize_topic', '/pf/relocalize_on_track')
        self.declare_parameter('localize_button', 0)
        self.declare_parameter('global_button', 3)
        self.declare_parameter('start_x', 54.177)
        self.declare_parameter('start_y', 12.958)
        self.declare_parameter('start_yaw', -0.116)
        self.declare_parameter('publish_repeats', 5)
        self.declare_parameter('publish_interval_s', 0.1)
        self.declare_parameter('cooldown_s', 1.0)

        joy_topic = str(self.get_parameter('joy_topic').value)
        initialpose_topic = str(self.get_parameter('initialpose_topic').value)
        track_relocalize_topic = str(self.get_parameter('track_relocalize_topic').value)

        self._localize_button = int(self.get_parameter('localize_button').value)
        self._global_button = int(self.get_parameter('global_button').value)
        self._start_x = float(self.get_parameter('start_x').value)
        self._start_y = float(self.get_parameter('start_y').value)
        self._start_yaw = float(self.get_parameter('start_yaw').value)
        self._publish_repeats = int(self.get_parameter('publish_repeats').value)
        self._publish_interval_s = float(self.get_parameter('publish_interval_s').value)
        self._cooldown_s = float(self.get_parameter('cooldown_s').value)

        self._last_local_fire = None
        self._last_track_fire = None
        self._pending_repeats = 0
        self._pending_mode = ''

        self._initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, initialpose_topic, 10)
        self._track_relocalize_pub = self.create_publisher(
            Empty, track_relocalize_topic, 10)

        joy_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=25,
        )
        self.create_subscription(Joy, joy_topic, self._joy_callback, joy_qos)
        self._repeat_timer = self.create_timer(
            self._publish_interval_s, self._repeat_callback)

        self.get_logger().info(
            f'Ready: button {self._localize_button} -> local pose '
            f'({self._start_x:.2f}, {self._start_y:.2f}, yaw={self._start_yaw:.2f}), '
            f'button {self._global_button} -> track spread',
        )

    def _joy_callback(self, msg: Joy) -> None:
        buttons = list(msg.buttons)
        if self._pressed_with_cooldown(buttons, self._localize_button, '_last_local_fire'):
            self._pending_mode = 'local'
            self._pending_repeats = self._publish_repeats
            self._publish_local_pose()
            self.get_logger().info(
                f'Relocalized at ({self._start_x:.2f}, {self._start_y:.2f}, '
                f'yaw={self._start_yaw:.2f})',
            )
        elif self._pressed_with_cooldown(buttons, self._global_button, '_last_track_fire'):
            self._pending_mode = 'track'
            self._pending_repeats = self._publish_repeats
            self._publish_track_relocalize()
            self.get_logger().info('Track-wide PF reinitialization triggered')

    def _pressed_with_cooldown(
        self, buttons: list[int], index: int, last_fire_attr: str,
    ) -> bool:
        if index >= len(buttons) or buttons[index] != 1:
            return False
        now = self.get_clock().now()
        last_fire = getattr(self, last_fire_attr)
        if last_fire is not None:
            if (now - last_fire).nanoseconds < int(self._cooldown_s * 1e9):
                return False
        setattr(self, last_fire_attr, now)
        return True

    def _publish_local_pose(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self._start_x
        msg.pose.pose.position.y = self._start_y
        msg.pose.pose.orientation.z = math.sin(self._start_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self._start_yaw / 2.0)
        self._initpose_pub.publish(msg)

    def _publish_track_relocalize(self) -> None:
        self._track_relocalize_pub.publish(Empty())

    def _repeat_callback(self) -> None:
        if self._pending_repeats <= 1:
            self._pending_repeats = 0
            self._pending_mode = ''
            return
        self._pending_repeats -= 1
        if self._pending_mode == 'local':
            self._publish_local_pose()
        elif self._pending_mode == 'track':
            self._publish_track_relocalize()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PfRelocalizeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

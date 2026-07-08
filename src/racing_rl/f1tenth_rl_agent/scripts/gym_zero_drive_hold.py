#!/usr/bin/env python3
"""Hold ego and opponent still in f1tenth_gym by publishing zero Ackermann commands.

The gym bridge (async_mode) steps physics at ~100 Hz using the last /drive and
/opp_drive commands. gap_driver nodes may still be running after pkill misses;
this node overwrites both topics at a high rate with speed=0 and steer=0.

Run inside the gym container during static (or solo) detector validation::

    python3 gym_zero_drive_hold.py --hz 50

Does not modify f1tenth_gym — interfaces only via ROS topics.
"""

from __future__ import annotations

import argparse

import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from rclpy.node import Node

from f1tenth_rl_agent import interfaces as ifc


class ZeroDriveHold(Node):
    def __init__(self, hz: float, ego_topic: str, opp_topic: str):
        super().__init__("gym_zero_drive_hold")
        self._ego_pub = self.create_publisher(AckermannDriveStamped, ego_topic, 10)
        self._opp_pub = self.create_publisher(AckermannDriveStamped, opp_topic, 10)
        period = 1.0 / hz if hz > 0.0 else 0.02
        self.create_timer(period, self._publish)
        self.get_logger().info(
            f"Publishing zero Ackermann on {ego_topic} and {opp_topic} at {hz:.1f} Hz"
        )

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for pub in (self._ego_pub, self._opp_pub):
            msg = AckermannDriveStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.drive = AckermannDrive()
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0
            msg.drive.acceleration = 0.0
            msg.drive.jerk = 0.0
            pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero /drive and /opp_drive hold for gym validation")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--ego-topic", default=ifc.TOPIC_DRIVE)
    parser.add_argument("--opp-topic", default=ifc.TOPIC_OPP_DRIVE)
    args = parser.parse_args()

    rclpy.init()
    node = ZeroDriveHold(args.hz, args.ego_topic, args.opp_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Throttle a high-rate LaserScan down to a target rate for slam_toolbox.

The f1tenth_gym_ros bridge publishes /scan at ~250 Hz, which floods slam_toolbox's TF
message-filter queue (it drops most scans and the map fragments). This node republishes
at most ``target_rate_hz`` messages per second on ``output_topic`` (passing the scan
through unchanged, frame_id intact), giving slam a clean, processable stream.

Self-contained (no topic_tools dependency).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanThrottleNode(Node):
    def __init__(self) -> None:
        super().__init__("scan_throttle")
        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/scan_slam")
        self.declare_parameter("target_rate_hz", 25.0)

        self._min_period = 1.0 / max(float(self.get_parameter("target_rate_hz").value), 1e-3)
        self._last_pub = None

        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value
        self._pub = self.create_publisher(LaserScan, out_topic, 10)
        self.create_subscription(LaserScan, in_topic, self._on_scan, 10)
        self.get_logger().info(
            f"scan_throttle: {in_topic} -> {out_topic} @ "
            f"{float(self.get_parameter('target_rate_hz').value):.0f} Hz"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._last_pub is not None and (now - self._last_pub) < self._min_period:
            return
        self._last_pub = now
        self._pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ScanThrottleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

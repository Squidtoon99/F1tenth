"""localization_preflight_node: report health of the external localization inputs.

Localization is an external prerequisite: the RL vehicle graph consumes map-frame
pose from a particle filter (/pf/pose/odom), body twist from the VESC (/odom), and
the LiDAR scan (/scan). This read-only node watches those three sources and logs
their rate and frame_id, warning loudly when any is missing or stale. It never
publishes drive commands; the actual fail-safe is inherent (vehicle_obs emits no
observation without pose + twist, so policy_inference emits no action and the drive
watchdog stops the car). This node just makes the failure diagnosable on the car.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class LocalizationPreflightNode(Node):
    def __init__(self, **kwargs):
        super().__init__("localization_preflight", **kwargs)
        self.declare_parameter("pose_topic", "/pf/pose/odom")
        self.declare_parameter("twist_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("stale_timeout_s", 0.5)
        self.declare_parameter("expected_pose_frame", "map")
        self.declare_parameter("check_period_s", 1.0)

        gp = self.get_parameter
        pose_topic = gp("pose_topic").get_parameter_value().string_value
        twist_topic = gp("twist_topic").get_parameter_value().string_value
        scan_topic = gp("scan_topic").get_parameter_value().string_value
        self.stale_timeout = gp("stale_timeout_s").get_parameter_value().double_value
        self.expected_pose_frame = (
            gp("expected_pose_frame").get_parameter_value().string_value
        )
        period = gp("check_period_s").get_parameter_value().double_value

        self._last = {"pose": None, "twist": None, "scan": None}
        self._frame = {"pose": "", "scan": ""}
        self._ever_healthy = False

        self.create_subscription(Odometry, pose_topic, self._on_pose, 10)
        self.create_subscription(Odometry, twist_topic, self._on_twist, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.create_timer(period, self._check)
        self.get_logger().info(
            f"localization preflight watching pose='{pose_topic}' "
            f"twist='{twist_topic}' scan='{scan_topic}'"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, msg: Odometry):
        self._last["pose"] = self._now()
        self._frame["pose"] = msg.header.frame_id

    def _on_twist(self, msg: Odometry):
        self._last["twist"] = self._now()

    def _on_scan(self, msg: LaserScan):
        self._last["scan"] = self._now()
        self._frame["scan"] = msg.header.frame_id

    def _check(self):
        now = self._now()
        missing = []
        stale = []
        for name in ("pose", "twist", "scan"):
            t = self._last[name]
            if t is None:
                missing.append(name)
            elif now - t > self.stale_timeout:
                stale.append(f"{name} ({now - t:.2f}s old)")

        if missing or stale:
            parts = []
            if missing:
                parts.append("missing: " + ", ".join(missing))
            if stale:
                parts.append("stale: " + ", ".join(stale))
            self.get_logger().error(
                "localization NOT ready -- " + "; ".join(parts)
                + " (the car will not receive observations/actions until this clears)"
            )
            return

        if self._frame["pose"] and self._frame["pose"] != self.expected_pose_frame:
            self.get_logger().warn(
                f"pose frame_id '{self._frame['pose']}' != expected "
                f"'{self.expected_pose_frame}'; centerline must be in the same frame"
            )
        if not self._ever_healthy:
            self._ever_healthy = True
            self.get_logger().info("localization ready: pose + twist + scan are live")


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationPreflightNode()
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

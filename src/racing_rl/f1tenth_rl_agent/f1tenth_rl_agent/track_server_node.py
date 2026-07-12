"""track_server_node: publish the track centerline + widths + viz markers.

Loads the centerline CSV (the exact asset the policy trained on) and publishes it
on latched (``transient_local``) topics so evaluation and other consumers get a
single source of truth, plus RViz/Foxglove markers.
"""

from __future__ import annotations

import os

import numpy as np
import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.track_io import compute_track_boundaries, load_track_csv


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class TrackServerNode(Node):
    def __init__(self, **kwargs):
        super().__init__("track_server", **kwargs)
        self.declare_parameter("track_csv", "")
        self.declare_parameter("frame_id", ifc.FRAME_MAP)
        self.declare_parameter("republish_hz", 1.0)

        track_csv = self.get_parameter("track_csv").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        republish_hz = (
            self.get_parameter("republish_hz").get_parameter_value().double_value
        )

        if not track_csv:
            raise ValueError("track_server: 'track_csv' parameter is required")

        track_csv = self._resolve_track_csv(track_csv)

        self.centerline, self.w_tr_left, self.w_tr_right = load_track_csv(track_csv)
        self.get_logger().info(
            f"Loaded track '{track_csv}' with {self.centerline.shape[0]} centerline points"
        )

        latched = _latched_qos()
        self.path_pub = self.create_publisher(
            Path, ifc.TOPIC_TRACK_CENTERLINE, latched
        )
        self.widths_pub = self.create_publisher(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, latched
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, ifc.TOPIC_TRACK_MARKERS, latched
        )

        self._path_msg = self._build_path()
        self._widths_msg = self._build_widths()
        self._markers_msg = self._build_markers()

        self._publish_all()
        period = 1.0 / republish_hz if republish_hz > 0 else 1.0
        self.timer = self.create_timer(period, self._publish_all)

    def _resolve_track_csv(self, track_csv: str) -> str:
        """Resolve a bare filename against the package's bundled assets.

        Absolute/relative paths that exist are used as-is. A bare filename (e.g.
        ``IV_2026_SIM_centerline.csv``) is looked up in this package's installed
        ``assets/`` dir so the same config works in any container layout.
        """
        if os.path.isabs(track_csv) or os.path.exists(track_csv):
            return track_csv
        try:
            share = get_package_share_directory("f1tenth_rl_agent")
        except PackageNotFoundError:
            return track_csv
        candidate = os.path.join(share, "assets", track_csv)
        if os.path.exists(candidate):
            return candidate
        return track_csv

    def _build_path(self) -> Path:
        path = Path()
        path.header.frame_id = self.frame_id
        for x, y in self.centerline:
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def _build_widths(self) -> Float32MultiArray:
        n = self.centerline.shape[0]
        interleaved = np.empty(2 * n, dtype=np.float32)
        interleaved[0::2] = self.w_tr_left
        interleaved[1::2] = self.w_tr_right

        msg = Float32MultiArray()
        point_dim = MultiArrayDimension(label="point", size=n, stride=2 * n)
        lr_dim = MultiArrayDimension(label="lr", size=2, stride=2)
        msg.layout.dim = [point_dim, lr_dim]
        msg.data = interleaved.tolist()
        return msg

    def _line_strip(self, idx: int, pts: np.ndarray, rgba: tuple) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame_id
        m.ns = "track"
        m.id = idx
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.pose.orientation.w = 1.0
        for x, y in pts:
            m.points.append(Point(x=float(x), y=float(y), z=0.0))
        # close the loop
        m.points.append(Point(x=float(pts[0, 0]), y=float(pts[0, 1]), z=0.0))
        return m

    def _build_markers(self) -> MarkerArray:
        left, right = compute_track_boundaries(
            self.centerline, self.w_tr_left, self.w_tr_right
        )
        arr = MarkerArray()
        arr.markers.append(
            self._line_strip(0, self.centerline, (1.0, 1.0, 1.0, 0.7))
        )
        arr.markers.append(self._line_strip(1, left, (1.0, 0.1, 0.1, 0.85)))
        arr.markers.append(self._line_strip(2, right, (0.1, 0.3, 1.0, 0.85)))
        return arr

    def _stamp(self):
        now = self.get_clock().now().to_msg()
        self._path_msg.header.stamp = now
        for ps in self._path_msg.poses:
            ps.header.stamp = now
        for m in self._markers_msg.markers:
            m.header.stamp = now

    def _publish_all(self):
        self._stamp()
        self.path_pub.publish(self._path_msg)
        self.widths_pub.publish(self._widths_msg)
        self.markers_pub.publish(self._markers_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackServerNode()
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

"""rclpy integration test for track_server_node (runs in the ROS 2 container)."""

import os
import tempfile

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile  # noqa: E402
from nav_msgs.msg import Path  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent.track_server_node import TrackServerNode  # noqa: E402
from _helpers import make_oval, write_track_csv  # noqa: E402


def _latched():
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def _spin_until(node, predicate, timeout_s=5.0):
    end = node.get_clock().now().nanoseconds + int(timeout_s * 1e9)
    while node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_track_server_publishes_track():
    n = 150
    cl, wl, wr = make_oval(n=n)
    rclpy.init()
    server = None
    sub_node = None
    try:
        with tempfile.TemporaryDirectory() as d:
            path = write_track_csv(os.path.join(d, "track.csv"), cl, wl, wr)
            server = TrackServerNode(
                parameter_overrides=[Parameter("track_csv", value=path)]
            )

            sub_node = rclpy.create_node("track_sub")
            received = {}
            sub_node.create_subscription(
                Path, ifc.TOPIC_TRACK_CENTERLINE,
                lambda m: received.__setitem__("path", m), _latched())
            sub_node.create_subscription(
                Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS,
                lambda m: received.__setitem__("widths", m), _latched())
            sub_node.create_subscription(
                MarkerArray, ifc.TOPIC_TRACK_MARKERS,
                lambda m: received.__setitem__("markers", m), _latched())

            def ready():
                rclpy.spin_once(server, timeout_sec=0.0)
                rclpy.spin_once(sub_node, timeout_sec=0.0)
                return {"path", "widths", "markers"}.issubset(received)

            assert _spin_until(server, ready, timeout_s=5.0)
            assert len(received["path"].poses) == n
            assert len(received["widths"].data) == 2 * n
            assert len(received["markers"].markers) == 3
    finally:
        if server is not None:
            server.destroy_node()
        if sub_node is not None:
            sub_node.destroy_node()
        rclpy.shutdown()

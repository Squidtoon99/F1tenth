"""rclpy integration test for evaluation_node (runs in the container)."""

import math

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent.evaluation_node import EvaluationNode  # noqa: E402
from _helpers import make_oval  # noqa: E402


def _latched():
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def test_evaluation_publishes_metrics():
    cl, wl, wr = make_oval(n=180)
    rclpy.init()
    node = None
    pub = None
    try:
        node = EvaluationNode()
        pub = rclpy.create_node("eval_pub")
        path_pub = pub.create_publisher(Path, ifc.TOPIC_TRACK_CENTERLINE, _latched())
        width_pub = pub.create_publisher(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, _latched()
        )
        odom_pub = pub.create_publisher(Odometry, ifc.TOPIC_ODOM, 10)

        path = Path()
        path.header.frame_id = ifc.FRAME_MAP
        for x, y in cl:
            ps = PoseStamped()
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        path_pub.publish(path)

        w = Float32MultiArray()
        inter = np.empty(2 * len(wl), dtype=np.float32)
        inter[0::2] = wl
        inter[1::2] = wr
        w.data = inter.tolist()
        width_pub.publish(w)

        received = {}
        # Subscribe on the separate publisher node: subscribing on `node` itself
        # (which also has a busy odom subscription) starves this callback under
        # single-threaded spin_once.
        pub.create_subscription(
            Float32MultiArray, ifc.TOPIC_METRICS,
            lambda m: received.__setitem__("m", m), 10)

        odom = Odometry()
        odom.pose.pose.position.x = 20.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = 3.0

        end = node.get_clock().now().nanoseconds + int(5e9)
        while node.get_clock().now().nanoseconds < end and "m" not in received:
            odom_pub.publish(odom)
            rclpy.spin_once(pub, timeout_sec=0.02)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert "m" in received
        assert len(received["m"].data) == ifc.METRICS_LEN
        assert math.isfinite(received["m"].data[ifc.METRICS_SPEED])
    finally:
        if node is not None:
            node.destroy_node()
        if pub is not None:
            pub.destroy_node()
        rclpy.shutdown()

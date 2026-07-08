"""rclpy integration test for observation_builder_node (runs in the container)."""

import math

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry, Path  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

from f1tenth_rl_agent import interfaces as ifc  # noqa: E402
from f1tenth_rl_agent.observation_builder_node import ObservationBuilderNode  # noqa: E402
from _helpers import make_oval  # noqa: E402


def _latched():
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


def _path_msg(cl):
    msg = Path()
    msg.header.frame_id = ifc.FRAME_MAP
    for x, y in cl:
        ps = PoseStamped()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def _widths_msg(wl, wr):
    msg = Float32MultiArray()
    inter = np.empty(2 * len(wl), dtype=np.float32)
    inter[0::2] = wl
    inter[1::2] = wr
    msg.data = inter.tolist()
    return msg


def _odom(x, y, yaw, vx, stamp_s):
    msg = Odometry()
    msg.header.stamp.sec = int(stamp_s)
    msg.header.stamp.nanosec = int((stamp_s % 1) * 1e9)
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)
    msg.twist.twist.linear.x = vx
    msg.twist.twist.angular.z = 0.1
    return msg


def test_observation_builder_emits_380():
    cl, wl, wr = make_oval(n=180)
    rclpy.init()
    node = None
    pub = None
    try:
        node = ObservationBuilderNode()
        pub = rclpy.create_node("ob_pub")
        path_pub = pub.create_publisher(Path, ifc.TOPIC_TRACK_CENTERLINE, _latched())
        width_pub = pub.create_publisher(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, _latched()
        )
        odom_pub = pub.create_publisher(Odometry, ifc.TOPIC_ODOM, 10)

        path_pub.publish(_path_msg(cl))
        width_pub.publish(_widths_msg(wl, wr))

        received = {}
        # Subscribe on the separate publisher node: subscribing on `node` itself
        # (which also has a busy odom subscription) starves this callback under
        # single-threaded spin_once.
        pub.create_subscription(
            Float32MultiArray, ifc.TOPIC_OBSERVATION,
            lambda m: received.__setitem__("obs", m), 10)

        end = node.get_clock().now().nanoseconds + int(8e9)
        t = 0.0
        while node.get_clock().now().nanoseconds < end and "obs" not in received:
            odom_pub.publish(_odom(20.0, 0.0, 0.5, 3.0, t))
            t += 0.1
            rclpy.spin_once(pub, timeout_sec=0.02)
            rclpy.spin_once(node, timeout_sec=0.05)

        assert "obs" in received, "no observation published"
        data = np.asarray(received["obs"].data, dtype=np.float32)
        assert data.shape == (ifc.NUM_OBS,)
        assert np.isfinite(data).all()
    finally:
        if node is not None:
            node.destroy_node()
        if pub is not None:
            pub.destroy_node()
        rclpy.shutdown()

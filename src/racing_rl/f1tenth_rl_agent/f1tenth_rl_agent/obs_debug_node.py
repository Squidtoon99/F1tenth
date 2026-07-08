"""obs_debug_node: decode /rl/observation into plottable scalars and markers.

This node is read-only diagnostics: it consumes the exact observation vector the
policy sees and republishes it as a flat scalar array (for time-series plots) plus
RViz/Foxglove markers (future track corridor, opponent position). It never touches
the drive command. It is intended for diagnosing wall collisions on the real car,
where neither the sim ``evaluation`` node nor the builder debug markers are running.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import obs_debug_viz as viz


class ObsDebugNode(Node):
    def __init__(self, **kwargs):
        super().__init__("obs_debug", **kwargs)
        self.declare_parameter("pose_topic", "/pf/pose/odom")
        self.declare_parameter("future_track_num_points", ifc.FUTURE_TRACK_NUM_POINTS)
        self.declare_parameter("enable_markers", True)
        self.declare_parameter("frame_id", ifc.FRAME_MAP)

        gp = self.get_parameter
        pose_topic = gp("pose_topic").get_parameter_value().string_value
        self.num_points = int(
            gp("future_track_num_points").get_parameter_value().integer_value
        )
        self.enable_markers = gp("enable_markers").get_parameter_value().bool_value
        self.frame_id = gp("frame_id").get_parameter_value().string_value

        self._last_pose: Odometry | None = None
        self._warned_len = False

        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_OBSERVATION, self._on_obs, 10
        )
        self.create_subscription(Odometry, pose_topic, self._on_pose, 10)

        self.scalars_pub = self.create_publisher(
            Float32MultiArray, ifc.TOPIC_OBS_DEBUG_SCALARS, 10
        )
        if self.enable_markers:
            self.future_pub = self.create_publisher(
                MarkerArray, ifc.TOPIC_FUTURE_POINTS, 1
            )
            self.opp_pub = self.create_publisher(
                Marker, ifc.TOPIC_OBS_DEBUG_OPPONENT, 1
            )

        self.get_logger().info(
            f"obs_debug ready; pose_topic={pose_topic} "
            f"num_points={self.num_points} markers={self.enable_markers}"
        )

    def _on_pose(self, msg: Odometry):
        self._last_pose = msg

    def _on_obs(self, msg: Float32MultiArray):
        obs = np.asarray(msg.data, dtype=np.float32)
        expected = ifc.OBS_FUTURE_POINTS[0] + 3 * self.num_points * 2 + ifc.NUM_TYRE_SLIP
        if obs.shape[0] not in (ifc.NUM_OBS_BASE, ifc.NUM_OBS_1V1) and obs.shape[0] < expected:
            if not self._warned_len:
                self.get_logger().warn(
                    f"unexpected observation length {obs.shape[0]} "
                    f"(expected {ifc.NUM_OBS_BASE} or {ifc.NUM_OBS_1V1}); "
                    "skipping decode"
                )
                self._warned_len = True
            return

        scalars = viz.decode_scalars(obs, self.num_points)
        out = Float32MultiArray()
        out.data = scalars.tolist()
        self.scalars_pub.publish(out)

        if not self.enable_markers:
            return

        pose = self._last_pose
        if pose is None:
            return
        pos = pose.pose.pose.position
        q = pose.pose.pose.orientation
        yaw = float(
            math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
        )
        now = self.get_clock().now().to_msg()

        future = viz.future_block(obs, self.num_points)
        markers = viz.build_future_markers(
            future, float(pos.x), float(pos.y), yaw, self.frame_id, now
        )
        self.future_pub.publish(markers)
        self._publish_opponent_marker(scalars, float(pos.x), float(pos.y), yaw, now)

    def _publish_opponent_marker(self, scalars, ego_x, ego_y, yaw, stamp):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = "opponent"
        m.id = 0
        present = scalars[ifc.OBS_DEBUG_OPP_PRESENT] > 0.5
        if not present:
            m.action = Marker.DELETE
            self.opp_pub.publish(m)
            return
        wx, wy = viz.opponent_world_xy(
            float(scalars[ifc.OBS_DEBUG_OPP_REL_X]),
            float(scalars[ifc.OBS_DEBUG_OPP_REL_Y]),
            ego_x,
            ego_y,
            yaw,
        )
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.0, 1.0, 0.9)
        self.opp_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ObsDebugNode()
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

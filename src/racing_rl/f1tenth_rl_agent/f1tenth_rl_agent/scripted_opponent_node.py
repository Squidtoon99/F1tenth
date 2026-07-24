"""scripted_opponent_node: centerline-following opponent for 1v1 gym deploy.

Matches training ``ScriptedCenterlineOpponent``: P-control on lateral error and
heading error with a fixed target speed, publishing Ackermann commands on
``/opp_drive`` so the f1tenth_gym bridge steps the second agent.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.obs_core import ObservationBuilder, build_boundary_state, frenet_projection


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class ScriptedOpponentNode(Node):
    def __init__(self, **kwargs):
        super().__init__("scripted_opponent", **kwargs)
        self.declare_parameter("control_hz", ifc.CONTROL_HZ)
        self.declare_parameter("opponent_target_speed", 3.0)
        self.declare_parameter("kp_ey", 1.0)
        self.declare_parameter("kh_heading", 1.0)
        self.declare_parameter("max_steer", ifc.MAX_STEER)

        gp = self.get_parameter
        self.target_speed = gp("opponent_target_speed").get_parameter_value().double_value
        self.kp_ey = gp("kp_ey").get_parameter_value().double_value
        self.kh_heading = gp("kh_heading").get_parameter_value().double_value
        self.max_steer = gp("max_steer").get_parameter_value().double_value
        hz = gp("control_hz").get_parameter_value().double_value

        self._centerline = None
        self._w_left = None
        self._w_right = None
        self._geom = None
        self._last_odom: Odometry | None = None

        latched = _latched_qos()
        self.create_subscription(Path, ifc.TOPIC_TRACK_CENTERLINE, self._on_centerline, latched)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, self._on_widths, latched
        )
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_odom, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, ifc.TOPIC_OPP_DRIVE, 10)
        self.create_timer(1.0 / hz if hz > 0 else 1.0 / ifc.CONTROL_HZ, self._on_timer)
        self.get_logger().info(
            f"scripted_opponent ready (target_speed={self.target_speed:.2f} m/s)"
        )

    def _on_centerline(self, msg: Path):
        self._centerline = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        self._maybe_init_geom()

    def _on_widths(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        self._w_left = data[0::2].copy()
        self._w_right = data[1::2].copy()
        self._maybe_init_geom()

    def _maybe_init_geom(self):
        if self._centerline is None or self._w_left is None:
            return
        if len(self._w_left) != len(self._centerline):
            return
        builder = ObservationBuilder(
            self._centerline, self._w_left, self._w_right, device=torch.device("cpu")
        )
        self._geom = builder.geom

    def _on_odom(self, msg: Odometry):
        self._last_odom = msg

    def _on_timer(self):
        if self._geom is None or self._last_odom is None:
            return
        msg = self._last_odom
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        base_pos = torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32)
        frenet = frenet_projection(base_pos, self._geom, torch.device("cpu"))
        boundary = build_boundary_state(
            frenet,
            torch.as_tensor(self._w_left, dtype=torch.float32),
            torch.as_tensor(self._w_right, dtype=torch.float32),
        )
        ey = float(boundary["ey"][0])
        seg_dir = frenet["seg_dir"][0]
        track_angle = float(torch.atan2(seg_dir[1], seg_dir[0]))
        heading_err = yaw - track_angle
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        delta_max = max(self.max_steer, 1e-6)
        steer = -(self.kp_ey * ey + self.kh_heading * heading_err) / delta_max
        steer = max(-1.0, min(1.0, steer))
        steering_angle = steer * self.max_steer

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "opp_racecar/base_link"
        out.drive.speed = float(self.target_speed)
        out.drive.steering_angle = float(steering_angle)
        self.drive_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedOpponentNode()
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

"""observation_builder_node: build the policy observation from odometry (380 or 387).

Subscribes to ground-truth odometry and the latched track topics, reconstructs the
exact training observation via obs_core, and publishes it at the control rate.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import obs_debug_viz
from f1tenth_rl_agent.obs_core import ObservationBuilder, quat_xyzw_to_wxyz


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class ObservationBuilderNode(Node):
    def __init__(self, **kwargs):
        super().__init__("observation_builder", **kwargs)
        self.declare_parameter("control_hz", ifc.CONTROL_HZ)
        self.declare_parameter("contact_margin_m", ifc.CONTACT_MARGIN_M)
        self.declare_parameter("future_track_num_points", ifc.FUTURE_TRACK_NUM_POINTS)
        self.declare_parameter("future_track_horizon_s", ifc.FUTURE_TRACK_HORIZON_S)
        self.declare_parameter("future_track_width", ifc.FUTURE_TRACK_WIDTH)
        self.declare_parameter("twist_in_world_frame", False)
        self.declare_parameter("publish_debug_markers", True)
        self.declare_parameter("enable_opponent_obs", False)
        self.declare_parameter("zero_opponent_obs", False)
        self.declare_parameter("opponent_odom_topic", ifc.TOPIC_OPP_ODOM)

        gp = self.get_parameter
        self.control_hz = gp("control_hz").get_parameter_value().double_value
        self.twist_in_world = (
            gp("twist_in_world_frame").get_parameter_value().bool_value
        )
        self.publish_markers = (
            gp("publish_debug_markers").get_parameter_value().bool_value
        )
        self.enable_opponent = (
            gp("enable_opponent_obs").get_parameter_value().bool_value
        )
        self.zero_opponent = (
            gp("zero_opponent_obs").get_parameter_value().bool_value
        )
        self.obs_cfg = ifc.default_obs_cfg(self.enable_opponent)
        self.obs_cfg["zero_opponent_obs"] = self.zero_opponent
        self.obs_cfg["contact_margin_m"] = (
            gp("contact_margin_m").get_parameter_value().double_value
        )
        self.obs_cfg["future_track_num_points"] = (
            gp("future_track_num_points").get_parameter_value().integer_value
        )
        self.obs_cfg["future_track_horizon_s"] = (
            gp("future_track_horizon_s").get_parameter_value().double_value
        )
        self.obs_cfg["future_track_width"] = (
            gp("future_track_width").get_parameter_value().double_value
        )

        self.builder: ObservationBuilder | None = None
        self._centerline = None
        self._w_left = None
        self._w_right = None

        self._last_odom: Odometry | None = None
        self._last_opp_odom: Odometry | None = None
        self._prev_body_vel: np.ndarray | None = None
        self._prev_vel_stamp: float | None = None
        self._body_accel = np.zeros(2, dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)

        latched = _latched_qos()
        self.create_subscription(Path, ifc.TOPIC_TRACK_CENTERLINE, self._on_centerline, latched)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, self._on_widths, latched
        )
        opp_topic = gp("opponent_odom_topic").get_parameter_value().string_value
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_odom, 10)
        if self.enable_opponent and not self.zero_opponent:
            self.create_subscription(Odometry, opp_topic, self._on_opp_odom, 10)
        self.create_subscription(Float32MultiArray, ifc.TOPIC_ACTION, self._on_action, 10)

        self.obs_pub = self.create_publisher(Float32MultiArray, ifc.TOPIC_OBSERVATION, 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, ifc.TOPIC_FUTURE_POINTS, 1
        )

        period = 1.0 / self.control_hz if self.control_hz > 0 else 0.1
        self.timer = self.create_timer(period, self._on_timer)
        self.get_logger().info("observation_builder ready; waiting for track + odom")

    # --- subscriptions ---------------------------------------------------------
    def _on_centerline(self, msg: Path):
        pts = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=np.float32,
        )
        self._centerline = pts
        self._maybe_build_builder()

    def _on_widths(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        self._w_left = data[0::2].copy()
        self._w_right = data[1::2].copy()
        self._maybe_build_builder()

    def _maybe_build_builder(self):
        if self.builder is not None:
            return
        if self._centerline is None or self._w_left is None or self._w_right is None:
            return
        if len(self._w_left) != len(self._centerline):
            self.get_logger().warn(
                "track widths length does not match centerline; waiting for consistent data"
            )
            return
        self.builder = ObservationBuilder(
            centerline=self._centerline,
            w_tr_left=self._w_left,
            w_tr_right=self._w_right,
            obs_cfg=self.obs_cfg,
            device=torch.device("cpu"),
        )
        self.get_logger().info(
            f"ObservationBuilder initialized with {len(self._centerline)} points; "
            f"num_obs={self.obs_cfg['num_obs']}"
        )

    def _on_action(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self._last_action = np.array(msg.data[:2], dtype=np.float32)

    def _on_odom(self, msg: Odometry):
        self._last_odom = msg

    def _on_opp_odom(self, msg: Odometry):
        self._last_opp_odom = msg

    # --- main loop -------------------------------------------------------------
    @staticmethod
    def _stamp_seconds(msg: Odometry) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _body_velocity(self, msg: Odometry, yaw: float) -> np.ndarray:
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if self.twist_in_world:
            c, s = math.cos(yaw), math.sin(yaw)
            bx = c * vx + s * vy
            by = -s * vx + c * vy
            return np.array([bx, by], dtype=np.float32)
        return np.array([vx, vy], dtype=np.float32)

    def _on_timer(self):
        if self.builder is None or self._last_odom is None:
            return
        msg = self._last_odom

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        quat_xyzw = torch.tensor([[q.x, q.y, q.z, q.w]], dtype=torch.float32)
        quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
        yaw = float(
            math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
        )

        body_vel = self._body_velocity(msg, yaw)
        stamp = self._stamp_seconds(msg)
        if self._prev_body_vel is not None and self._prev_vel_stamp is not None:
            dt = stamp - self._prev_vel_stamp
            if dt > 1e-4:
                self._body_accel = (body_vel - self._prev_body_vel) / dt
        self._prev_body_vel = body_vel
        self._prev_vel_stamp = stamp

        base_lin_vel = torch.tensor([[body_vel[0], body_vel[1], 0.0]], dtype=torch.float32)
        base_ang_vel = torch.tensor(
            [[0.0, 0.0, msg.twist.twist.angular.z]], dtype=torch.float32
        )
        base_lin_acc = torch.tensor(
            [[self._body_accel[0], self._body_accel[1], 0.0]], dtype=torch.float32
        )
        last_actions = torch.tensor([self._last_action], dtype=torch.float32)
        base_pos = torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32)

        opponent_block = None
        if self.enable_opponent:
            if self.zero_opponent or self._last_opp_odom is None:
                opponent_block = base_lin_vel.new_zeros((1, ifc.OPPONENT_OBS_DIM))
            else:
                opp = self._last_opp_odom
                opp_pos = opp.pose.pose.position
                oq = opp.pose.pose.orientation
                opp_yaw = float(
                    math.atan2(
                        2.0 * (oq.w * oq.z + oq.x * oq.y),
                        1.0 - 2.0 * (oq.y * oq.y + oq.z * oq.z),
                    )
                )
                ego_vel_world = torch.tensor(
                    [[msg.twist.twist.linear.x, msg.twist.twist.linear.y]],
                    dtype=torch.float32,
                )
                opp_vel_world = torch.tensor(
                    [[opp.twist.twist.linear.x, opp.twist.twist.linear.y]],
                    dtype=torch.float32,
                )
                ego_yaw_t = torch.tensor([yaw], dtype=torch.float32)
                opp_pos_t = torch.tensor([[opp_pos.x, opp_pos.y, opp_pos.z]], dtype=torch.float32)
                opponent_block = self.builder.build_opponent_block(
                    ego_pos=base_pos,
                    ego_yaw=ego_yaw_t,
                    ego_vel_world=ego_vel_world,
                    opp_pos=opp_pos_t,
                    opp_vel_world=opp_vel_world,
                    present=torch.tensor([1.0], dtype=torch.float32),
                )

        obs = self.builder.build(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            base_lin_acc=base_lin_acc,
            last_actions=last_actions,
            base_pos=base_pos,
            base_quat_wxyz=quat_wxyz,
            opponent_block=opponent_block,
        )
        obs_np = obs.squeeze(0).numpy().astype(np.float32)

        out = Float32MultiArray()
        out.data = obs_np.tolist()
        self.obs_pub.publish(out)

        if self.publish_markers:
            self._publish_future_markers(obs_np, float(pos.x), float(pos.y), yaw)

    def _publish_future_markers(self, obs_np, px, py, yaw):
        samples = self.obs_cfg["future_track_num_points"]
        future = obs_debug_viz.future_block(obs_np, samples)
        now = self.get_clock().now().to_msg()
        arr = obs_debug_viz.build_future_markers(
            future, px, py, yaw, ifc.FRAME_MAP, now
        )
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ObservationBuilderNode()
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

"""evaluation_node: monitor the closed loop and report success metrics.

Uses the same Frenet projection as the observation builder to track lap progress,
lateral error, out-of-bounds and stuck conditions. Publishes ``/rl/metrics`` and,
when ``auto_reset`` is enabled, resets the car via ``/initialpose`` on episode end.
"""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
import torch
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.eval_logic import EpisodeMonitor, opponent_pose_ahead
from f1tenth_rl_agent.obs_core import ObservationBuilder
from f1tenth_rl_agent.spawn_utils import MapAssets, load_map_assets, sample_reset_pose


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class EvaluationNode(Node):
    def __init__(self, **kwargs):
        super().__init__("evaluation", **kwargs)
        self.declare_parameter("oob_margin_m", 0.0)
        self.declare_parameter("stuck_speed_mps", 0.2)
        self.declare_parameter("stuck_timeout_s", 3.0)
        self.declare_parameter("reset_x", 0.0)
        self.declare_parameter("reset_y", 2.0)
        self.declare_parameter("reset_yaw", -0.9)
        self.declare_parameter("auto_reset", True)
        self.declare_parameter("random_reset", True)
        self.declare_parameter("random_spawn_on_start", True)
        self.declare_parameter("reset_seed", -1)
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("spawn_max_dist", 1.0)
        self.declare_parameter("reset_grace_s", 2.0)
        self.declare_parameter("enable_opponent_spawn", False)
        self.declare_parameter("opponent_spawn_gap_m", 7.0)
        self.declare_parameter("spawn_opponent_on_start", False)

        gp = self.get_parameter
        self.reset_x = gp("reset_x").get_parameter_value().double_value
        self.reset_y = gp("reset_y").get_parameter_value().double_value
        self.reset_yaw = gp("reset_yaw").get_parameter_value().double_value
        self.auto_reset = gp("auto_reset").get_parameter_value().bool_value
        self.random_reset = gp("random_reset").get_parameter_value().bool_value
        self.random_spawn_on_start = (
            gp("random_spawn_on_start").get_parameter_value().bool_value
        )
        seed = int(gp("reset_seed").get_parameter_value().integer_value)
        self._rng = np.random.default_rng(None if seed < 0 else seed)
        self._did_initial_spawn = False
        self.reset_grace_s = gp("reset_grace_s").get_parameter_value().double_value
        self._reset_grace_until = 0.0
        self.spawn_max_dist = gp("spawn_max_dist").get_parameter_value().double_value
        self.enable_opponent_spawn = gp("enable_opponent_spawn").get_parameter_value().bool_value
        self.opponent_spawn_gap_m = gp("opponent_spawn_gap_m").get_parameter_value().double_value
        self.spawn_opponent_on_start = gp("spawn_opponent_on_start").get_parameter_value().bool_value
        self._opp_spawn_timer = None
        self._pending_ego_spawn: tuple[float, float, float] | None = None
        self._map_assets: MapAssets | None = None
        map_yaml = gp("map_yaml").get_parameter_value().string_value.strip()
        map_yaml = self._resolve_asset(map_yaml)
        if map_yaml:
            try:
                self._map_assets = load_map_assets(map_yaml)
                self.get_logger().info(f"Loaded spawn map from {map_yaml}")
            except (OSError, ValueError, KeyError) as exc:
                self.get_logger().warn(
                    f"Could not load map_yaml '{map_yaml}' ({exc}); "
                    "falling back to centerline spawns"
                )

        self.monitor = EpisodeMonitor(
            oob_margin_m=gp("oob_margin_m").get_parameter_value().double_value,
            stuck_speed_mps=gp("stuck_speed_mps").get_parameter_value().double_value,
            stuck_timeout_s=gp("stuck_timeout_s").get_parameter_value().double_value,
        )

        self.builder: ObservationBuilder | None = None
        self._centerline = None
        self._w_left = None
        self._w_right = None

        latched = _latched_qos()
        self.create_subscription(Path, ifc.TOPIC_TRACK_CENTERLINE, self._on_centerline, latched)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, self._on_widths, latched
        )
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_odom, 10)

        self.metrics_pub = self.create_publisher(Float32MultiArray, ifc.TOPIC_METRICS, 10)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, ifc.TOPIC_INITIALPOSE, 1
        )
        self.opp_reset_pub = self.create_publisher(PoseStamped, ifc.TOPIC_GOAL_POSE, 1)

    @staticmethod
    def _resolve_asset(path: str) -> str:
        """Resolve a bare filename against this package's bundled assets.

        Absolute/relative paths that exist are used as-is; a bare filename (e.g.
        ``IV_2026_SIM.yaml``) is looked up in the installed ``assets/`` dir so the
        same config works regardless of container layout. Empty stays empty.
        """
        if not path or os.path.isabs(path) or os.path.exists(path):
            return path
        try:
            share = get_package_share_directory("f1tenth_rl_agent")
        except PackageNotFoundError:
            return path
        candidate = os.path.join(share, "assets", path)
        return candidate if os.path.exists(candidate) else path

    def _on_centerline(self, msg: Path):
        self._centerline = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float32
        )
        self._maybe_build()

    def _on_widths(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        self._w_left = data[0::2].copy()
        self._w_right = data[1::2].copy()
        self._maybe_build()

    def _maybe_build(self):
        if self.builder is not None:
            return
        if self._centerline is None or self._w_left is None or self._w_right is None:
            return
        if len(self._w_left) != len(self._centerline):
            return
        self.builder = ObservationBuilder(
            centerline=self._centerline,
            w_tr_left=self._w_left,
            w_tr_right=self._w_right,
            obs_cfg={"num_obs": ifc.NUM_OBS},
            device=torch.device("cpu"),
        )
        self.get_logger().info("evaluation monitor ready")
        if not self._did_initial_spawn and (
            self.random_spawn_on_start or self.spawn_opponent_on_start
        ):
            reason = "initial random spawn" if self.random_spawn_on_start else "initial 1v1 spawn"
            self._reset_car(reason=reason)
            self._did_initial_spawn = True
            self.monitor.reset(self.get_clock().now().nanoseconds * 1e-9)

    def _on_odom(self, msg: Odometry):
        if self.builder is None:
            return
        pos = msg.pose.pose.position
        base_pos = torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32)
        frenet_state, boundary_state = self.builder.frenet(base_pos)

        s = float(frenet_state["s"][0])
        track_len = float(frenet_state["L"])
        ey = float(boundary_state["ey"][0])
        w_l = float(boundary_state["w_l_s"][0])
        w_r = float(boundary_state["w_r_s"][0])
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        event = self.monitor.update(s, track_len, ey, w_l, w_r, speed, t)

        metrics = Float32MultiArray()
        metrics.data = [
            float(event.lap_count),
            float(self.monitor.last_lap_time or 0.0),
            float(event.max_progress),
            float(ey),
            1.0 if event.oob else 0.0,
            float(speed),
        ]
        self.metrics_pub.publish(metrics)

        if event.lap_completed:
            self.get_logger().info(
                f"LAP {event.lap_count} completed in {event.lap_time:.2f}s"
            )
        if event.oob:
            self.get_logger().warn(f"OUT OF BOUNDS (ey={ey:.2f})")
        if event.stuck:
            self.get_logger().warn("Car stuck")

        if (event.oob or event.stuck) and self.auto_reset and t >= self._reset_grace_until:
            self._reset_car()
            self.monitor.reset(t)

    def _reset_pose(self) -> tuple[float, float, float]:
        if (
            self.random_reset
            and self._centerline is not None
            and len(self._centerline) >= 2
        ):
            return sample_reset_pose(
                self._centerline,
                self._rng,
                w_left=self._w_left,
                w_right=self._w_right,
                map_assets=self._map_assets,
                spawn_max_dist=self.spawn_max_dist,
            )
        return self.reset_x, self.reset_y, self.reset_yaw

    def _reset_car(self, reason: str = "episode reset"):
        x, y, yaw = self._reset_pose()
        self._reset_grace_until = self.get_clock().now().nanoseconds * 1e-9 + self.reset_grace_s
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = ifc.FRAME_MAP
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.reset_pub.publish(msg)
        self.get_logger().info(
            f"Published /initialpose ({reason}) at ({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
        )
        if self.enable_opponent_spawn and self._centerline is not None:
            self._pending_ego_spawn = (x, y, yaw)
            if self._opp_spawn_timer is None:
                self._opp_spawn_timer = self.create_timer(0.35, self._publish_opponent_spawn)
            else:
                self._opp_spawn_timer.reset()


    def _publish_opponent_spawn(self):
        if self._opp_spawn_timer is not None:
            self._opp_spawn_timer.cancel()
            self._opp_spawn_timer = None
        if not self.enable_opponent_spawn or self._centerline is None:
            return
        if self._pending_ego_spawn is None:
            return
        x, y, _ = self._pending_ego_spawn
        ox, oy, oyaw = opponent_pose_ahead(
            self._centerline, x, y, gap_m=self.opponent_spawn_gap_m
        )
        opp = PoseStamped()
        opp.header.frame_id = ifc.FRAME_MAP
        opp.header.stamp = self.get_clock().now().to_msg()
        opp.pose.position.x = ox
        opp.pose.position.y = oy
        opp.pose.orientation.z = math.sin(oyaw / 2)
        opp.pose.orientation.w = math.cos(oyaw / 2)
        self.opp_reset_pub.publish(opp)
        gap = math.hypot(ox - x, oy - y)
        self.get_logger().info(
            f"Published /goal_pose (opponent) at ({ox:.2f}, {oy:.2f}, yaw={oyaw:.2f}), "
            f"euclidean gap {gap:.2f} m"
        )

def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
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

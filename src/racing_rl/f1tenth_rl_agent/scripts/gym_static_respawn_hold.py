#!/usr/bin/env python3
"""Re-publish ego/opponent spawn poses when static validation detects drift.

Reads live /ego_racecar/odom and /opp_racecar/odom; if speed or gap deviates from
the configured static layout, republishes /initialpose and /goal_pose using the
same centerline-ahead logic as evaluation_node.

Run alongside gym_zero_drive_hold.py during static detector validation.
Does not modify f1tenth_gym.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.eval_logic import opponent_pose_ahead


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _speed(msg: Odometry) -> float:
    return math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)


class StaticRespawnHold(Node):
    def __init__(
        self,
        *,
        track_csv: str,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        gap_m: float,
        max_speed_mps: float,
        max_gap_err_m: float,
        check_hz: float,
    ):
        super().__init__("gym_static_respawn_hold")
        cl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(0, 1))
        self._centerline = cl.astype(np.float64)
        self._ego_x = ego_x
        self._ego_y = ego_y
        self._ego_yaw = ego_yaw
        self._gap_m = gap_m
        self._max_speed = max_speed_mps
        self._max_gap_err = max_gap_err_m
        ox, oy, oyaw = opponent_pose_ahead(self._centerline, ego_x, ego_y, gap_m=gap_m)
        self._opp_x, self._opp_y, self._opp_yaw = ox, oy, oyaw

        self._ego_pub = self.create_publisher(PoseWithCovarianceStamped, ifc.TOPIC_INITIALPOSE, 1)
        self._opp_pub = self.create_publisher(PoseStamped, ifc.TOPIC_GOAL_POSE, 1)
        self._ego: Odometry | None = None
        self._opp: Odometry | None = None
        self._respawns = 0
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_opp, 10)
        period = 1.0 / check_hz if check_hz > 0 else 0.5
        self.create_timer(period, self._check)
        self.get_logger().info(
            f"Static layout ego=({ego_x:.2f},{ego_y:.2f},{ego_yaw:.2f}) "
            f"opp=({ox:.2f},{oy:.2f},{oyaw:.2f}) gap={gap_m:.1f}m"
        )
        self._publish_spawns("initial")

    def _on_ego(self, msg: Odometry) -> None:
        self._ego = msg

    def _on_opp(self, msg: Odometry) -> None:
        self._opp = msg

    def _publish_spawns(self, reason: str) -> None:
        stamp = self.get_clock().now().to_msg()
        ego = PoseWithCovarianceStamped()
        ego.header.frame_id = ifc.FRAME_MAP
        ego.header.stamp = stamp
        ego.pose.pose.position.x = self._ego_x
        ego.pose.pose.position.y = self._ego_y
        ego.pose.pose.orientation.z = math.sin(self._ego_yaw / 2.0)
        ego.pose.pose.orientation.w = math.cos(self._ego_yaw / 2.0)
        self._ego_pub.publish(ego)

        opp = PoseStamped()
        opp.header.frame_id = ifc.FRAME_MAP
        opp.header.stamp = stamp
        opp.pose.position.x = self._opp_x
        opp.pose.position.y = self._opp_y
        opp.pose.orientation.z = math.sin(self._opp_yaw / 2.0)
        opp.pose.orientation.w = math.cos(self._opp_yaw / 2.0)
        self._opp_pub.publish(opp)
        self._respawns += 1
        self.get_logger().info(f"Republished spawns ({reason}), count={self._respawns}")

    def _check(self) -> None:
        if self._ego is None or self._opp is None:
            return
        ex = self._ego.pose.pose.position.x
        ey = self._ego.pose.pose.position.y
        ox = self._opp.pose.pose.position.x
        oy = self._opp.pose.pose.position.y
        gap = math.hypot(ox - ex, oy - ey)
        speed = max(_speed(self._ego), _speed(self._opp))
        pos_err = math.hypot(ex - self._ego_x, ey - self._ego_y) + math.hypot(
            ox - self._opp_x, oy - self._opp_y
        )
        if speed > self._max_speed or abs(gap - self._gap_m) > self._max_gap_err or pos_err > 1.0:
            self._publish_spawns(
                f"drift speed={speed:.2f} gap={gap:.2f} pos_err={pos_err:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-csv", required=True)
    parser.add_argument("--ego-x", type=float, default=0.0)
    parser.add_argument("--ego-y", type=float, default=2.0)
    parser.add_argument("--ego-yaw", type=float, default=-1.465477)
    parser.add_argument("--gap-m", type=float, default=7.0)
    parser.add_argument("--max-speed-mps", type=float, default=0.15)
    parser.add_argument("--max-gap-err-m", type=float, default=1.0)
    parser.add_argument("--check-hz", type=float, default=2.0)
    args = parser.parse_args()

    rclpy.init()
    node = StaticRespawnHold(
        track_csv=args.track_csv,
        ego_x=args.ego_x,
        ego_y=args.ego_y,
        ego_yaw=args.ego_yaw,
        gap_m=args.gap_m,
        max_speed_mps=args.max_speed_mps,
        max_gap_err_m=args.max_gap_err_m,
        check_hz=args.check_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot LiDAR visibility check: does /scan see the opponent at ground-truth pose?

Subscribes to /scan, /ego_racecar/odom, /opp_racecar/odom; prints bearing toward GT
opponent, nearest beam range in that sector, and whether any returns fall in a
car-sized window (~5–9 m ahead). Use after static spawn to confirm the gym renders
the opponent into ego LiDAR before running the C++ detector.

Example (inside gym container, static harness running)::

    python3 gym_scan_opponent_debug.py --duration-s 5
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from f1tenth_rl_agent import interfaces as ifc


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanOpponentDebug(Node):
    def __init__(self) -> None:
        super().__init__("gym_scan_opponent_debug")
        self._scan: LaserScan | None = None
        self._ego: Odometry | None = None
        self._opp: Odometry | None = None
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_opp, 10)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg

    def _on_ego(self, msg: Odometry) -> None:
        self._ego = msg

    def _on_opp(self, msg: Odometry) -> None:
        self._opp = msg

    def analyze(self) -> int:
        if self._scan is None or self._ego is None or self._opp is None:
            print("FAIL: missing /scan, ego odom, or opp odom")
            return 1

        ex = self._ego.pose.pose.position.x
        ey = self._ego.pose.pose.position.y
        ego_yaw = _yaw(self._ego)
        ox = self._opp.pose.pose.position.x
        oy = self._opp.pose.pose.position.y
        dx, dy = ox - ex, oy - ey
        gap = math.hypot(dx, dy)
        rel_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        rel_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
        bearing_world = math.atan2(dy, dx)
        bearing_lidar = bearing_world - ego_yaw
        while bearing_lidar > math.pi:
            bearing_lidar -= 2.0 * math.pi
        while bearing_lidar < -math.pi:
            bearing_lidar += 2.0 * math.pi

        scan = self._scan
        ranges = np.array(scan.ranges, dtype=np.float64)
        invalid = ~np.isfinite(ranges) | (ranges >= scan.range_max - 1e-3)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        half_fov = math.radians(20.0)
        mask = (angles >= bearing_lidar - half_fov) & (angles <= bearing_lidar + half_fov) & ~invalid
        sector = ranges[mask]
        ahead_mask = ~invalid & (angles >= -math.pi / 2) & (angles <= math.pi / 2)
        ahead = ranges[ahead_mask]
        car_window = ahead[(ahead >= 4.0) & (ahead <= 9.0)]

        print("Scan opponent debug (single frame)")
        print(f"  GT gap:     {gap:.2f} m  rel=({rel_x:.2f}, {rel_y:.2f})")
        print(f"  Bearing:    {math.degrees(bearing_lidar):.1f} deg (ego frame)")
        if sector.size == 0:
            print("  Sector:     no valid beams toward opponent")
        else:
            print(
                f"  Sector:     min={sector.min():.2f} m  med={np.median(sector):.2f} m "
                f"({sector.size} beams ±20 deg)"
            )
        print(f"  Ahead 4–9m: {car_window.size} beams in forward hemisphere")
        visible = rel_x > 0.0 and sector.size > 0 and abs(sector.min() - gap) < 2.0
        print("PASS: opponent likely visible in LiDAR" if visible else "WARN: weak / no LiDAR returns on opponent")
        return 0 if visible else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=5.0)
    args = parser.parse_args()

    rclpy.init()
    node = ScanOpponentDebug()
    t_end = time.time() + args.duration_s
    try:
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._scan and node._ego and node._opp:
                break
    finally:
        rc = node.analyze()
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

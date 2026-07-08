#!/usr/bin/env python3
"""Log ego/opponent pose, speed, and along-track gap during gym validation.

Useful for confirming static harness keeps cars visible and roughly 7 m apart
without modifying f1tenth_gym.

Example (inside gym container)::

    python3 gym_pose_monitor.py --duration-s 10 --expected-gap-m 7.0
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from f1tenth_rl_agent import interfaces as ifc


def _yaw(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _speed(msg: Odometry) -> float:
    return math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)


class PoseMonitor(Node):
    def __init__(self, track_csv: str, expected_gap_m: float):
        super().__init__("gym_pose_monitor")
        self.expected_gap_m = expected_gap_m
        cl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(0, 1))
        self._centerline = cl.astype(np.float64)
        self._ego: Odometry | None = None
        self._opp: Odometry | None = None
        self.samples: list[dict] = []
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_opp, 10)

    def _on_ego(self, msg: Odometry) -> None:
        self._ego = msg
        self._maybe_sample()

    def _on_opp(self, msg: Odometry) -> None:
        self._opp = msg
        self._maybe_sample()

    def _maybe_sample(self) -> None:
        if self._ego is None or self._opp is None:
            return
        ex, ey = self._ego.pose.pose.position.x, self._ego.pose.pose.position.y
        ox, oy = self._opp.pose.pose.position.x, self._opp.pose.pose.position.y
        gap = math.hypot(ox - ex, oy - ey)
        ego_yaw = _yaw(self._ego)
        dx, dy = ox - ex, oy - ey
        rel_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        rel_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
        self.samples.append(
            {
                "ego_speed": _speed(self._ego),
                "opp_speed": _speed(self._opp),
                "gap_m": gap,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "gap_err": abs(gap - self.expected_gap_m),
            }
        )

    def report(self) -> int:
        if not self.samples:
            print("FAIL: no paired ego/opp samples (is num_agent=2 and /opp_racecar/odom up?)")
            return 1
        n = len(self.samples)
        ego_spd = np.array([s["ego_speed"] for s in self.samples])
        opp_spd = np.array([s["opp_speed"] for s in self.samples])
        gap = np.array([s["gap_m"] for s in self.samples])
        rel_x = np.array([s["rel_x"] for s in self.samples])
        gap_err = np.array([s["gap_err"] for s in self.samples])
        ahead_frac = float((rel_x > 0.0).mean())
        print(f"Pose monitor: {n} samples")
        print(
            f"  ego speed:  med={np.median(ego_spd):.3f} m/s  p90={np.percentile(ego_spd, 90):.3f}"
        )
        print(
            f"  opp speed:  med={np.median(opp_spd):.3f} m/s  p90={np.percentile(opp_spd, 90):.3f}"
        )
        print(
            f"  gap:        med={np.median(gap):.2f} m  p90={np.percentile(gap, 90):.2f} "
            f"(expected {self.expected_gap_m:.1f})"
        )
        print(f"  rel_x>0:    {100.0 * ahead_frac:.1f}% (opponent ahead in ego frame)")
        print(f"  |gap err|:  med={np.median(gap_err):.2f} m  p90={np.percentile(gap_err, 90):.2f}")
        ok = (
            np.median(ego_spd) < 0.15
            and np.median(opp_spd) < 0.15
            and np.median(gap_err) < 1.5
            and ahead_frac > 0.8
        )
        print("PASS: static pose stable" if ok else "WARN: cars moving or opponent mis-placed")
        return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--expected-gap-m", type=float, default=7.0)
    parser.add_argument(
        "--track-csv",
        default="/sim_ws/src/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PoseMonitor(args.track_csv, args.expected_gap_m)
    t_end = time.time() + args.duration_s
    try:
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        rc = node.report()
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

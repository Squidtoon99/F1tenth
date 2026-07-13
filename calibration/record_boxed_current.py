#!/usr/bin/env python3
"""Record a stepped current/brake bag for longitudinal force calibration (Part 2).

Requires a live ROS 2 graph with the VESC driver (wheels boxed / off the ground).
Publishes stepped ``/commands/motor/current`` then ``/commands/motor/brake`` while
``ros2 bag record`` captures core telemetry.

Example (on the car, after bringup):

  ros2 bag record -o ~/f1tenth_calib_bags/boxed_current_$(date +%H%M%S) \\
    /commands/motor/current /commands/motor/brake /sensors/core /odom \\
    /sensors/imu/raw /ackermann_cmd &
  python3 calibration/record_boxed_current.py --i-max 8 --steps 5 --hold-s 1.5
"""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class BoxedCurrentRecorder(Node):
    def __init__(self) -> None:
        super().__init__("boxed_current_recorder")
        self._cur = self.create_publisher(Float64, "/commands/motor/current", 10)
        self._brk = self.create_publisher(Float64, "/commands/motor/brake", 10)

    def _pub(self, i_drive: float, i_brake: float) -> None:
        c = Float64()
        b = Float64()
        c.data = float(i_drive)
        b.data = float(i_brake)
        self._cur.publish(c)
        self._brk.publish(b)

    def run_steps(self, i_max: float, steps: int, hold_s: float) -> None:
        levels = [i_max * k / steps for k in range(steps + 1)]
        self.get_logger().info(f"drive steps A={levels}")
        for amp in levels:
            self._pub(amp, 0.0)
            time.sleep(hold_s)
        self._pub(0.0, 0.0)
        time.sleep(hold_s)
        self.get_logger().info(f"brake steps A={levels}")
        for amp in levels:
            self._pub(0.0, amp)
            time.sleep(hold_s)
        self._pub(0.0, 0.0)
        self.get_logger().info("done; stop bag recording")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-max", type=float, default=8.0, help="peak amps (bench-safe)")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--hold-s", type=float, default=1.5)
    args = parser.parse_args()

    rclpy.init()
    node = BoxedCurrentRecorder()
    try:
        node.run_steps(args.i_max, args.steps, args.hold_s)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

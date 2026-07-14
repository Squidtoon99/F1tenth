#!/usr/bin/env python3
"""Command safe boxed-wheel current/brake steps while a rosbag records telemetry.

Requires a live ROS 2 graph with the VESC driver (wheels boxed / off the ground).
Exactly one command publisher may be active. Commands are continuously refreshed
and the node aborts on VESC faults, excessive current, voltage, or temperature.

Example (on the car, after bringup):

  ros2 bag record -o ~/f1tenth_calib_bags/boxed_current_$(date +%H%M%S) \\
    /commands/motor/current /commands/motor/brake /sensors/core /odom \\
    /sensors/imu/raw /ackermann_cmd &
  python3 calibration/record_boxed_current.py --drive-levels 1 2 4 6 8 \
    --brake-levels 1 2 4 6 8 --arm
"""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped


class BoxedCurrentRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("boxed_current_recorder")
        self.args = args
        self._cur = self.create_publisher(Float64, "/commands/motor/current", 10)
        self._brk = self.create_publisher(Float64, "/commands/motor/brake", 10)
        self._state = None
        self.create_subscription(VescStateStamped, "/sensors/core", self._state_cb, 10)

    def _state_cb(self, msg: VescStateStamped) -> None:
        self._state = msg.state

    def _pub(self, i_drive: float, i_brake: float) -> None:
        if i_drive != 0.0 and i_brake != 0.0:
            raise RuntimeError("drive and brake current must be mutually exclusive")
        if i_drive > 0.0:
            self._cur.publish(Float64(data=float(i_drive)))
        elif i_brake > 0.0:
            self._brk.publish(Float64(data=float(i_brake)))
        else:
            self._cur.publish(Float64(data=0.0))

    def _check_state(self) -> None:
        if self._state is None:
            raise RuntimeError("no /sensors/core telemetry")
        s = self._state
        values = (
            s.current_motor,
            s.current_input,
            s.speed,
            s.voltage_input,
            s.temp_fet,
            s.temp_motor,
        )
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("non-finite VESC telemetry")
        if s.fault_code != 0:
            raise RuntimeError(f"VESC fault_code={s.fault_code}")
        if abs(s.current_motor) > self.args.max_motor_current:
            raise RuntimeError(f"motor current {s.current_motor:.2f} A exceeds limit")
        if abs(s.current_input) > self.args.max_input_current:
            raise RuntimeError(f"input current {s.current_input:.2f} A exceeds limit")
        if abs(s.speed) > self.args.max_erpm:
            raise RuntimeError(f"wheel speed {s.speed:.0f} ERPM exceeds limit")
        if not self.args.min_voltage <= s.voltage_input <= self.args.max_voltage:
            raise RuntimeError(f"pack voltage {s.voltage_input:.2f} V outside limits")
        for name, value in (("FET", s.temp_fet), ("motor", s.temp_motor)):
            if value > 0.0 and value > self.args.max_temp:
                raise RuntimeError(f"{name} temperature {value:.1f} C exceeds limit")

    def _hold(self, drive: float, brake: float, duration: float, label: str) -> None:
        self.get_logger().info(f"{label}: drive={drive:.2f} A brake={brake:.2f} A")
        end = time.monotonic() + duration
        period = 1.0 / self.args.rate_hz
        while time.monotonic() < end:
            self._pub(drive, brake)
            rclpy.spin_once(self, timeout_sec=0.0)
            self._check_state()
            time.sleep(period)

    def _zero(self, duration: float | None = None) -> None:
        self._hold(0.0, 0.0, self.args.settle_s if duration is None else duration, "zero")

    def run_steps(self) -> None:
        deadline = time.monotonic() + 5.0
        while self._state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._check_state()
        self._zero(2.0)

        for repeat in range(self.args.repeats):
            for amp in self.args.drive_levels:
                self._hold(amp, 0.0, self.args.hold_s, f"drive r{repeat + 1}")
                self._zero()

        if not self.args.skip_brake:
            for repeat in range(self.args.repeats):
                for amp in self.args.brake_levels:
                    self._hold(
                        self.args.spin_current,
                        0.0,
                        self.args.spin_s,
                        f"brake spin-up r{repeat + 1}",
                    )
                    if abs(self._state.speed) < self.args.min_brake_erpm:
                        raise RuntimeError(
                            f"wheel speed {self._state.speed:.0f} ERPM below brake threshold"
                        )
                    self._hold(0.0, amp, self.args.hold_s, f"brake r{repeat + 1}")
                    self._zero()

        self.get_logger().info("boxed-wheel sequence complete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-levels", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--brake-levels", type=float, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--spin-current", type=float, default=2.0)
    parser.add_argument("--spin-s", type=float, default=1.5)
    parser.add_argument("--min-brake-erpm", type=float, default=300.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--max-motor-current", type=float, default=12.0)
    parser.add_argument("--max-input-current", type=float, default=12.0)
    parser.add_argument("--max-erpm", type=float, default=100000.0)
    parser.add_argument("--min-voltage", type=float, default=9.9)
    parser.add_argument("--max-voltage", type=float, default=13.2)
    parser.add_argument("--max-temp", type=float, default=70.0)
    parser.add_argument("--skip-brake", action="store_true")
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args()
    if not args.arm:
        parser.error("--arm is required after completing the physical safety checklist")
    if min(args.drive_levels + args.brake_levels) <= 0.0:
        parser.error("all current levels must be positive")

    rclpy.init()
    node = BoxedCurrentRecorder(args)
    try:
        node.run_steps()
    finally:
        for _ in range(10):
            node._pub(0.0, 0.0)
            time.sleep(0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

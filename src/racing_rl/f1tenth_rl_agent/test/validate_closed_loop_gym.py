#!/usr/bin/env python3
"""Closed-loop gym acceptance validator for the deployable RL stack.

Runs against a live f1tenth_gym bridge + a running agent graph (vehicle or python
stack). It observes the real ROS topics for a fixed window and asserts the health
criteria the release gate requires:

  * /rl/observation: correct length, all finite, published at ~control rate
  * /rl/action:      2 values, within [-clip, clip], published at ~control rate
  * /drive:          acceleration in [-1, 1], |steer| within max_steer, speed==0
  * /ego_racecar/odom: the car actually moves (speed above a floor for long enough)
  * /rl/metrics:     track progress advances and/or a lap completes
  * no observation dimension surprises (matches --expect-obs-dim)

Exit 0 only when every criterion passes for enough samples; non-zero with a concise
report otherwise. No mocks: this needs live topics from the gym sim.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_rl_agent import interfaces as ifc


class ClosedLoopValidator(Node):
    def __init__(
        self,
        *,
        expect_obs_dim: int,
        max_steer: float,
        move_speed_floor: float,
    ):
        super().__init__("closed_loop_validator")
        self.expect_obs_dim = expect_obs_dim
        self.max_steer = max_steer
        self.move_speed_floor = move_speed_floor

        self.obs_count = 0
        self.obs_bad_dim = 0
        self.obs_non_finite = 0
        self.action_count = 0
        self.action_oob = 0
        self.drive_count = 0
        self.drive_oob = 0
        self.odom_count = 0
        self.moving_count = 0
        self.max_progress = 0.0
        self.lap_count = 0.0
        self.saw_metrics = False

        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_OBSERVATION, self._on_obs, 10
        )
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_ACTION, self._on_action, 10
        )
        self.create_subscription(AckermannDriveStamped, ifc.TOPIC_DRIVE, self._on_drive, 10)
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_odom, 10)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_METRICS, self._on_metrics, 10
        )

    def _on_obs(self, msg: Float32MultiArray):
        self.obs_count += 1
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != self.expect_obs_dim:
            self.obs_bad_dim += 1
        if not np.isfinite(data).all():
            self.obs_non_finite += 1

    def _on_action(self, msg: Float32MultiArray):
        self.action_count += 1
        a = np.asarray(msg.data, dtype=np.float64)
        if a.shape[0] != ifc.NUM_ACTIONS or not np.isfinite(a).all():
            self.action_oob += 1
        elif float(np.max(np.abs(a))) > ifc.CLIP_ACTIONS + 1e-5:
            self.action_oob += 1

    def _on_drive(self, msg: AckermannDriveStamped):
        self.drive_count += 1
        speed = msg.drive.speed
        accel = msg.drive.acceleration
        steer = msg.drive.steering_angle
        if not (
            math.isfinite(speed) and math.isfinite(accel) and math.isfinite(steer)
        ):
            self.drive_oob += 1
        elif abs(speed) > 1e-3:
            # Force mode keeps speed unused (ERPM path remapped away).
            self.drive_oob += 1
        elif accel < -1.0 - 1e-3 or accel > 1.0 + 1e-3:
            self.drive_oob += 1
        elif abs(steer) > self.max_steer + 1e-3:
            self.drive_oob += 1

    def _on_odom(self, msg: Odometry):
        self.odom_count += 1
        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        if v > self.move_speed_floor:
            self.moving_count += 1

    def _on_metrics(self, msg: Float32MultiArray):
        if len(msg.data) <= ifc.METRICS_MAX_PROGRESS:
            return
        self.saw_metrics = True
        self.max_progress = max(self.max_progress, float(msg.data[ifc.METRICS_MAX_PROGRESS]))
        self.lap_count = max(self.lap_count, float(msg.data[ifc.METRICS_LAP_COUNT]))

    def report(self, *, min_samples: int, min_progress: float) -> int:
        failures: list[str] = []
        if self.obs_count < min_samples:
            failures.append(f"observations {self.obs_count} < {min_samples}")
        if self.obs_bad_dim:
            failures.append(
                f"{self.obs_bad_dim} observations != {self.expect_obs_dim} dims"
            )
        if self.obs_non_finite:
            failures.append(f"{self.obs_non_finite} non-finite observations")
        if self.action_count < min_samples:
            failures.append(f"actions {self.action_count} < {min_samples}")
        if self.action_oob:
            failures.append(f"{self.action_oob} out-of-bounds/non-finite actions")
        if self.drive_count < min_samples:
            failures.append(f"drive msgs {self.drive_count} < {min_samples}")
        if self.drive_oob:
            failures.append(f"{self.drive_oob} out-of-bounds/non-finite drive commands")
        if self.moving_count < min_samples // 2:
            failures.append(
                f"car barely moved ({self.moving_count} samples > "
                f"{self.move_speed_floor} m/s)"
            )
        if not self.saw_metrics:
            failures.append("no /rl/metrics received (evaluation node down?)")
        elif self.lap_count < 1.0 and self.max_progress < min_progress:
            failures.append(
                f"insufficient progress: max_progress={self.max_progress:.3f} "
                f"(< {min_progress}) and laps={self.lap_count:.0f}"
            )

        print("Closed-loop gym validation:")
        print(f"  obs={self.obs_count} action={self.action_count} drive={self.drive_count}")
        print(f"  moving_samples={self.moving_count} odom={self.odom_count}")
        print(f"  max_progress={self.max_progress:.3f} laps={self.lap_count:.0f}")

        if failures:
            print("FAIL:", "; ".join(failures), file=sys.stderr)
            return 1
        print("PASS")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop gym acceptance gate")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--expect-obs-dim", type=int, default=ifc.NUM_OBS_1V1)
    parser.add_argument("--max-steer", type=float, default=ifc.MAX_STEER)
    parser.add_argument("--move-speed-floor", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-progress", type=float, default=0.95)
    args = parser.parse_args()

    rclpy.init()
    node = ClosedLoopValidator(
        expect_obs_dim=args.expect_obs_dim,
        max_steer=args.max_steer,
        move_speed_floor=args.move_speed_floor,
    )
    try:
        end = node.get_clock().now().nanoseconds + int(args.duration_s * 1e9)
        while node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.report(
            min_samples=args.min_samples, min_progress=args.min_progress
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Closed-loop gym gate for the 1,097-D sensor_racer graph.

Listens to live ROS topics from f1tenth_gym + sensor_policy_sim.launch.py.
Does not require the 392-D evaluation node or /rl/metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from f1tenth_interfaces.msg import ActuatorCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import sensor_interfaces as si

COURTYARD_LENGTH_M = 17.62
TRAINING_LAP_S = 5.3


def _now_s(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def _stamp_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _spin_until(
    node: Node,
    *,
    sim_start_s: float,
    duration_s: float,
    wall_deadline: float,
) -> None:
    # 5 ms waits drain 20 Hz × 5 topics without a 1 kHz busy loop that
    # starves sensor_racer in the same container. Duration is sim /clock
    # so the score does not depend on host wall speed.
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while time.monotonic() < wall_deadline:
            executor.spin_once(timeout_sec=0.005)
            if _now_s(node) - sim_start_s >= duration_s:
                return
    finally:
        executor.remove_node(node)


def _load_centerline(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    pts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("x_m"):
                continue
            parts = line.split(",")
            pts.append((float(parts[0]), float(parts[1])))
    xy = np.asarray(pts, dtype=np.float64)
    if xy.shape[0] < 2:
        raise SystemExit(f"{path}: need at least two centerline rows")
    deltas = np.diff(xy, axis=0)
    seglen = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    return xy, cum, float(cum[-1])


def _project_s(x: float, y: float, xy: np.ndarray, cum: np.ndarray) -> tuple[float, float]:
    dxy = xy - np.array([x, y], dtype=np.float64)
    dist = np.hypot(dxy[:, 0], dxy[:, 1])
    i = int(np.argmin(dist))
    i0 = max(i - 1, 0)
    i1 = min(i + 1, xy.shape[0] - 1)
    best_s = float(cum[i])
    best_d = float(dist[i])
    for a, b in ((i0, i), (i, i1)):
        if a == b:
            continue
        p0 = xy[a]
        p1 = xy[b]
        span = p1 - p0
        denom = float(span @ span)
        if denom < 1e-12:
            continue
        t = float(np.clip(((np.array([x, y]) - p0) @ span) / denom, 0.0, 1.0))
        proj = p0 + t * span
        d = float(np.hypot(proj[0] - x, proj[1] - y))
        if d < best_d:
            best_d = d
            best_s = float(cum[a] + t * (cum[b] - cum[a]))
    return best_s, best_d


class SensorPolicyValidator(Node):
    def __init__(
        self,
        *,
        expect_obs_dim: int,
        max_steer: float,
        move_speed_floor: float,
        centerline: Path | None,
        offtrack_m: float,
    ):
        super().__init__(
            "sensor_policy_closed_loop_validator",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True),
            ],
        )
        self.expect_obs_dim = expect_obs_dim
        self.max_steer = max_steer
        self.move_speed_floor = move_speed_floor
        self.offtrack_m = offtrack_m
        self.xy = self.cum = None
        self.loop_length = COURTYARD_LENGTH_M
        if centerline is not None and centerline.is_file():
            self.xy, self.cum, self.loop_length = _load_centerline(centerline)

        self.obs_count = 0
        self.obs_bad_dim = 0
        self.obs_non_finite = 0
        self.desired_count = 0
        self.desired_stamps: list[float] = []
        self.drive_count = 0
        self.drive_oob = 0
        self.odom_count = 0
        self.moving_count = 0
        self.infer_ms: list[float] = []
        self.deadline_misses = 0
        self.valid_ticks = 0
        self.diag_count = 0
        self.laps = 0
        self.offtrack_samples = 0
        self.crashes = 0
        self.lap_times: list[float] = []
        self.max_speed = 0.0
        self._prev_s: float | None = None
        self._progress_m = 0.0
        self._offtrack = False
        self._lap_t0 = 0.0
        self._start_xy: tuple[float, float] | None = None
        self._end_xy: tuple[float, float] | None = None

        self.create_subscription(
            Float32MultiArray, si.TOPIC_OBSERVATION, self._on_obs, 10
        )
        self.create_subscription(
            ActuatorCommand, si.TOPIC_DESIRED_ACTUATOR, self._on_desired, 10
        )
        self.create_subscription(
            Float32MultiArray, si.TOPIC_DIAGNOSTICS, self._on_diag, 10
        )
        self.create_subscription(AckermannDriveStamped, ifc.TOPIC_DRIVE, self._on_drive, 10)
        self.create_subscription(
            Odometry, ifc.TOPIC_ODOM, self._on_odom, qos_profile_sensor_data
        )

    def _on_obs(self, msg: Float32MultiArray) -> None:
        self.obs_count += 1
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != self.expect_obs_dim:
            self.obs_bad_dim += 1
        if not np.isfinite(data).all():
            self.obs_non_finite += 1

    def _on_desired(self, msg: ActuatorCommand) -> None:
        self.desired_count += 1
        self.desired_stamps.append(_stamp_s(msg.header.stamp))

    def _on_diag(self, msg: Float32MultiArray) -> None:
        self.diag_count += 1
        data = list(msg.data)
        if len(data) <= si.DIAG_VALID_TICK:
            return
        if len(data) > si.DIAG_INFER_MS:
            self.infer_ms.append(float(data[si.DIAG_INFER_MS]))
        if len(data) > si.DIAG_DEADLINE_MISS and float(data[si.DIAG_DEADLINE_MISS]) > 0.5:
            self.deadline_misses += 1
        if float(data[si.DIAG_VALID_TICK]) > 0.5:
            self.valid_ticks += 1

    def _on_drive(self, msg: AckermannDriveStamped) -> None:
        self.drive_count += 1
        speed = msg.drive.speed
        steer = msg.drive.steering_angle
        if not (math.isfinite(speed) and math.isfinite(steer)):
            self.drive_oob += 1
        elif abs(steer) > self.max_steer + 1e-3:
            self.drive_oob += 1

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_count += 1
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)
        self.max_speed = max(self.max_speed, speed)
        if speed > self.move_speed_floor:
            self.moving_count += 1
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        if self._start_xy is None:
            self._start_xy = (x, y)
        self._end_xy = (x, y)
        if self.xy is None:
            return
        s, dist = _project_s(x, y, self.xy, self.cum)
        if dist > self.offtrack_m:
            self.offtrack_samples += 1
            if not self._offtrack:
                self.crashes += 1
                self._offtrack = True
        else:
            self._offtrack = False
        if self._prev_s is not None:
            delta = s - self._prev_s
            if delta < -0.5 * self.loop_length:
                delta += self.loop_length
                now = _now_s(self)
                self.laps += 1
                self.lap_times.append(now - self._lap_t0)
                self._lap_t0 = now
            elif delta > 0.5 * self.loop_length:
                delta -= self.loop_length
            if delta > 0.0:
                self._progress_m += delta
        self._prev_s = s

    def _desired_hz(self) -> float:
        if len(self.desired_stamps) < 4:
            return 0.0
        span = self.desired_stamps[-1] - self.desired_stamps[0]
        if span <= 1e-6:
            return 0.0
        return (len(self.desired_stamps) - 1) / span

    def report(
        self,
        *,
        min_samples: int,
        min_laps: float,
        duration_s: float,
        json_out: str = "",
    ) -> int:
        hz = self._desired_hz()
        mean_infer = float(np.mean(self.infer_ms)) if self.infer_ms else float("nan")
        p95_infer = float(np.percentile(self.infer_ms, 95)) if self.infer_ms else float("nan")
        mean_lap_s = (
            float(np.mean(self.lap_times)) if self.lap_times else float("nan")
        )
        failures: list[str] = []
        if self.desired_count < min_samples:
            failures.append(f"desired cmds {self.desired_count} < {min_samples}")
        if self.obs_count < min_samples:
            failures.append(f"observations {self.obs_count} < {min_samples}")
        if self.obs_bad_dim:
            failures.append(
                f"{self.obs_bad_dim} observations != {self.expect_obs_dim} dims"
            )
        if self.obs_non_finite:
            failures.append(f"{self.obs_non_finite} non-finite observations")
        if self.drive_count < min_samples:
            failures.append(f"drive msgs {self.drive_count} < {min_samples}")
        if self.odom_count < min_samples // 2:
            failures.append(f"odom msgs {self.odom_count} < {min_samples // 2}")
        if self.drive_oob:
            failures.append(f"{self.drive_oob} out-of-bounds drive commands")
        if self.moving_count < min_samples // 4:
            failures.append(
                f"car barely moved ({self.moving_count} samples > "
                f"{self.move_speed_floor} m/s)"
            )
        if abs(hz - si.CONTROL_HZ) > 4.0 and self.desired_count >= min_samples:
            failures.append(f"policy rate {hz:.2f} Hz; expected ~{si.CONTROL_HZ:.0f}")
        if self.laps < min_laps and self._progress_m < 0.8 * self.loop_length:
            failures.append(
                f"insufficient progress: laps={self.laps} progress={self._progress_m:.2f} m"
            )

        print("Closed-loop sensor-policy gym validation:")
        print(f"  desired={self.desired_count} obs={self.obs_count} drive={self.drive_count}")
        print(f"  desired_hz={hz:.2f} valid_ticks={self.valid_ticks} diags={self.diag_count}")
        print(
            f"  infer_ms mean={mean_infer:.2f} p95={p95_infer:.2f} "
            f"deadline_misses={self.deadline_misses}"
        )
        print(
            f"  moving_samples={self.moving_count} odom={self.odom_count} "
            f"max_speed={self.max_speed:.2f} m/s"
        )
        print(
            f"  laps={self.laps} lap_times_s={self.lap_times} "
            f"mean_lap_s={mean_lap_s:.2f} crashes={self.crashes} "
            f"progress_m={self._progress_m:.2f} offtrack={self.offtrack_samples} "
            f"(training courtyard ~{TRAINING_LAP_S:.1f} s / {COURTYARD_LENGTH_M:.2f} m)"
        )
        score = {
            "pass": not failures,
            "laps": self.laps,
            "lap_times_s": [round(t, 3) for t in self.lap_times],
            "mean_lap_s": None if math.isnan(mean_lap_s) else round(mean_lap_s, 3),
            "crashes": self.crashes,
            "progress_m": round(self._progress_m, 3),
            "offtrack_samples": self.offtrack_samples,
            "max_speed_mps": round(self.max_speed, 3),
            "desired_hz": round(hz, 2),
            "desired": self.desired_count,
            "odom": self.odom_count,
            "deadline_misses": self.deadline_misses,
            "infer_ms_mean": None if math.isnan(mean_infer) else round(mean_infer, 2),
            "infer_ms_p95": None if math.isnan(p95_infer) else round(p95_infer, 2),
            "duration_s": duration_s,
            "loop_length_m": round(self.loop_length, 3),
            "training_lap_s": TRAINING_LAP_S,
            "start_xy": (
                None
                if self._start_xy is None
                else [round(self._start_xy[0], 3), round(self._start_xy[1], 3)]
            ),
            "end_xy": (
                None
                if self._end_xy is None
                else [round(self._end_xy[0], 3), round(self._end_xy[1], 3)]
            ),
        }
        print("SCORE_JSON", json.dumps(score))
        if json_out:
            Path(json_out).write_text(json.dumps(score, indent=2) + "\n")

        if failures:
            print("FAIL:", "; ".join(failures), file=sys.stderr)
            return 1
        print("PASS")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensor-policy closed-loop gym gate")
    parser.add_argument("--duration-s", type=float, default=90.0)
    parser.add_argument("--wall-timeout-s", type=float, default=600.0)
    parser.add_argument("--expect-obs-dim", type=int, default=si.NUM_OBS)
    parser.add_argument("--max-steer", type=float, default=si.MAX_STEER_RAD)
    parser.add_argument("--move-speed-floor", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-laps", type=float, default=1.0)
    parser.add_argument("--offtrack-m", type=float, default=1.2)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--centerline",
        default=str(
            Path(__file__).resolve().parents[4]
            / "training"
            / "assets"
            / "courtyard_2_centerline.csv"
        ),
    )
    args = parser.parse_args()
    centerline = Path(args.centerline)

    rclpy.init()
    node = SensorPolicyValidator(
        expect_obs_dim=args.expect_obs_dim,
        max_steer=args.max_steer,
        move_speed_floor=args.move_speed_floor,
        centerline=centerline if centerline.is_file() else None,
        offtrack_m=args.offtrack_m,
    )
    try:
        warmup_deadline = time.monotonic() + max(float(args.warmup_s), 0.0)
        while time.monotonic() < warmup_deadline and node.odom_count == 0:
            rclpy.spin_once(node, timeout_sec=0.01)
        sim_start = _now_s(node)
        node._lap_t0 = sim_start
        wall_deadline = time.monotonic() + max(float(args.wall_timeout_s), 1.0)
        _spin_until(
            node,
            sim_start_s=sim_start,
            duration_s=float(args.duration_s),
            wall_deadline=wall_deadline,
        )
        return node.report(
            min_samples=args.min_samples,
            min_laps=args.min_laps,
            duration_s=_now_s(node) - sim_start,
            json_out=args.json_out,
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

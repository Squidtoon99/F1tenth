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
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.gym_certify import (
    NetLapCounter,
    load_centerline,
    pose_error_xy_yaw,
    project_s,
    spawn_pose_from_centerline,
    summarize_repeat_scores,
    yaw_from_quat,
)

AUSTIN_LENGTH_M = 421.04
TRAINING_LAP_S = 78.3


def _now_s(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def _stamp_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _spin_while(node: Node, predicate, wall_deadline: float) -> bool:
    # 5 ms waits drain 20 Hz × 5 topics without a 1 kHz busy loop that
    # starves sensor_racer in the same container.
    from rclpy.executors import SingleThreadedExecutor

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while time.monotonic() < wall_deadline:
            executor.spin_once(timeout_sec=0.005)
            if predicate():
                return True
        return False
    finally:
        executor.remove_node(node)


def _spin_until(
    node: Node,
    *,
    sim_start_s: float,
    duration_s: float,
    wall_deadline: float,
) -> None:
    _spin_while(
        node,
        lambda: _now_s(node) - sim_start_s >= duration_s,
        wall_deadline,
    )


class SensorPolicyValidator(Node):
    def __init__(
        self,
        *,
        expect_obs_dim: int,
        max_steer: float,
        move_speed_floor: float,
        centerline: Path | None,
        offtrack_m: float,
        spawn_xyyaw: tuple[float, float, float],
        spawn_tol_m: float,
        spawn_yaw_tol_rad: float,
        max_deadline_misses: int,
        min_warmup_scans: int = 5,
        min_warmup_diags: int = 1,
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
        self.spawn_x, self.spawn_y, self.spawn_yaw = spawn_xyyaw
        self.spawn_tol_m = spawn_tol_m
        self.spawn_yaw_tol_rad = spawn_yaw_tol_rad
        self.max_deadline_misses = max_deadline_misses
        self.min_warmup_scans = int(min_warmup_scans)
        self.min_warmup_diags = int(min_warmup_diags)
        self.xy = self.cum = None
        self.loop_length = AUSTIN_LENGTH_M
        if centerline is not None and centerline.is_file():
            self.xy, self.cum, self.loop_length = load_centerline(centerline)
        self._laps = NetLapCounter(self.loop_length)
        self._scoring = False
        self._latest_xyyaw: tuple[float, float, float] | None = None
        self._start_yaw: float | None = None

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
        self.offtrack_samples = 0
        self.max_speed = 0.0
        self._start_xy: tuple[float, float] | None = None
        self._end_xy: tuple[float, float] | None = None
        self._warmup_scans = 0
        self._warmup_diags = 0
        self._warmup_obs = 0

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
        self.create_subscription(
            LaserScan, si.TOPIC_SCAN, self._on_scan, qos_profile_sensor_data
        )
        self._pause_pub = self.create_publisher(Bool, "/pause_sim", 10)
        self._reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

    def reset_run(self) -> None:
        self._scoring = False
        self.obs_count = 0
        self.obs_bad_dim = 0
        self.obs_non_finite = 0
        self.desired_count = 0
        self.desired_stamps.clear()
        self.drive_count = 0
        self.drive_oob = 0
        self.odom_count = 0
        self.moving_count = 0
        self.infer_ms.clear()
        self.deadline_misses = 0
        self.valid_ticks = 0
        self.diag_count = 0
        self.offtrack_samples = 0
        self.max_speed = 0.0
        self._start_xy = None
        self._start_yaw = None
        self._end_xy = None
        self._warmup_scans = 0
        self._warmup_diags = 0
        self._warmup_obs = 0
        self._laps.clear()

    def warmup_ready(self) -> bool:
        return (
            self.at_spawn()
            and self._warmup_scans >= self.min_warmup_scans
            and self._warmup_diags >= self.min_warmup_diags
        )

    def at_spawn(self) -> bool:
        if self._latest_xyyaw is None:
            return False
        xy_err, yaw_err = pose_error_xy_yaw(
            *self._latest_xyyaw, self.spawn_x, self.spawn_y, self.spawn_yaw
        )
        return xy_err <= self.spawn_tol_m and yaw_err <= self.spawn_yaw_tol_rad

    def set_paused(self, paused: bool) -> None:
        msg = Bool()
        msg.data = bool(paused)
        self._pause_pub.publish(msg)

    def publish_spawn(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(self.spawn_x)
        msg.pose.pose.position.y = float(self.spawn_y)
        msg.pose.pose.orientation.z = math.sin(self.spawn_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.spawn_yaw / 2.0)
        self._reset_pub.publish(msg)

    def _on_scan(self, _msg: LaserScan) -> None:
        self._warmup_scans += 1

    def _on_obs(self, msg: Float32MultiArray) -> None:
        self._warmup_obs += 1
        if not self._scoring:
            return
        self.obs_count += 1
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape[0] != self.expect_obs_dim:
            self.obs_bad_dim += 1
        if not np.isfinite(data).all():
            self.obs_non_finite += 1

    def _on_desired(self, msg: ActuatorCommand) -> None:
        if not self._scoring:
            return
        self.desired_count += 1
        self.desired_stamps.append(_stamp_s(msg.header.stamp))

    def _on_diag(self, msg: Float32MultiArray) -> None:
        self._warmup_diags += 1
        if not self._scoring:
            return
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
        if not self._scoring:
            return
        self.drive_count += 1
        speed = msg.drive.speed
        steer = msg.drive.steering_angle
        if not (math.isfinite(speed) and math.isfinite(steer)):
            self.drive_oob += 1
        elif abs(steer) > self.max_steer + 1e-3:
            self.drive_oob += 1

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self._latest_xyyaw = (x, y, yaw)
        if not self._scoring:
            return
        self.odom_count += 1
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        speed = math.hypot(vx, vy)
        self.max_speed = max(self.max_speed, speed)
        if speed > self.move_speed_floor:
            self.moving_count += 1
        if self._start_xy is None:
            self._start_xy = (x, y)
            self._start_yaw = yaw
        self._end_xy = (x, y)
        if self.xy is None:
            return
        s, dist = project_s(x, y, self.xy, self.cum)
        offtrack = dist > self.offtrack_m
        if offtrack:
            self.offtrack_samples += 1
        self._laps.update(s, offtrack, _now_s(self))

    def _desired_hz(self) -> float:
        if len(self.desired_stamps) < 4:
            return 0.0
        span = self.desired_stamps[-1] - self.desired_stamps[0]
        if span <= 1e-6:
            return 0.0
        return (len(self.desired_stamps) - 1) / span

    def score_dict(self, *, duration_s: float) -> dict:
        hz = self._desired_hz()
        mean_infer = float(np.mean(self.infer_ms)) if self.infer_ms else float("nan")
        p95_infer = float(np.percentile(self.infer_ms, 95)) if self.infer_ms else float("nan")
        laps = self._laps.laps
        lap_times = self._laps.lap_times
        crashes = self._laps.crashes
        progress_m = laps * self.loop_length + self._laps.progress
        mean_lap_s = float(np.mean(lap_times)) if lap_times else float("nan")
        return {
            "laps": laps,
            "lap_times_s": [round(t, 3) for t in lap_times],
            "mean_lap_s": None if math.isnan(mean_lap_s) else round(mean_lap_s, 3),
            "crashes": crashes,
            "progress_m": round(progress_m, 3),
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
            "start_yaw": (
                None if self._start_yaw is None else round(self._start_yaw, 4)
            ),
            "end_xy": (
                None
                if self._end_xy is None
                else [round(self._end_xy[0], 3), round(self._end_xy[1], 3)]
            ),
        }

    def report(
        self,
        *,
        min_samples: int,
        min_laps: float,
        duration_s: float,
        json_out: str = "",
        print_score: bool = True,
    ) -> tuple[int, dict]:
        hz = self._desired_hz()
        mean_infer = float(np.mean(self.infer_ms)) if self.infer_ms else float("nan")
        p95_infer = float(np.percentile(self.infer_ms, 95)) if self.infer_ms else float("nan")
        laps = self._laps.laps
        lap_times = self._laps.lap_times
        crashes = self._laps.crashes
        progress_m = laps * self.loop_length + self._laps.progress
        mean_lap_s = float(np.mean(lap_times)) if lap_times else float("nan")
        failures: list[str] = []
        if self._start_xy is None or self._start_yaw is None:
            failures.append("missing start pose")
        else:
            xy_err, yaw_err = pose_error_xy_yaw(
                self._start_xy[0],
                self._start_xy[1],
                self._start_yaw,
                self.spawn_x,
                self.spawn_y,
                self.spawn_yaw,
            )
            if xy_err > self.spawn_tol_m or yaw_err > self.spawn_yaw_tol_rad:
                failures.append(
                    f"start pose ({self._start_xy[0]:.3f}, {self._start_xy[1]:.3f}, "
                    f"{self._start_yaw:.3f}) off spawn by {xy_err:.3f} m / "
                    f"{yaw_err:.3f} rad"
                )
        if self.deadline_misses > self.max_deadline_misses:
            failures.append(
                f"deadline_misses {self.deadline_misses} > {self.max_deadline_misses}"
            )
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
        if laps < min_laps and progress_m < 0.8 * self.loop_length:
            failures.append(
                f"insufficient progress: laps={laps} progress={progress_m:.2f} m"
            )

        if print_score:
            print("Closed-loop sensor-policy gym validation:")
            print(
                f"  desired={self.desired_count} obs={self.obs_count} "
                f"drive={self.drive_count}"
            )
            print(
                f"  desired_hz={hz:.2f} valid_ticks={self.valid_ticks} "
                f"diags={self.diag_count}"
            )
            print(
                f"  infer_ms mean={mean_infer:.2f} p95={p95_infer:.2f} "
                f"deadline_misses={self.deadline_misses}"
            )
            print(
                f"  moving_samples={self.moving_count} odom={self.odom_count} "
                f"max_speed={self.max_speed:.2f} m/s"
            )
            print(
                f"  laps={laps} lap_times_s={lap_times} "
                f"mean_lap_s={mean_lap_s:.2f} crashes={crashes} "
                f"progress_m={progress_m:.2f} offtrack={self.offtrack_samples} "
                f"(training Austin ~{TRAINING_LAP_S:.1f} s / {AUSTIN_LENGTH_M:.2f} m)"
            )
        score = self.score_dict(duration_s=duration_s)
        score["pass"] = not failures
        score["failures"] = failures
        if print_score:
            print("SCORE_JSON", json.dumps(score))
            if json_out:
                Path(json_out).write_text(json.dumps(score, indent=2) + "\n")
            if failures:
                print("FAIL:", "; ".join(failures), file=sys.stderr)
            else:
                print("PASS")
        return (0 if not failures else 1, score)


def _run_one(
    node: SensorPolicyValidator,
    *,
    duration_s: float,
    spawn_wait_s: float,
    wall_deadline: float,
) -> float:
    node.reset_run()
    node.set_paused(True)
    node.publish_spawn()
    spawn_deadline = min(wall_deadline, time.monotonic() + spawn_wait_s)
    if not _spin_while(node, node.at_spawn, spawn_deadline):
        raise TimeoutError("timed out waiting for exact spawn pose")
    warm_deadline = min(wall_deadline, time.monotonic() + spawn_wait_s)
    if not _spin_while(node, node.warmup_ready, warm_deadline):
        raise TimeoutError("timed out waiting for warmup scans")
    node.reset_run()
    node._scoring = True
    node.set_paused(False)
    start_deadline = min(wall_deadline, time.monotonic() + spawn_wait_s)
    if not _spin_while(node, lambda: node._start_xy is not None, start_deadline):
        raise TimeoutError("timed out waiting for scored start pose")
    sim_start = _now_s(node)
    node._laps.lap_t0 = sim_start
    _spin_until(
        node,
        sim_start_s=sim_start,
        duration_s=duration_s,
        wall_deadline=wall_deadline,
    )
    node.set_paused(True)
    return _now_s(node) - sim_start


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensor-policy closed-loop gym gate")
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--wall-timeout-s", type=float, default=900.0)
    parser.add_argument("--expect-obs-dim", type=int, default=si.NUM_OBS)
    parser.add_argument("--max-steer", type=float, default=si.MAX_STEER_RAD)
    parser.add_argument("--move-speed-floor", type=float, default=0.3)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-laps", type=float, default=1.0)
    parser.add_argument("--offtrack-m", type=float, default=1.2)
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--spawn-tol-m", type=float, default=0.15)
    parser.add_argument("--spawn-yaw-tol-rad", type=float, default=0.15)
    parser.add_argument("--spawn-wait-s", type=float, default=60.0)
    parser.add_argument("--warmup-scans", type=int, default=5)
    parser.add_argument("--max-deadline-misses", type=int, default=0)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--centerline",
        default=str(
            Path(__file__).resolve().parents[4]
            / "training"
            / "assets"
            / "Austin_centerline.csv"
        ),
    )
    args = parser.parse_args()
    centerline = Path(args.centerline)
    if int(args.repeats) < 1:
        print("FAIL: --repeats must be >= 1", file=sys.stderr)
        return 2
    spawn = (0.0, 0.0, -0.6524)
    if centerline.is_file():
        spawn = spawn_pose_from_centerline(centerline)

    rclpy.init()
    node = SensorPolicyValidator(
        expect_obs_dim=args.expect_obs_dim,
        max_steer=args.max_steer,
        move_speed_floor=args.move_speed_floor,
        centerline=centerline if centerline.is_file() else None,
        offtrack_m=args.offtrack_m,
        spawn_xyyaw=spawn,
        spawn_tol_m=float(args.spawn_tol_m),
        spawn_yaw_tol_rad=float(args.spawn_yaw_tol_rad),
        max_deadline_misses=int(args.max_deadline_misses),
        min_warmup_scans=int(args.warmup_scans),
    )
    try:
        warmup_deadline = time.monotonic() + max(float(args.warmup_s), 0.0)
        while time.monotonic() < warmup_deadline and node._latest_xyyaw is None:
            rclpy.spin_once(node, timeout_sec=0.01)
        wall_deadline = time.monotonic() + max(float(args.wall_timeout_s), 1.0)
        scores = []
        rc = 0
        for i in range(int(args.repeats)):
            print(f"==> certify repeat {i + 1}/{int(args.repeats)}")
            try:
                duration = _run_one(
                    node,
                    duration_s=float(args.duration_s),
                    spawn_wait_s=float(args.spawn_wait_s),
                    wall_deadline=wall_deadline,
                )
            except TimeoutError as exc:
                print(f"FAIL: {exc}", file=sys.stderr)
                return 1
            run_rc, score = node.report(
                min_samples=args.min_samples,
                min_laps=args.min_laps,
                duration_s=duration,
                print_score=True,
            )
            scores.append(score)
            rc = rc or run_rc
        summary = summarize_repeat_scores(scores)
        print("REPEAT_JSON", json.dumps(summary))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        if not summary["pass"]:
            print("FAIL: one or more repeats failed", file=sys.stderr)
            return 1
        return rc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gym validation: compare LiDAR opponent detector output to ground truth.

Run inside the f1tenth_gym_ros container with the gym bridge, evaluation (spawn),
and opponent_detector already up. Subscribes to /rl/opponent/odom (detector),
/opp_racecar/odom (GT), and /ego_racecar/odom (ego); time-aligns by stamp
(nearest-neighbor within 50 ms) and reports recall, position error, false-positive
rate, and presence duty cycle.

Exit 0 when scenario thresholds pass; non-zero otherwise. No mocks — requires live
ROS topics from the gym sim.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.obs_core import ObservationBuilder, build_boundary_state, frenet_projection

TOPIC_DETECTOR = "/rl/opponent/odom"
LIDAR_FOV_HALF_RAD = math.radians(135.0)
DEFAULT_CLUSTER_GAP_M = 0.30


def _stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _pos_xy(msg: Odometry) -> tuple[float, float]:
    return float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)


def _vel_xy(msg: Odometry) -> tuple[float, float]:
    return float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)


def _world_to_ego_rel(
    ego_x: float, ego_y: float, ego_yaw: float, wx: float, wy: float
) -> tuple[float, float]:
    dx = wx - ego_x
    dy = wy - ego_y
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)
    rel_x = cos_y * dx + sin_y * dy
    rel_y = -sin_y * dx + cos_y * dy
    return rel_x, rel_y


@dataclass
class FrameSample:
    det_present: bool
    gt_visible: bool
    map_err_m: float | None = None
    ego_rel_err_m: float | None = None


@dataclass
class ScanClusterDebug:
    centroid_x: float
    centroid_y: float
    extent_m: float
    beam_count: int
    ey: float


class _MsgBuffer:
    def __init__(self, max_age_s: float = 3.0):
        self._max_age_ns = int(max_age_s * 1e9)
        self._items: deque[tuple[int, object]] = deque()

    def add(self, msg) -> None:
        ts = _stamp_ns(msg)
        self._items.append((ts, msg))
        cutoff = ts - self._max_age_ns
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def nearest(self, ref_ns: int, tol_ns: int):
        best_msg = None
        best_dt = tol_ns + 1
        for ts, msg in self._items:
            dt = abs(ts - ref_ns)
            if dt <= tol_ns and dt < best_dt:
                best_msg = msg
                best_dt = dt
        return best_msg


class OpponentDetectorValidator(Node):
    def __init__(
        self,
        *,
        track_csv: str,
        scenario: str,
        duration_s: float,
        sync_tol_ms: float,
        max_range_m: float,
        min_samples: int,
        max_fp_rate: float,
        max_presence_duty: float,
        min_recall: float,
        max_mean_err_m: float,
        max_p95_err_m: float,
        debug_fp: bool,
    ):
        super().__init__("opponent_detector_validator")
        self.scenario = scenario
        self.duration_s = duration_s
        self.sync_tol_ns = int(sync_tol_ms * 1e6)
        self.max_range_m = max_range_m
        self.min_samples = min_samples
        self.max_fp_rate = max_fp_rate
        self.max_presence_duty = max_presence_duty
        self.min_recall = min_recall
        self.max_mean_err_m = max_mean_err_m
        self.max_p95_err_m = max_p95_err_m
        self.debug_fp = debug_fp

        cl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(0, 1))
        wl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(2,))
        wr = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(3,))
        obs_cfg = ifc.default_obs_cfg(enable_opponent_obs=True)
        self.builder = ObservationBuilder(cl, wl, wr, obs_cfg=obs_cfg)

        self._det_buf = _MsgBuffer()
        self._gt_buf = _MsgBuffer()
        self._scan_buf = _MsgBuffer()
        self.samples: list[FrameSample] = []
        self._fp_debug_logs: list[str] = []

        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, TOPIC_DETECTOR, self._on_det, 10)
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_gt, 10)
        if self.debug_fp:
            self.create_subscription(LaserScan, "/scan", self._on_scan, 10)

    def _on_det(self, msg: Odometry) -> None:
        self._det_buf.add(msg)

    def _on_gt(self, msg: Odometry) -> None:
        self._gt_buf.add(msg)

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_buf.add(msg)

    def _on_ego(self, msg: Odometry) -> None:
        ref_ns = _stamp_ns(msg)
        det = self._det_buf.nearest(ref_ns, self.sync_tol_ns)
        gt = self._gt_buf.nearest(ref_ns, self.sync_tol_ns)

        gt_visible = self._gt_visible(msg, gt)
        det_present = det is not None

        sample = FrameSample(det_present=det_present, gt_visible=gt_visible)
        if det_present and gt_visible and gt is not None:
            ex, ey = _pos_xy(msg)
            gx, gy = _pos_xy(gt)
            dx, dy = _pos_xy(det)
            sample.map_err_m = math.hypot(dx - gx, dy - gy)
            sample.ego_rel_err_m = self._ego_rel_pos_error(msg, gt, det)

        self.samples.append(sample)

        if (
            self.debug_fp
            and self.scenario == "solo"
            and det_present
            and not gt_visible
            and det is not None
        ):
            dbg = self._debug_fp_cluster(ref_ns, msg, det)
            if dbg is not None:
                line = (
                    f"FP debug: centroid=({dbg.centroid_x:.3f},{dbg.centroid_y:.3f}) "
                    f"extent={dbg.extent_m:.3f}m beams={dbg.beam_count} ey={dbg.ey:.3f}"
                )
                self._fp_debug_logs.append(line)
                print(line, file=sys.stderr)

    def _gt_visible(self, ego: Odometry, gt: Odometry | None) -> bool:
        if self.scenario == "solo":
            return False
        if gt is None:
            return False

        ex, ey = _pos_xy(ego)
        gx, gy = _pos_xy(gt)
        map_dist = math.hypot(gx - ex, gy - ey)
        if map_dist >= self.max_range_m:
            return False

        ego_yaw = _yaw_from_odom(ego)
        rel_x, rel_y = _world_to_ego_rel(ex, ey, ego_yaw, gx, gy)
        ego_range = math.hypot(rel_x, rel_y)
        if ego_range >= self.max_range_m:
            return False

        bearing = math.atan2(rel_y, rel_x)
        if abs(bearing) > LIDAR_FOV_HALF_RAD:
            return False
        return True

    def _ego_rel_pos_error(self, ego: Odometry, gt: Odometry, det: Odometry) -> float:
        ego_pos = torch.tensor(
            [[ego.pose.pose.position.x, ego.pose.pose.position.y, 0.0]],
            dtype=torch.float32,
        )
        gt_pos = torch.tensor(
            [[gt.pose.pose.position.x, gt.pose.pose.position.y, 0.0]],
            dtype=torch.float32,
        )
        det_pos = torch.tensor(
            [[det.pose.pose.position.x, det.pose.pose.position.y, 0.0]],
            dtype=torch.float32,
        )
        ego_yaw = torch.tensor([_yaw_from_odom(ego)], dtype=torch.float32)
        evx, evy = _vel_xy(ego)
        gvx, gvy = _vel_xy(gt)
        dvx, dvy = _vel_xy(det)
        ego_vel = torch.tensor([[evx, evy, 0.0]], dtype=torch.float32)
        gt_vel = torch.tensor([[gvx, gvy, 0.0]], dtype=torch.float32)
        det_vel = torch.tensor([[dvx, dvy, 0.0]], dtype=torch.float32)

        gt_block = (
            self.builder.build_opponent_block(
                ego_pos, ego_yaw, ego_vel, gt_pos, gt_vel, present=None
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
        det_block = (
            self.builder.build_opponent_block(
                ego_pos, ego_yaw, ego_vel, det_pos, det_vel, present=None
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
        return float(math.hypot(det_block[0] - gt_block[0], det_block[1] - gt_block[1]))

    def _debug_fp_cluster(
        self, ref_ns: int, ego: Odometry, det: Odometry
    ) -> ScanClusterDebug | None:
        scan = self._scan_buf.nearest(ref_ns, self.sync_tol_ns)
        if scan is None:
            return None

        ex, ey = _pos_xy(ego)
        ego_yaw = _yaw_from_odom(ego)
        det_x, det_y = _pos_xy(det)

        points: list[tuple[float, float]] = []
        n = len(scan.ranges)
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            lx = r * math.cos(ang)
            ly = r * math.sin(ang)
            wx = ex + math.cos(ego_yaw) * lx - math.sin(ego_yaw) * ly
            wy = ey + math.sin(ego_yaw) * lx + math.cos(ego_yaw) * ly
            points.append((wx, wy))

        if not points:
            return None

        clusters = _cluster_points(points, DEFAULT_CLUSTER_GAP_M)
        if not clusters:
            return None

        best = min(
            clusters,
            key=lambda c: math.hypot(_cluster_centroid(c)[0] - det_x, _cluster_centroid(c)[1] - det_y),
        )
        cx, cy = _cluster_centroid(best)
        extent = _cluster_extent(best)
        opp_pos = torch.tensor([[cx, cy, 0.0]], dtype=torch.float32)
        frenet = frenet_projection(opp_pos, self.builder.geom, self.builder.device)
        boundary = build_boundary_state(
            frenet, self.builder.w_tr_left, self.builder.w_tr_right
        )
        ey_val = float(boundary["ey"][0])
        return ScanClusterDebug(
            centroid_x=cx,
            centroid_y=cy,
            extent_m=extent,
            beam_count=len(best),
            ey=ey_val,
        )

    def report(self) -> int:
        if len(self.samples) < self.min_samples:
            print(
                f"FAIL: only {len(self.samples)} aligned samples "
                f"(need >= {self.min_samples})",
                file=sys.stderr,
            )
            return 1

        n = len(self.samples)
        presence_duty = sum(1 for s in self.samples if s.det_present) / n

        gt_visible_frames = [s for s in self.samples if s.gt_visible]
        n_visible = len(gt_visible_frames)
        if n_visible > 0:
            recall = sum(1 for s in gt_visible_frames if s.det_present) / n_visible
        else:
            recall = float("nan")

        not_visible_frames = [s for s in self.samples if not s.gt_visible]
        n_not_visible = len(not_visible_frames)
        if n_not_visible > 0:
            fp_rate = sum(1 for s in not_visible_frames if s.det_present) / n_not_visible
        else:
            fp_rate = float("nan")

        map_errs = [s.map_err_m for s in self.samples if s.map_err_m is not None]
        ego_errs = [s.ego_rel_err_m for s in self.samples if s.ego_rel_err_m is not None]

        def _stats(vals: list[float]) -> tuple[float, float, float]:
            if not vals:
                return float("nan"), float("nan"), float("nan")
            arr = np.asarray(vals, dtype=np.float64)
            return float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max())

        map_mean, map_p95, map_max = _stats(map_errs)
        ego_mean, ego_p95, ego_max = _stats(ego_errs)

        print(f"Opponent detector validation ({self.scenario}, n={n}):")
        print(f"  presence_duty: {presence_duty * 100:.1f}%")
        print(f"  recall:        {recall * 100:.1f}%" if not math.isnan(recall) else "  recall:        n/a")
        print(f"  fp_rate:       {fp_rate * 100:.1f}%" if not math.isnan(fp_rate) else "  fp_rate:       n/a")
        print(
            f"  map_err_m:     mean={map_mean:.3f} p95={map_p95:.3f} max={map_max:.3f} "
            f"(n={len(map_errs)})"
        )
        print(
            f"  ego_rel_err_m: mean={ego_mean:.3f} p95={ego_p95:.3f} max={ego_max:.3f} "
            f"(n={len(ego_errs)})"
        )

        failures: list[str] = []
        if self.scenario == "solo":
            if not math.isnan(fp_rate) and fp_rate > self.max_fp_rate:
                failures.append(
                    f"fp_rate {fp_rate * 100:.1f}% > {self.max_fp_rate * 100:.1f}%"
                )
            if presence_duty > self.max_presence_duty:
                failures.append(
                    f"presence_duty {presence_duty * 100:.1f}% > "
                    f"{self.max_presence_duty * 100:.1f}%"
                )
        elif self.scenario == "static":
            if math.isnan(recall) or recall < self.min_recall:
                recall_pct = recall * 100 if not math.isnan(recall) else 0.0
                failures.append(f"recall {recall_pct:.1f}% < {self.min_recall * 100:.1f}%")
            if not math.isnan(map_mean) and map_mean > self.max_mean_err_m:
                failures.append(f"mean map_err {map_mean:.3f}m > {self.max_mean_err_m:.3f}m")
            if not math.isnan(map_p95) and map_p95 > self.max_p95_err_m:
                failures.append(f"p95 map_err {map_p95:.3f}m > {self.max_p95_err_m:.3f}m")
        else:
            failures.append(f"unknown scenario '{self.scenario}'")

        if failures:
            print("FAIL:", "; ".join(failures), file=sys.stderr)
            return 1

        print(f"PASS: scenario={self.scenario}, samples={n}")
        return 0


def _cluster_centroid(cluster: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _cluster_extent(cluster: list[tuple[float, float]]) -> float:
    max_d = 0.0
    for i, a in enumerate(cluster):
        for b in cluster[i + 1 :]:
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            max_d = max(max_d, d)
    return max_d


def _cluster_points(
    points: list[tuple[float, float]], gap_m: float
) -> list[list[tuple[float, float]]]:
    if not points:
        return []
    clusters: list[list[tuple[float, float]]] = []
    current = [points[0]]
    for prev, pt in zip(points, points[1:]):
        if math.hypot(pt[0] - prev[0], pt[1] - prev[1]) <= gap_m:
            current.append(pt)
        else:
            clusters.append(current)
            current = [pt]
    clusters.append(current)
    return clusters


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate LiDAR opponent detector vs gym ground truth"
    )
    parser.add_argument(
        "--scenario",
        choices=("solo", "static"),
        required=True,
        help="solo: no opponent (FP baseline); static: stationary opponent 7 m ahead",
    )
    parser.add_argument(
        "--track-csv",
        default="/sim_ws/src/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--max-range-m", type=float, default=10.0)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--max-fp-rate", type=float, default=0.05)
    parser.add_argument("--max-presence-duty", type=float, default=0.05)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-mean-err-m", type=float, default=0.7)
    parser.add_argument("--max-p95-err-m", type=float, default=1.0)
    parser.add_argument(
        "--debug-fp",
        action="store_true",
        help="On solo FPs, log scan cluster centroid/extent/beam count/ey",
    )
    args = parser.parse_args()

    rclpy.init()
    node = OpponentDetectorValidator(
        track_csv=args.track_csv,
        scenario=args.scenario,
        duration_s=args.duration_s,
        sync_tol_ms=args.sync_tolerance_ms,
        max_range_m=args.max_range_m,
        min_samples=args.min_samples,
        max_fp_rate=args.max_fp_rate,
        max_presence_duty=args.max_presence_duty,
        min_recall=args.min_recall,
        max_mean_err_m=args.max_mean_err_m,
        max_p95_err_m=args.max_p95_err_m,
        debug_fp=args.debug_fp,
    )
    try:
        t_end = time.time() + args.duration_s
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

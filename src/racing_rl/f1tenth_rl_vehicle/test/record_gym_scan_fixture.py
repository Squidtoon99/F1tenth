#!/usr/bin/env python3
"""Record live gym LiDAR frames for offline C++ OpponentDetector replay tests.

Run inside the f1tenth_gym_ros container while the gym bridge is publishing /scan,
/ego_racecar/odom, and (for 1v1) /opp_racecar/odom. Writes a compact JSON fixture
that test_rl_obs_core.cpp can load in a ``ReplayGymScanFixture`` test.

On-disk format (JSON, UTF-8)
----------------------------
The schema is intentionally *flat* so the minimal hand-rolled JSON reader in
``test_rl_obs_core.cpp`` (``loadGymScanFixture``) can parse it. That reader only
understands strings, numeric arrays, and numbers - it cannot skip nested objects,
``null`` literals, or booleans. Keep every per-frame value a number or a numeric
array.

Top-level object::

    {
      "format_version": 1,                  // number (ignored by C++)
      "description": "...",                 // string (ignored by C++)
      "track_csv": "<path used when recording>",   // string (read by C++)
      "frames": [
        {
          "stamp_sec": <int>,               // number (ignored)
          "stamp_nanosec": <int>,           // number (ignored)
          "ranges": [<float>, ...],         // length == num_beams; invalid -> INVALID_RANGE
          "angle_min": <float rad>,
          "angle_increment": <float rad>,
          "ego_x": <float>, "ego_y": <float>, "ego_yaw": <float rad>,
          "opp_x": <float>, "opp_y": <float>   // (0, 0) when no opponent present
        },
        ...
      ]
    }

C++ consumption contract (see ``rangesToMapBeams`` / ``ReplayGymScanFixture``)
------------------------
* Iterate ``frames`` in order; for each frame call ``OpponentDetector::update()``
  with beams built from ``ranges`` and ego pose ``(ego_x, ego_y)``.
* A range is treated as a *no-return* beam when it is non-finite or outside the
  scan's ``[range_min, range_max]``. Invalid beams are written as ``INVALID_RANGE``
  (a large sentinel, NOT ``null``) so the numeric parser accepts them; the C++
  side rejects anything ``> range_max``.
* Opponent presence is inferred from ``opp_x``/``opp_y``: ``(0, 0)`` means
  solo / absent (GT for negative tests). There is no ``present`` boolean because
  the C++ frame parser cannot read booleans.
* ``ranges`` are raw ``sensor_msgs/LaserScan.ranges`` (metres, ego laser frame).
  The detector node transforms them to map frame using ego pose; replay tests
  mirror ``opponent_detector_node.cpp`` beam projection (zero lidar offset).
* ``angle_min``/``angle_increment`` are repeated per frame (constant) so each
  frame is self-contained for the flat reader.

Example::

    python record_gym_scan_fixture.py \\
        --output /tmp/gym_scan_static.json \\
        --num-frames 120 --duration-s 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from f1tenth_rl_agent import interfaces as ifc


def _stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def _yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# Sentinel for an invalid / no-return beam. The C++ flat JSON reader cannot parse
# ``null``; rangesToMapBeams treats anything > range_max as invalid, so a large
# finite value reproduces a "no return" beam.
INVALID_RANGE = 1.0e9


def _range_to_json(r: float):
    if not math.isfinite(r):
        return INVALID_RANGE
    return float(r)


@dataclass
class RecordedFrame:
    stamp_sec: int
    stamp_nanosec: int
    ranges: list
    ego_x: float
    ego_y: float
    ego_yaw: float
    opp_x: float
    opp_y: float
    opp_present: bool


class ScanFixtureRecorder(Node):
    def __init__(
        self,
        *,
        num_frames: int,
        sync_tol_ms: float,
        track_csv: str,
    ):
        super().__init__("scan_fixture_recorder")
        self.num_frames = num_frames
        self.sync_tol_ns = int(sync_tol_ms * 1e6)
        self.track_csv = track_csv
        self.frames: list[RecordedFrame] = []

        self._scan_meta: dict | None = None
        self._ego: Odometry | None = None
        self._gt: Odometry | None = None
        self._have_gt = False

        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, ifc.TOPIC_OPP_RACE_ODOM, self._on_gt, 10)

    @property
    def done(self) -> bool:
        return len(self.frames) >= self.num_frames

    def _on_ego(self, msg: Odometry) -> None:
        self._ego = msg

    def _on_gt(self, msg: Odometry) -> None:
        self._gt = msg
        self._have_gt = True

    def _on_scan(self, msg: LaserScan) -> None:
        if self.done or self._ego is None:
            return

        scan_ns = _stamp_ns(msg)
        ego_ns = _stamp_ns(self._ego)
        if abs(scan_ns - ego_ns) > self.sync_tol_ns:
            return

        if self._scan_meta is None:
            self._scan_meta = {
                "angle_min": float(msg.angle_min),
                "angle_increment": float(msg.angle_increment),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "num_beams": len(msg.ranges),
            }

        opp_present = False
        opp_x = 0.0
        opp_y = 0.0
        if self._have_gt and self._gt is not None:
            gt_ns = _stamp_ns(self._gt)
            if abs(scan_ns - gt_ns) <= self.sync_tol_ns:
                opp_present = True
                opp_x = float(self._gt.pose.pose.position.x)
                opp_y = float(self._gt.pose.pose.position.y)

        ego = self._ego
        self.frames.append(
            RecordedFrame(
                stamp_sec=int(msg.header.stamp.sec),
                stamp_nanosec=int(msg.header.stamp.nanosec),
                ranges=[_range_to_json(r) for r in msg.ranges],
                ego_x=float(ego.pose.pose.position.x),
                ego_y=float(ego.pose.pose.position.y),
                ego_yaw=_yaw_from_odom(ego),
                opp_x=opp_x,
                opp_y=opp_y,
                opp_present=opp_present,
            )
        )

    def to_document(self) -> dict:
        if self._scan_meta is None:
            raise RuntimeError("no scan messages recorded")
        angle_min = self._scan_meta["angle_min"]
        angle_increment = self._scan_meta["angle_increment"]
        # Flat per-frame schema: every value is a number or numeric array so the
        # minimal C++ reader (loadGymScanFixture) can parse it. angle_min /
        # angle_increment are repeated per frame (constant) for self-containment.
        # Opponent presence is encoded as opp_x/opp_y == (0, 0) when absent.
        return {
            "format_version": 1,
            "description": "Gym LiDAR scan fixture for OpponentDetector replay",
            "track_csv": self.track_csv,
            "frames": [
                {
                    "stamp_sec": f.stamp_sec,
                    "stamp_nanosec": f.stamp_nanosec,
                    "ranges": f.ranges,
                    "angle_min": angle_min,
                    "angle_increment": angle_increment,
                    "ego_x": f.ego_x,
                    "ego_y": f.ego_y,
                    "ego_yaw": f.ego_yaw,
                    "opp_x": (f.opp_x if f.opp_present else 0.0),
                    "opp_y": (f.opp_y if f.opp_present else 0.0),
                }
                for f in self.frames
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record gym /scan fixture for C++ replay")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--track-csv",
        default="/sim_ws/src/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
    )
    args = parser.parse_args()

    rclpy.init()
    node = ScanFixtureRecorder(
        num_frames=args.num_frames,
        sync_tol_ms=args.sync_tolerance_ms,
        track_csv=args.track_csv,
    )
    exit_code = 0
    try:
        t_end = time.time() + args.duration_s
        while time.time() < t_end and rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

        if len(node.frames) == 0:
            print("FAIL: recorded 0 frames", file=sys.stderr)
            exit_code = 1
        else:
            doc = node.to_document()
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            print(f"wrote {args.output}: {len(node.frames)} frames")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

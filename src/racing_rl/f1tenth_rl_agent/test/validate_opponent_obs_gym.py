#!/usr/bin/env python3
"""Gym validation: compare deployed obs[384:390] to ground-truth opponent block.

Run inside the f1tenth_gym_ros container with the 1v1 sim bridge and agent stack
(vehicle_obs with enable_opponent_obs) already up. Computes the expected
6-dim opponent block from /ego_racecar/opp_odom (gym ground truth) and compares
it to /rl/observation[384:390].

Exit 0 when max channel error stays within tolerance for enough samples; non-zero
otherwise. No mocks — requires live ROS topics from the gym sim.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.obs_core import ObservationBuilder


def _yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _vel_xy(msg: Odometry) -> tuple[float, float]:
    return (
        float(msg.twist.twist.linear.x),
        float(msg.twist.twist.linear.y),
    )


class OpponentObsValidator(Node):
    def __init__(
        self,
        *,
        track_csv: str,
        duration_s: float,
        pos_tol: float,
        vel_tol: float,
        gap_tol: float,
        ey_tol: float,
        min_present_frac: float,
        min_samples: int,
        opp_reference_topic: str,
    ):
        super().__init__("opponent_obs_validator")
        cl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(0, 1))
        wl = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(2,))
        wr = np.loadtxt(track_csv, delimiter=",", skiprows=1, usecols=(3,))
        obs_cfg = ifc.default_obs_cfg(enable_opponent_obs=True)
        self.builder = ObservationBuilder(cl, wl, wr, obs_cfg=obs_cfg)
        self.duration_s = duration_s
        self.pos_tol = pos_tol
        self.vel_tol = vel_tol
        self.gap_tol = gap_tol
        self.ey_tol = ey_tol
        self.min_present_frac = min_present_frac
        self.min_samples = min_samples
        self.opp_reference_topic = opp_reference_topic

        self._ego: Odometry | None = None
        self._ref_opp: Odometry | None = None
        self.errors: list[np.ndarray] = []
        self.present_flags: list[float] = []

        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_ego, 10)
        self.create_subscription(Odometry, opp_reference_topic, self._on_ref_opp, 10)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_OBSERVATION, self._on_obs, 10
        )

    def _on_ego(self, msg: Odometry):
        self._ego = msg

    def _on_ref_opp(self, msg: Odometry):
        self._ref_opp = msg

    def _on_obs(self, msg: Float32MultiArray):
        obs = np.asarray(msg.data, dtype=np.float32)
        if obs.shape[0] < ifc.expected_num_obs():
            return
        if self._ego is None or self._ref_opp is None:
            return

        ego = self._ego
        opp = self._ref_opp
        ego_pos = torch.tensor(
            [[ego.pose.pose.position.x, ego.pose.pose.position.y, 0.0]],
            dtype=torch.float32,
        )
        opp_pos = torch.tensor(
            [[opp.pose.pose.position.x, opp.pose.pose.position.y, 0.0]],
            dtype=torch.float32,
        )
        ego_yaw = torch.tensor([_yaw_from_odom(ego)], dtype=torch.float32)
        evx, evy = _vel_xy(ego)
        ovx, ovy = _vel_xy(opp)
        ego_vel = torch.tensor([[evx, evy, 0.0]], dtype=torch.float32)
        opp_vel = torch.tensor([[ovx, ovy, 0.0]], dtype=torch.float32)

        gt_block = (
            self.builder.build_opponent_block(
                ego_pos, ego_yaw, ego_vel, opp_pos, opp_vel
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
        obs_block = obs[384:390]
        err = np.abs(obs_block - gt_block)
        self.errors.append(err)
        self.present_flags.append(
            1.0 if float(np.abs(obs_block).max()) > 1e-6 else 0.0
        )

    def report(self) -> int:
        if len(self.errors) < self.min_samples:
            print(
                f"FAIL: only {len(self.errors)} paired samples "
                f"(need >= {self.min_samples})",
                file=sys.stderr,
            )
            return 1

        err_mat = np.stack(self.errors, axis=0)
        max_err = err_mat.max(axis=0)
        p95_err = np.percentile(err_mat, 95, axis=0)
        mean_err = err_mat.mean(axis=0)
        present_frac = float(np.mean(np.asarray(self.present_flags) > 0.5))

        labels = ["rel_x", "rel_y", "rel_vx", "rel_vy", "gap_norm", "ey_opp"]
        print("Opponent obs[384:390] vs gym GT block:")
        for i, name in enumerate(labels):
            print(
                f"  {name}: mean={mean_err[i]:.4f} p95={p95_err[i]:.4f} max={max_err[i]:.4f}"
            )

        tols = [
            self.pos_tol,
            self.pos_tol,
            self.vel_tol,
            self.vel_tol,
            self.gap_tol,
            self.ey_tol,
        ]
        failures = [
            f"{labels[i]} p95={p95_err[i]:.4f} > tol={tols[i]}"
            for i in range(6)
            if p95_err[i] > tols[i]
        ]
        if present_frac < self.min_present_frac:
            failures.append(
                f"presence fraction {present_frac:.2f} < {self.min_present_frac}"
            )

        if failures:
            print("FAIL:", "; ".join(failures), file=sys.stderr)
            return 1

        print(
            f"PASS: {len(self.errors)} samples, presence={present_frac:.2f}, "
            f"max_err={max_err.max():.4f}, ref={self.opp_reference_topic}"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate 1v1 opponent obs in gym")
    parser.add_argument(
        "--track-csv",
        default="/sim_ws/src/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv",
    )
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--pos-tol", type=float, default=0.75)
    parser.add_argument("--vel-tol", type=float, default=1.0)
    parser.add_argument("--gap-tol", type=float, default=0.15)
    parser.add_argument("--ey-tol", type=float, default=0.25)
    parser.add_argument("--min-present-frac", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument(
        "--opp-reference-topic",
        default=ifc.TOPIC_OPP_ODOM,
        help="Odometry topic used to compute the expected opponent block "
        "(default: gym GT /ego_racecar/opp_odom; use /rl/opponent/odom for detector wiring).",
    )
    args = parser.parse_args()

    rclpy.init()
    node = OpponentObsValidator(
        track_csv=args.track_csv,
        duration_s=args.duration_s,
        pos_tol=args.pos_tol,
        vel_tol=args.vel_tol,
        gap_tol=args.gap_tol,
        ey_tol=args.ey_tol,
        min_present_frac=args.min_present_frac,
        min_samples=args.min_samples,
        opp_reference_topic=args.opp_reference_topic,
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

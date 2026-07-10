#!/usr/bin/env python3
"""Estimate tyre-road friction utilization from a general driving rosbag.

Unlike the steady full-lock skidpad (analyze_full_lock / estimate_friction, which
only give a LOWER BOUND because the car never slides), this scans an entire lap
for the moments of highest horizontal acceleration and rising body slip -- i.e.
where the tyre is closest to its friction limit. At the grip limit the total
horizontal accel plateaus at ~mu*g while body-slip angle grows, so the peak
|a| / g there is the best data-driven estimate of the real mu.

Signals used (particle-filter pose is ground truth; IMU gives direct accel):
  /odom               -> longitudinal speed v
  /sensors/imu/raw    -> ax, ay (body frame), yaw rate
  /pf/pose/odom       -> x, y, yaw -> course-over-ground -> body slip beta

Run with an env that has `rosbags` (e.g. the on-car repo venv):
    python analysis/friction/estimate_mu_from_bag.py /path/to/bag_dir
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

G = 9.81

ACKERMANN_DRIVE = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""
ACKERMANN_DRIVE_STAMPED = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""


def build_typestore():
    ts = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg(ACKERMANN_DRIVE, "ackermann_msgs/msg/AckermannDrive"))
    types.update(get_types_from_msg(
        ACKERMANN_DRIVE_STAMPED, "ackermann_msgs/msg/AckermannDriveStamped"))
    ts.register(types)
    return ts


def yaw_from_quat(z: float, w: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def read(path: Path, ts):
    odom, imu, pf = [], [], []
    with AnyReader([path], default_typestore=ts) as reader:
        for conn, t, raw in reader.messages():
            tsec = t * 1e-9
            if conn.topic == "/odom":
                m = reader.deserialize(raw, conn.msgtype)
                odom.append((tsec, m.twist.twist.linear.x, m.twist.twist.angular.z))
            elif conn.topic == "/sensors/imu/raw":
                m = reader.deserialize(raw, conn.msgtype)
                imu.append((tsec, m.linear_acceleration.x, m.linear_acceleration.y,
                            m.angular_velocity.z))
            elif conn.topic == "/pf/pose/odom":
                m = reader.deserialize(raw, conn.msgtype)
                p = m.pose.pose
                pf.append((tsec, p.position.x, p.position.y,
                           yaw_from_quat(p.orientation.z, p.orientation.w)))
    return (np.array(odom, dtype=float), np.array(imu, dtype=float),
            np.array(pf, dtype=float))


def body_slip(pf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, x, y, yaw = pf[:, 0], pf[:, 1], pf[:, 2], pf[:, 3]
    vx, vy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(vx, vy)
    course = np.arctan2(vy, vx)
    beta = np.arctan2(np.sin(course - yaw), np.cos(course - yaw))
    # slip is only meaningful when actually moving
    beta = np.where(speed > 0.5, beta, 0.0)
    return t, beta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=str, help="rosbag2 directory")
    args = ap.parse_args()
    path = Path(args.bag)

    ts = build_typestore()
    odom, imu, pf = read(path, ts)
    print(f"bag: {path.name}")
    print(f"  /odom {len(odom)}  /imu {len(imu)}  /pf {len(pf)} samples")

    v = np.abs(odom[:, 1])
    print(f"\nspeed (odom): max {v.max():.2f} m/s  95th pct {np.percentile(v, 95):.2f}  "
          f"median {np.median(v):.2f}")

    V_GATE = 2.0
    moving_frac = float((v > V_GATE).mean())
    print(f"fraction of time above {V_GATE:.0f} m/s: {moving_frac*100:.1f}%")

    # Interpolate odom speed onto the IMU clock so accel is gated on real motion.
    v_at_imu = np.interp(imu[:, 0], odom[:, 0], v)
    fast = v_at_imu > V_GATE
    a_lat_imu = np.abs(imu[:, 2])
    a_horiz_imu = np.hypot(imu[:, 1], imu[:, 2])
    # a_lat from odom kinematics (v * yaw_rate), gated on real motion.
    a_lat_kin = np.abs(odom[:, 1] * odom[:, 2])
    fast_odom = v > V_GATE

    def stat(arr, mask, label):
        sel = arr[mask]
        if sel.size == 0:
            print(f"{label}: (no samples above gate)")
            return float("nan")
        print(f"{label}: max {sel.max():.2f} m/s^2 ({sel.max()/G:.2f} g)  "
              f"95th {np.percentile(sel,95)/G:.2f} g")
        return sel.max() / G

    print(f"\n--- accel while actually moving (v > {V_GATE:.0f} m/s) ---")
    mu_ay = stat(a_lat_imu, fast, "lateral |ay| (imu)  ")
    mu_ah = stat(a_horiz_imu, fast, "horizontal |a| (imu)")
    stat(a_lat_kin, fast_odom, "v*omega (odom)      ")

    if len(pf) > 8:
        tpf, beta = body_slip(pf)
        v_at_pf = np.interp(tpf, odom[:, 0], v)
        beta_deg = np.degrees(np.abs(beta))
        fast_pf = v_at_pf > V_GATE
        if fast_pf.any():
            bf = beta_deg[fast_pf]
            print(f"\nbody slip |beta| when v>{V_GATE:.0f}: max {bf.max():.1f} deg  "
                  f"95th {np.percentile(bf,95):.1f} deg  median {np.median(bf):.1f} deg")

    mu_lb = np.nanmax([mu_ay, mu_ah])
    print(f"\n{'='*60}")
    print(f"friction estimate: peak utilized mu ~= {mu_lb:.2f} while moving "
          f"(LOWER BOUND -- car never held a sustained slide here)")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

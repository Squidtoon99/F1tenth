"""Reactive wall-following explorer for recon-lap mapping.

Traces a closed, wall-bounded track in a single clean lap directly from the LiDAR
``/scan`` (no global planner, no frontier search). It hugs one wall at a fixed distance
using a two-ray distance/heading estimate with look-ahead, plus a front-clearance
override that turns out of corners and hairpins. Because it commits to following one wall,
it goes around the loop in a single consistent direction -- never ping-ponging, U-turning,
or shortcutting -- and the ``occupancy_mapper`` builds the map from the same live scans.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ReactiveExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__("reactive_explorer")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("scan_topic", "/scan"),
                ("odom_topic", "/ego_racecar/odom"),
                ("drive_topic", "/drive"),
                ("cruise_speed_mps", 2.0),
                ("min_speed_mps", 0.8),
                ("max_steer_rad", 0.4),
                ("wheelbase_m", 0.33),
                ("follow_side", "left"),     # which wall to hug: "left" or "right"
                ("wall_target_m", 0.9),      # distance to hold from the followed wall
                ("wall_kp", 0.9),            # proportional gain on distance error
                ("wall_lookahead_m", 1.1),   # project distance this far ahead (anticipation)
                ("wall_theta_deg", 50.0),    # fwd tilt of the 2nd ray for the wall estimate
                ("front_stop_m", 2.0),       # front clearance below which we turn out of corners
                ("front_turn_gain", 0.6),    # extra steer per metre of front encroachment
                ("lookahead_clip_m", 10.0),  # cap ranges so far-away openings don't dominate
                ("steer_smooth", 0.3),       # low-pass on steering command
                ("speed_smooth", 0.0),       # low-pass on speed (gentle accel/decel, no VESC crunch)
                ("min_lap_m", 100.0),        # min travel before loop closure can be declared
                ("closure_radius_m", 5.0),   # return within this of start (after a lap) = done
                ("map_save_path", "/tmp/slam_map"),
                ("slam_map_topic", "/slam_map"),
                ("auto_save_map", True),
                ("stop_on_loop", True),
                # Safety: in dry-run the steering/speed are computed and logged but the car is
                # commanded to stay still (speed 0). Flip to false only when ready to drive.
                ("dry_run", True),
                ("status_period_s", 1.0),
                # Deterministic on-track spawn. The gym's MAP_RANDOM_STATIC reset can place
                # the car off the racing corridor (infield/outside are also "free"), so we
                # teleport it onto a known centerline pose via /initialpose before driving.
                ("reset_on_start", True),
                ("initialpose_topic", "/initialpose"),
                ("start_x", 0.42),
                ("start_y", 0.16),
                ("start_yaw", -1.583),
                ("start_delay_s", 1.5),
            ],
        )

        self._max_steer = float(self.get_parameter("max_steer_rad").value)
        self._prev_steer = 0.0
        # Seed at the floor so the speed low-pass never dips into the VESC deadband.
        self._prev_speed = float(self.get_parameter("min_speed_mps").value)

        self._start_xy: np.ndarray | None = None
        self._prev_xy: np.ndarray | None = None
        self._traveled = 0.0
        self._loop_closed = False
        self._done = False
        # Hold off driving / odometry tracking until the on-track spawn reset has settled.
        self._ready = not bool(self.get_parameter("reset_on_start").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)

        # Status bookkeeping for periodic logging.
        self._last_speed = 0.0
        self._last_steer = 0.0
        self._last_front = float("nan")
        self._last_xy: np.ndarray | None = None
        self._scan_count = 0
        self._map_known = 0
        self._map_total = 0
        self._map_msgs = 0

        drive_topic = self.get_parameter("drive_topic").value
        scan_topic = self.get_parameter("scan_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        slam_topic = str(self.get_parameter("slam_map_topic").value)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self._initpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, str(self.get_parameter("initialpose_topic").value), 10
        )
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        # slam_toolbox latches /map with transient-local durability; match it so we actually
        # receive the map we are "reading from".
        map_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, slam_topic, self._on_map, map_qos)
        self.get_logger().info(
            f"reactive_explorer up [{'DRY-RUN' if self._dry_run else 'LIVE-DRIVE'}]: "
            f"scan={scan_topic} odom={odom_topic} drive={drive_topic} slam_map={slam_topic}"
        )
        self.create_timer(float(self.get_parameter("status_period_s").value), self._log_status)
        if bool(self.get_parameter("reset_on_start").value):
            # Publish a couple of times (latched-ish) then become ready after the delay.
            self.create_timer(0.3, self._publish_initialpose)
            delay = float(self.get_parameter("start_delay_s").value)
            self._ready_timer = self.create_timer(delay, self._become_ready)

    def _publish_initialpose(self) -> None:
        if self._ready:
            return
        x = float(self.get_parameter("start_x").value)
        y = float(self.get_parameter("start_y").value)
        yaw = float(self.get_parameter("start_yaw").value)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._initpose_pub.publish(msg)

    def _become_ready(self) -> None:
        self._ready_timer.cancel()
        self._start_xy = None
        self._prev_xy = None
        self._traveled = 0.0
        self._ready = True
        self.get_logger().info("On-track spawn set; beginning recon lap.")

    # --- odometry: distance tracking + loop closure -----------------------------------
    def _on_odom(self, msg: Odometry) -> None:
        if not self._ready:
            return
        p = msg.pose.pose.position
        xy = np.array([p.x, p.y], dtype=np.float64)
        self._last_xy = xy.copy()
        if self._start_xy is None:
            self._start_xy = xy.copy()
            self._prev_xy = xy.copy()
            return
        self._traveled += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy.copy()
        if (not self._loop_closed
                and self._traveled > float(self.get_parameter("min_lap_m").value)
                and float(np.linalg.norm(xy - self._start_xy))
                < float(self.get_parameter("closure_radius_m").value)):
            self._loop_closed = True
            self.get_logger().info(
                f"Loop closed: traveled={self._traveled:.0f} m, back near start "
                f"({xy[0]:.1f}, {xy[1]:.1f})."
            )
            if bool(self.get_parameter("stop_on_loop").value):
                self._finish()

    # --- wall following on the live scan ----------------------------------------------
    def _on_scan(self, msg: LaserScan) -> None:
        self._scan_count += 1
        if not self._ready or self._done:
            self._publish_stop()
            return
        n = len(msg.ranges)
        if n == 0:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        clip = float(self.get_parameter("lookahead_clip_m").value)
        ranges[~np.isfinite(ranges)] = clip
        ranges = np.clip(ranges, 0.0, clip)
        amin = float(msg.angle_min)
        ainc = float(msg.angle_increment)

        def r_at(deg: float) -> float:
            """Min range in a small window around an angle (deg), robust to single noisy beams."""
            idx = int(round((math.radians(deg) - amin) / max(ainc, 1e-9)))
            lo = max(0, idx - 2)
            hi = min(n, idx + 3)
            if lo >= hi:
                return clip
            return float(ranges[lo:hi].min())

        side = 1.0 if str(self.get_parameter("follow_side").value) == "left" else -1.0
        theta = math.radians(float(self.get_parameter("wall_theta_deg").value))
        target_d = float(self.get_parameter("wall_target_m").value)
        kp = float(self.get_parameter("wall_kp").value)
        look = float(self.get_parameter("wall_lookahead_m").value)

        # Two-ray estimate of distance/heading to the followed wall (side: +left / -right).
        b = r_at(side * 90.0)               # beam perpendicular to the followed wall
        a = r_at(side * (90.0 - math.degrees(theta)))  # beam tilted forward toward the wall
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        dist = b * math.cos(alpha)
        dist_proj = dist + look * math.sin(alpha)
        # Steer to hold target_d from the followed wall (sign flips for left vs right wall).
        steer = side * kp * (dist_proj - target_d)

        # Front clearance: if a wall looms ahead (a corner toward the followed wall or a
        # dead-end), turn hard toward whichever side is more open so we never nose in.
        front = min(r_at(0.0), r_at(15.0), r_at(-15.0), r_at(30.0), r_at(-30.0))
        d_left = min(r_at(60.0), r_at(90.0))
        d_right = min(r_at(-60.0), r_at(-90.0))
        front_stop = float(self.get_parameter("front_stop_m").value)
        if front < front_stop:
            turn = float(self.get_parameter("front_turn_gain").value) * (front_stop - front)
            steer += turn if d_left > d_right else -turn

        steer = float(np.clip(steer, -self._max_steer, self._max_steer))
        sm = float(self.get_parameter("steer_smooth").value)
        steer = sm * self._prev_steer + (1.0 - sm) * steer
        self._prev_steer = steer

        cruise = float(self.get_parameter("cruise_speed_mps").value)
        min_speed = float(self.get_parameter("min_speed_mps").value)
        # Slow for sharp steering AND for low forward clearance (corners / hairpins).
        steer_factor = 1.0 - abs(steer) / max(self._max_steer, 1e-3)
        clear_factor = float(np.clip((front - 0.6) / 2.0, 0.0, 1.0))
        speed = min_speed + (cruise - min_speed) * min(steer_factor, clear_factor)
        speed = max(min_speed, min(cruise, speed))
        # Low-pass the speed so it eases between cruise and the floor instead of snapping,
        # which is what makes the VESC cog/"crunch" on corner entry.
        ssm = float(self.get_parameter("speed_smooth").value)
        speed = ssm * self._prev_speed + (1.0 - ssm) * speed
        self._prev_speed = speed

        self._last_speed = speed
        self._last_steer = steer
        self._last_front = front
        if self._dry_run:
            # Compute everything, but keep the car still and let the human drive/observe.
            self._publish_stop()
        else:
            self._publish_drive(speed, steer)

    # --- slam map: confirm we are reading slam_toolbox + track coverage ---------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16)
        self._map_total = int(data.size)
        self._map_known = int(np.count_nonzero(data >= 0))  # -1 == unknown
        self._map_msgs += 1

    def _log_status(self) -> None:
        pose = "pose=? "
        if self._last_xy is not None:
            dist = (float(np.linalg.norm(self._last_xy - self._start_xy))
                    if self._start_xy is not None else 0.0)
            pose = f"pose=({self._last_xy[0]:.1f},{self._last_xy[1]:.1f}) dist0={dist:.1f} "
        known_pct = (100.0 * self._map_known / self._map_total) if self._map_total else 0.0
        mode = "DRY-RUN" if self._dry_run else "LIVE"
        state = "done" if self._done else ("ready" if self._ready else "waiting-reset")
        self.get_logger().info(
            f"[{mode}/{state}] scans={self._scan_count} {pose}"
            f"traveled={self._traveled:.1f}m front={self._last_front:.2f} "
            f"cmd(speed={self._last_speed:.2f},steer={self._last_steer:+.2f}) "
            f"slam_map: msgs={self._map_msgs} known={self._map_known}/{self._map_total} "
            f"({known_pct:.1f}%)"
        )

    # --- drive helpers ----------------------------------------------------------------
    def _publish_drive(self, speed: float, steer: float) -> None:
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.speed = float(speed)
        drive.drive.steering_angle = float(steer)
        self._drive_pub.publish(drive)

    def _publish_stop(self) -> None:
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        self._drive_pub.publish(drive)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._publish_stop()
        if bool(self.get_parameter("auto_save_map").value):
            # Give slam_toolbox time to publish the final full map before saving, otherwise
            # map_saver_cli latches a stale (smaller) snapshot.
            self._save_timer = self.create_timer(3.0, self._save_map_once)

    def _save_map_once(self) -> None:
        self._save_timer.cancel()
        self._save_map()

    def _save_map(self) -> None:
        path = str(self.get_parameter("map_save_path").value)
        topic = str(self.get_parameter("slam_map_topic").value)
        self.get_logger().info(f"Saving SLAM map ({topic}) to {path}")
        try:
            result = subprocess.run(
                ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", path,
                 "--ros-args", "-p", "map_subscribe_transient_local:=true",
                 "-p", "save_map_timeout:=10000.0",
                 "-r", f"map:={topic}"],
                check=False, timeout=60.0, capture_output=True,
            )
            if result.returncode == 0:
                self.get_logger().info(f"Map saved: {path}.pgm / {path}.yaml")
            else:
                self.get_logger().error(
                    f"map_saver_cli exit {result.returncode}: "
                    f"{result.stderr.decode(errors='ignore')[:300]}"
                )
        except (subprocess.SubprocessError, OSError) as exc:
            self.get_logger().error(f"Map save failed: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReactiveExplorerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Navigate to frontier goals using A*, pure pursuit, and PID speed control."""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from f1tenth_mapping.mapping_math import (
    MapMeta,
    astar,
    distance_to_goal,
    grid_from_occupancy,
)
from f1tenth_mapping.mapping_state import NavStatus
from f1tenth_mapping.pure_pursuit import SpeedPID, compute_steering


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class NavigatorNode(Node):
    def __init__(self) -> None:
        super().__init__("navigator")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_topic", "/map"),
                ("odom_topic", "/odom"),
                ("goal_topic", "/mapping/goal"),
                ("nav_status_topic", "/mapping/nav_status"),
                ("drive_topic", "/drive"),
                ("control_hz", 20.0),
                ("mapping_speed_mps", 2.0),
                ("speed_limit_mps", 2.5),
                ("goal_tolerance_m", 0.5),
                ("stuck_timeout_s", 15.0),
                ("watchdog_timeout_s", 0.5),
                ("wheelbase_m", 0.33),
                ("max_steer_rad", 0.44),
                ("lookahead_m", 1.5),
                ("min_lookahead_m", 0.8),
                ("path_inflation_cells", 2),
                ("speed_pid_kp", 1.5),
                ("speed_pid_ki", 0.1),
                ("speed_pid_kd", 0.05),
                ("speed_pid_limit", 1.0),
            ],
        )

        self._map: OccupancyGrid | None = None
        self._odom: Odometry | None = None
        self._goal: PoseStamped | None = None
        self._path: np.ndarray | None = None
        self._status = NavStatus.IDLE
        self._last_progress_dist = math.inf
        self._last_progress_time = time.monotonic()
        self._last_drive_time = time.monotonic()
        self._speed_pid = SpeedPID(
            kp=float(self.get_parameter("speed_pid_kp").value),
            ki=float(self.get_parameter("speed_pid_ki").value),
            kd=float(self.get_parameter("speed_pid_kd").value),
            limit=float(self.get_parameter("speed_pid_limit").value),
        )

        map_topic = self.get_parameter("map_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        goal_topic = self.get_parameter("goal_topic").value
        nav_status_topic = self.get_parameter("nav_status_topic").value
        drive_topic = self.get_parameter("drive_topic").value

        self._drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self._status_pub = self.create_publisher(Float32MultiArray, nav_status_topic, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._on_map, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)

        hz = float(self.get_parameter("control_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self._control_tick)
        watchdog = float(self.get_parameter("watchdog_timeout_s").value)
        self.create_timer(max(watchdog / 2.0, 0.1), self._watchdog_tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._path = None
        self._status = NavStatus.NAVIGATING
        self._last_progress_dist = math.inf
        self._last_progress_time = time.monotonic()
        self._speed_pid.reset()
        self.get_logger().info(
            f"New goal ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def _pose(self) -> tuple[np.ndarray, float, float] | None:
        if self._odom is None:
            return None
        p = self._odom.pose.pose.position
        q = self._odom.pose.pose.orientation
        yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        speed = float(self._odom.twist.twist.linear.x)
        return np.array([p.x, p.y], dtype=np.float64), yaw, speed

    def _plan_path(self, pose_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray | None:
        if self._map is None:
            return None
        meta = MapMeta(
            width=int(self._map.info.width),
            height=int(self._map.info.height),
            resolution=float(self._map.info.resolution),
            origin_x=float(self._map.info.origin.position.x),
            origin_y=float(self._map.info.origin.position.y),
        )
        grid = grid_from_occupancy(np.array(self._map.data), meta.width, meta.height)
        inflation = int(self.get_parameter("path_inflation_cells").value)
        path = astar(grid, meta, pose_xy, goal_xy, inflation_cells=inflation)
        if path is None:
            return None
        return path

    def _control_tick(self) -> None:
        pose_data = self._pose()
        if pose_data is None:
            self._publish_status()
            return
        pose_xy, yaw, speed = pose_data

        if self._goal is None or self._status != NavStatus.NAVIGATING:
            self._publish_stop()
            self._publish_status(pose_xy)
            return

        goal_xy = np.array(
            [self._goal.pose.position.x, self._goal.pose.position.y],
            dtype=np.float64,
        )
        dist = distance_to_goal(pose_xy, goal_xy)
        tolerance = float(self.get_parameter("goal_tolerance_m").value)

        if dist <= tolerance:
            self._status = NavStatus.REACHED
            self._publish_stop()
            self._publish_status(pose_xy)
            self.get_logger().info("Goal reached")
            return

        if self._check_stuck(dist):
            self._status = NavStatus.STUCK
            self._publish_stop()
            self._publish_status(pose_xy)
            self.get_logger().warn("Navigation stuck; reporting failure")
            return

        if self._path is None:
            self._path = self._plan_path(pose_xy, goal_xy)
            if self._path is None:
                self._status = NavStatus.FAILED
                self._publish_stop()
                self._publish_status(pose_xy)
                self.get_logger().warn("No feasible path to goal")
                return

        lookahead = max(
            float(self.get_parameter("min_lookahead_m").value),
            min(float(self.get_parameter("lookahead_m").value), 0.5 + 0.4 * abs(speed)),
        )
        steering, _ = compute_steering(
            pose_xy,
            yaw,
            self._path,
            wheelbase_m=float(self.get_parameter("wheelbase_m").value),
            lookahead_m=lookahead,
            max_steer_rad=float(self.get_parameter("max_steer_rad").value),
        )

        cruise = float(self.get_parameter("mapping_speed_mps").value)
        target_speed = min(cruise, max(0.3, dist))
        speed_limit = float(self.get_parameter("speed_limit_mps").value)
        dt = 1.0 / max(float(self.get_parameter("control_hz").value), 1.0)
        throttle_adj = self._speed_pid.compute(target_speed, abs(speed), dt)
        cmd_speed = max(0.0, min(speed_limit, target_speed + throttle_adj))

        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.speed = float(cmd_speed)
        drive.drive.steering_angle = float(steering)
        self._drive_pub.publish(drive)
        self._last_drive_time = time.monotonic()
        self._publish_status(pose_xy)

    def _check_stuck(self, dist: float) -> bool:
        now = time.monotonic()
        if dist < self._last_progress_dist - 0.05:
            self._last_progress_dist = dist
            self._last_progress_time = now
        stuck_timeout = float(self.get_parameter("stuck_timeout_s").value)
        return (now - self._last_progress_time) > stuck_timeout

    def _publish_stop(self) -> None:
        drive = AckermannDriveStamped()
        drive.header.stamp = self.get_clock().now().to_msg()
        drive.header.frame_id = "base_link"
        drive.drive.speed = 0.0
        drive.drive.steering_angle = 0.0
        self._drive_pub.publish(drive)
        self._last_drive_time = time.monotonic()

    def _watchdog_tick(self) -> None:
        if self._status != NavStatus.NAVIGATING:
            return
        timeout = float(self.get_parameter("watchdog_timeout_s").value)
        if time.monotonic() - self._last_drive_time > timeout:
            self._status = NavStatus.STUCK
            self._publish_stop()
            self._publish_status()

    def _publish_status(self, pose_xy: np.ndarray | None = None) -> None:
        if pose_xy is None and self._pose() is not None:
            pose_xy = self._pose()[0]
        if pose_xy is None:
            pose_xy = np.zeros(2, dtype=np.float64)
        msg = Float32MultiArray()
        msg.data = [
            float(self._status),
            float(pose_xy[0]),
            float(pose_xy[1]),
        ]
        self._status_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

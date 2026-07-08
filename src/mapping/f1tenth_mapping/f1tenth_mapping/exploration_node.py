"""DFS frontier exploration planner."""

from __future__ import annotations

import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from f1tenth_mapping.mapping_math import (
    DfsGoalStack,
    MapMeta,
    cluster_frontiers,
    coverage_ratio,
    free_bounding_box,
    grid_from_occupancy,
)
from f1tenth_mapping.mapping_state import ExplorationStatus, NavStatus


class ExplorationNode(Node):
    def __init__(self) -> None:
        super().__init__("exploration")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_topic", "/map"),
                ("goal_topic", "/mapping/goal"),
                ("nav_status_topic", "/mapping/nav_status"),
                ("status_topic", "/mapping/status"),
                ("map_save_path", "/tmp/f1tenth_map"),
                ("control_hz", 1.0),
                ("min_cluster_size", 3),
                ("min_coverage_ratio", 0.98),
                ("max_runtime_s", 3600.0),
                ("startup_delay_s", 5.0),
                ("auto_save_map", True),
            ],
        )

        self._map: OccupancyGrid | None = None
        self._pose_xy = np.zeros(2, dtype=np.float64)
        self._goal_stack = DfsGoalStack()
        self._active_goal: np.ndarray | None = None
        self._status = ExplorationStatus.WAITING_FOR_MAP
        self._start_time = time.monotonic()
        self._nav_status = NavStatus.IDLE
        self._saved_map = False

        map_topic = self.get_parameter("map_topic").value
        goal_topic = self.get_parameter("goal_topic").value
        nav_status_topic = self.get_parameter("nav_status_topic").value
        status_topic = self.get_parameter("status_topic").value

        self._goal_pub = self.create_publisher(PoseStamped, goal_topic, 10)
        self._status_pub = self.create_publisher(Float32MultiArray, status_topic, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._on_map, 10)
        self.create_subscription(Float32MultiArray, nav_status_topic, self._on_nav_status, 10)

        hz = float(self.get_parameter("control_hz").value)
        self.create_timer(1.0 / max(hz, 0.1), self._tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _on_nav_status(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            return
        self._nav_status = NavStatus(int(msg.data[0]))
        self._pose_xy[0] = float(msg.data[1])
        self._pose_xy[1] = float(msg.data[2])
        if self._nav_status in (NavStatus.REACHED, NavStatus.FAILED, NavStatus.STUCK):
            self._active_goal = None

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start_time
        startup_delay = float(self.get_parameter("startup_delay_s").value)
        if elapsed < startup_delay:
            self._publish_status(0.0, 0)
            return

        max_runtime = float(self.get_parameter("max_runtime_s").value)
        if elapsed > max_runtime and self._status != ExplorationStatus.COMPLETE:
            self._status = ExplorationStatus.TIMEOUT
            self.get_logger().warn("Exploration timed out")
            self._maybe_save_map()
            self._publish_status(0.0, 0)
            return

        if self._map is None:
            self._publish_status(0.0, 0)
            return

        meta, grid, roi = self._parse_map(self._map)
        cov = coverage_ratio(grid, roi) if roi is not None else 0.0
        frontiers = cluster_frontiers(
            grid,
            meta,
            min_cluster_size=int(self.get_parameter("min_cluster_size").value),
            roi=roi,
        )
        frontier_count = len(frontiers)

        min_cov = float(self.get_parameter("min_coverage_ratio").value)
        if frontier_count == 0 and cov >= min_cov:
            if self._status != ExplorationStatus.COMPLETE:
                self._status = ExplorationStatus.COMPLETE
                self.get_logger().info(
                    f"Exploration complete: coverage={cov:.3f}, frontiers=0"
                )
                self._maybe_save_map()
            self._publish_status(cov, frontier_count)
            return

        self._status = ExplorationStatus.EXPLORING

        if self._active_goal is None and self._nav_status not in (NavStatus.NAVIGATING,):
            if len(self._goal_stack) == 0 and frontiers:
                self._goal_stack.push_clusters(frontiers, self._pose_xy)
            goal_xy = self._goal_stack.pop()
            if goal_xy is not None:
                self._send_goal(goal_xy)
                self._active_goal = goal_xy.copy()
            elif frontiers:
                self._goal_stack.push_clusters(frontiers, self._pose_xy)
                goal_xy = self._goal_stack.pop()
                if goal_xy is not None:
                    self._send_goal(goal_xy)
                    self._active_goal = goal_xy.copy()

        self._publish_status(cov, frontier_count)

    def _parse_map(
        self, msg: OccupancyGrid
    ) -> tuple[MapMeta, np.ndarray, tuple[int, int, int, int] | None]:
        meta = MapMeta(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
        )
        grid = grid_from_occupancy(np.array(msg.data), meta.width, meta.height)
        free_mask = (grid >= 0) & (grid <= 99)
        roi = free_bounding_box(free_mask)
        return meta, grid, roi

    def _send_goal(self, goal_xy: np.ndarray) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(goal_xy[0])
        msg.pose.position.y = float(goal_xy[1])
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)
        self.get_logger().info(f"Sent frontier goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")

    def _publish_status(self, coverage: float, frontier_count: int) -> None:
        msg = Float32MultiArray()
        msg.data = [
            float(self._status),
            coverage,
            float(frontier_count),
            float(len(self._goal_stack)),
            1.0 if self._active_goal is not None else 0.0,
        ]
        self._status_pub.publish(msg)

    def _maybe_save_map(self) -> None:
        if self._saved_map or not bool(self.get_parameter("auto_save_map").value):
            return
        path = str(self.get_parameter("map_save_path").value)
        self.get_logger().info(f"Saving map to {path}")
        try:
            result = subprocess.run(
                ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", path],
                check=False,
                timeout=30.0,
                capture_output=True,
            )
            if result.returncode == 0:
                self._saved_map = True
            else:
                self.get_logger().error(
                    f"Map save failed with exit code {result.returncode}"
                )
        except (subprocess.SubprocessError, OSError) as exc:
            self.get_logger().error(f"Map save failed: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ExplorationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

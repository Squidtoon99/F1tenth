"""Occupancy-grid mapper: build a map from live LiDAR scans (log-odds inverse sensor model).

A real, standard occupancy-grid mapping implementation. For every LaserScan it:
  1. looks up the sensor pose in the map frame (TF map -> <laser frame>),
  2. ray-casts each beam, decrementing log-odds along the free ray and incrementing it at
     the measured hit cell (the inverse sensor model),
  3. publishes the accumulated grid as nav_msgs/OccupancyGrid.

Unlike slam_toolbox this does no scan matching (the gym already provides a consistent
pose), so it integrates *every* scan densely and never fragments or freezes. It is the
in-sim equivalent of the offline raycast mapper, but driven by the real /scan stream.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class OccupancyMapperNode(Node):
    def __init__(self) -> None:
        super().__init__("occupancy_mapper")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("scan_topic", "/scan_slam"),
                ("map_topic", "/slam_map"),
                ("map_frame", "map"),
                ("resolution", 0.05),
                ("width_m", 70.0),
                ("height_m", 70.0),
                ("origin_x", -35.0),
                ("origin_y", -35.0),
                ("max_range_m", 25.0),
                # No-return beams (no wall within sensor range) only get free-carved up to
                # this distance, to avoid long "spray" fans into genuinely-unknown space.
                ("free_clear_range_m", 12.0),
                ("publish_period_s", 1.0),
                # Ignore scans for this long after startup, so the car's pre-reset spawn
                # position (before the on-track teleport) is not baked into the map.
                ("start_delay_s", 2.5),
                ("l_free", 0.4),
                ("l_occ", 0.85),
                ("l_min", -4.0),
                ("l_max", 6.0),
                ("occ_thresh", 0.5),
                ("free_thresh", -0.5),
            ],
        )
        self._res = float(self.get_parameter("resolution").value)
        self._ox = float(self.get_parameter("origin_x").value)
        self._oy = float(self.get_parameter("origin_y").value)
        self._W = int(round(float(self.get_parameter("width_m").value) / self._res))
        self._H = int(round(float(self.get_parameter("height_m").value) / self._res))
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._free_clear = float(self.get_parameter("free_clear_range_m").value)
        self._l_free = float(self.get_parameter("l_free").value)
        self._l_occ = float(self.get_parameter("l_occ").value)
        self._l_min = float(self.get_parameter("l_min").value)
        self._l_max = float(self.get_parameter("l_max").value)
        self._map_frame = str(self.get_parameter("map_frame").value)

        self._logodds = np.zeros((self._H, self._W), dtype=np.float32)
        # Sample distances for ray free-space carving.
        self._d = np.arange(self._res, self._max_range, self._res, dtype=np.float64)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._start_time = self.get_clock().now()
        self._start_delay = float(self.get_parameter("start_delay_s").value)

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._map_pub = self.create_publisher(
            OccupancyGrid, str(self.get_parameter("map_topic").value), latched
        )
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, 10
        )
        self.create_timer(
            float(self.get_parameter("publish_period_s").value), self._publish_map
        )
        self.get_logger().info(
            f"occupancy_mapper: {self._W}x{self._H} @ {self._res} m, "
            f"origin=({self._ox},{self._oy}), frame={self._map_frame}"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        if (self.get_clock().now() - self._start_time).nanoseconds * 1e-9 < self._start_delay:
            return
        # Sensor pose in the map frame from TF (the gym publishes ground-truth TF).
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, msg.header.frame_id, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        lx, ly = float(t.x), float(t.y)
        lyaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)

        n = len(msg.ranges)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        beam_ang = msg.angle_min + np.arange(n) * msg.angle_increment
        amap = lyaw + beam_ang
        ca, sa = np.cos(amap), np.sin(amap)

        finite = np.isfinite(ranges)
        hit = finite & (ranges >= msg.range_min) & (ranges < self._max_range)
        # Returning beams carve free space up to their wall hit; no-return beams only carve
        # a limited "confident" distance so they don't spray free fans into unknown space.
        r_eff = np.where(hit, ranges, self._free_clear)

        # --- free space: sample points along each beam up to its (range - 1 cell) ---
        # free_pts[k, m] valid where d_m < r_eff_k - res
        valid = self._d[None, :] < (r_eff[:, None] - self._res)
        fk, fm = np.nonzero(valid)
        fx = lx + ca[fk] * self._d[fm]
        fy = ly + sa[fk] * self._d[fm]
        self._stamp(fx, fy, -self._l_free)

        # --- occupied: the measured hit cell of each returning beam ---
        if hit.any():
            ex = lx + ca[hit] * ranges[hit]
            ey = ly + sa[hit] * ranges[hit]
            self._stamp(ex, ey, +self._l_occ)

    def _stamp(self, xs, ys, delta) -> None:
        cols = np.floor((xs - self._ox) / self._res).astype(np.int64)
        rows = np.floor((ys - self._oy) / self._res).astype(np.int64)
        inb = (rows >= 0) & (rows < self._H) & (cols >= 0) & (cols < self._W)
        r = rows[inb]
        c = cols[inb]
        if r.size == 0:
            return
        np.add.at(self._logodds, (r, c), delta)
        np.clip(self._logodds, self._l_min, self._l_max, out=self._logodds)

    def _publish_map(self) -> None:
        occ_thresh = float(self.get_parameter("occ_thresh").value)
        free_thresh = float(self.get_parameter("free_thresh").value)
        grid = np.full((self._H, self._W), -1, dtype=np.int8)
        grid[self._logodds >= occ_thresh] = 100
        grid[self._logodds <= free_thresh] = 0

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.info.resolution = self._res
        msg.info.width = self._W
        msg.info.height = self._H
        msg.info.origin.position.x = self._ox
        msg.info.origin.position.y = self._oy
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).tolist()
        self._map_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OccupancyMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

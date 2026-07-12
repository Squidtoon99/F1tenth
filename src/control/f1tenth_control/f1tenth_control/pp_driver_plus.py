"""pp_driver_plus: pure pursuit with a Follow-the-Gap override for overtaking.

Standalone node. It reuses the pure-pursuit logic from ``PurePursuitNode``
(``pp_driver``) and the Follow-the-Gap / disparity-extender logic from
``ReactiveFollowGap`` (``gap_driver``) WITHOUT modifying or spinning either of
those nodes.

Reuse mechanism:
  - It subclasses ``PurePursuitNode`` so the pure-pursuit method chain
    (``compute_pure_pursuit`` -> ``_find_lookahead_point`` -> ``_find_closest_idx``)
    resolves its ``self.*`` calls against this node. ``PurePursuitNode.__init__``
    is intentionally bypassed (``Node.__init__`` is called directly) so none of
    its subscriptions/publishers are created.
  - The Follow-the-Gap helpers are self-contained (they only read attributes on
    ``self``), so they are invoked as unbound methods of ``ReactiveFollowGap``.

Behaviour:
  Default mode is raceline pure pursuit. When an obstacle (opponent) is detected
  directly ahead within a forward LiDAR cone, the node switches to Follow-the-Gap
  steering/speed until the path ahead clears again (distance hysteresis).

Known limitations:
  - The gap search uses the full scan (same as gap_driver); the forward cone is
    used only as the switch trigger, not to bound the gap search.
  - Pass side is whatever the largest gap dictates (no left/right preference).
  - Walls on tight corners may trigger gap mode; tune the cone angle and the
    enter/exit distances on the actual track.
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from f1tenth_control.pp_driver import PurePursuitNode
from f1tenth_control.gap_driver import ReactiveFollowGap


class PpDriverPlusNode(PurePursuitNode):
    """Pure pursuit with a reactive Follow-the-Gap override when blocked ahead."""

    def __init__(self) -> None:
        # Bypass PurePursuitNode.__init__ (we do not want its subs/pubs); only
        # initialise the underlying rclpy Node. PurePursuitNode methods remain
        # available via inheritance and run against this node's state.
        Node.__init__(self, 'pp_driver_plus')

        # --- topics / timing ---
        self.declare_parameter('odom_topic', '/pf/pose/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('scan_timeout_s', 0.3)

        # --- pure pursuit (names match pp_driver) ---
        self.declare_parameter(
            'raceline_path',
            '/config/maps/raceline.csv',
        )
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('speed_scale', 0.9)
        self.declare_parameter('min_speed', 1.7)
        self.declare_parameter('max_speed', 8.0)
        self.declare_parameter('use_raceline_speed', True)
        self.declare_parameter('max_steering_angle', 0.4)
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)

        # --- obstacle trigger (new) ---
        self.declare_parameter('forward_cone_half_angle', 0.35)
        self.declare_parameter('obstacle_enter_dist', 2.5)
        self.declare_parameter('obstacle_exit_dist', 3.5)

        # --- follow-the-gap (names match gap_driver) ---
        self.declare_parameter('velocity', 2.0)
        self.declare_parameter('max_velocity', 3.0)
        self.declare_parameter('max_distance_threshold', 5.0)
        self.declare_parameter('min_distance_threshold', 0.3)
        self.declare_parameter('steering_sensitivity', 1.0)
        self.declare_parameter('bubble_radius', 50)
        self.declare_parameter('disparity_radius_m', 0.15)
        self.declare_parameter('disparity_threshold', 1.0)

        gp = self.get_parameter
        self.odom_topic = gp('odom_topic').value
        self.scan_topic = gp('scan_topic').value
        self.drive_topic = gp('drive_topic').value
        control_rate_hz = float(gp('control_rate_hz').value)
        self.scan_timeout_s = float(gp('scan_timeout_s').value)

        # pure pursuit state/params expected by PurePursuitNode methods
        self.ld = float(gp('lookahead_distance').value)
        self.L = float(gp('wheelbase').value)
        self.speed_scale = float(gp('speed_scale').value)
        self.min_speed = float(gp('min_speed').value)
        self.max_speed = float(gp('max_speed').value)
        self.use_raceline_speed = bool(gp('use_raceline_speed').value)
        self.max_steering_angle = float(gp('max_steering_angle').value)
        orig_x = float(gp('origin_x').value)
        orig_y = float(gp('origin_y').value)

        # obstacle trigger
        self.forward_cone_half_angle = float(gp('forward_cone_half_angle').value)
        self.obstacle_enter_dist = float(gp('obstacle_enter_dist').value)
        self.obstacle_exit_dist = float(gp('obstacle_exit_dist').value)

        # gap params expected by ReactiveFollowGap methods
        self.velocity = float(gp('velocity').value)
        self.max_velocity = float(gp('max_velocity').value)
        self.max_distance_threshold = float(gp('max_distance_threshold').value)
        self.min_distance_threshold = float(gp('min_distance_threshold').value)
        self.steering_sensitivity = float(gp('steering_sensitivity').value)
        self.bubble_radius = int(gp('bubble_radius').value)
        self.disparity_radius_m = float(gp('disparity_radius_m').value)
        self.disparity_threshold = float(gp('disparity_threshold').value)

        # --- runtime state (attributes the reused methods read) ---
        self.car_x = orig_x
        self.car_y = orig_y
        self.car_heading = 0.0
        self.curr_error = 0.0
        self.prev_error = 0.0
        self.processed_ranges = None

        self.latest_scan = None
        self.latest_scan_time = None
        self.latest_odom = None
        self.mode = 'pure_pursuit'

        # --- raceline (reuse PurePursuitNode loader) ---
        self.raceline = PurePursuitNode._load_raceline(
            self, gp('raceline_path').value, orig_x, orig_y
        )
        if self.raceline is None:
            self.get_logger().error(
                'pp_driver_plus: failed to load raceline; node will idle'
            )
        else:
            self.get_logger().info(
                f'pp_driver_plus: loaded raceline with {len(self.raceline["x"])} points'
            )

        # --- publishers ---
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10
        )
        self.pub_target_point = self.create_publisher(
            Marker, '/pp_plus/target_point', 10
        )
        self.pub_mode = self.create_publisher(String, '/pp_plus/mode', 10)

        # --- subscribers ---
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)

        # --- control timer ---
        self.create_timer(1.0 / control_rate_hz, self.control_loop)
        self.get_logger().info('pp_driver_plus started in pure_pursuit mode')

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_time = self.get_clock().now()

    # ------------------------------------------------------------------ #
    #  Control loop                                                       #
    # ------------------------------------------------------------------ #

    def control_loop(self) -> None:
        if self.raceline is None or self.latest_odom is None:
            return

        self._update_pose(self.latest_odom)

        # Pure-pursuit baseline. compute_pure_pursuit also calls
        # self.publish_target_point(...) which we override to draw the marker.
        steer_pp, speed_pp = PurePursuitNode.compute_pure_pursuit(self)

        scan = self.latest_scan
        scan_fresh = self._scan_is_fresh()
        min_ahead = self._min_range_in_cone(scan) if scan_fresh else float('inf')

        self._update_mode(min_ahead)

        if self.mode == 'gap_follow' and scan_fresh:
            steer_gap, speed_gap = self._compute_gap_command(scan)
            steer = steer_gap
            # Keep the more conservative speed so we never exceed the raceline
            # plan while reacting to a blocker.
            speed = min(speed_pp, speed_gap) if self.use_raceline_speed else speed_gap
        else:
            steer, speed = steer_pp, speed_pp

        steer = float(np.clip(steer, -self.max_steering_angle, self.max_steering_angle))
        speed = float(max(np.clip(speed, 0.0, self.max_speed), self.min_speed))

        self._publish_drive(speed, steer)
        self._publish_mode()

    def _update_pose(self, msg: Odometry) -> None:
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_heading = math.atan2(siny_cosp, cosy_cosp)

    def _scan_is_fresh(self) -> bool:
        if self.latest_scan is None or self.latest_scan_time is None:
            return False
        age_s = (self.get_clock().now() - self.latest_scan_time).nanoseconds * 1e-9
        return age_s < self.scan_timeout_s

    def _update_mode(self, min_ahead: float) -> None:
        if self.mode == 'pure_pursuit':
            if min_ahead < self.obstacle_enter_dist:
                self.mode = 'gap_follow'
                self.get_logger().info(
                    f'switch -> gap_follow (obstacle {min_ahead:.2f} m ahead)'
                )
        else:
            if min_ahead > self.obstacle_exit_dist:
                self.mode = 'pure_pursuit'
                self.get_logger().info(
                    f'switch -> pure_pursuit (clear {min_ahead:.2f} m ahead)'
                )

    # ------------------------------------------------------------------ #
    #  Obstacle detection                                                 #
    # ------------------------------------------------------------------ #

    def _min_range_in_cone(self, scan: LaserScan) -> float:
        """Minimum valid range within +/- forward_cone_half_angle of straight ahead."""
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return float('inf')
        angles = scan.angle_min + np.arange(ranges.size) * scan.angle_increment
        cone = ranges[np.abs(angles) <= self.forward_cone_half_angle]
        valid = cone[np.isfinite(cone) & (cone > scan.range_min) & (cone > 0.0)]
        if valid.size == 0:
            return float('inf')
        return float(np.min(valid))

    # ------------------------------------------------------------------ #
    #  Follow-the-gap (reuses ReactiveFollowGap methods, unbound)         #
    # ------------------------------------------------------------------ #

    def _compute_gap_command(self, scan: LaserScan) -> tuple[float, float]:
        ranges = np.array(scan.ranges)
        self.processed_ranges = ReactiveFollowGap.preprocess_lidar(
            self, ranges, scan.angle_increment
        )
        start_i, end_i = ReactiveFollowGap.find_max_gap(self)
        best_point = ReactiveFollowGap.find_best_point(self, start_i, end_i)
        if best_point is None:
            return 0.0, self.min_speed

        steer = (scan.angle_min + best_point * scan.angle_increment) * self.steering_sensitivity

        point_dist = float(ranges[best_point])
        if not math.isfinite(point_dist):
            point_dist = self.max_distance_threshold
        speed = min(float(self.velocity * point_dist / 4.0), self.max_velocity)

        point_rad = scan.angle_min + best_point * scan.angle_increment
        target_x = point_dist * math.cos(point_rad + self.car_heading) + self.car_x
        target_y = point_dist * math.sin(point_rad + self.car_heading) + self.car_y
        self._publish_target_marker(target_x, target_y, 'gap_follow')

        return steer, speed

    # ------------------------------------------------------------------ #
    #  Publishing                                                         #
    # ------------------------------------------------------------------ #

    def publish_target_point(self, x: float, y: float) -> None:
        """Override of PurePursuitNode.publish_target_point.

        Called from the reused compute_pure_pursuit; draws the lookahead target
        on this node's own debug topic, coloured by the current mode.
        """
        self._publish_target_marker(x, y, self.mode)

    def _publish_target_marker(self, x: float, y: float, mode: str) -> None:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pp_plus_target'
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        if mode == 'gap_follow':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
        else:
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 1.0
        marker.color.a = 1.0

        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = 0.0
        marker.points.append(p)
        self.pub_target_point.publish(marker)

    def _publish_drive(self, speed: float, steering_angle: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering_angle)
        self.pub_drive.publish(msg)

    def _publish_mode(self) -> None:
        msg = String()
        msg.data = self.mode
        self.pub_mode.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PpDriverPlusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

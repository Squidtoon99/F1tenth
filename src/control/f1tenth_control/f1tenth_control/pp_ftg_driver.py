import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np
import math


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------
STATE_RACING   = 'RACING'    # Normal pure pursuit on raceline
STATE_AVOIDING = 'AVOIDING'  # Follow-the-gap avoidance active
STATE_RETURN   = 'RETURN'    # Smoothly blending back to raceline


class PurePursuitFTGNode(Node):
    """
    Pure Pursuit + Follow-the-Gap obstacle avoidance.

    State machine:
        RACING   -> AVOIDING  : obstacle detected within safety bubble
        AVOIDING -> RETURN    : no obstacle in front for N consecutive scans
        RETURN   -> RACING    : cross-track error back below return threshold

    During AVOIDING  : steering comes entirely from FTG
    During RETURN    : steering is a weighted blend (FTG weight decays to 0)
    During RACING    : steering comes entirely from pure pursuit
    """

    def __init__(self):
        super().__init__('pure_pursuit_ftg_node')

        # ------------------------------------------------------------------ #
        # Parameters                                                           #
        # ------------------------------------------------------------------ #
        self.declare_parameter('lookahead_distance',   1.2)
        self.declare_parameter('wheelbase',            0.33)
        self.declare_parameter('max_steering_angle',   0.5)
        self.declare_parameter('speed_scale',          0.8)
        self.declare_parameter('min_speed',            0.8)
        self.declare_parameter('max_speed',           15.0)
        self.declare_parameter('raceline_path',
            '/sim_ws/src/f1tenth_gym_ros/maps/IV_26_2.csv')
        self.declare_parameter('use_raceline_speed',   True)

        # Follow-the-gap parameters
        self.declare_parameter('bubble_radius',        0.35)   # m  - robot half-width + margin
        self.declare_parameter('obstacle_dist_thresh', 2.5)    # m  - trigger avoidance
        self.declare_parameter('ftg_lookahead',        1.5)    # m  - FTG target distance
        self.declare_parameter('ftg_speed',            1.5)    # m/s while avoiding
        self.declare_parameter('ftg_min_gap_width',    0.6)    # m  - min viable gap
        self.declare_parameter('clear_scan_count',     5)      # consecutive clear scans to exit AVOIDING
        self.declare_parameter('return_error_thresh',  0.15)   # m  - CTE to exit RETURN

        self.ld                 = self.get_parameter('lookahead_distance').value
        self.L                  = self.get_parameter('wheelbase').value
        self.max_steer          = self.get_parameter('max_steering_angle').value
        self.speed_scale        = self.get_parameter('speed_scale').value
        self.min_speed          = self.get_parameter('min_speed').value
        self.max_speed          = self.get_parameter('max_speed').value
        self.use_raceline_speed = self.get_parameter('use_raceline_speed').value
        raceline_path           = self.get_parameter('raceline_path').value

        self.bubble_radius        = self.get_parameter('bubble_radius').value
        self.obstacle_dist_thresh = self.get_parameter('obstacle_dist_thresh').value
        self.ftg_lookahead        = self.get_parameter('ftg_lookahead').value
        self.ftg_speed            = self.get_parameter('ftg_speed').value
        self.ftg_min_gap_width    = self.get_parameter('ftg_min_gap_width').value
        self.clear_scan_count     = self.get_parameter('clear_scan_count').value
        self.return_error_thresh  = self.get_parameter('return_error_thresh').value

        # ------------------------------------------------------------------ #
        # Load raceline                                                        #
        # ------------------------------------------------------------------ #
        self.raceline = self._load_raceline(raceline_path)
        if self.raceline is None:
            self.get_logger().error(f'Failed to load raceline from {raceline_path}')
            return
        self.get_logger().info(
            f'Loaded raceline with {len(self.raceline["x"])} points')

        # ------------------------------------------------------------------ #
        # State                                                                #
        # ------------------------------------------------------------------ #
        self.car_x       = 0.0
        self.car_y       = 0.0
        self.car_heading = 0.0

        self.curr_error  = 0.0
        self.prev_error  = 0.0

        # FTG / state-machine state
        self.mode              = STATE_RACING
        self.clear_count       = 0       # consecutive obstacle-free scans
        self.return_blend      = 0.0     # 0 = full FTG, 1 = full pure pursuit
        self.latest_scan       = None    # most recent LaserScan message
        self.ftg_steering      = 0.0    # last FTG steering output (used in RETURN blend)

        # ------------------------------------------------------------------ #
        # Subscribers                                                          #
        # ------------------------------------------------------------------ #
        self.sub_odom = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_cb,
            10
        )
        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        # ------------------------------------------------------------------ #
        # Publishers                                                           #
        # ------------------------------------------------------------------ #
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        self.pub_raceline_points = self.create_publisher(
            Marker, '/raceline_points', 10)
        self.pub_target_point = self.create_publisher(
            Marker, '/target_point', 10)
        self.pub_gap_viz = self.create_publisher(
            MarkerArray, '/ftg_debug', 10)

        self.publish_raceline()
        self.get_logger().info('Pure Pursuit + FTG node started')

    # ======================================================================= #
    # Raceline helpers (unchanged from base)                                   #
    # ======================================================================= #

    def _load_raceline(self, path):
        try:
            data = np.genfromtxt(
                path,
                delimiter=',',
                skip_header=1,
                dtype=np.float32
            )
            return {
                's':     data[:, 0],
                'x':     data[:, 1],
                'y':     data[:, 2],
                'psi':   data[:, 3],
                'kappa': data[:, 4],
                'vx':    data[:, 5],
                'ax':    data[:, 6],
            }
        except Exception as e:
            self.get_logger().error(f'Error loading raceline: {e}')
            return None

    def _find_closest_idx(self):
        """Signed cross-track error + closest raceline index."""
        x = self.raceline['x']
        y = self.raceline['y']
        n = len(x)

        pts         = np.stack([x, y], axis=-1)
        car_pos     = np.array([self.car_x, self.car_y])
        dists       = np.linalg.norm(pts - car_pos, axis=1)
        closest_idx = int(np.argmin(dists))

        next_idx = (closest_idx + 1) % n
        tangent  = np.array([
            x[next_idx] - x[closest_idx],
            y[next_idx] - y[closest_idx],
        ])
        tangent_len = np.linalg.norm(tangent)
        if tangent_len > 1e-6:
            tangent /= tangent_len

        to_car = car_pos - pts[closest_idx]
        self.curr_error = float(tangent[0] * to_car[1] - tangent[1] * to_car[0])

        return closest_idx

    def _find_lookahead_point(self):
        """Interpolated lookahead point on raceline."""
        n           = len(self.raceline['x'])
        closest_idx = self._find_closest_idx()
        car_pos     = np.array([self.car_x, self.car_y])
        x           = self.raceline['x']
        y           = self.raceline['y']
        vx          = self.raceline['vx']

        for i in range(n):
            idx      = (closest_idx + i) % n
            next_idx = (closest_idx + i + 1) % n

            p0 = np.array([x[idx],      y[idx]])
            p1 = np.array([x[next_idx], y[next_idx]])
            seg     = p1 - p0
            seg_len = np.linalg.norm(seg)

            if seg_len < 1e-6:
                continue

            d0 = np.linalg.norm(p0 - car_pos)
            d1 = np.linalg.norm(p1 - car_pos)

            if d1 >= self.ld:
                if d0 < self.ld:
                    f            = p0 - car_pos
                    a            = np.dot(seg, seg)
                    b            = 2.0 * np.dot(f, seg)
                    c            = np.dot(f, f) - self.ld ** 2
                    discriminant = b * b - 4 * a * c
                    if discriminant >= 0:
                        t         = (-b + math.sqrt(discriminant)) / (2.0 * a)
                        t         = float(np.clip(t, 0.0, 1.0))
                        pt        = p0 + t * seg
                        target_vx = vx[idx] + t * (vx[next_idx] - vx[idx])
                        return float(pt[0]), float(pt[1]), float(target_vx)
                return float(x[next_idx]), float(y[next_idx]), float(vx[next_idx])

        return float(x[closest_idx]), float(y[closest_idx]), float(vx[closest_idx])

    # ======================================================================= #
    # Pure pursuit steering                                                    #
    # ======================================================================= #

    def _pure_pursuit_steering(self):
        """Return (steering_angle, speed) from raceline."""
        target_x, target_y, target_vx = self._find_lookahead_point()
        self.publish_target_point(target_x, target_y)

        dx    = target_x - self.car_x
        dy    = target_y - self.car_y
        alpha = math.atan2(dy, dx) - self.car_heading
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        steering = math.atan2(2.0 * self.L * math.sin(alpha), self.ld)
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        if self.use_raceline_speed:
            speed = float(np.clip(
                target_vx * self.speed_scale,
                self.min_speed,
                self.max_speed
            ))
        else:
            speed = self.min_speed

        self.prev_error = self.curr_error
        return steering, speed

    # ======================================================================= #
    # Follow-the-Gap                                                           #
    # ======================================================================= #

    def _obstacle_in_front(self, ranges, angles, range_max):
        """
        Returns True if any point within obstacle_dist_thresh lies in the
        forward-facing FOV (±90 deg).
        """
        front_mask = np.abs(angles) < (math.pi / 2.0)
        front_ranges = ranges[front_mask]
        valid = front_ranges[(front_ranges > 0.01) & (front_ranges < range_max)]
        if len(valid) == 0:
            return False
        return bool(np.min(valid) < self.obstacle_dist_thresh)

    def _follow_the_gap(self, scan: LaserScan):
        """
        Classic Follow-the-Gap algorithm.
        Returns (steering_angle, speed).

        Steps:
          1. Pre-process scan — clip to max range, zero out invalid returns.
          2. Find closest point, zero out a bubble of indices around it.
          3. Find the longest free gap (run of non-zero ranges).
          4. Aim for the furthest point inside that gap.
          5. Convert gap heading to steering angle.
        """
        ranges = np.array(scan.ranges, dtype=np.float64)
        n      = len(ranges)

        angle_min  = scan.angle_min
        angle_inc  = scan.angle_increment
        range_max  = scan.range_max
        angles     = angle_min + np.arange(n) * angle_inc

        # -- 1. Pre-process ------------------------------------------------ #
        ranges = np.where(np.isfinite(ranges), ranges, range_max)
        ranges = np.clip(ranges, 0.0, range_max)

        # -- 2. Safety bubble ---------------------------------------------- #
        closest_idx = int(np.argmin(ranges))
        closest_dist = ranges[closest_idx]

        # Angular width of bubble at closest distance (arc = radius / dist)
        if closest_dist > 0.01:
            bubble_half_angle = math.asin(
                min(self.bubble_radius / closest_dist, 1.0))
            bubble_half_idx   = int(math.ceil(bubble_half_angle / angle_inc))
        else:
            bubble_half_idx = n // 4

        lo = max(0,     closest_idx - bubble_half_idx)
        hi = min(n - 1, closest_idx + bubble_half_idx)
        ranges[lo:hi + 1] = 0.0

        # -- 3. Longest gap ------------------------------------------------- #
        # A "free" beam is one with range > ftg_min_gap_width
        free = (ranges > self.ftg_min_gap_width).astype(int)
        best_start, best_end = 0, 0
        cur_start = None

        for i in range(n):
            if free[i]:
                if cur_start is None:
                    cur_start = i
            else:
                if cur_start is not None:
                    if (i - cur_start) > (best_end - best_start):
                        best_start, best_end = cur_start, i - 1
                    cur_start = None
        # handle gap running to end of array
        if cur_start is not None:
            if (n - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, n - 1

        if best_end <= best_start:
            # No usable gap — steer straight and slow down
            self.get_logger().warn('FTG: no usable gap found, going straight')
            return 0.0, self.ftg_speed * 0.5

        # -- 4. Target: furthest point inside best gap --------------------- #
        gap_ranges  = ranges[best_start:best_end + 1]
        best_in_gap = int(np.argmax(gap_ranges))
        target_idx  = best_start + best_in_gap
        target_angle = angles[target_idx]

        # -- 5. Steering ---------------------------------------------------- #
        # target_angle is in the lidar frame; lidar is assumed forward-mounted
        # so this is directly the heading error
        steering = math.atan2(
            2.0 * self.L * math.sin(target_angle),
            self.ftg_lookahead
        )
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        self._publish_gap_viz(angles, ranges, best_start, best_end,
                              target_idx, range_max)

        return steering, float(self.ftg_speed)

    # ======================================================================= #
    # State machine                                                            #
    # ======================================================================= #

    def _update_state(self, scan: LaserScan):
        """Transition the state machine based on scan and CTE."""
        ranges    = np.array(scan.ranges, dtype=np.float64)
        n         = len(ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        angles    = angle_min + np.arange(n) * angle_inc
        range_max = scan.range_max

        ranges = np.where(np.isfinite(ranges), ranges, range_max)
        obstacle_present = self._obstacle_in_front(ranges, angles, range_max)

        if self.mode == STATE_RACING:
            if obstacle_present:
                self.mode        = STATE_AVOIDING
                self.clear_count = 0
                self.get_logger().info('→ AVOIDING')

        elif self.mode == STATE_AVOIDING:
            if not obstacle_present:
                self.clear_count += 1
                if self.clear_count >= self.clear_scan_count:
                    self.mode         = STATE_RETURN
                    self.return_blend = 0.0   # start at full FTG
                    self.get_logger().info('→ RETURN')
            else:
                self.clear_count = 0

        elif self.mode == STATE_RETURN:
            if obstacle_present:
                # obstacle came back — revert to avoiding
                self.mode        = STATE_AVOIDING
                self.clear_count = 0
                self.get_logger().info('→ AVOIDING (re-triggered)')
            else:
                # advance blend toward pure pursuit
                self.return_blend = min(1.0, self.return_blend + 0.05)
                if (self.return_blend >= 1.0
                        and abs(self.curr_error) < self.return_error_thresh):
                    self.mode = STATE_RACING
                    self.get_logger().info('→ RACING')

    # ======================================================================= #
    # Main control loop                                                        #
    # ======================================================================= #

    def odom_cb(self, msg):
        self.car_x   = msg.pose.pose.position.x
        self.car_y   = msg.pose.pose.position.y

        q            = msg.pose.pose.orientation
        siny_cosp    = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp    = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_heading = math.atan2(siny_cosp, cosy_cosp)

        if self.latest_scan is not None:
            self._update_state(self.latest_scan)

        steering, speed = self._compute_control()
        self.publish_drive(speed, steering)

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _compute_control(self):
        """Blend FTG and pure pursuit according to current state."""
        pp_steer, pp_speed = self._pure_pursuit_steering()

        if self.mode == STATE_RACING:
            return pp_steer, pp_speed

        if self.latest_scan is None:
            return pp_steer, pp_speed

        ftg_steer, ftg_speed = self._follow_the_gap(self.latest_scan)
        self.ftg_steering    = ftg_steer

        if self.mode == STATE_AVOIDING:
            return ftg_steer, ftg_speed

        # STATE_RETURN: linear blend, return_blend goes 0→1
        w       = self.return_blend          # weight on pure pursuit
        steer   = (1.0 - w) * ftg_steer + w * pp_steer
        speed   = (1.0 - w) * ftg_speed + w * pp_speed
        return float(steer), float(speed)

    # ======================================================================= #
    # Publishers                                                               #
    # ======================================================================= #

    def publish_drive(self, speed, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.header.frame_id      = 'base_link'
        msg.drive.speed          = float(speed)
        msg.drive.steering_angle = float(np.clip(
            steering_angle, -self.max_steer, self.max_steer))
        self.pub_drive.publish(msg)

    def publish_raceline(self):
        marker                 = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = 'raceline'
        marker.id              = 0
        marker.type            = Marker.POINTS
        marker.action          = Marker.ADD
        marker.scale.x         = 0.10
        marker.scale.y         = 0.10
        marker.color.r         = 1.0
        marker.color.g         = 0.0
        marker.color.b         = 1.0
        marker.color.a         = 1.0

        for x, y in np.stack([self.raceline['x'], self.raceline['y']], axis=-1):
            p     = Point()
            p.x   = float(x)
            p.y   = float(y)
            p.z   = 0.0
            marker.points.append(p)

        self.pub_raceline_points.publish(marker)

    def publish_target_point(self, x, y):
        marker                 = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = 'target'
        marker.id              = 1
        marker.type            = Marker.SPHERE
        marker.action          = Marker.ADD
        marker.scale.x         = 0.20
        marker.scale.y         = 0.20
        marker.scale.z         = 0.20

        # colour indicates mode
        if self.mode == STATE_RACING:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        elif self.mode == STATE_AVOIDING:
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        else:
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        self.pub_target_point.publish(marker)

    def _publish_gap_viz(self, angles, ranges, gap_start, gap_end,
                         target_idx, range_max):
        """Publish MarkerArray showing the chosen gap and target beam."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # -- gap beams (blue) --
        gap_marker              = Marker()
        gap_marker.header.frame_id = 'laser'
        gap_marker.header.stamp    = now
        gap_marker.ns              = 'ftg_gap'
        gap_marker.id              = 0
        gap_marker.type            = Marker.LINE_LIST
        gap_marker.action          = Marker.ADD
        gap_marker.scale.x         = 0.02
        gap_marker.color           = ColorRGBA(r=0.0, g=0.4, b=1.0, a=0.6)
        gap_marker.pose.orientation.w = 1.0

        for i in range(gap_start, gap_end + 1, max(1, (gap_end - gap_start) // 40)):
            r = min(ranges[i], range_max)
            if r < 0.01:
                continue
            origin = Point(x=0.0, y=0.0, z=0.0)
            end    = Point(
                x=float(r * math.cos(angles[i])),
                y=float(r * math.sin(angles[i])),
                z=0.0
            )
            gap_marker.points.append(origin)
            gap_marker.points.append(end)

        arr.markers.append(gap_marker)

        # -- target beam (cyan) --
        tgt_marker              = Marker()
        tgt_marker.header.frame_id = 'laser'
        tgt_marker.header.stamp    = now
        tgt_marker.ns              = 'ftg_target'
        tgt_marker.id              = 1
        tgt_marker.type            = Marker.ARROW
        tgt_marker.action          = Marker.ADD
        tgt_marker.scale.x         = 0.05
        tgt_marker.scale.y         = 0.10
        tgt_marker.scale.z         = 0.10
        tgt_marker.color           = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)
        tgt_marker.pose.orientation.w = 1.0

        r = min(ranges[target_idx], range_max)
        origin = Point(x=0.0, y=0.0, z=0.0)
        end    = Point(
            x=float(r * math.cos(angles[target_idx])),
            y=float(r * math.sin(angles[target_idx])),
            z=0.0
        )
        tgt_marker.points.append(origin)
        tgt_marker.points.append(end)
        arr.markers.append(tgt_marker)

        self.pub_gap_viz.publish(arr)


# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitFTGNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
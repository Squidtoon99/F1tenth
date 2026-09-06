import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np
import math


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('robot_namespace', '/ego_racecar')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')

        # --- parameters ---
        self.declare_parameter('lookahead_distance', 1.2)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.4)
        self.declare_parameter('speed_scale', 1.0)       # scale factor for raceline speeds
        self.declare_parameter('min_speed', 0.8)         # minimum speed m/s
        self.declare_parameter('max_speed', 5.0)         # maximum speed m/s
        self.declare_parameter('min_accel', -8.0)         # minimum speed m/s/s
        self.declare_parameter('max_accel', 8.0)         # maximum speed m/s/s
        self.declare_parameter('raceline_path', '/config/maps/raceline.csv')
        self.declare_parameter('use_raceline_speed', True)
        self.declare_parameter('origin_x', 0)
        self.declare_parameter('origin_y', 0)

        self.robot_namespace = self.get_parameter('robot_namespace').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value

        self.ld                = self.get_parameter('lookahead_distance').value
        self.L                 = self.get_parameter('wheelbase').value
        self.max_steer         = self.get_parameter('max_steering_angle').value
        self.speed_scale       = self.get_parameter('speed_scale').value
        self.min_speed         = self.get_parameter('min_speed').value
        self.max_speed         = self.get_parameter('max_speed').value
        self.min_accel         = self.get_parameter('min_accel').value
        self.max_accel         = self.get_parameter('max_accel').value
        self.use_raceline_speed = self.get_parameter('use_raceline_speed').value
        raceline_path          = self.get_parameter('raceline_path').value
        orig_x = self.get_parameter('origin_x').value
        orig_y = self.get_parameter('origin_y').value

        # --- load raceline ---
        self.raceline = self._load_raceline(raceline_path, orig_x, orig_y)
        if self.raceline is None:
            self.get_logger().error(f'Failed to load raceline from {raceline_path}')
            return
        self.get_logger().info(f'Loaded raceline with {len(self.raceline["x"])} points')

        # --- state ---
        self.car_x       = orig_x
        self.car_y       = orig_y
        self.car_heading = 0.0

        self.curr_error = 0.0
        self.prev_error = 0.0

        # --- subscribers ---
        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            10
        )

        # --- publisher ---
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        self.pub_raceline_points = self.create_publisher(
            Marker,
            '/raceline_points',
            10
        )

        self.pub_target_point = self.create_publisher(
            Marker,
            '/target_point',
            10
        )

        self.publish_raceline()
        self.get_logger().info('Pure pursuit node started')

    def _load_raceline(self, path, orig_x, orig_y):
        """
        Load raceline CSV from f1tenth_racetracks format:
            s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        File has 3 header rows before data, semicolon delimited.
        """
        try:
            data = np.genfromtxt(
                path,
                delimiter=',',
                skip_header=1,
                dtype=np.float32    
            )

            data[:, 1] += orig_x
            data[:, 2] += orig_y
            
            return {
                's':     data[:, 0],  # arc length ms
                'x':     data[:, 1],  # x position m
                'y':     data[:, 2],  # y position m
                'psi':   data[:, 3],  # heading rad
                'kappa': data[:, 4],  # curvature rad/m
                'vx':    data[:, 5],  # target speed m/s
                'ax':    data[:, 6],  # target acceleration m/s^2
            }
        except Exception as e:
            self.get_logger().error(f'Error loading raceline: {e}')
            return None

    def _find_closest_idx(self):
        """Find the index of the closest raceline point to the car."""
        pts  = np.stack([self.raceline['x'], self.raceline['y']], axis=-1)
        dist = np.linalg.norm(pts - np.array([self.car_x, self.car_y]), axis=1)
        self.curr_error = np.min(dist)
        return np.argmin(dist)

    def _find_lookahead_point(self):
        """
        Walk forward along the raceline from the closest point and
        interpolate to find the exact point at lookahead distance ld.
        Returns (target_x, target_y, target_vx).
        """
        n           = len(self.raceline['x'])
        closest_idx = self._find_closest_idx()
        car_pos     = np.array([self.car_x, self.car_y])
        x           = self.raceline['x']
        y           = self.raceline['y']
        vx          = self.raceline['vx']
        ax          = self.raceline['ax']

        for i in range(n):
            idx      = (closest_idx + i) % n
            next_idx = (closest_idx + i + 1) % n

            p0 = np.array([x[idx],      y[idx]])
            p1 = np.array([x[next_idx], y[next_idx]])
            seg     = p1 - p0
            seg_len = np.linalg.norm(seg)

            if seg_len < 1e-6:
                continue

            d1 = np.linalg.norm(p1 - car_pos)
            d0 = np.linalg.norm(p0 - car_pos)

            if d1 >= self.ld:
                if d0 < self.ld:
                    # interpolate exact lookahead point
                    f = p0 - car_pos
                    a = np.dot(seg, seg)
                    b = 2.0 * np.dot(f, seg)
                    c = np.dot(f, f) - self.ld ** 2
                    discriminant = b * b - 4 * a * c

                    if discriminant >= 0:
                        t      = (-b + math.sqrt(discriminant)) / (2.0 * a)
                        t      = np.clip(t, 0.0, 1.0)
                        pt     = p0 + t * seg
                        target_vx = vx[idx] + t * (vx[next_idx] - vx[idx])
                        return pt[0], pt[1], float(target_vx)

                return x[next_idx], y[next_idx], float(vx[next_idx])

        # fallback
        return x[closest_idx], y[closest_idx], float(ax[closest_idx])

    def _get_target_speed(self, closest_idx):
        """
        Get target speed at the current position, scaled and clamped.
        """
        vx = float(self.raceline['vx'][closest_idx]) * self.speed_scale
        return float(np.clip(vx, self.min_speed, self.max_speed))

    def odom_cb(self, msg):
        self.car_x     = msg.pose.pose.position.x
        self.car_y     = msg.pose.pose.position.y

        q         = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_heading = math.atan2(siny_cosp, cosy_cosp)

        steering, accel = self.compute_pure_pursuit()
        self.publish_drive(accel, steering)

    def compute_pure_pursuit(self):
        target_x, target_y, target_ax = self._find_lookahead_point()
        self.publish_target_point(target_x, target_y)
        # self.publish_raceline()

        # angle to target relative to car heading
        dx    = target_x - self.car_x
        dy    = target_y - self.car_y

        alpha = math.atan2(dy, dx) - self.car_heading
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # normalize

        # pure pursuit formula
        steering = math.atan2(2.0 * self.L * math.sin(alpha), self.ld)

        K_p = 0.8
        K_i = 0.0
        K_d = 1.2
        
        p_term = K_p * self.curr_error
        # i_term = K_i * accumulated_error
        d_term = K_d * (self.curr_error - self.prev_error)

        output = p_term + d_term # + i_term

        # speed from raceline or fixed
        if self.use_raceline_speed:
            accel = float(np.clip(
                target_ax * self.speed_scale,
                self.min_accel,
                self.max_accel
            ))
        else:
            accel = np.min(self.min_speed, target_ax * self.speed_scale)

        self.prev_error = self.curr_error
        return steering, accel

    def publish_drive(self, accel, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed          = float(0.0)
        msg.drive.steering_angle = float(steering_angle)
        msg.drive.acceleration   = float(accel)
        self.pub_drive.publish(msg)
        # self.publish_raceline()


    def publish_raceline(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'raceline'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        for x, y in np.stack([self.raceline['x'], self.raceline['y']], axis=-1):
            self.get_logger().info(f'x: {x}, y: {y}')
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.0
            marker.points.append(p)

        self.pub_raceline_points.publish(marker)

    def publish_target_point(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'target'
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = 0.0
        marker.points.append(p)

        self.pub_target_point.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

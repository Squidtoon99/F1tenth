import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import numpy as np
import math


class SafetyNode(Node):
    """
    F1Tenth automatic safety node.

    Behaviors:
      - Emergency stop: full brake effort if TTC drops below threshold
      - Progressive braking based on proximity of nearest obstacle
      - Publishes force/brake overrides on /brake (Ackermann acceleration)
    """

    def __init__(self):
        super().__init__('safety_node')

        # --- parameters ---
        self.declare_parameter('ttc_threshold',        0.3)   # s  — emergency stop
        self.declare_parameter('ttc_slow_threshold',   1.0)   # s  — begin braking
        self.declare_parameter('min_obstacle_dist',    0.3)   # m  — hard stop distance
        self.declare_parameter('slow_obstacle_dist',   1.0)   # m  — begin braking distance
        self.declare_parameter('imu_accel_threshold', 15.0)   # m/s^2 — crash detection
        self.declare_parameter('scan_angle_range',     1.5)   # rad — forward cone
        self.declare_parameter('max_speed',            7.0)   # m/s — overspeed brake

        self.ttc_threshold       = self.get_parameter('ttc_threshold').value
        self.ttc_slow_threshold  = self.get_parameter('ttc_slow_threshold').value
        self.min_obstacle_dist   = self.get_parameter('min_obstacle_dist').value
        self.slow_obstacle_dist  = self.get_parameter('slow_obstacle_dist').value
        self.imu_accel_threshold = self.get_parameter('imu_accel_threshold').value
        self.scan_angle_range    = self.get_parameter('scan_angle_range').value
        self.max_speed           = float(self.get_parameter('max_speed').value)

        # --- state ---
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.current_acceleration = 0.0
        self.latest_scan = None
        self.latest_imu = None
        self.estop_active = False
        # VESC /odom twist sign (matches vehicle_obs twist_vx_sign).
        self.declare_parameter('odom_vx_sign', -1.0)
        self.odom_vx_sign = float(self.get_parameter('odom_vx_sign').value)
        # Force/current brake effort when slowing or estopping (ADR 0006).
        self.declare_parameter('estop_acceleration', -1.0)
        self.estop_acceleration = float(self.get_parameter('estop_acceleration').value)

        # --- subscribers ---
        self.sub_scan  = self.create_subscription(LaserScan, '/scan',              self.scan_cb,  10)
        self.sub_odom  = self.create_subscription(Odometry,  '/odom',  self.odom_cb,  10)
        self.sub_imu   = self.create_subscription(Imu,       '/imu',               self.imu_cb,   10)
        self.sub_drive = self.create_subscription(
            AckermannDriveStamped, '/drive', self.drive_cb, 10
        )

        # --- publishers ---
        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/brake', 10)
        # self.pub_estop = self.create_publisher(Bool, '/brake', 10)

        # --- timer: safety check at 50Hz ---
        self.timer = self.create_timer(0.02, self.safety_cb)

        self.get_logger().info('Safety node started')

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry):
        self.current_speed = abs(
            self.odom_vx_sign * float(msg.twist.twist.linear.x)
        )

    def imu_cb(self, msg: Imu):
        self.latest_imu = msg
        # check for sudden impact / crash
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        total_accel = math.sqrt(ax**2 + ay**2)
        if total_accel > self.imu_accel_threshold:
            self._trigger_estop(f'IMU impact detected: {total_accel:.1f} m/s^2')

    def drive_cb(self, msg: AckermannDriveStamped):
        # store the commanded steering / longitudinal effort so we can override
        self.current_steering = msg.drive.steering_angle
        self.current_acceleration = float(msg.drive.acceleration)
        # speed field is unused in current mode; keep abs for legacy callers
        self.current_speed = max(self.current_speed, abs(float(msg.drive.speed)))

    # ------------------------------------------------------------------ #
    #  Safety logic                                                        #
    # ------------------------------------------------------------------ #

    def _compute_ttc(self) -> tuple[float, float]:
        """
        Compute minimum TTC and minimum distance in the forward scan cone.
        Returns (min_ttc, min_dist).
        """
        if self.latest_scan is None:
            return float('inf'), float('inf')

        scan      = self.latest_scan
        ranges    = np.array(scan.ranges)
        angles    = (scan.angle_min +
                     np.arange(len(ranges)) * scan.angle_increment)

        # forward cone only
        mask   = np.abs(angles) <= self.scan_angle_range / 2.0
        ranges = ranges[mask]
        angles = angles[mask]

        # filter invalid readings
        valid  = np.isfinite(ranges) & (ranges > 0.01) & (ranges < scan.range_max)
        if not np.any(valid):
            return float('inf'), float('inf')

        ranges = ranges[valid]
        angles = angles[valid]

        # range rate: how fast each point is approaching
        # r_dot = -v * cos(angle)  (negative = closing)
        speed     = abs(self.current_speed)
        range_dot = -speed * np.cos(angles)  # positive = closing

        # TTC = range / range_rate (only for closing points)
        closing   = range_dot > 0.01
        if not np.any(closing):
            return float('inf'), np.min(ranges)

        ttc = np.where(closing, ranges / range_dot, float('inf'))
        return float(np.min(ttc)), float(np.min(ranges))

    def _trigger_estop(self, reason: str = ''):
        self.estop_active = True
        if reason:
            self.get_logger().warn(f'ESTOP: {reason}')

    def _safe_acceleration(self, min_ttc: float, min_dist: float) -> float | None:
        """Return brake effort in [-1, 0], or None if no override needed."""
        if self.estop_active:
            return self.estop_acceleration

        # Scale braking from coast (0) to full brake (-1) as hazards approach.
        brake = 0.0
        if min_ttc < self.ttc_slow_threshold and self.ttc_slow_threshold > self.ttc_threshold:
            t = (min_ttc - self.ttc_threshold) / (
                self.ttc_slow_threshold - self.ttc_threshold
            )
            brake = max(brake, 1.0 - float(np.clip(t, 0.0, 1.0)))
        if min_dist < self.slow_obstacle_dist and self.slow_obstacle_dist > self.min_obstacle_dist:
            t = (min_dist - self.min_obstacle_dist) / (
                self.slow_obstacle_dist - self.min_obstacle_dist
            )
            brake = max(brake, 1.0 - float(np.clip(t, 0.0, 1.0)))
        # Overspeed: request full brake when measured speed exceeds the limit.
        if self.max_speed > 0.0 and self.current_speed > self.max_speed:
            brake = max(brake, 1.0)

        if brake <= 0.0:
            return None
        return -brake

    def safety_cb(self):
        min_ttc, min_dist = self._compute_ttc()

        # --- emergency stop checks ---
        if min_ttc <= self.ttc_threshold:
            self._trigger_estop('TTC threshold')
        elif min_dist <= self.min_obstacle_dist:
            self._trigger_estop('obstacle distance')
        else:
            if self.estop_active:
                self.get_logger().info('ESTOP cleared')
            self.estop_active = False

        accel = self._safe_acceleration(min_ttc, min_dist)
        if accel is None:
            return
        # Only override when requesting more braking than the policy already asked.
        if accel >= self.current_acceleration:
            return

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = float(self.current_steering)
        drive_msg.drive.acceleration = float(accel)
        self.pub_drive.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
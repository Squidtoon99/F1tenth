import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import logging
import numpy as np
import math
import time

class PidNode(Node):
    def __init__(self):
        super().__init__('pid_driver_node')
        self.get_logger().info("PID Driver starting")

        self.curr_error = 0.0
        self.prev_error = 0.0
        self.steering = 0.0

        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            10
        )

        self.sub_scan = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
            10
        )

        self.pub_drive = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        
        self.timer = self.create_timer(0.1, self.publish_drive)
        
        # self.pub_future_points = self.create_publisher(
        #     MarkerArray,
        #     '/future_track_points',
        #     10
        # )

        time.sleep(3.0)

    def scan_cb(self, msg):
        try:
            offset_a = int((msg.angle_max - 1.57) / msg.angle_increment ) 
            offset_b = 180
            theta = offset_b * msg.angle_increment

            dist_a = msg.ranges[offset_a]
            dist_b = msg.ranges[offset_a + offset_b]
            
            alpha = math.atan((dist_a * math.cos(theta) - dist_b) / (dist_a * math.sin(theta)))
            L = 0.05
            d_t1 = dist_b * math.cos(alpha) + L * math.sin(alpha)
            
            desired_dist = 0.5
            self.curr_error = desired_dist - d_t1

            self.steering = self.calc_steering()

            self.prev_error = self.curr_error

        except Exception:
            self.steering = 0.0
        
    def calc_steering(self):
        K_p = 2.0
        K_i = 0.0
        K_d = 3.0
        
        p_term = K_p * self.curr_error
        # i_term = K_i * accumulated_error
        d_term = K_d * (self.curr_error - self.prev_error)

        output = p_term + d_term # + i_term
        return float(np.clip(output, -0.4, 0.4))

    def odom_cb(self, msg):
        pass

    def publish_drive(self):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = self.steering
        msg.drive.speed = 1.0
        if (math.fabs(self.steering) * 180 / math.pi) <= 10:
            msg.drive.speed = 2.0
        elif (math.fabs(self.steering) * 180 / math.pi) <= 20:
            msg.drive.speed = 1.5
        
        # self.get_logger().info(f'steering: {msg.drive.speed}')
        self.pub_drive.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    logging.basicConfig(level=logging.INFO)
    node = PidNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

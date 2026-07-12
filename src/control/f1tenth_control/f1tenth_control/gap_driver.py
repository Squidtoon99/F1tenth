#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')

        self.declare_parameter('robot_namespace', '/ego_racecar')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')

        self.declare_parameter('velocity', 1.0)
        self.declare_parameter('max_velocity', 5.0)
        self.declare_parameter('max_distance_threshold', 5.0)
        self.declare_parameter('min_distance_threshold', 1.1) # 0.9m roughly the turning radius of the car
        self.declare_parameter('max_steering_angle', np.radians(40.0))
        self.declare_parameter('steering_sensitivity', 1.0)
        self.declare_parameter('threshold', 5.0)
        self.declare_parameter('bubble_radius', 50)
        self.declare_parameter('disparity_radius_m', 0.15)
        self.declare_parameter('disparity_threshold', 1.0)


        self.robot_namespace = self.get_parameter('robot_namespace').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value

        self.velocity = self.get_parameter('velocity').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_distance_threshold = self.get_parameter('max_distance_threshold').value
        self.min_distance_threshold = self.get_parameter('min_distance_threshold').value 
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.steering_sensitivity = self.get_parameter('steering_sensitivity').value
        self.threshold = self.get_parameter('threshold').value
        self.bubble_radius = self.get_parameter('bubble_radius').value
        self.disparity_radius_m = self.get_parameter('disparity_radius_m').value
        self.disparity_threshold = self.get_parameter('disparity_threshold').value


        self.processed_ranges = None
        self.steering_angle_buffer = []
        self.steering_angle_buffer_size = 1
        self.car_x = 0.0
        self.car_y = 0.0
        self.car_heading = 0.0
        self.drive_msg = None



        self.sub_odom = self.create_subscription(
            Odometry,
            self.robot_namespace + self.odom_topic,
            self.odom_cb,
            10
        )
        self.lidar_sub = self.create_subscription(LaserScan, self.scan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)

        self.pub_target_point = self.create_publisher(
            Marker,
            '/target_point',
            10
        )

        # self.timer = self.create_timer(0.1, self.publish_drive)

    def odom_cb(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.car_heading = math.atan2(siny_cosp, cosy_cosp)


    def preprocess_lidar(self, ranges, angle_increment):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)

            Returns: numpy array of preprocessed ranges
        """
        proc_ranges = np.array(ranges)


        proc_ranges[proc_ranges == float('inf')] = self.max_distance_threshold
        proc_ranges[np.isnan(proc_ranges)] = 0.0

        # Find nearest lidar point and surround points with bubble of zero length ranges
        nearest_range = float('inf')
        nearest_range_index = -1

        for i in range(len(proc_ranges)):
            if proc_ranges[i] < nearest_range:
                nearest_range = proc_ranges[i]
                nearest_range_index = i


        # Clamp bubble indices to array bounds
        obj_dist = max(nearest_range, 0.1)
        disparity_rads = np.arcsin(self.disparity_radius_m / obj_dist)
        disparity_rads = disparity_rads if not np.isnan(disparity_rads) else 0.0
        offset = int(disparity_rads / angle_increment)

        start_bubble = max(0, nearest_range_index - self.bubble_radius)
        end_bubble = min(len(proc_ranges), nearest_range_index + self.bubble_radius + 1)
        proc_ranges[start_bubble:end_bubble] = 0.0



        i = 0
        # testing disparity extender
        while i < len(proc_ranges) - 1:
            if np.abs(proc_ranges[i] - proc_ranges[i+1]) > self.disparity_threshold:
                smaller = proc_ranges[i] if proc_ranges[i] < proc_ranges[i+1] else proc_ranges[i+1]

                disparity_rads = np.arcsin(self.disparity_radius_m / smaller)
                disparity_rads = disparity_rads if not np.isnan(disparity_rads) else 0.0
                offset = int(disparity_rads / angle_increment)

                proc_ranges[(i - offset):(i + offset)] = smaller
                i += offset
            
            i += 1

        proc_ranges[proc_ranges > self.max_distance_threshold] = self.max_distance_threshold
        proc_ranges[proc_ranges < self.min_distance_threshold] = 0.0

        """
        i believe the issue with collisions on long straights is due to a bug with the 'bubble' functionality
        going to pad each range of values that are equal to 0 with addition 0s
        IMPORTANT: make sure to move counter variable the same amount as the padding number,
                   otherwise it will likely result in a bug where the rest of the array is all set to 0
        """

        return proc_ranges

    def find_max_gap(self):
        """ Return the start index & end index of the max gap in processed_ranges
        """


        # Find max length sequence of non-zero lengths
        max_gap_index_start = 0
        max_gap_counter = 0
        current_gap_index_start = 0
        current_gap_counter = 0

        for i in range(len(self.processed_ranges)):
            if self.processed_ranges[i] == 0.0:
                if current_gap_counter > max_gap_counter:
                    max_gap_counter = current_gap_counter
                    max_gap_index_start = current_gap_index_start
                current_gap_counter = 0
            else:
                if current_gap_counter == 0:
                    current_gap_index_start = i
                current_gap_counter += 1

        # Handle case where gap is at the end
        if current_gap_counter > max_gap_counter:
            max_gap_counter = current_gap_counter
            max_gap_index_start = current_gap_index_start


        return (max_gap_index_start, max_gap_index_start + max_gap_counter)

    def find_best_point(self, start_i, end_i):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        
        if start_i < 0 or end_i > len(self.processed_ranges):
            return None
        
        furthest_point = start_i
        
        # Choose furthest point, will result in sidewinding when turning on corners
        for i in range(start_i, end_i):
            if self.processed_ranges[i] > self.processed_ranges[furthest_point]:
                furthest_point = i

        # Choose a biased point
        gap_length = end_i - start_i
        biased_point = int(start_i + 0.5 * gap_length)

        if furthest_point > biased_point:
            best_point = biased_point
            #self.get_logger().info(f"Choosing biased point: {best_point}")
        else:
            best_point = furthest_point
            #self.get_logger().info(f"Choosing furthest point: {best_point}")
        
        best_point = biased_point

        return best_point

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        ranges = np.array(data.ranges)

        # don't consider gaps behind the car
        # offset = (int) ((-np.pi / 2 - data.angle_min) / data.angle_increment)

        self.processed_ranges = self.preprocess_lidar(ranges, data.angle_increment)

        # TODO:
        #Find closest point to LiDAR

        #Eliminate all points inside 'bubble' (set them to zero) 
        #Find max length gap 
        max_gap = self.find_max_gap()

        #Find the best point in the gap 
        best_point = self.find_best_point(max_gap[0], max_gap[1])

        curr_steering_angle = data.angle_min + best_point * data.angle_increment
        curr_steering_angle = np.clip(curr_steering_angle, -self.max_steering_angle, self.max_steering_angle)


        point_rad = data.angle_min + best_point * data.angle_increment
        point_dist = ranges[best_point]

        target_x = point_dist * math.cos(point_rad + self.car_heading) + self.car_x
        target_y = point_dist * math.sin(point_rad + self.car_heading) + self.car_y

        self.publish_target_point(target_x, target_y)


        # Average steering angle # if i set the buffer size to 1 it is better than not using the average at all somehow
        self.steering_angle_buffer.append(curr_steering_angle)

        if len(self.steering_angle_buffer) > self.steering_angle_buffer_size:
            self.steering_angle_buffer.pop(0)
        avg_steering_angle = sum(self.steering_angle_buffer) / len(self.steering_angle_buffer) * self.steering_sensitivity

        #Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = avg_steering_angle


        drive_msg.drive.speed = min(float(self.velocity * point_dist / 4.0), self.max_velocity)

        # if np.abs(np.degrees(avg_steering_angle)) >= 0.0 and np.abs(np.degrees(avg_steering_angle)) < 10.0:
        #     drive_msg.drive.speed = self.velocity
        # elif np.abs(np.degrees(avg_steering_angle)) >= 10.0 and np.abs(np.degrees(avg_steering_angle)) < 20.0:
        #     drive_msg.drive.speed = self.velocity * 0.5
        # else:
        #     drive_msg.drive.speed = self.velocity * 0.25

        self.drive_msg = drive_msg
        self.publish_drive()

    def publish_drive(self):
        if self.drive_msg is None:
            return
        self.drive_pub.publish(self.drive_msg)

    def publish_target_point(self, x, y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'target'
        marker.id = 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.40
        marker.scale.y = 0.40
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
    print("Gap Follow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
"""Warp plant behind sensor_racer: /scan,/imu,/odom out, /drive in, lockstep."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.gym_certify import spawn_pose_from_centerline
from f1tenth_rl_agent.sensor_preprocessing import beam_angles_rad, body_accel_to_driver


class WarpSensorPlantNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("warp_sensor_plant", **kwargs)
        self.declare_parameter("scan_topic", si.TOPIC_SCAN)
        self.declare_parameter("imu_topic", si.TOPIC_IMU)
        self.declare_parameter("odom_topic", ifc.TOPIC_ODOM)
        self.declare_parameter("drive_topic", ifc.TOPIC_DRIVE)
        self.declare_parameter("centerline_csv", "")
        gp = self.get_parameter
        scan_topic = gp("scan_topic").get_parameter_value().string_value
        imu_topic = gp("imu_topic").get_parameter_value().string_value
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        drive_topic = gp("drive_topic").get_parameter_value().string_value
        cl_csv = gp("centerline_csv").get_parameter_value().string_value

        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]
        for path in (
            root,
            root / "training",
            root / "libs/f1tenth_policy",
        ):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from analysis.warp_sensor_lockstep import flying_spawn_speed, make_austin_env

        self._env = make_austin_env()
        pose = spawn_pose_from_centerline(cl_csv) if cl_csv else (0.0, 0.0, -0.6524)
        self._spawn_speed = flying_spawn_speed(self._env.env_cfg)
        self._env.reset_to(
            [pose[0], pose[1]], pose[2], self._spawn_speed, seed=0, with_sensors=True
        )
        self._angles = beam_angles_rad()
        self._paused = True
        self._awaiting_drive = False
        self._last_drive_stamp = -1.0
        self._sim_t = 0.0
        self._steer_rad = 0.0

        self._scan_pub = self.create_publisher(
            LaserScan, scan_topic, qos_profile_sensor_data
        )
        self._imu_pub = self.create_publisher(Imu, imu_topic, qos_profile_sensor_data)
        self._odom_pub = self.create_publisher(
            Odometry, odom_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            AckermannDriveStamped, drive_topic, self._on_drive, 10
        )
        self.create_subscription(Bool, "/pause_sim", self._on_pause, 10)
        self._publish_observations()
        self.get_logger().info("WarpSensorPlant ready (paused at spawn)")

    def _stamp(self):
        msg = self.get_clock().now().to_msg()
        msg.sec = int(self._sim_t)
        msg.nanosec = int((self._sim_t % 1.0) * 1e9)
        return msg

    def _publish_observations(self) -> None:
        from f1tenth_policy.layout import LIDAR_DIM

        ego = self._env._ego.tensor
        x = float(ego["x"][0])
        y = float(ego["y"][0])
        yaw = float(ego["yaw"][0])
        vx = float(ego["vx"][0])
        vy = float(ego["vy"][0])
        yaw_rate = float(ego["yaw_rate"][0])
        ax_d, ay_d = body_accel_to_driver(float(ego["ax"][0]), float(ego["ay"][0]))
        lidar = self._env.actor_obs_buf[0, :LIDAR_DIM].detach().cpu().numpy()
        stamp = self._stamp()
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = ifc.FRAME_BASE_LINK
        scan.angle_min = float(si.LIDAR_ANGLE_MIN)
        scan.angle_max = float(self._angles[-1])
        scan.angle_increment = float(self._angles[1] - self._angles[0])
        scan.range_min = float(si.LIDAR_RANGE_MIN)
        scan.range_max = float(si.LIDAR_RANGE_MAX)
        scan.ranges = [float(v) for v in lidar]
        self._scan_pub.publish(scan)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = ifc.FRAME_BASE_LINK
        imu.orientation_covariance[0] = -1.0
        imu.linear_acceleration.x = ax_d
        imu.linear_acceleration.y = ay_d
        imu.linear_acceleration.z = float(si.GRAVITY_MS2)
        imu.angular_velocity.z = float(ego["imu_gz"][0])
        self._imu_pub.publish(imu)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = ifc.FRAME_MAP
        odom.child_frame_id = ifc.FRAME_BASE_LINK
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw * 0.5)
        odom.pose.pose.orientation.w = math.cos(yaw * 0.5)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = yaw_rate
        self._odom_pub.publish(odom)
        if not self._paused:
            self._awaiting_drive = True

    def _on_pause(self, msg: Bool) -> None:
        pausing = bool(msg.data)
        if self._paused and not pausing:
            from f1tenth_rl_agent.gym_certify import spawn_pose_from_centerline

            cl = self.get_parameter("centerline_csv").get_parameter_value().string_value
            pose = spawn_pose_from_centerline(cl) if cl else (0.0, 0.0, -0.6524)
            self._env.reset_to(
                [pose[0], pose[1]],
                pose[2],
                self._spawn_speed,
                seed=0,
                with_sensors=True,
            )
            self._sim_t = 0.0
            self._awaiting_drive = False
            self._publish_observations()
        self._paused = pausing

    def _on_drive(self, msg: AckermannDriveStamped) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= self._last_drive_stamp:
            return
        self._last_drive_stamp = stamp
        if self._paused:
            return
        import torch

        steer_rad = float(msg.drive.steering_angle)
        delta = steer_rad - self._steer_rad
        self._steer_rad = steer_rad
        delta_norm = float(
            np.clip(delta / max(float(si.STEERING_DELTA_MAX_RAD), 1e-6), -1.0, 1.0)
        )
        long_cmd = float(msg.drive.acceleration)
        action = torch.tensor([[long_cmd, delta_norm]], dtype=torch.float32)
        self._env.step(action, with_sensors=True)
        self._sim_t += 1.0 / float(si.CONTROL_HZ)
        self._awaiting_drive = False
        self._publish_observations()

    def destroy_node(self) -> bool:
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()
        return super().destroy_node()


def main(args=None) -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    training = root / "training"
    for path in (root, training, root / "libs/f1tenth_policy"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    rclpy.init(args=args)
    node = WarpSensorPlantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

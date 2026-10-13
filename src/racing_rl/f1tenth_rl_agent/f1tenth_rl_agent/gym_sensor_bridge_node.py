"""Gym adapter: odom → IMU, applied actuator → /drive, plus a safe-zero seed."""

from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from f1tenth_interfaces.msg import ActuatorCommand
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.gym_certify import WarpLongitudinalPlant, WarpLongitudinalParams
from f1tenth_rl_agent.sensor_preprocessing import body_accel_to_driver


class GymSensorBridgeNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("gym_sensor_bridge", **kwargs)

        self.declare_parameter("odom_topic", ifc.TOPIC_ODOM)
        self.declare_parameter("imu_topic", si.TOPIC_IMU)
        self.declare_parameter("applied_topic", si.TOPIC_APPLIED_ACTUATOR)
        self.declare_parameter("drive_topic", ifc.TOPIC_DRIVE)
        self.declare_parameter("imu_frame_id", ifc.FRAME_BASE_LINK)
        self.declare_parameter("seed_applied_hz", 20.0)
        self.declare_parameter("seed_applied_count", 5)
        self.declare_parameter("max_steer", float(si.MAX_STEER_RAD))
        self.declare_parameter("mass", 3.444)
        self.declare_parameter("f_drive_max", 26.5)
        self.declare_parameter("f_brake_max", 23.1)
        self.declare_parameter("i_drive_max_a", 80.0)
        self.declare_parameter("i_brake_max_a", 40.0)
        self.declare_parameter("i_slew_a_per_s", 200.0)
        self.declare_parameter("control_dt", 0.05)

        gp = self.get_parameter
        odom_topic = gp("odom_topic").get_parameter_value().string_value
        imu_topic = gp("imu_topic").get_parameter_value().string_value
        applied_topic = gp("applied_topic").get_parameter_value().string_value
        drive_topic = gp("drive_topic").get_parameter_value().string_value
        self._imu_frame_id = gp("imu_frame_id").get_parameter_value().string_value
        seed_hz = float(gp("seed_applied_hz").get_parameter_value().double_value)
        self._seed_remaining = int(
            gp("seed_applied_count").get_parameter_value().integer_value
        )
        self._max_steer = float(gp("max_steer").get_parameter_value().double_value)
        self._plant = WarpLongitudinalPlant(
            WarpLongitudinalParams(
                mass=float(gp("mass").get_parameter_value().double_value),
                f_drive_max=float(gp("f_drive_max").get_parameter_value().double_value),
                f_brake_max=float(gp("f_brake_max").get_parameter_value().double_value),
                i_drive_max=float(gp("i_drive_max_a").get_parameter_value().double_value),
                i_brake_max=float(gp("i_brake_max_a").get_parameter_value().double_value),
                i_slew=float(gp("i_slew_a_per_s").get_parameter_value().double_value),
                control_dt=float(gp("control_dt").get_parameter_value().double_value),
            )
        )

        self._prev_t: float | None = None
        self._vx = 0.0
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._have_imu = False
        self._imu_msg = Imu()
        self._imu_msg.orientation_covariance[0] = -1.0
        self._drive_msg = AckermannDriveStamped()
        self._seed_msg = ActuatorCommand()
        self._seed_msg.source = ActuatorCommand.SOURCE_SAFE
        self._seed_ns = 0
        self._drive_seq = 0

        self.create_subscription(
            Odometry, odom_topic, self._on_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            ActuatorCommand, applied_topic, self._on_applied, 10
        )
        self.create_subscription(Bool, "/pause_sim", self._on_pause_sim, 10)
        self._imu_pub = self.create_publisher(
            Imu, imu_topic, qos_profile_sensor_data
        )
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )
        self._applied_pub = self.create_publisher(
            ActuatorCommand, applied_topic, 10
        )

        self._publish_safe_zero_applied()
        if self._seed_remaining > 0:
            period = 1.0 / max(seed_hz, 1.0)
            self._seed_timer = self.create_timer(
                period,
                self._on_seed_timer,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
        else:
            self._seed_timer = None
        self._imu_hold_timer = self.create_timer(1.0 / 20.0, self._on_imu_hold)

        self.get_logger().info(
            f"gym_sensor_bridge odom={odom_topic} → imu={imu_topic}; "
            f"applied={applied_topic} → drive={drive_topic}"
        )

    def _publish_safe_zero_applied(self) -> None:
        msg = self._seed_msg
        self._seed_ns += 1000
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = self._seed_ns
        msg.header.frame_id = self._imu_frame_id
        msg.drive_current_a = 0.0
        msg.brake_current_a = 0.0
        msg.servo_position = 0.0
        msg.longitudinal = 0.0
        msg.steering = 0.0
        msg.source = ActuatorCommand.SOURCE_SAFE
        self._applied_pub.publish(msg)
        self._on_applied(msg)

    def _reset_episode(self) -> None:
        self._plant.reset()
        self._vx = 0.0
        self._prev_vx = 0.0
        self._prev_vy = 0.0
        self._prev_t = None
        self._drive_seq = 0
        self._publish_safe_zero_applied()

    def _on_pause_sim(self, msg: Bool) -> None:
        if not bool(msg.data):
            self._reset_episode()

    def _on_seed_timer(self) -> None:
        if self._seed_remaining <= 0:
            if self._seed_timer is not None:
                self._seed_timer.cancel()
            return
        self._publish_safe_zero_applied()
        self._seed_remaining -= 1
        if self._seed_remaining <= 0 and self._seed_timer is not None:
            self._seed_timer.cancel()

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        self._vx = vx
        ax = 0.0
        ay = 0.0
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                ax = (vx - self._prev_vx) / dt
                ay = (vy - self._prev_vy) / dt
        self._prev_t = t
        self._prev_vx = vx
        self._prev_vy = vy

        ax_driver, ay_driver = body_accel_to_driver(ax, ay)
        imu = self._imu_msg
        imu.header.stamp = stamp
        imu.header.frame_id = self._imu_frame_id
        imu.linear_acceleration.x = ax_driver
        imu.linear_acceleration.y = ay_driver
        imu.linear_acceleration.z = float(si.GRAVITY_MS2)
        imu.angular_velocity.x = float(msg.twist.twist.angular.x)
        imu.angular_velocity.y = float(msg.twist.twist.angular.y)
        imu.angular_velocity.z = float(msg.twist.twist.angular.z)
        self._have_imu = True
        self._imu_pub.publish(imu)

    def _on_imu_hold(self) -> None:
        if not self._have_imu:
            return
        imu = self._imu_msg
        imu.header.stamp = self.get_clock().now().to_msg()
        self._imu_pub.publish(imu)

    def _on_applied(self, msg: ActuatorCommand) -> None:
        drive = self._drive_msg
        incoming = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        self._drive_seq = max(self._drive_seq + 1, incoming)
        sec, nsec = divmod(self._drive_seq, 1_000_000_000)
        drive.header.stamp.sec = int(sec)
        drive.header.stamp.nanosec = int(nsec)
        drive.header.frame_id = msg.header.frame_id or self._imu_frame_id
        drive.drive = AckermannDrive()
        long_cmd = float(msg.longitudinal)
        drive.drive.acceleration = long_cmd
        drive.drive.steering_angle = float(msg.steering) * self._max_steer
        drive.drive.speed = self._plant.step(
            self._vx,
            float(msg.drive_current_a),
            float(msg.brake_current_a),
        )
        self._drive_pub.publish(drive)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GymSensorBridgeNode()
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

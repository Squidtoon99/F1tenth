"""Gym bringup for the 1,097-D sensor policy (no hardware drivers).

Starts sensor_racer, rl_current_gate, and gym_sensor_bridge against the
f1tenth_gym_ros topics. IMU scales are SI (gym odom is m/s² / rad/s, not the
car's g / deg/s IMU). Current limits default to the 80 A / 40 A plant.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    control_share = get_package_share_directory("f1tenth_control")
    agent_share = get_package_share_directory("f1tenth_rl_agent")

    checkpoint_path = LaunchConfiguration("checkpoint_path")
    device = LaunchConfiguration("device")
    gym_odom_topic = LaunchConfiguration("gym_odom_topic")
    i_drive_max_a = LaunchConfiguration("i_drive_max_a")
    i_brake_max_a = LaunchConfiguration("i_brake_max_a")
    i_brake_safe_a = LaunchConfiguration("i_brake_safe_a")

    default_sensor_racer = os.path.join(agent_share, "config", "sensor_policy.yaml")
    default_gate = os.path.join(control_share, "config", "rl_current_gate.yaml")

    sensor_racer = Node(
        package="f1tenth_rl_agent",
        executable="sensor_racer",
        name="sensor_racer",
        output="screen",
        parameters=[
            default_sensor_racer,
            {
                "checkpoint_path": ParameterValue(checkpoint_path, value_type=str),
                "device": ParameterValue(device, value_type=str),
                "odom_topic": ParameterValue(gym_odom_topic, value_type=str),
                "i_drive_max_a": ParameterValue(i_drive_max_a, value_type=float),
                "i_brake_max_a": ParameterValue(i_brake_max_a, value_type=float),
                "imu_accel_to_ms2": 1.0,
                "imu_gyro_to_rads": 1.0,
                "imu_ax_bias": 0.0,
                "imu_ay_bias": 0.0,
                "imu_az_bias": 0.0,
                "imu_gx_bias": 0.0,
                "imu_gy_bias": 0.0,
                "imu_gz_bias": 0.0,
                "twist_vx_sign": 1.0,
            },
        ],
    )

    current_gate = Node(
        package="f1tenth_control",
        executable="rl_current_gate",
        name="rl_current_gate",
        output="screen",
        parameters=[
            default_gate,
            {
                "i_drive_max_a": ParameterValue(i_drive_max_a, value_type=float),
                "i_brake_max_a": ParameterValue(i_brake_max_a, value_type=float),
                "i_brake_safe_a": ParameterValue(i_brake_safe_a, value_type=float),
            },
        ],
    )

    gym_sensor_bridge = Node(
        package="f1tenth_rl_agent",
        executable="gym_sensor_bridge",
        name="gym_sensor_bridge",
        output="screen",
        parameters=[
            {
                "odom_topic": ParameterValue(gym_odom_topic, value_type=str),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="/policies/policy.pt",
                description="Format-4 1,097-D recurrent end-to-end policy checkpoint.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="Inference device: cpu or cuda.",
            ),
            DeclareLaunchArgument(
                "gym_odom_topic",
                default_value="/ego_racecar/odom",
                description="Gym ground-truth ego odometry (pose + body twist).",
            ),
            DeclareLaunchArgument(
                "i_drive_max_a",
                default_value="80.0",
                description="Drive current limit (A) for sensor_racer and the gate.",
            ),
            DeclareLaunchArgument(
                "i_brake_max_a",
                default_value="40.0",
                description="Brake current limit (A) for sensor_racer and the gate.",
            ),
            DeclareLaunchArgument(
                "i_brake_safe_a",
                default_value="20.0",
                description="Fail-closed safe brake current (A).",
            ),
            sensor_racer,
            current_gate,
            gym_sensor_bridge,
        ]
    )

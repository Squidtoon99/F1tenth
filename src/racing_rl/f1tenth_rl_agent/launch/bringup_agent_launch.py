"""Bring up the full f1tenth_rl_agent stack.

Launches all five nodes (track_server, observation_builder, policy_inference,
drive_command, evaluation) sharing a single parameter file. The simulator
(f1tenth_gym_ros gym_bridge_launch.py) is launched separately.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("f1tenth_rl_agent")
    default_params = os.path.join(pkg_share, "config", "agent.yaml")

    params_file = LaunchConfiguration("params_file")
    checkpoint_path = LaunchConfiguration("checkpoint_path")

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the agent parameter YAML file.",
    )
    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/policies/policy.pt",
        description="Path to the trained policy .pt checkpoint.",
    )

    nodes = [
        Node(
            package="f1tenth_rl_agent",
            executable="track_server",
            name="track_server",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="observation_builder",
            name="observation_builder",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="policy_inference",
            name="policy_inference",
            parameters=[params_file, {"checkpoint_path": checkpoint_path}],
            output="screen",
        ),
        Node(
            # drive_command lives in the control package (f1tenth_control).
            package="f1tenth_control",
            executable="drive_command",
            name="drive_command",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="evaluation",
            name="evaluation",
            parameters=[params_file],
            output="screen",
        ),
    ]

    return LaunchDescription([declare_params, declare_ckpt, *nodes])

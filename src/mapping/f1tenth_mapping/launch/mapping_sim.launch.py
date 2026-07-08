"""Launch autonomous mapping against f1tenth_gym_ros (simulator)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("f1tenth_mapping")
    default_params = os.path.join(pkg_share, "config", "exploration.yaml")
    slam_params = os.path.join(pkg_share, "config", "slam_toolbox_mapping.yaml")

    params_file = LaunchConfiguration("params_file")
    map_save_path = LaunchConfiguration("map_save_path")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Exploration + navigator parameter file.",
            ),
            DeclareLaunchArgument(
                "map_save_path",
                default_value="/tmp/f1tenth_map",
                description="Prefix path for map_saver_cli output.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock (required for gym sim).",
            ),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[slam_params, {"use_sim_time": use_sim_time}],
            ),
            Node(
                package="f1tenth_mapping",
                executable="exploration",
                name="exploration",
                output="screen",
                parameters=[
                    params_file,
                    {"map_save_path": map_save_path},
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="f1tenth_mapping",
                executable="navigator",
                name="navigator",
                output="screen",
                parameters=[
                    params_file,
                    {"odom_topic": "/ego_racecar/odom"},
                    {"use_sim_time": use_sim_time},
                ],
            ),
        ]
    )

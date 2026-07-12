"""Reactive recon-lap mapping against a running f1tenth_gym_ros bridge.

Assumes the gym bridge is already running (provides /scan, /ego_racecar/odom, and the
ground-truth map -> ego_racecar/base_link TF). Starts:

- slam_toolbox (async) building the real map from the live /scan, published on /slam_map
  in the slam_map frame (see config/slam_toolbox_gym.yaml),
- reactive_explorer driving the car around the loop with follow-the-gap.

The car traces the corridor once; slam_toolbox maps it; reactive_explorer saves the map
on loop closure.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("f1tenth_mapping")
    slam_params = os.path.join(pkg_share, "config", "slam_toolbox_gym.yaml")
    explorer_params = os.path.join(pkg_share, "config", "reactive_explorer.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("slam_params_file", default_value=slam_params),
            DeclareLaunchArgument("explorer_params_file", default_value=explorer_params),
            DeclareLaunchArgument("map_save_path", default_value="/tmp/slam_map"),
            DeclareLaunchArgument("scan_topic", default_value="/scan"),
            DeclareLaunchArgument("slam_scan_rate_hz", default_value="25.0"),
            # The gym publishes /scan at ~250 Hz, which floods slam_toolbox's TF-sync queue
            # ("message queue full" -> sparse, fragmented map). Throttle to a sane rate and
            # feed slam_toolbox the throttled topic /scan_slam.
            Node(
                package="f1tenth_mapping",
                executable="scan_throttle",
                name="scan_throttle",
                output="screen",
                parameters=[{
                    "input_topic": LaunchConfiguration("scan_topic"),
                    "output_topic": "/scan_slam",
                    "target_rate_hz": 25.0,
                }],
            ),
            # Occupancy-grid mapper (log-odds inverse sensor model) building the map from the
            # throttled /scan_slam at the gym's ground-truth pose, published on /slam_map.
            Node(
                package="f1tenth_mapping",
                executable="occupancy_mapper",
                name="occupancy_mapper",
                output="screen",
                parameters=[{
                    "scan_topic": "/scan_slam",
                    "map_topic": "/slam_map",
                    "map_frame": "map",
                    "resolution": 0.05,
                    "max_range_m": 25.0,
                    "publish_period_s": 1.0,
                    "start_delay_s": 3.0,
                }],
            ),
            Node(
                package="f1tenth_mapping",
                executable="reactive_explorer",
                name="reactive_explorer",
                output="screen",
                parameters=[
                    LaunchConfiguration("explorer_params_file"),
                    {"map_save_path": LaunchConfiguration("map_save_path")},
                ],
            ),
        ]
    )

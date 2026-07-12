"""Reactive recon-lap driving on the real F1TENTH car, reading slam_toolbox's map.

Assumes the f1tenth_stack bringup and slam_toolbox (online_async, mode: mapping) are ALREADY
running on the car. This launch adds only our driver:

- reactive_explorer: hugs one wall to trace the closed track in a single clean lap, publishing
  AckermannDriveStamped on /drive (the ackermann_mux "navigation" input). It subscribes to
  slam_toolbox's /map purely to confirm/monitor coverage, and saves the map on loop closure.

It does NOT start slam_toolbox (already running) and does NOT run the occupancy_mapper
(slam_toolbox is the mapper on the car). Start in dry_run:=true, then flip to false to drive.
Stop the stock f1tenth gap_driver first so there is a single publisher on /drive.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("f1tenth_mapping")
    explorer_params = os.path.join(pkg_share, "config", "reactive_explorer_real.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("explorer_params_file", default_value=explorer_params),
            DeclareLaunchArgument("map_save_path", default_value="/home/shereef/maps/f1tenth_map"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            Node(
                package="f1tenth_mapping",
                executable="reactive_explorer",
                name="reactive_explorer",
                output="screen",
                parameters=[
                    LaunchConfiguration("explorer_params_file"),
                    {
                        "map_save_path": LaunchConfiguration("map_save_path"),
                        "dry_run": LaunchConfiguration("dry_run"),
                    },
                ],
            ),
        ]
    )

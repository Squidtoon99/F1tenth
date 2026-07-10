"""Vehicle-only bringup (drivers + sensors + safety), no autonomy.

Includes the vendored f1tenth_stack bringup (joy teleop, VESC driver + odom,
ackermann_mux, LiDAR, base_link->laser TF) without modifying vendored code. Per-car
calibration is supplied by overriding the vendored config arguments with files from
the mounted /config overlay. Localization (particle filter) and the racing stack are
layered on top by race.launch.py.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    stack_share = get_package_share_directory("f1tenth_stack")
    default_vesc = os.path.join(stack_share, "config", "vesc.yaml")
    default_sensors = os.path.join(stack_share, "config", "sensors.yaml")
    default_mux = os.path.join(stack_share, "config", "mux.yaml")
    default_joy = os.path.join(stack_share, "config", "joy_teleop.yaml")

    vesc_config = LaunchConfiguration("vesc_config")
    sensors_config = LaunchConfiguration("sensors_config")
    mux_config = LaunchConfiguration("mux_config")
    joy_config = LaunchConfiguration("joy_config")

    declare_vesc = DeclareLaunchArgument(
        "vesc_config",
        default_value=default_vesc,
        description="VESC driver/odom calibration YAML (override with the per-car /config file).",
    )
    declare_sensors = DeclareLaunchArgument(
        "sensors_config", default_value=default_sensors,
        description="LiDAR / sensor driver YAML.",
    )
    declare_mux = DeclareLaunchArgument(
        "mux_config", default_value=default_mux,
        description="ackermann_mux priority/timeout YAML.",
    )
    declare_joy = DeclareLaunchArgument(
        "joy_config", default_value=default_joy,
        description="joy + joy_teleop YAML.",
    )

    vendored_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(stack_share, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "vesc_config": vesc_config,
            "sensors_config": sensors_config,
            "mux_config": mux_config,
            "joy_config": joy_config,
        }.items(),
    )

    return LaunchDescription(
        [
            declare_vesc,
            declare_sensors,
            declare_mux,
            declare_joy,
            vendored_bringup,
        ]
    )

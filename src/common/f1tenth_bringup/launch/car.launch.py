"""Vehicle-only bringup (drivers + sensors + safety), no autonomy.

Includes the vendored f1tenth_stack bringup (joy teleop, VESC driver + odom,
ackermann_mux, LiDAR, base_link->laser TF) without modifying vendored code.
Per-car calibration is supplied by overriding the vendored config arguments with
files from the mounted /config overlay. The RL deadman gate runs here so /drive is
blocked unless R1 or L1 is held.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    stack_share = get_package_share_directory("f1tenth_stack")
    default_overlay = "/config/params.yaml"
    default_sensors = os.path.join(stack_share, "config", "sensors.yaml")

    overlay_params = LaunchConfiguration("overlay_params_file")
    vesc_config = LaunchConfiguration("vesc_config")
    sensors_config = LaunchConfiguration("sensors_config")
    mux_config = LaunchConfiguration("mux_config")
    joy_config = LaunchConfiguration("joy_config")
    gate_config = LaunchConfiguration("gate_config")

    declare_overlay = DeclareLaunchArgument(
        "overlay_params_file",
        default_value=default_overlay,
        description="Per-car overlay YAML (used for all driver configs when present).",
    )
    declare_vesc = DeclareLaunchArgument(
        "vesc_config",
        default_value=overlay_params,
        description="VESC driver/odom calibration YAML.",
    )
    declare_sensors = DeclareLaunchArgument(
        "sensors_config", default_value=default_sensors,
        description="LiDAR / sensor driver YAML.",
    )
    declare_mux = DeclareLaunchArgument(
        "mux_config", default_value=overlay_params,
        description="ackermann_mux priority/timeout YAML.",
    )
    declare_joy = DeclareLaunchArgument(
        "joy_config", default_value=overlay_params,
        description="joy + joy_teleop YAML.",
    )
    declare_gate = DeclareLaunchArgument(
        "gate_config",
        default_value=overlay_params,
        description="RL deadman gate parameters.",
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

    deadman_gate = Node(
        package="f1tenth_control",
        executable="rl_deadman_gate",
        name="rl_deadman_gate",
        output="screen",
        parameters=[gate_config],
    )

    return LaunchDescription(
        [
            declare_overlay,
            declare_vesc,
            declare_sensors,
            declare_mux,
            declare_joy,
            declare_gate,
            vendored_bringup,
            deadman_gate,
        ]
    )

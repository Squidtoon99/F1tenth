"""Vehicle-only bringup (drivers + sensors + safety), no autonomy.

Includes the vendored f1tenth_stack bringup (joy teleop, VESC driver + odom,
ackermann_mux, LiDAR, base_link->laser TF) without modifying vendored code.
Per-car calibration is supplied by overriding the vendored config arguments with
files from the mounted /config overlay. The RL deadman gate runs here so /drive is
blocked unless R1 or L1 is held.

When motor_mode:=force, remaps the vendored ackermann_to_vesc motor/servo
outputs away and enables f1tenth_control/vesc_actuator as the exclusive owner of
current, brake, and servo commands (see ADR 0005). Default remains speed/ERPM.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _launch_setup(context, *args, **kwargs):
    stack_share = get_package_share_directory("f1tenth_stack")
    control_share = get_package_share_directory("f1tenth_control")

    vesc_config = LaunchConfiguration("vesc_config")
    sensors_config = LaunchConfiguration("sensors_config")
    mux_config = LaunchConfiguration("mux_config")
    joy_config = LaunchConfiguration("joy_config")
    gate_config = LaunchConfiguration("gate_config")
    motor_mode = LaunchConfiguration("motor_mode")
    actuator_config = LaunchConfiguration("actuator_config")

    mode = motor_mode.perform(context)
    force = mode == "force"

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

    if force:
        # Divert ERPM/servo from ackermann_to_vesc so vesc_actuator owns them.
        vendored = GroupAction(
            [
                SetRemap(src="commands/motor/speed", dst="commands/motor/speed_erpm_unused"),
                SetRemap(
                    src="commands/servo/position",
                    dst="commands/servo/position_erpm_unused",
                ),
                vendored_bringup,
            ]
        )
    else:
        vendored = vendored_bringup

    deadman_gate = Node(
        package="f1tenth_control",
        executable="rl_deadman_gate",
        name="rl_deadman_gate",
        output="screen",
        parameters=[gate_config],
    )

    actions = [vendored, deadman_gate]

    # Always declare the node; enabled defaults false. Force mode remaps + overlay
    # can flip enabled:true without rebuilding.
    default_actuator = os.path.join(control_share, "config", "vesc_actuator.yaml")
    actuator = Node(
        package="f1tenth_control",
        executable="vesc_actuator",
        name="vesc_actuator",
        output="screen",
        parameters=[
            default_actuator,
            actuator_config,
            {"enabled": force},
        ],
    )
    actions.append(actuator)
    return actions


def generate_launch_description() -> LaunchDescription:
    stack_share = get_package_share_directory("f1tenth_stack")
    control_share = get_package_share_directory("f1tenth_control")
    default_overlay = "/config/params.yaml"
    default_sensors = os.path.join(stack_share, "config", "sensors.yaml")
    default_actuator = os.path.join(control_share, "config", "vesc_actuator.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "overlay_params_file",
                default_value=default_overlay,
                description="Per-car overlay YAML (used for all driver configs when present).",
            ),
            DeclareLaunchArgument(
                "vesc_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="VESC driver/odom calibration YAML.",
            ),
            DeclareLaunchArgument(
                "sensors_config",
                default_value=default_sensors,
                description="LiDAR / sensor driver YAML.",
            ),
            DeclareLaunchArgument(
                "mux_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="ackermann_mux priority/timeout YAML.",
            ),
            DeclareLaunchArgument(
                "joy_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="joy + joy_teleop YAML.",
            ),
            DeclareLaunchArgument(
                "gate_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="RL deadman gate parameters.",
            ),
            DeclareLaunchArgument(
                "motor_mode",
                default_value="speed",
                description="Motor command mode: speed (ERPM) or force (current/brake).",
            ),
            DeclareLaunchArgument(
                "actuator_config",
                default_value=default_actuator,
                description="vesc_actuator YAML (force mode).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

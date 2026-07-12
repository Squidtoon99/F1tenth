"""Bringup for the algorithmic racing stack.

Composes localization (external prerequisite or started by race.launch.py),
pure-pursuit + follow-the-gap control, and the RL deadman gate.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    control_share = get_package_share_directory('f1tenth_control')
    default_driver_config = os.path.join(control_share, 'config', 'pp_driver_plus.yaml')
    default_gate_config = os.path.join(control_share, 'config', 'rl_deadman_gate.yaml')

    driver = LaunchConfiguration('driver')
    driver_config = LaunchConfiguration('driver_config')
    gate_config = LaunchConfiguration('gate_config')

    declare_driver = DeclareLaunchArgument(
        'driver',
        default_value='pp_driver_plus',
        description='Algorithmic controller executable: pp_driver, pp_driver_plus, '
        'pp_ftg_driver, gap_driver, or pid_driver.',
    )
    declare_driver_config = DeclareLaunchArgument(
        'driver_config',
        default_value=default_driver_config,
        description='ROS parameter file for the selected driver node.',
    )
    declare_gate_config = DeclareLaunchArgument(
        'gate_config',
        default_value=default_gate_config,
        description='RL deadman gate parameters.',
    )

    driver_node = Node(
        package='f1tenth_control',
        executable=driver,
        name=driver,
        output='screen',
        parameters=[driver_config],
    )

    deadman_gate = Node(
        package='f1tenth_control',
        executable='rl_deadman_gate',
        name='rl_deadman_gate',
        output='screen',
        parameters=[gate_config],
    )

    return LaunchDescription([
        declare_driver,
        declare_driver_config,
        declare_gate_config,
        driver_node,
        deadman_gate,
    ])

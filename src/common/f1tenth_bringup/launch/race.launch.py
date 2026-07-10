"""Top-level launch for on-car racing.

Composes the vehicle drivers (car.launch.py -> vendored f1tenth_stack), a read-only
localization preflight, and the selected racing stack. Localization (the particle
filter publishing /pf/pose/odom) is an external prerequisite and is NOT started
here; the preflight node reports its health and the RL graph inherently emits no
drive command until pose + twist are live.

Per-car calibration/track come from the overlay mounted at /config; the policy from
/policies. Select the stack with ``stack:=rl`` (default) or ``stack:=algo``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    bringup_share = get_package_share_directory("f1tenth_bringup")
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")
    agent_share = get_package_share_directory("f1tenth_rl_agent")
    algo_share = get_package_share_directory("f1tenth_racing_algo")

    default_track_csv = os.path.join(
        agent_share, "assets", "IV_2026_SIM_centerline.csv"
    )

    stack = LaunchConfiguration("stack")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    overlay_params_file = LaunchConfiguration("overlay_params_file")
    track_csv = LaunchConfiguration("track_csv")
    enable_drivers = LaunchConfiguration("enable_drivers")
    enable_opponent = LaunchConfiguration("enable_opponent")

    declare_stack = DeclareLaunchArgument(
        "stack", default_value="rl",
        description="Which racing stack to run: 'rl' or 'algo'.",
    )
    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path", default_value="/policies/policy.pt",
        description="Trained 390-dim policy .pt (must include obs_norm).",
    )
    declare_overlay = DeclareLaunchArgument(
        "overlay_params_file", default_value="/config/params.yaml",
        description="Per-car overlay (node-scoped ROS params) layered on the stack defaults.",
    )
    declare_track = DeclareLaunchArgument(
        "track_csv", default_value=default_track_csv,
        description="Surveyed centerline CSV in the localization map frame.",
    )
    declare_drivers = DeclareLaunchArgument(
        "enable_drivers", default_value="true",
        description="Start the vendored vehicle drivers (car.launch.py). Off for bench tests.",
    )
    declare_opponent = DeclareLaunchArgument(
        "enable_opponent", default_value="false",
        description="Start the LiDAR opponent detector (1v1). Solo racing keeps it off.",
    )

    is_rl = IfCondition(PythonExpression(["'", stack, "' == 'rl'"]))
    is_algo = IfCondition(PythonExpression(["'", stack, "' == 'algo'"]))

    drivers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "car.launch.py")
        ),
        condition=IfCondition(enable_drivers),
    )

    # Read-only localization health check (external PF is a prerequisite).
    preflight = Node(
        package="f1tenth_rl_agent",
        executable="localization_preflight",
        name="localization_preflight",
        output="screen",
    )

    rl_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vehicle_share, "launch", "bringup_vehicle.launch.py")
        ),
        condition=is_rl,
        launch_arguments={
            "checkpoint_path": checkpoint_path,
            "overlay_params_file": overlay_params_file,
            "track_csv": track_csv,
            "enable_opponent": enable_opponent,
            "enable_obs_debug": "true",
        }.items(),
    )

    algo_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(algo_share, "launch", "algo.launch.py")
        ),
        condition=is_algo,
    )

    return LaunchDescription(
        [
            declare_stack,
            declare_ckpt,
            declare_overlay,
            declare_track,
            declare_drivers,
            declare_opponent,
            drivers,
            preflight,
            rl_stack,
            algo_stack,
        ]
    )

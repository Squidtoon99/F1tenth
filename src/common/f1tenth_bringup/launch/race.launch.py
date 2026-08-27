"""Top-level launch for on-car racing.

Composes the vehicle drivers (car.launch.py -> vendored f1tenth_stack), particle-
filter localization, a read-only localization preflight, and the selected racing
stack.

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
    loc_share = get_package_share_directory("f1tenth_localization")
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")
    algo_share = get_package_share_directory("f1tenth_racing_algo")

    stack = LaunchConfiguration("stack")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    overlay_params_file = LaunchConfiguration("overlay_params_file")
    track_csv = LaunchConfiguration("track_csv")
    map_yaml = LaunchConfiguration("map_yaml")
    enable_drivers = LaunchConfiguration("enable_drivers")
    enable_localization = LaunchConfiguration("enable_localization")
    enable_opponent = LaunchConfiguration("enable_opponent")
    use_sim_time = LaunchConfiguration("use_sim_time")

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
        "track_csv", default_value="/config/maps/centerline.csv",
        description="Surveyed centerline CSV (map frame) for RL obs and PF relocalize.",
    )
    declare_map = DeclareLaunchArgument(
        "map_yaml", default_value="/config/maps/map.yaml",
        description="Occupancy grid YAML from the per-car overlay.",
    )
    declare_drivers = DeclareLaunchArgument(
        "enable_drivers", default_value="true",
        description="Start the vendored vehicle drivers (car.launch.py). Off for bench tests.",
    )
    declare_localization = DeclareLaunchArgument(
        "enable_localization", default_value="true",
        description="Start map_server + particle filter (localization.launch.py).",
    )
    declare_opponent = DeclareLaunchArgument(
        "enable_opponent", default_value="false",
        description="Start the LiDAR opponent detector (1v1). Solo racing keeps it off.",
    )
    declare_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="false",
        description="Use simulation clock (true in gym).",
    )

    is_rl = IfCondition(PythonExpression(["'", stack, "' == 'rl'"]))
    is_algo = IfCondition(PythonExpression(["'", stack, "' == 'algo'"]))

    drivers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "car.launch.py")
        ),
        condition=IfCondition(enable_drivers),
        launch_arguments={
            "overlay_params_file": overlay_params_file,
        }.items(),
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(loc_share, "launch", "localization.launch.py")
        ),
        condition=IfCondition(enable_localization),
        launch_arguments={
            "map_yaml": map_yaml,
            "track_centerline_csv": track_csv,
            "use_sim_time": use_sim_time,
        }.items(),
    )

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
        launch_arguments={
            "driver_config": overlay_params_file,
        }.items(),
    )

    return LaunchDescription(
        [
            declare_stack,
            declare_ckpt,
            declare_overlay,
            declare_track,
            declare_map,
            declare_drivers,
            declare_localization,
            declare_opponent,
            declare_sim_time,
            drivers,
            localization,
            preflight,
            rl_stack,
            algo_stack,
        ]
    )

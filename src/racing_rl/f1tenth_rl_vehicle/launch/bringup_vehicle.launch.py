"""Bring up the lean on-car RL agent: 3 nodes.

Launches the two C++ vehicle nodes (vehicle_obs, drive) from f1tenth_rl_vehicle and
the Python policy_inference node from f1tenth_rl_agent. The f1tenth_stack (particle
filter + VESC + ackermann_mux) is launched separately and is not part of this file.

By default the track centerline is the IV_2026_SIM asset installed with
f1tenth_rl_agent; override track_csv for a surveyed real track.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")
    agent_share = get_package_share_directory("f1tenth_rl_agent")

    default_vehicle_params = os.path.join(vehicle_share, "config", "vehicle.yaml")
    default_agent_params = os.path.join(agent_share, "config", "agent.yaml")
    default_track_csv = os.path.join(
        agent_share, "assets", "IV_2026_SIM_centerline.csv"
    )

    params_file = LaunchConfiguration("params_file")
    agent_params_file = LaunchConfiguration("agent_params_file")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    track_csv = LaunchConfiguration("track_csv")
    enable_opponent = LaunchConfiguration("enable_opponent")
    enable_obs_debug = LaunchConfiguration("enable_obs_debug")

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_vehicle_params,
        description="vehicle_obs + drive parameter YAML.",
    )
    declare_agent_params = DeclareLaunchArgument(
        "agent_params_file",
        default_value=default_agent_params,
        description="policy_inference parameter YAML (from f1tenth_rl_agent).",
    )
    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/checkpoints/policy.pt",
        description="Trained policy .pt (must include obs_norm to match training).",
    )
    declare_track = DeclareLaunchArgument(
        "track_csv",
        default_value=default_track_csv,
        description="Centerline CSV in the same frame as the localization map.",
    )
    declare_opponent = DeclareLaunchArgument(
        "enable_opponent",
        default_value="false",
        description="Launch LiDAR opponent_detector (requires enable_opponent_obs in YAML).",
    )
    declare_obs_debug = DeclareLaunchArgument(
        "enable_obs_debug",
        default_value="true",
        description="Launch the read-only obs_debug node (scalars + markers for diagnosis).",
    )

    # Optional open-loop calibration profiler (vehicle_calibration package). When
    # enabled it OWNS /rl/action, so policy_inference must not run at the same time.
    # profiler_script defaults to $F1TENTH_REPO/vehicle_calibration/ros/... ; set the
    # env var or pass profiler_script:=/abs/path on the car.
    enable_profiler = LaunchConfiguration("enable_profiler")
    profiler_script = LaunchConfiguration("profiler_script")
    default_profiler_script = os.path.join(
        os.environ.get("F1TENTH_REPO", ""),
        "vehicle_calibration",
        "ros",
        "profile_maneuver_node.py",
    )
    declare_enable_profiler = DeclareLaunchArgument(
        "enable_profiler",
        default_value="false",
        description="Run the open-loop maneuver profiler instead of policy_inference.",
    )
    declare_profiler_script = DeclareLaunchArgument(
        "profiler_script",
        default_value=default_profiler_script,
        description="Absolute path to vehicle_calibration/ros/profile_maneuver_node.py.",
    )
    run_policy = PythonExpression(["'", enable_profiler, "' != 'true'"])

    nodes = [
        Node(
            package="f1tenth_rl_vehicle",
            executable="vehicle_obs",
            name="vehicle_obs",
            parameters=[
                params_file,
                {"track_csv": track_csv},
            ],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="policy_inference",
            name="policy_inference",
            parameters=[
                agent_params_file,
                {"checkpoint_path": checkpoint_path},
            ],
            output="screen",
            condition=IfCondition(run_policy),
        ),
        Node(
            executable="python3",
            arguments=[profiler_script],
            name="profile_maneuver",
            output="screen",
            condition=IfCondition(enable_profiler),
        ),
        Node(
            package="f1tenth_rl_vehicle",
            executable="drive",
            name="drive",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_vehicle",
            executable="opponent_detector",
            name="opponent_detector",
            parameters=[params_file, {"track_csv": track_csv}],
            output="screen",
            condition=IfCondition(enable_opponent),
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="obs_debug",
            name="obs_debug",
            parameters=[agent_params_file],
            output="screen",
            condition=IfCondition(enable_obs_debug),
        ),
    ]

    return LaunchDescription(
        [
            declare_params,
            declare_agent_params,
            declare_ckpt,
            declare_track,
            declare_opponent,
            declare_obs_debug,
            declare_enable_profiler,
            declare_profiler_script,
            *nodes,
        ]
    )

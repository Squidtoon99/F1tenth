"""Top-level launch for evaluating the RL stack against the f1tenth_gym bridge.

The gym bridge (physics + /scan + /ego_racecar/odom + /drive) runs in the separate
sim container (see sim/ and deploy/docker/docker-compose.sim.yml). This launch runs
the agent-side ROS graph in the agent container and closes the loop on /drive.

The on-car C++ autonomy graph (vehicle_obs -> policy_inference -> drive) from
bringup_vehicle.launch.py is pointed at the gym's ground-truth odom in place of the
particle filter + VESC odom. Solo racing: the opponent detector is off and the
8-dim opponent block [384:392) is zeroed (the vehicle.yaml sentinel), so the
392-dim policy runs without a detector.

The evaluation node (read-only) provides spawn/reset + lap/progress/OOB/stuck metrics;
it is composed here alongside track_server.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    agent_share = get_package_share_directory("f1tenth_rl_agent")
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")

    default_agent_params = os.path.join(agent_share, "config", "agent.yaml")
    default_track_csv = os.path.join(
        agent_share, "assets", "IV_2026_SIM_centerline.csv"
    )
    # Prefer the car01 overlay (slip/load estimation on) when running from the
    # monorepo workspace mount; fall back to the package empty overlay.
    default_overlay = os.path.join(
        vehicle_share, "config", "overlay_defaults.yaml"
    )
    car01_overlay = "/ws/deploy/cars/car01/params.yaml"
    if os.path.isfile(car01_overlay):
        default_overlay = car01_overlay

    checkpoint_path = LaunchConfiguration("checkpoint_path")
    agent_params_file = LaunchConfiguration("agent_params_file")
    overlay_params_file = LaunchConfiguration("overlay_params_file")
    track_csv = LaunchConfiguration("track_csv")
    gym_odom_topic = LaunchConfiguration("gym_odom_topic")

    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/policies/policy.pt",
        description="Trained 392-dim policy .pt (must include obs_norm).",
    )
    declare_agent_params = DeclareLaunchArgument(
        "agent_params_file",
        default_value=default_agent_params,
        description="Agent parameter YAML (track_server / policy_inference / evaluation).",
    )
    declare_overlay = DeclareLaunchArgument(
        "overlay_params_file",
        default_value=default_overlay,
        description="Per-car overlay (e.g. car01 params with slip/load estimation).",
    )
    declare_track = DeclareLaunchArgument(
        "track_csv",
        default_value=default_track_csv,
        description="Centerline CSV (same frame as the gym map).",
    )
    declare_gym_odom = DeclareLaunchArgument(
        "gym_odom_topic",
        default_value="/ego_racecar/odom",
        description="Gym ground-truth ego odometry (pose + body twist).",
    )

    vehicle_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vehicle_share, "launch", "bringup_vehicle.launch.py")
        ),
        launch_arguments={
            "checkpoint_path": checkpoint_path,
            "agent_params_file": agent_params_file,
            "overlay_params_file": overlay_params_file,
            "track_csv": track_csv,
            "pose_topic": gym_odom_topic,
            "twist_topic": gym_odom_topic,
            "enable_opponent": "false",
            "enable_obs_debug": "false",
        }.items(),
    )
    vehicle_track_server = Node(
        package="f1tenth_rl_agent",
        executable="track_server",
        name="track_server",
        parameters=[agent_params_file],
        output="screen",
    )
    vehicle_evaluation = Node(
        package="f1tenth_rl_agent",
        executable="evaluation",
        name="evaluation",
        parameters=[agent_params_file],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_ckpt,
            declare_agent_params,
            declare_overlay,
            declare_track,
            declare_gym_odom,
            vehicle_bringup,
            vehicle_track_server,
            vehicle_evaluation,
        ]
    )

"""Top-level launch for evaluating the RL stack against the f1tenth_gym bridge.

The gym bridge (physics + /scan + /ego_racecar/odom + /drive) runs in the separate
sim container (see sim/ and deploy/docker/docker-compose.sim.yml). This launch runs
the agent-side ROS graph in the agent container and closes the loop on /drive.

Two graphs are selectable with ``stack``:

- ``vehicle`` (default, the release gate): the exact on-car C++ autonomy graph
  (vehicle_obs -> policy_inference -> drive) from bringup_vehicle.launch.py, pointed
  at the gym's ground-truth odom in place of the particle filter + VESC odom. Solo
  racing: the opponent detector is off and the 6-dim opponent block [384:390) is
  zeroed (the vehicle.yaml sentinel), so the 390-dim policy runs without a detector.
- ``python``: the pure-Python agent graph (bringup_agent_launch.py), kept as a
  regression/parity check against the same checkpoint, track, and steering envelope.

The evaluation node (read-only) provides spawn/reset + lap/progress/OOB/stuck metrics
for both graphs; for the vehicle graph it is composed here alongside track_server.
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
    agent_share = get_package_share_directory("f1tenth_rl_agent")
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")

    default_agent_params = os.path.join(agent_share, "config", "agent.yaml")
    default_track_csv = os.path.join(
        agent_share, "assets", "IV_2026_SIM_centerline.csv"
    )

    stack = LaunchConfiguration("stack")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    agent_params_file = LaunchConfiguration("agent_params_file")
    track_csv = LaunchConfiguration("track_csv")
    gym_odom_topic = LaunchConfiguration("gym_odom_topic")

    declare_stack = DeclareLaunchArgument(
        "stack",
        default_value="vehicle",
        description="Which agent graph to certify: 'vehicle' (on-car C++) or 'python'.",
    )
    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/policies/policy.pt",
        description="Trained 390-dim policy .pt (must include obs_norm).",
    )
    declare_agent_params = DeclareLaunchArgument(
        "agent_params_file",
        default_value=default_agent_params,
        description="Agent parameter YAML (track_server / policy_inference / evaluation).",
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

    is_vehicle = IfCondition(PythonExpression(["'", stack, "' == 'vehicle'"]))
    is_python = IfCondition(PythonExpression(["'", stack, "' == 'python'"]))

    # --- vehicle graph: the exact on-car C++ stack against gym GT odom -----------
    vehicle_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vehicle_share, "launch", "bringup_vehicle.launch.py")
        ),
        condition=is_vehicle,
        launch_arguments={
            "checkpoint_path": checkpoint_path,
            "agent_params_file": agent_params_file,
            "track_csv": track_csv,
            "pose_topic": gym_odom_topic,
            "twist_topic": gym_odom_topic,
            "enable_opponent": "false",
            "enable_obs_debug": "false",
        }.items(),
    )
    # track_server + evaluation give the vehicle graph the same reset/metrics sidecar
    # the python graph gets from bringup_agent_launch.py.
    vehicle_track_server = Node(
        package="f1tenth_rl_agent",
        executable="track_server",
        name="track_server",
        parameters=[agent_params_file],
        output="screen",
        condition=is_vehicle,
    )
    vehicle_evaluation = Node(
        package="f1tenth_rl_agent",
        executable="evaluation",
        name="evaluation",
        parameters=[agent_params_file],
        output="screen",
        condition=is_vehicle,
    )

    # --- python graph: regression / parity check --------------------------------
    python_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(agent_share, "launch", "bringup_agent_launch.py")
        ),
        condition=is_python,
        launch_arguments={
            "checkpoint_path": checkpoint_path,
            "params_file": agent_params_file,
        }.items(),
    )

    return LaunchDescription(
        [
            declare_stack,
            declare_ckpt,
            declare_agent_params,
            declare_track,
            declare_gym_odom,
            vehicle_bringup,
            vehicle_track_server,
            vehicle_evaluation,
            python_bringup,
        ]
    )

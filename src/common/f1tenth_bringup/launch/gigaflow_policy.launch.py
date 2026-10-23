"""No-localization IRL Gigaflow bringup (drivers + racer + current gate).

Starts vendored sensor drivers, joy teleop, RL deadman gate, optional safety
brake, gigaflow_racer, and rl_current_gate. Does not start ackermann_mux,
ackermann_to_vesc, vesc_actuator, localization, or the 390-D RL graph.
rl_current_gate is the sole publisher of VESC motor/brake/servo commands.
The racer injects a frozen IRL embodied condition (never samples vmax_scale).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import AndCondition, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_setup(context, *args, **kwargs):
    bringup_share = get_package_share_directory("f1tenth_bringup")
    control_share = get_package_share_directory("f1tenth_control")
    agent_share = get_package_share_directory("f1tenth_rl_agent")

    vesc_config = LaunchConfiguration("vesc_config")
    sensors_config = LaunchConfiguration("sensors_config")
    joy_config = LaunchConfiguration("joy_config")
    overlay_params_file = LaunchConfiguration("overlay_params_file")
    gate_config = LaunchConfiguration("gate_config")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    device = LaunchConfiguration("device")
    dry_run = LaunchConfiguration("dry_run")
    enable_drivers = LaunchConfiguration("enable_drivers")
    enable_teleop = LaunchConfiguration("enable_teleop")
    enable_deadman = LaunchConfiguration("enable_deadman")
    enable_racer = LaunchConfiguration("enable_racer")
    enable_gate = LaunchConfiguration("enable_gate")
    enable_safety = LaunchConfiguration("enable_safety")
    recording_mode = LaunchConfiguration("recording_mode")
    i_drive_max_a = LaunchConfiguration("i_drive_max_a")
    i_brake_max_a = LaunchConfiguration("i_brake_max_a")
    i_slew_a_per_s = LaunchConfiguration("i_slew_a_per_s")
    i_brake_safe_a = LaunchConfiguration("i_brake_safe_a")
    online_ppo_enable = LaunchConfiguration("online_ppo_enable")
    online_ppo_rollout_steps = LaunchConfiguration("online_ppo_rollout_steps")
    online_ppo_lr = LaunchConfiguration("online_ppo_lr")
    online_ppo_kl_coef = LaunchConfiguration("online_ppo_kl_coef")
    online_ppo_kl_clip = LaunchConfiguration("online_ppo_kl_clip")
    online_ppo_output_dir = LaunchConfiguration("online_ppo_output_dir")

    dry_run_enabled = dry_run.perform(context).lower() in ("true", "1", "yes")
    odom_parameters = [vesc_config]
    if dry_run_enabled:
        odom_parameters.append(
            os.path.join(bringup_share, "config", "vesc_to_odom_dryrun.yaml")
        )

    drivers = [
        Node(
            condition=IfCondition(enable_drivers),
            package="joy",
            executable="joy_node",
            name="joy",
            parameters=[joy_config],
        ),
        Node(
            condition=IfCondition(enable_drivers),
            package="vesc_ackermann",
            executable="vesc_to_odom_node",
            name="vesc_to_odom_node",
            parameters=odom_parameters,
        ),
        Node(
            condition=IfCondition(enable_drivers),
            package="vesc_driver",
            executable="vesc_driver_node",
            name="vesc_driver_node",
            parameters=[vesc_config],
        ),
        Node(
            condition=IfCondition(enable_drivers),
            package="urg_node",
            executable="urg_node_driver",
            name="urg_node",
            parameters=[sensors_config],
        ),
        Node(
            condition=IfCondition(enable_drivers),
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_baselink_to_laser",
            arguments=[
                "0.27",
                "0.0",
                "0.11",
                "0.0",
                "0.0",
                "0.0",
                "base_link",
                "laser",
            ],
        ),
        Node(
            condition=AndCondition(
                IfCondition(enable_drivers), IfCondition(enable_teleop)
            ),
            package="joy_teleop",
            executable="joy_teleop",
            name="joy_teleop",
            parameters=[joy_config],
        ),
        Node(
            condition=IfCondition(enable_deadman),
            package="f1tenth_control",
            executable="rl_deadman_gate",
            name="rl_deadman_gate",
            output="screen",
            parameters=[gate_config],
        ),
    ]

    safety = Node(
        condition=IfCondition(enable_safety),
        package="f1tenth_control",
        executable="safety",
        name="safety_node",
        output="screen",
        parameters=[overlay_params_file],
    )

    default_gigaflow_racer = os.path.join(agent_share, "config", "gigaflow_policy.yaml")
    gigaflow_racer = Node(
        condition=IfCondition(enable_racer),
        package="f1tenth_rl_agent",
        executable="gigaflow_racer",
        name="gigaflow_racer",
        output="screen",
        parameters=[
            default_gigaflow_racer,
            overlay_params_file,
            {
                "checkpoint_path": ParameterValue(checkpoint_path, value_type=str),
                "device": ParameterValue(device, value_type=str),
                "recording_mode": ParameterValue(recording_mode, value_type=bool),
                "i_drive_max_a": ParameterValue(i_drive_max_a, value_type=float),
                "i_brake_max_a": ParameterValue(i_brake_max_a, value_type=float),
                "i_slew_a_per_s": ParameterValue(i_slew_a_per_s, value_type=float),
                "online_ppo_enable": ParameterValue(
                    online_ppo_enable, value_type=bool
                ),
                "online_ppo_rollout_steps": ParameterValue(
                    online_ppo_rollout_steps, value_type=int
                ),
                "online_ppo_lr": ParameterValue(online_ppo_lr, value_type=float),
                "online_ppo_kl_coef": ParameterValue(
                    online_ppo_kl_coef, value_type=float
                ),
                "online_ppo_kl_clip": ParameterValue(
                    online_ppo_kl_clip, value_type=float
                ),
                "online_ppo_output_dir": ParameterValue(
                    online_ppo_output_dir, value_type=str
                ),
            },
        ],
    )

    default_gate = os.path.join(control_share, "config", "rl_current_gate.yaml")
    gate_remappings = []
    if dry_run_enabled:
        gate_remappings = [
            ("commands/motor/current", "commands/motor/current_dryrun"),
            ("commands/motor/brake", "commands/motor/brake_dryrun"),
            ("commands/servo/position", "commands/servo/position_dryrun"),
        ]

    current_gate = Node(
        condition=IfCondition(enable_gate),
        package="f1tenth_control",
        executable="rl_current_gate",
        name="rl_current_gate",
        output="screen",
        parameters=[
            default_gate,
            overlay_params_file,
            {
                "i_drive_max_a": ParameterValue(i_drive_max_a, value_type=float),
                "i_brake_max_a": ParameterValue(i_brake_max_a, value_type=float),
                "i_brake_safe_a": ParameterValue(i_brake_safe_a, value_type=float),
                "i_slew_a_per_s": ParameterValue(i_slew_a_per_s, value_type=float),
            },
        ],
        remappings=gate_remappings,
    )
    return [*drivers, safety, gigaflow_racer, current_gate]


def generate_launch_description() -> LaunchDescription:
    stack_share = get_package_share_directory("f1tenth_stack")
    default_overlay = "/config/params.yaml"
    default_sensors = os.path.join(stack_share, "config", "sensors.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "overlay_params_file",
                default_value=default_overlay,
                description="Per-car overlay YAML (node-scoped ros__parameters).",
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
                "joy_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="joy + joy_teleop YAML.",
            ),
            DeclareLaunchArgument(
                "gate_config",
                default_value=LaunchConfiguration("overlay_params_file"),
                description="RL deadman gate and rl_current_gate overlay YAML.",
            ),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="/policies/gigaflow_policy.pt",
                description="IRL Gigaflow actor artifact (actor_*.pt, baked condition).",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="Inference device: cpu or cuda.",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="Remap VESC current/brake/servo to *_dryrun topics.",
            ),
            DeclareLaunchArgument(
                "enable_drivers",
                default_value="true",
                description="Start vendored sensor/teleop drivers.",
            ),
            DeclareLaunchArgument(
                "enable_teleop",
                default_value="false",
                description="Start joy_teleop on /teleop (manual stick; off for RL).",
            ),
            DeclareLaunchArgument(
                "enable_deadman",
                default_value="true",
                description="Start rl_deadman_gate (set false for boxed bench tests).",
            ),
            DeclareLaunchArgument(
                "enable_racer",
                default_value="true",
                description="Start the gigaflow_racer inference node.",
            ),
            DeclareLaunchArgument(
                "enable_gate",
                default_value="true",
                description="Start rl_current_gate (sole VESC command owner).",
            ),
            DeclareLaunchArgument(
                "enable_safety",
                default_value="false",
                description="Start the optional safety brake publisher on /brake.",
            ),
            DeclareLaunchArgument(
                "recording_mode",
                default_value="false",
                description="Publish raw/processed IMU record topics.",
            ),
            DeclareLaunchArgument(
                "i_drive_max_a",
                default_value="60.0",
                description="Drive current limit (A).",
            ),
            DeclareLaunchArgument(
                "i_brake_max_a",
                default_value="20.0",
                description="Brake current limit (A).",
            ),
            DeclareLaunchArgument(
                "i_slew_a_per_s",
                default_value="200.0",
                description="Current slew limit (A/s).",
            ),
            DeclareLaunchArgument(
                "i_brake_safe_a",
                default_value="5.0",
                description="Fail-closed safe brake current (A).",
            ),
            DeclareLaunchArgument(
                "online_ppo_enable",
                default_value="false",
                description="Live on-car KL-clipped PPO from SOURCE_RL ticks.",
            ),
            DeclareLaunchArgument(
                "online_ppo_rollout_steps",
                default_value="128",
                description="SOURCE_RL ticks collected before one PPO update.",
            ),
            DeclareLaunchArgument(
                "online_ppo_lr",
                default_value="3e-5",
                description="On-car PPO Adam learning rate.",
            ),
            DeclareLaunchArgument(
                "online_ppo_kl_coef",
                default_value="0.1",
                description="KL penalty coefficient for on-car PPO.",
            ),
            DeclareLaunchArgument(
                "online_ppo_kl_clip",
                default_value="0.05",
                description="Skip the actor step when approx KL exceeds this.",
            ),
            DeclareLaunchArgument(
                "online_ppo_output_dir",
                default_value="/policies/online_ppo",
                description="Directory for actor_latest.pt from live PPO.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

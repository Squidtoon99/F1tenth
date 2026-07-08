"""Top-level launch for simulation.

Scaffold placeholder. Brings up the gym bridge (see sim/) plus the selected
racing stack, without the physical vehicle drivers.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stack",
                default_value="rl",
                description="Which racing stack to run in sim: 'algo' or 'rl'.",
            ),
            # TODO: include the gym bridge launch + the selected racing stack.
        ]
    )

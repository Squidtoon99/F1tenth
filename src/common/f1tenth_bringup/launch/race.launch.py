"""Top-level launch for on-car racing.

Scaffold placeholder. Composes vehicle drivers + localization + the selected
racing stack (algorithmic or RL) + control. Which stack runs is selected via a
launch argument; per-car params/map come from the mounted config overlay.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "stack",
                default_value="algo",
                description="Which racing stack to run: 'algo' or 'rl'.",
            ),
            # TODO: include vehicle bringup, localization, mapping (optional),
            # the selected racing stack, and control here.
        ]
    )

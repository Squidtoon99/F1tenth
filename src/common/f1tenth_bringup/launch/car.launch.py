"""Vehicle-only bringup (drivers + sensors + safety), no autonomy.

Scaffold placeholder. Useful for hardware shakedown and teleop before layering a
racing stack on top via race.launch.py.
"""

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            # TODO: include the vendored vehicle bringup (src/vehicle) here.
        ]
    )

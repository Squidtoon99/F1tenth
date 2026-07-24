from setuptools import find_packages, setup
import os
from glob import glob

package_name = "f1tenth_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description="Path-tracking control and drive-command layer.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "drive_command = f1tenth_control.drive_command_node:main",
            "pp_driver = f1tenth_control.pp_driver:main",
            "pp_driver_plus = f1tenth_control.pp_driver_plus:main",
            "pp_ftg_driver = f1tenth_control.pp_ftg_driver:main",
            "gap_driver = f1tenth_control.gap_driver:main",
            "pid_driver = f1tenth_control.pid_driver:main",
            "safety = f1tenth_control.safety:main",
            "rl_deadman_gate = f1tenth_control.rl_deadman_gate_node:main",
            "vesc_actuator = f1tenth_control.vesc_actuator_node:main",
            "rl_current_gate = f1tenth_control.rl_current_gate_node:main",
        ],
    },
)

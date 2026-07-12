import os
from glob import glob

from setuptools import find_packages, setup

package_name = "f1tenth_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="f1tenth-genesis",
    maintainer_email="dev@todo.todo",
    description="F1TENTH autonomous track mapping with slam_toolbox and DFS exploration.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "exploration = f1tenth_mapping.exploration_node:main",
            "navigator = f1tenth_mapping.navigator_node:main",
            "reactive_explorer = f1tenth_mapping.reactive_explorer_node:main",
            "scan_throttle = f1tenth_mapping.scan_throttle_node:main",
            "occupancy_mapper = f1tenth_mapping.occupancy_mapper_node:main",
        ],
    },
)

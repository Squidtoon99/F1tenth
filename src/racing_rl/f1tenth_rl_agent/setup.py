import os
from glob import glob

from setuptools import find_packages, setup

package_name = "f1tenth_rl_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "assets"), glob("assets/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="f1tenth-genesis",
    maintainer_email="dev@todo.todo",
    description="ROS 2 nodes to run a trained F1TENTH QRSAC policy in f1tenth_gym_ros.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "track_server = f1tenth_rl_agent.track_server_node:main",
            "observation_builder = f1tenth_rl_agent.observation_builder_node:main",
            "obs_debug = f1tenth_rl_agent.obs_debug_node:main",
            "policy_inference = f1tenth_rl_agent.policy_inference_node:main",
            "localization_preflight = f1tenth_rl_agent.localization_preflight_node:main",
            "evaluation = f1tenth_rl_agent.evaluation_node:main",
            "scripted_opponent = f1tenth_rl_agent.scripted_opponent_node:main",
        ],
    },
)

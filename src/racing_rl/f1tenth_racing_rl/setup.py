from setuptools import find_packages, setup

package_name = "f1tenth_racing_rl"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # torch is provided by the Docker image; f1tenth_contract is installed editable
    # / discovered by colcon. Declared here for documentation.
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description="On-car RL inference nodes (observation builder, policy inference, action decode).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # TODO: e.g.
            # "observation_builder = f1tenth_racing_rl.observation_builder_node:main",
            # "policy_inference = f1tenth_racing_rl.policy_inference_node:main",
        ],
    },
)

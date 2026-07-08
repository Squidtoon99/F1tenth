from setuptools import find_packages, setup

package_name = "f1tenth_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description="Perception nodes (e.g. opponent/obstacle detection).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # TODO: e.g. "opponent_detector = f1tenth_perception.opponent_detector_node:main",
        ],
    },
)

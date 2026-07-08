from setuptools import find_packages, setup

package_name = "f1tenth_control"

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
    description="Path-tracking control (pure pursuit, optional MPC) and drive-command layer.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # TODO: e.g. "pure_pursuit = f1tenth_control.pure_pursuit_node:main",
            #       "drive_command = f1tenth_control.drive_command_node:main",
        ],
    },
)

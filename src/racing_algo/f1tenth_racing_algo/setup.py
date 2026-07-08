import os
from glob import glob

from setuptools import find_packages, setup

package_name = "f1tenth_racing_algo"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description="Composition/bringup of the algorithmic racing stack.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)

from setuptools import find_packages, setup
import os
from glob import glob

package_name = "f1tenth_localization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description="Particle-filter localization launch and wrappers.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pf_relocalize = f1tenth_localization.pf_relocalize_node:main",
        ],
    },
)

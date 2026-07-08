from setuptools import find_packages, setup

package_name = "f1tenth_planning"

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
    description="Raceline optimization and global/local planning.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # TODO: e.g. "raceline_optimizer = f1tenth_planning.raceline_optimizer_node:main",
        ],
    },
)

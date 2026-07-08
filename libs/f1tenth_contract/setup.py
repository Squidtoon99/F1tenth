from setuptools import find_packages, setup

package_name = "f1tenth_contract"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    # These data_files make the directory a valid ament_python package so colcon
    # can discover/build it from the repo root. They are harmless for plain pip
    # installs.
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description=(
        "Single source of truth for the observation/action layout shared by RL "
        "training and the on-car inference node."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)

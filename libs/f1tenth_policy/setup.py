from setuptools import find_packages, setup

package_name = "f1tenth_policy"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "torch"],
    zip_safe=True,
    maintainer="maintainers",
    maintainer_email="dev@example.com",
    description=(
        "Shared Lee sensor-policy layout, GRU actor, normalization, artifact "
        "validation, and delta-steering semantics for training and deploy."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},
)

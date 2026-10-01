from glob import glob

from setuptools import find_packages, setup

package_name = "greenseg"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Fernando Canadas Aranega",
    maintainer_email="fernando.ca@ual.es",
    description="GreenSeg ground segmentation (GPF + normal/curvature-verified region growing)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "greenseg_node = greenseg.greenseg_node:main",
        ],
    },
)

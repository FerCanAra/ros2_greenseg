# GreenSeg: Ground Segmentation Algorithm for Agricultural Robots in Mediterranean Greenhouses using RGB-D Point Clouds🌱

[![arXiv](https://arxiv.org/abs/2605.25279)
[![Last Updated](https://img.shields.io/badge/last%20updated-2026--10-blue)](.)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22679881.svg)](https://doi.org/10.5281/zenodo.22679881)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

This paper presents GreenSeg, a robust perception framework for autonomous navigation using RGB-D sensing. The proposed method introduces a dual-layer validation strategy: a robust global plane fitting combined with a surface curvature filter for terrain adaptability, and a seed-point-based Region Growing constraint to ensure the spatial continuity of the navigable plane. Experimental validation was conducted using the AGRICOBIOT I
platform across four diurnal scenarios with varying solar elevations. The results show that GreenSeg consistently outperforms benchmark segmentation methods, achieving peak improvements of 11.58% in mean Recall and 19.24% in mIoU during critical rotational maneuvers at the end of corridors. These findings confirm that the proposed algorithm enables stable and safe autonomous navigation in unstructured, dynamic agricultural environments
that are subject to budget constraints and sensitive to lighting conditions.

The simulator has been tested on ROS2 Humble, Ubuntu 22.04 LTS.

> **Note:** Process tested with on 5th October 2026.

📑 Citation
------------------
*Paper citation*
------------------
```
@article{canadas2026ros2,
  title={GreenSeg: Ground Segmentation Algorithm for Agricultural Robots in Mediterranean Greenhouses using RGB-D Point Clouds},
  author={Ca{\~n}adas-Ar{\'a}nega, Fernando and Moreno, Jos{\'e} C. and Blanco-Claraco, Jos{\'e} L.},
  journal={Computer Modeling in Engineering \& Sciences},
  volume={-},
  number={-},
  pages={-},
  year={2026},
  publisher={Tech Science Press},
  note={Article in press}
}
```
*Software citation*
------------------
```
@software{agricultural_benchmark,
  author  = {Ca{\~n}adas-Ar{\'a}nega, Fernando},
  title   = {GreenSeg: Ground Segmentation Algorithm for Agricultural Robots in Mediterranean Greenhouses using RGB-D Point Clouds},
  version = {1.0.1},
  year    = {2026},
  doi     = {https://doi.org/10.5281/zenodo.22679881},
  url     = {https://github.com/FerCanAra/robotics_benchmark_greenhouse/tree/main}
}
```
📜 License
--------------------
This project is distributed under the **BSD 3-Clause License**.

Copyright © 2026, Individual contributors  
Project owner: Fernando Cañadas Aránega <fernando.ca@ual.es> (University of Almeria) and collaborators

See the [LICENSE](LICENSE) file for full license text.

⚙️ Prerequisites
--------------------
In order to use the algotihm, you must have the following packages installed:

1. [Robot Operating System (ROS 2) Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html). It is recommended to install the full desktop.
2. Realsense Library. It is easily installed with the following command:
```
cd ~/ros2_ws/src # Create this path if you didn't have it available.
git clone https://github.com/realsenseai/realsense-ros.git
git clone https://github.com/realsenseai/librealsense.git
```
3. Dependencies: Base Pointcloud Groundfloor Segmentationn:
```
sudo -E wget -O- https://amrdocs.intel.com/repos/gpg-keys/GPG-PUB-KEY-INTEL-AMR.gpg | sudo tee /usr/share/keyrings/amr-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/amr-archive-keyring.gpg] https://amrdocs.intel.com/repos/$(source /etc/os-release && echo $VERSION_CODENAME) amr main" | sudo tee /etc/apt/sources.list.d/amr.list > /dev/null
sudo apt update
apt-cache search groundfloor
sudo apt install ros-humble-pointcloud-groundfloor-segmentation
```

🛠️ Install and build
--------------------
In order to run the greenseg, the official project repository must be installed. This can be done easily using the following commands.

````
mkdir -p ros2_ws/src && cd ros2_ws/src # If you created it in the requirements section, skip this line.
git clone https://github.com/FerCanAra/greenseg.git
cd ../..
sudo apt install python3-scipy ros-$ROS_DISTRO-cv-bridge
rosdep update
rosdep install --from-paths src -y --ignore-src
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.bash
````

🚀 Usage
--------------------

Launch your camera application with the topics "/camera/depth/image_raw" and "/camera/depth/camera_info" enabled (you can also use a ROS 2 bag containing this information). Then, in another terminal, run the following command:

**Real robot**
------------------

```bash
ros2 launch greenseg greenseg.launch.py
```

**Simulation**
------------------

```
ros2 launch greenseg greenseg.launch.py use_sim_time:=true \
    depth_topic:=/camera/depth/image_raw camera_info_topic:=/camera/depth/camera_info
```
> **Note:**  The `depth_filter_node` is not needed: the range filter and the `[::2, ::2]` (`pixel_stride: 2`) are built-in. If it continues to be used, set `pixel_stride: 1`.

| Topic | tipe | Content |
|---|---|---|
| `/greenseg/obstacles` | PointCloud2 | P_obs^RG (ec. 20) → costmap local |
| `/greenseg/ground` | PointCloud2 | P_ground^RG (ec. 19) |
| `/greenseg/labeled` | PointCloud2 | x y z `label` `rho` `kappa` (0 ground, 1 obstacle, 2 above, 3 noise) |

In RViz, use `/greenseg/labeled` with *Color Transformer = Intensity* and the `label` channel.




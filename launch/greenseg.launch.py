"""Launch GreenSeg.

  ros2 launch greenseg greenseg.launch.py
  ros2 launch greenseg greenseg.launch.py depth_topic:=/camera/depth/image_raw \
      camera_info_topic:=/camera/depth/camera_info use_sim_time:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("greenseg"), "config", "greenseg.yaml")
    args = [
        DeclareLaunchArgument("depth_topic", default_value="/camera/camera/depth/image_rect_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/depth/camera_info"),
        DeclareLaunchArgument("params_file", default_value=cfg),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    node = Node(
        package="greenseg",
        executable="greenseg_node",
        name="greenseg",
        output="screen",
        parameters=[LaunchConfiguration("params_file"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        remappings=[
            ("~/depth", LaunchConfiguration("depth_topic")),
            ("~/camera_info", LaunchConfiguration("camera_info_topic")),
            ("~/labeled", "/greenseg/labeled"),
            ("~/ground", "/greenseg/ground"),
            ("~/obstacles", "/greenseg/obstacles"),
            ("~/label_image", "/greenseg/label_image"),
        ],
    )
    return LaunchDescription(args + [node])

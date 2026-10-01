"""
+-------------------------------------------------------------------------+
|                         GreenSeg ROS 2 node.                            |
| Copyright (C) 2026  Fernando Cañadas Aránega                            |
| PhD Student University of Almería, Spain                                |
| Contact: fernando.ca@ual.es                                             |
| Distributed under 3-clause BSD License                                  |
| See COPYING                                                             |
+-------------------------------------------------------------------------+

Subscribes
  ~/depth         sensor_msgs/Image       16UC1 [mm] (RealSense) or 32FC1 [m] (sim)
  ~/camera_info   sensor_msgs/CameraInfo

Publishes (frame: base_frame, default base_link)
  ~/labeled       sensor_msgs/PointCloud2  x y z label rho kappa
                  label: 0 ground, 1 obstacle, 2 above, 3 noise
  ~/ground        sensor_msgs/PointCloud2  P_ground^RG 
  ~/obstacles     sensor_msgs/PointCloud2  P_obs^RG  -> local costmap
  ~/label_image   sensor_msgs/Image mono8  per-pixel label of the (strided) depth
                  image, 255 = invalid (organized method only)

Parameter `method`: "organized" (default, image-space neighbourhoods, fast)
or "kdtree" (reference implementation with spherical KD-tree neighbourhoods).
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from greenseg.core import (GROUND, OBSTACLE, DepthProjector, GreenSegParams,
                           greenseg, greenseg_organized, quaternion_to_rotation,
                           transform_points)

_XYZ_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_LABELED_FIELDS = _XYZ_FIELDS + [
    PointField(name="label", offset=12, datatype=PointField.UINT32, count=1),
    PointField(name="rho", offset=16, datatype=PointField.FLOAT32, count=1),
    PointField(name="kappa", offset=20, datatype=PointField.FLOAT32, count=1),
]
_LABELED_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("label", "<u4"), ("rho", "<f4"), ("kappa", "<f4")])


def _cloud(header: Header, fields, data: np.ndarray) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(data.shape[0])
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = data.dtype.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def _xyz_cloud(header: Header, pts: np.ndarray) -> PointCloud2:
    data = np.ascontiguousarray(pts, dtype=np.float32).view(
        np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])).ravel()
    return _cloud(header, _XYZ_FIELDS, data)


class GreenSegNode(Node):

    def __init__(self):
        super().__init__("greenseg")
        d = GreenSegParams()
        # --- algorithm parameters (Table 2 defaults) ---
        self.declare_parameter("max_surface_height", d.h_ground)
        self.declare_parameter("max_incline", d.max_incline_deg)
        self.declare_parameter("robot_height", d.robot_height)
        self.declare_parameter("n_neighbors", d.n_neighbors)
        self.declare_parameter("r_neighbors", d.r_neighbors)
        self.declare_parameter("rho_min", d.rho_min)
        self.declare_parameter("kappa_max", d.kappa_max)
        self.declare_parameter("r_growing", d.r_growing)
        self.declare_parameter("max_distance_filtered", d.max_depth)
        self.declare_parameter("min_distance_filtered", d.min_depth)
        self.declare_parameter("radial_min_distance", d.radial_min_distance)
        self.declare_parameter("gpf_iterations", d.gpf_iterations)
        self.declare_parameter("gpf_lpr_fraction", d.gpf_lpr_fraction)
        self.declare_parameter("gpf_seed_threshold", d.gpf_seed_threshold)
        self.declare_parameter("gpf_max_points", d.gpf_max_points)
        self.declare_parameter("seed_band", d.seed_band)
        self.declare_parameter("height_reference", d.height_reference)
        self.declare_parameter("voxel_size", d.voxel_size)
        self.declare_parameter("normal_max_neighbors", d.normal_max_neighbors)
        self.declare_parameter("window_half", d.window_half)
        self.declare_parameter("window_max_step", d.window_max_step)
        self.declare_parameter("growing_pixel_radius", d.growing_pixel_radius)
        self.declare_parameter("normal_decimation", d.normal_decimation)
        self.declare_parameter("method", "organized")
        # --- node parameters ---
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("pixel_stride", 2)          # 2 = [::2, ::2]
        self.declare_parameter("depth_scale_16u", 0.001)   # 16UC1 units -> m
        self.declare_parameter("tf_timeout", 0.1)
        self.declare_parameter("use_sensor_data_qos", True)
        self.declare_parameter("publish_labeled", True)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.prm = GreenSegParams(
            h_ground=g("max_surface_height"), max_incline_deg=g("max_incline"),
            robot_height=g("robot_height"), n_neighbors=int(g("n_neighbors")),
            r_neighbors=g("r_neighbors"), rho_min=g("rho_min"),
            kappa_max=g("kappa_max"), r_growing=g("r_growing"),
            max_depth=g("max_distance_filtered"),
            min_depth=g("min_distance_filtered"),
            radial_min_distance=g("radial_min_distance"),
            gpf_iterations=int(g("gpf_iterations")),
            gpf_lpr_fraction=g("gpf_lpr_fraction"),
            gpf_seed_threshold=g("gpf_seed_threshold"),
            gpf_max_points=int(g("gpf_max_points")),
            seed_band=float(g("seed_band")),
            height_reference=g("height_reference"),
            voxel_size=g("voxel_size"),
            normal_max_neighbors=int(g("normal_max_neighbors")),
            window_half=int(g("window_half")),
            window_max_step=int(g("window_max_step")),
            growing_pixel_radius=int(g("growing_pixel_radius")),
            normal_decimation=int(g("normal_decimation")))
        self.method = str(g("method")).lower()
        if self.method not in ("organized", "kdtree"):
            raise ValueError(f"Unknown method '{self.method}' (organized|kdtree)")
        n_win = (2 * self.prm.window_half + 1) ** 2
        if self.method == "organized" and self.prm.n_neighbors > 0.75 * n_win:
            self.get_logger().warn(
                f"n_neighbors={self.prm.n_neighbors} is close to the window size "
                f"({n_win} samples): increase window_half")
        if self.prm.normal_max_neighbors < self.prm.n_neighbors:
            self.get_logger().warn("normal_max_neighbors < n_neighbors: raising it")
            self.prm.normal_max_neighbors = self.prm.n_neighbors
        self.base_frame = g("base_frame")
        self.stride = max(1, int(g("pixel_stride")))
        self.depth_scale = g("depth_scale_16u")
        self.tf_timeout = Duration(seconds=float(g("tf_timeout")))
        self.publish_labeled = bool(g("publish_labeled"))

        self.bridge = CvBridge()
        self.projector = DepthProjector()
        self.camera_info = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # depth=1: never queue frames; if processing is slower than the
        # camera, stale frames are dropped instead of accumulating latency.
        qos_in = QoSProfile(
            reliability=(QoSReliabilityPolicy.BEST_EFFORT if g("use_sensor_data_qos")
                         else QoSReliabilityPolicy.RELIABLE),
            history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, "~/camera_info", self._info_cb, qos_in)
        self.create_subscription(Image, "~/depth", self._depth_cb, qos_in)

        self.pub_labeled = self.create_publisher(PointCloud2, "~/labeled", 1)
        self.pub_ground = self.create_publisher(PointCloud2, "~/ground", 1)
        self.pub_obs = self.create_publisher(PointCloud2, "~/obstacles", 1)
        self.pub_label_img = self.create_publisher(Image, "~/label_image", 1)

        self.get_logger().info(
            f"GreenSeg started | method={self.method} frame={self.base_frame} "
            f"stride={self.stride} "
            f"depth=[{self.prm.min_depth}, {self.prm.max_depth}] m "
            f"voxel={self.prm.voxel_size} m")

    # ------------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo):
        self.camera_info = msg

    def _lookup(self, source_frame: str, stamp):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, source_frame, Time.from_msg(stamp), self.tf_timeout)
        except TransformException:
            try:   # static camera mount: latest transform is equivalent
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, source_frame, Time())
            except TransformException as ex:
                self.get_logger().warn(f"TF {source_frame}->{self.base_frame}: {ex}",
                                       throttle_duration_sec=5.0)
                return None
        q = tf.transform.rotation
        tr = tf.transform.translation
        return (quaternion_to_rotation(q.x, q.y, q.z, q.w),
                np.array([tr.x, tr.y, tr.z]))

    def _depth_cb(self, msg: Image):
        if self.camera_info is None:
            self.get_logger().warn("Waiting for camera_info...", throttle_duration_sec=5.0)
            return
        info = self.camera_info
        if info.width and (info.width != msg.width or info.height != msg.height):
            self.get_logger().warn(
                f"camera_info {info.width}x{info.height} != depth {msg.width}x{msg.height}",
                throttle_duration_sec=5.0)
            return
        Rt = self._lookup(msg.header.frame_id, msg.header.stamp)
        if Rt is None:
            return

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        kw = dict(stride=self.stride, depth_scale_16u=self.depth_scale,
                  min_depth=self.prm.min_depth, max_depth=self.prm.max_depth)
        try:
            if self.method == "organized":
                img = self.projector.project_organized(depth, info.k, msg.encoding, **kw)
                P_img = transform_points(img.reshape(-1, 3), *Rt).reshape(img.shape)  # Eq. (1)
                res = greenseg_organized(P_img, img[..., 2],
                                         float(info.k[0]) / self.stride, self.prm)
            else:
                pts_cam = self.projector.project(depth, info.k, msg.encoding, **kw)
                res = greenseg(transform_points(pts_cam, *Rt), self.prm)    # Eq. (1)
        except ValueError as ex:
            self.get_logger().error(str(ex), throttle_duration_sec=5.0)
            return

        header = Header(stamp=msg.header.stamp, frame_id=self.base_frame)
        self.pub_ground.publish(_xyz_cloud(header, res.points[res.labels == GROUND]))
        self.pub_obs.publish(_xyz_cloud(header, res.points[res.labels == OBSTACLE]))
        if self.publish_labeled:
            data = np.empty(res.points.shape[0], dtype=_LABELED_DTYPE)
            data["x"], data["y"], data["z"] = res.points.T
            data["label"] = res.labels
            data["rho"] = np.nan_to_num(res.rho, nan=-1.0)
            data["kappa"] = np.nan_to_num(res.kappa, nan=-1.0)
            self.pub_labeled.publish(_cloud(header, _LABELED_FIELDS, data))
        if res.pixel_index is not None and self.pub_label_img.get_subscription_count() > 0:
            limg = self.bridge.cv2_to_imgmsg(res.label_image(), encoding="mono8")
            limg.header = msg.header          # camera frame, strided resolution
            self.pub_label_img.publish(limg)

        t = res.timings
        self.get_logger().info(
            f"n={t.get('n_points', 0)} total={t.get('total_ms', 0):.0f} ms "
            f"(gpf {t.get('base_gpf_ms', 0):.0f}, normals {t.get('normals_ms', 0):.0f}, "
            f"rg {t.get('region_growing_ms', 0):.0f})", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = GreenSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
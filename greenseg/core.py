"""
+-------------------------------------------------------------------------+
|                 GreenSeg core algorithm (ROS-independent                |
| Copyright (C) 2026  Fernando Cañadas Aránega                            |
| PhD Student University of Almería, Spain                                |
| Contact: fernando.ca@ual.es                                             |
| Distributed under 3-clause BSD License                                  |
| See COPYING                                                             |
+-------------------------------------------------------------------------+
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GROUND = 0
OBSTACLE = 1
ABOVE = 2
NOISE = 3
LABEL_NAMES = {GROUND: "ground", OBSTACLE: "obstacle", ABOVE: "above", NOISE: "noise"}


# ---------------------------------------------------------------------------
# Parameters 
# ---------------------------------------------------------------------------
@dataclass
class GreenSegParams:
    # --- Table 2 ---
    h_ground: float = 0.12          # max_surface_height [m]
    max_incline_deg: float = 30.0   # max_incline [deg]
    robot_height: float = 0.5       # robot_height H_robot [m]
    n_neighbors: int = 30           # min |N_i| for a valid normal
    r_neighbors: float = 0.05       # PCA neighbourhood radius r [m]
    rho_min: float = 0.90           # normal consistency threshold
    kappa_max: float = 0.05         # curvature threshold
    r_growing: float = 0.05         # region-growing adjacency radius r_g [m]
    max_depth: float = 3.0          # max_distance_filtered [m] (camera depth)
    min_depth: float = 0.3          # min_distance_filtered [m] (camera depth)

    # --- filter in the xy plane of base_link ---
    radial_min_distance: float = 0.3

    # --- GPF seeding  ---
    gpf_iterations: int = 3
    gpf_lpr_fraction: float = 0.05  # fraction of lowest points used as LPR
    gpf_seed_threshold: float = 0.10  # seeds: z < mean(LPR) + threshold
    gpf_max_points: int = 20000     # subsample for plane fitting (0 = all)

    # --- Region-growing seed ---
    seed_band: float = 0.5

    # --- Implementation options ---
    height_reference: str = "base_link"
    # Voxel downsampling before processing (0 = disabled). Caps the point
    # density of the near field so that the PCA cost stays bounded.
    voxel_size: float = 0.01
    # Upper bound on |N_i| used for PCA (nearest points within r). Must be
    # >= n_neighbors; the validity test |N_i| >= n_neighbors is unaffected.
    normal_max_neighbors: int = 64

    # --- Organized (image-space) variant, see greenseg_organized() ---
    # Window of (2K+1)^2 samples; the sampling step adapts to depth so the
    # window spans ~r_neighbors around each pixel.
    window_half: int = 4                # K -> 9x9 = 81 samples (~63 inside r)
    window_max_step: int = 8            # max pixel step between samples
    growing_pixel_radius: int = 1       # image-space reach of region-growing edges (1 = 8-connectivity)
    # PCA evaluated on one pixel out of normal_decimation^2 (lattice) and
    # copied to the remaining query pixels of each cell (1 = every pixel).
    normal_decimation: int = 2


@dataclass
class GreenSegResult:
    points: np.ndarray                  # (N,3) processed points (base_link)
    labels: np.ndarray                  # (N,) uint8, GreenSeg labels
    base_labels: np.ndarray             # (N,) uint8, base algorithm labels
    rho: np.ndarray                     # (N,) float, NaN where not computed
    kappa: np.ndarray                   # (N,) float, NaN where not computed
    plane_normal: np.ndarray            # (3,) refined n_hat
    plane_offset: float                 # refined d_hat
    plane_valid: bool                   # inclination within max_incline
    timings: dict = field(default_factory=dict)
    # Organized variant only: flat (row-major) pixel index of each point in
    # the (strided) image, and the image shape -> labels can be painted
    # back into a mask image.
    pixel_index: np.ndarray | None = None
    image_shape: tuple | None = None

    def label_image(self, fill: int = 255) -> np.ndarray:
        """(H, W) uint8 label image (organized variant only)."""
        img = np.full(self.image_shape, fill, dtype=np.uint8)
        img.ravel()[self.pixel_index] = self.labels
        return img

    def mask(self, label: int) -> np.ndarray:
        return self.labels == label

    @property
    def ground(self) -> np.ndarray:
        return self.points[self.labels == GROUND]

    @property
    def obstacles(self) -> np.ndarray:
        return self.points[self.labels == OBSTACLE]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix from a unit quaternion (ROS order x, y, z, w)."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class DepthProjector:
    """Back-projects a depth image into the camera optical frame.

    Handles both RealSense-style 16UC1 (millimetres) and simulator-style
    32FC1 (metres) images; invalid pixels (0, NaN, inf) are discarded.
    """

    def __init__(self):
        self._key = None
        self._ray_x = None
        self._ray_y = None

    def _update_rays(self, h: int, w: int, fx: float, fy: float,
                     cx: float, cy: float, stride: int):
        key = (h, w, fx, fy, cx, cy, stride)
        if key == self._key:
            return
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        self._ray_x = ((u - cx) / fx).astype(np.float64)
        self._ray_y = ((v - cy) / fy).astype(np.float64)
        self._key = key

    def project(self, depth: np.ndarray, K, encoding: str, stride: int = 1,
                depth_scale_16u: float = 0.001,
                min_depth: float = 0.0, max_depth: float = np.inf) -> np.ndarray:
        fx, fy, cx, cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        h, w = depth.shape[:2]
        self._update_rays(h, w, fx, fy, cx, cy, stride)
        d = depth[::stride, ::stride]
        enc = encoding.lower()
        if enc in ("16uc1", "mono16"):
            z = d.astype(np.float64) * depth_scale_16u
        elif enc == "32fc1":
            z = d.astype(np.float64)
        else:
            raise ValueError(f"Unsupported depth encoding '{encoding}'")
        valid = np.isfinite(z) & (z > 0.0) & (z >= min_depth) & (z <= max_depth)
        z = z[valid]
        return np.column_stack((self._ray_x[valid] * z, self._ray_y[valid] * z, z))

    def project_organized(self, depth: np.ndarray, K, encoding: str, stride: int = 1,
                          depth_scale_16u: float = 0.001, min_depth: float = 0.0,
                          max_depth: float = np.inf) -> np.ndarray:
        """Like project() but keeps the image structure: returns an
        (H', W', 3) array in the optical frame with NaN at invalid pixels
        (H' = ceil(H/stride), W' = ceil(W/stride))."""
        fx, fy, cx, cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        h, w = depth.shape[:2]
        self._update_rays(h, w, fx, fy, cx, cy, stride)
        d = depth[::stride, ::stride]
        enc = encoding.lower()
        if enc in ("16uc1", "mono16"):
            z = d.astype(np.float64) * depth_scale_16u
        elif enc == "32fc1":
            z = d.astype(np.float64)
        else:
            raise ValueError(f"Unsupported depth encoding '{encoding}'")
        valid = np.isfinite(z) & (z > 0.0) & (z >= min_depth) & (z <= max_depth)
        z = np.where(valid, z, np.nan)
        return np.stack((self._ray_x * z, self._ray_y * z, z), axis=-1)


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Eq. (1): p_base = R p_cam + t."""
    return points @ R.T + t


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0.0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def fit_plane(points: np.ndarray):
    """Least-squares plane (Eq. 7): n = eigenvector of the smallest
    eigenvalue of the covariance, d = -n^T centroid, with n_z >= 0."""
    c = points.mean(axis=0)
    C = np.cov((points - c).T)
    _, V = np.linalg.eigh(C)
    n = V[:, 0]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


# ---------------------------------------------------------------------------
# Base algorithm (Sec. 3.2.2)
# ---------------------------------------------------------------------------
def ground_plane_fitting(P: np.ndarray, prm: GreenSegParams):
    """Ground Plane Fitting with a horizontal prior (Eqs. 3-5, 7).

    Returns (n, d, valid) where valid is False if no plane within
    max_incline could be found (a horizontal fallback is then returned).
    """
    if prm.gpf_max_points > 0 and P.shape[0] > prm.gpf_max_points:
        # plane estimation on a fixed-stride subsample (deterministic);
        # the classification of Eq. (6) is still applied to every point
        P = P[:: int(np.ceil(P.shape[0] / prm.gpf_max_points))]
    z = P[:, 2]
    k = max(3, int(prm.gpf_lpr_fraction * P.shape[0]))
    k = min(k, P.shape[0])
    lpr = np.partition(z, k - 1)[:k].mean()
    seeds = P[z < lpr + prm.gpf_seed_threshold]
    if seeds.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0]), float(-lpr), False

    n, d = fit_plane(seeds)
    for _ in range(prm.gpf_iterations):
        inl = np.abs(P @ n + d) <= prm.h_ground
        if inl.sum() < 3:
            break
        n, d = fit_plane(P[inl])

    theta = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))
    if theta > prm.max_incline_deg:
        # Horizontal prior violated (Eq. 4): fall back to a horizontal plane
        return np.array([0.0, 0.0, 1.0]), float(-np.median(seeds[:, 2])), False
    return n, d, True


def classify_base(P: np.ndarray, n: np.ndarray, d: float, plane_valid: bool,
                  prm: GreenSegParams) -> np.ndarray:
    """Eq. (6)."""
    delta = P @ n + d
    height = P[:, 2] if prm.height_reference == "base_link" else delta
    labels = np.full(P.shape[0], NOISE, dtype=np.uint8)
    ground = (np.abs(delta) <= prm.h_ground) & plane_valid
    obstacle = ~ground & (height > prm.h_ground) & (height <= prm.robot_height)
    above = ~ground & (height > prm.robot_height)
    labels[ground] = GROUND
    labels[obstacle] = OBSTACLE
    labels[above] = ABOVE
    return labels


# ---------------------------------------------------------------------------
# GreenSeg verification (Sec. 4.2)
# ---------------------------------------------------------------------------
def eig_sym3(C: np.ndarray):
    """Closed-form eigen-decomposition of a batch of symmetric 3x3 matrices
    (much faster than np.linalg.eigh for large batches).

    Returns (w (M,3) ascending eigenvalues, v_min (M,3) unit eigenvector of
    the smallest eigenvalue).
    """
    a00, a11, a22 = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    a01, a02, a12 = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    ps = np.where(p > 1e-30, p, 1.0)
    b00, b11, b22 = (a00 - q) / ps, (a11 - q) / ps, (a22 - q) / ps
    b01, b02, b12 = a01 / ps, a02 / ps, a12 / ps
    detB = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    phi = np.arccos(np.clip(detB / 2.0, -1.0, 1.0)) / 3.0
    l_max = q + 2.0 * p * np.cos(phi)
    l_min = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    l_mid = 3.0 * q - l_max - l_min
    w = np.column_stack((l_min, l_mid, l_max))

    # eigenvector of l_min: cross product of two rows of (C - l_min I)
    r0 = np.column_stack((a00 - l_min, a01, a02))
    r1 = np.column_stack((a01, a11 - l_min, a12))
    r2 = np.column_stack((a02, a12, a22 - l_min))
    c = np.stack((np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)), axis=1)
    nrm = np.linalg.norm(c, axis=2)
    best = np.argmax(nrm, axis=1)
    i = np.arange(C.shape[0])
    v = c[i, best]
    nb = nrm[i, best]
    v = np.where(nb[:, None] > 1e-30, v / np.where(nb > 1e-30, nb, 1.0)[:, None],
                 np.array([0.0, 0.0, 1.0]))
    return w, v


def local_normals(query: np.ndarray, support: np.ndarray, support_tree: cKDTree,
                  radius: float, max_neighbors: int = 64):
    """Eqs. (10)-(12), (15), fully vectorised.

    N_i = the (at most max_neighbors) nearest points of the support cloud
    within `radius` of p_i (the point itself included). Capping the
    neighbourhood bounds the cost in the dense near field; with the default
    1 cm voxel grid a planar 5 cm disc holds ~78 points, so the cap barely
    shrinks the neighbourhood extent.

    Returns (normals (M,3), kappa (M,), counts (M,)).
    """
    M = query.shape[0]
    k = min(max_neighbors, support.shape[0])
    dist, idx = support_tree.query(query, k=k, distance_upper_bound=radius,
                                   workers=-1)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    valid = np.isfinite(dist)                    # missing neighbours -> inf
    counts = valid.sum(axis=1).astype(np.float64)
    cnt = np.maximum(counts, 1.0)

    # Centre on the query point (numerically safer), masked moments.
    idx = np.where(valid, idx, 0)
    D = (support[idx] - query[:, None, :]) * valid[:, :, None]   # (M, k, 3)
    x, y, z = D[:, :, 0], D[:, :, 1], D[:, :, 2]
    mx, my, mz = x.sum(1) / cnt, y.sum(1) / cnt, z.sum(1) / cnt
    C = np.empty((M, 3, 3))                      # Eq. (11)
    C[:, 0, 0] = (x * x).sum(1) / cnt - mx * mx
    C[:, 1, 1] = (y * y).sum(1) / cnt - my * my
    C[:, 2, 2] = (z * z).sum(1) / cnt - mz * mz
    C[:, 0, 1] = C[:, 1, 0] = (x * y).sum(1) / cnt - mx * my
    C[:, 0, 2] = C[:, 2, 0] = (x * z).sum(1) / cnt - mx * mz
    C[:, 1, 2] = C[:, 2, 1] = (y * z).sum(1) / cnt - my * mz

    w, normals = eig_sym3(C)                     # ascending eigenvalues
    w = np.clip(w, 0.0, None)
    total = w.sum(axis=1)
    kappa = np.divide(w[:, 0], total, out=np.full(M, np.inf), where=total > 1e-12)
    return normals, kappa, counts


def select_seed_component(points: np.ndarray, comp: np.ndarray,
                          band: float) -> np.ndarray:
    """Robust version of Eq. (17).

    The paper takes the single candidate closest to the robot as seed. With
    real depth data that point often lies on a small fragment isolated by
    invalid pixels (or on a near-field reflection); the region then covers
    only that fragment and Eq. (20) turns the rest of the floor into
    obstacles. Here the seed region is the connected component with the
    most candidates inside the band [d_min, d_min + band] of xy distance to
    the robot (band = 0 reproduces the paper exactly).
    """
    d = np.hypot(points[:, 0], points[:, 1])
    if band <= 0.0:
        return comp == comp[int(np.argmin(d))]
    near = d <= d.min() + band
    best = np.bincount(comp[near]).argmax()
    return comp == best


def region_growing(Pk: np.ndarray, radius: float, seed_band: float = 0.0) -> np.ndarray:
    """Eqs. (17)-(19): connected component (adjacency radius r_g) that
    contains the seed (see select_seed_component).
    Returns a boolean mask over Pk."""
    if Pk.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    pairs = cKDTree(Pk).query_pairs(radius, output_type="ndarray")
    N = Pk.shape[0]
    G = sparse.coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])),
                          shape=(N, N))
    _, comp = connected_components(G, directed=False)
    return select_seed_component(Pk, comp, seed_band)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def greenseg(points_base: np.ndarray, prm: GreenSegParams) -> GreenSegResult:
    """Run base segmentation + GreenSeg verification on a cloud already
    expressed in base_link (after Eq. 1)."""
    t0 = time.perf_counter()
    P = voxel_downsample(points_base, prm.voxel_size)

    # Eq. (2): radial pre-filter
    P = P[np.hypot(P[:, 0], P[:, 1]) >= prm.radial_min_distance]
    N = P.shape[0]
    rho = np.full(N, np.nan)
    kappa = np.full(N, np.nan)
    if N < 3:
        empty = np.full(N, NOISE, dtype=np.uint8)
        return GreenSegResult(P, empty, empty.copy(), rho, kappa,
                              np.array([0.0, 0.0, 1.0]), 0.0, False)
    t1 = time.perf_counter()

    # Base algorithm: GPF + Eq. (6)
    n, d, plane_valid = ground_plane_fitting(P, prm)
    base = classify_base(P, n, d, plane_valid, prm)
    # Eq. (7): refinement on P_ground^(0)
    g0 = np.flatnonzero(base == GROUND)
    if g0.size >= 3:
        st = max(1, int(np.ceil(g0.size / prm.gpf_max_points))) if prm.gpf_max_points > 0 else 1
        n, d = fit_plane(P[g0[::st]])
    t2 = time.perf_counter()

    labels = base.copy()
    if g0.size > 0:
        # Eqs. (10)-(12), (15): normals of P_ground^(0) over P_valid
        tree = cKDTree(P)
        normals, kap, counts = local_normals(P[g0], P, tree, prm.r_neighbors,
                                             prm.normal_max_neighbors)
        rh = np.abs(normals @ n)                                  # Eq. (13)
        rho[g0] = rh
        kappa[g0] = kap
        t3 = time.perf_counter()

        enough = counts >= prm.n_neighbors
        delta = P[g0] @ n + d
        cand = (enough & (rh >= prm.rho_min) & (kap <= prm.kappa_max)
                & (np.abs(delta) <= prm.h_ground))                # Eqs. (14),(16),(18.i)
        cand_idx = g0[cand]
        in_region = np.zeros(g0.size, dtype=bool)
        in_region[cand] = region_growing(P[cand_idx], prm.r_growing,
                                         prm.seed_band)        # Eqs. (17)-(19)
        t4 = time.perf_counter()

        # Eq. (20): rejected ground -> obstacle (conservative);
        # points without a valid neighbourhood -> noise.
        rejected = g0[~in_region]
        labels[rejected] = OBSTACLE
        labels[g0[~enough]] = NOISE
    else:
        t3 = t4 = time.perf_counter()

    t5 = time.perf_counter()
    timings = {
        "prefilter_ms": 1e3 * (t1 - t0),
        "base_gpf_ms": 1e3 * (t2 - t1),
        "normals_ms": 1e3 * (t3 - t2),
        "region_growing_ms": 1e3 * (t4 - t3),
        "total_ms": 1e3 * (t5 - t0),
        "n_points": int(N),
    }
    return GreenSegResult(P, labels, base, rho, kappa, n, float(d), plane_valid, timings)


# ---------------------------------------------------------------------------
# Organized (image-space) variant
# ---------------------------------------------------------------------------
# Same pipeline and thresholds as greenseg(), but it exploits the fact that a
# depth image is an organized cloud: neighbours of a pixel are looked up by
# shifting the image instead of querying a KD-tree.
#
#   Eq. (10)  N_i = { p_j in a (2K+1)x(2K+1) window around pixel i, sampled
#             with step s_i, such that ||p_j - p_i|| <= r }
#             s_i = clip(floor(r / (K * z_i / f_x)), 1, s_max), i.e. the
#             window spans ~r around p_i whatever its depth.
#   Eq. (18)  region growing on the graph whose edges join candidate pixels
#             that are at most `growing_pixel_radius` pixels apart in the
#             image and at most r_g apart in 3D.
# ---------------------------------------------------------------------------
def _shift(a: np.ndarray, pad: int, r0: int, r1: int, dy: int, dx: int, W: int):
    """Rows [r0, r1) of the un-padded image shifted by (dy, dx) pixels."""
    return a[pad + r0 + dy: pad + r1 + dy, pad + dx: pad + dx + W]


def _pixel_spacing(P_img: np.ndarray, axis: int) -> np.ndarray:
    """3D distance to the closest adjacent pixel along `axis`
    (min of forward/backward differences, robust to depth edges)."""
    d = np.linalg.norm(np.diff(P_img, axis=axis), axis=-1)
    pad_shape = list(d.shape)
    pad_shape[axis] = 1
    nanpad = np.full(pad_shape, np.nan)
    fwd = np.concatenate((d, nanpad), axis=axis)
    bwd = np.concatenate((nanpad, d), axis=axis)
    with np.errstate(invalid="ignore"):
        return np.fmin(fwd, bwd)


def organized_normals(P_img: np.ndarray, z_cam: np.ndarray, query: np.ndarray,
                      fx_eff: float, prm: GreenSegParams):
    """Masked-window PCA for the pixels in `query` (bool (H,W)).

    Returns (normals (M,3), kappa (M,), counts (M,)) for query pixels in
    row-major order.
    """
    H, W, _ = P_img.shape
    K = prm.window_half
    r2 = prm.r_neighbors ** 2
    rows, cols = np.nonzero(query)
    M = rows.size
    if M == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    # Sampling steps (rows, cols) adapted to the measured 3D spacing between
    # adjacent pixels, so that the window spans ~r in both image directions
    # (on the ground, seen at a grazing angle, rows are much further apart
    # than columns).
    du = _pixel_spacing(P_img, axis=1)[rows, cols]
    dv = _pixel_spacing(P_img, axis=0)[rows, cols]
    fallback = z_cam[rows, cols] / fx_eff
    du = np.where(np.isfinite(du), du, fallback)
    dv = np.where(np.isfinite(dv), dv, fallback)
    smax = prm.window_max_step
    sc = np.clip(np.floor(prm.r_neighbors / (K * np.maximum(du, 1e-6))), 1, smax).astype(np.int64)
    sr = np.clip(np.floor(prm.r_neighbors / (K * np.maximum(dv, 1e-6))), 1, smax).astype(np.int64)
    pad = K * smax

    # Gather the window samples of each query pixel by linear index into the
    # NaN-padded image (per-pixel steps, only query pixels are processed).
    Wp = W + 2 * pad
    # invalid pixels -> far-away sentinel: they fail the distance test and
    # keep every product finite (no NaN masking needed in the loop)
    X, Y, Z = (np.nan_to_num(np.pad(P_img[:, :, c].astype(np.float32), pad,
                                    constant_values=np.nan), nan=1e4).ravel()
               for c in range(3))
    base_lin = (rows + pad) * Wp + (cols + pad)
    cx, cy, cz = X[base_lin], Y[base_lin], Z[base_lin]

    acc = np.zeros((10, M), dtype=np.float32)  # n, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz
    for dy in range(-K, K + 1):
        row_off = dy * sr * Wp
        for dx in range(-K, K + 1):
            lin = base_lin + row_off + dx * sc
            ex = X[lin] - cx
            ey = Y[lin] - cy
            ez = Z[lin] - cz
            m = ((ex * ex + ey * ey + ez * ez) <= r2).astype(np.float32)
            ex *= m
            ey *= m
            ez *= m
            acc[0] += m
            acc[1] += ex; acc[2] += ey; acc[3] += ez
            acc[4] += ex * ex; acc[5] += ey * ey; acc[6] += ez * ez
            acc[7] += ex * ey; acc[8] += ex * ez; acc[9] += ey * ez
    acc = acc.T.astype(np.float64)

    cnt = np.maximum(acc[:, 0], 1.0)
    mx, my, mz = acc[:, 1] / cnt, acc[:, 2] / cnt, acc[:, 3] / cnt
    C = np.empty((M, 3, 3))                                   # Eq. (11)
    C[:, 0, 0] = acc[:, 4] / cnt - mx * mx
    C[:, 1, 1] = acc[:, 5] / cnt - my * my
    C[:, 2, 2] = acc[:, 6] / cnt - mz * mz
    C[:, 0, 1] = C[:, 1, 0] = acc[:, 7] / cnt - mx * my
    C[:, 0, 2] = C[:, 2, 0] = acc[:, 8] / cnt - mx * mz
    C[:, 1, 2] = C[:, 2, 1] = acc[:, 9] / cnt - my * mz

    w, v = eig_sym3(C)
    w = np.clip(w, 0.0, None)
    total = w.sum(axis=1)
    kappa = np.divide(w[:, 0], total, out=np.full(M, np.inf), where=total > 1e-12)
    return v, kappa, acc[:, 0]


def _decimated_normals(P_img, z_cam, query, fx_eff, prm):
    """organized_normals() evaluated on a decimated lattice: each query
    pixel takes the result of its cell representative (the first query
    pixel of its D x D cell in row-major order). Output order = row-major
    order of the query pixels."""
    D = max(1, int(prm.normal_decimation))
    if D == 1:
        return organized_normals(P_img, z_cam, query, fx_eff, prm)
    H, W = query.shape
    rows, cols = np.nonzero(query)
    cell = (rows // D) * ((W + D - 1) // D) + (cols // D)
    _, rep_pos, inv = np.unique(cell, return_index=True, return_inverse=True)
    rep = np.zeros_like(query)
    rep[rows[rep_pos], cols[rep_pos]] = True
    n_r, k_r, c_r = organized_normals(P_img, z_cam, rep, fx_eff, prm)
    # organized_normals returns the representatives in row-major order
    order = np.argsort(rows[rep_pos] * W + cols[rep_pos])
    pos_in_out = np.empty_like(order)
    pos_in_out[order] = np.arange(order.size)
    j = pos_in_out[inv]
    return n_r[j], k_r[j], c_r[j]


def organized_region_growing(P_img: np.ndarray, cand: np.ndarray,
                             prm: GreenSegParams) -> np.ndarray:
    """Eqs. (17)-(19) on the pixel graph. Returns a bool (H,W) mask."""
    H, W = cand.shape
    n = int(cand.sum())
    out = np.zeros_like(cand)
    if n == 0:
        return out
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[cand] = np.arange(n)
    R = prm.growing_pixel_radius
    rg2 = prm.r_growing ** 2
    src, dst = [], []
    for dy in range(0, R + 1):
        for dx in range(-R, R + 1):
            if dy == 0 and dx <= 0:
                continue                      # half-plane: each edge once
            ya, yb = 0, H - dy
            xa, xb = max(0, -dx), W - max(0, dx)
            A = (slice(ya, yb), slice(xa, xb))
            B = (slice(ya + dy, yb + dy), slice(xa + dx, xb + dx))
            d = P_img[A] - P_img[B]
            ok = cand[A] & cand[B] & ((d * d).sum(-1) <= rg2)
            src.append(idx[A][ok]); dst.append(idx[B][ok])
    src = np.concatenate(src); dst = np.concatenate(dst)
    G = sparse.coo_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    _, comp = connected_components(G, directed=False)

    out[cand] = select_seed_component(P_img[cand], comp, prm.seed_band)  # Eq. (17)
    return out


def greenseg_organized(P_img: np.ndarray, z_cam: np.ndarray, fx_eff: float,
                       prm: GreenSegParams) -> GreenSegResult:
    """GreenSeg on an organized cloud.

    P_img : (H,W,3) points in base_link, NaN where invalid (after Eq. 1).
    z_cam : (H,W) depth along the optical axis [m] (NaN where invalid).
    fx_eff: focal length in pixels of the (strided) image.
    """
    t0 = time.perf_counter()
    H, W, _ = P_img.shape
    valid = np.isfinite(P_img).all(-1)
    valid &= np.hypot(P_img[..., 0], P_img[..., 1]) >= prm.radial_min_distance  # Eq. (2)
    P_img = np.where(valid[..., None], P_img, np.nan)
    flat = np.flatnonzero(valid)
    P = P_img.reshape(-1, 3)[flat]
    N = P.shape[0]
    rho = np.full(N, np.nan)
    kappa = np.full(N, np.nan)
    if N < 3:
        empty = np.full(N, NOISE, dtype=np.uint8)
        return GreenSegResult(P, empty, empty.copy(), rho, kappa,
                              np.array([0.0, 0.0, 1.0]), 0.0, False,
                              pixel_index=flat, image_shape=(H, W))
    t1 = time.perf_counter()

    n, d, plane_valid = ground_plane_fitting(P, prm)
    base = classify_base(P, n, d, plane_valid, prm)
    g0 = np.flatnonzero(base == GROUND)
    if g0.size >= 3:
        st = max(1, int(np.ceil(g0.size / prm.gpf_max_points))) if prm.gpf_max_points > 0 else 1
        n, d = fit_plane(P[g0[::st]])                          # Eq. (7)
    t2 = time.perf_counter()

    labels = base.copy()
    if g0.size > 0:
        q = np.zeros(H * W, dtype=bool)
        q[flat[g0]] = True                    # row-major == order of g0
        normals, kap, counts = _decimated_normals(
            P_img, z_cam, q.reshape(H, W), fx_eff, prm)
        rh = np.abs(normals @ n)                               # Eq. (13)
        rho[g0] = rh
        kappa[g0] = kap
        t3 = time.perf_counter()

        enough = counts >= prm.n_neighbors
        delta = P[g0] @ n + d
        ok = (enough & (rh >= prm.rho_min) & (kap <= prm.kappa_max)
              & (np.abs(delta) <= prm.h_ground))
        cand = np.zeros(H * W, dtype=bool)
        cand[flat[g0[ok]]] = True
        region = organized_region_growing(P_img, cand.reshape(H, W), prm).ravel()
        in_region = region[flat[g0]]
        t4 = time.perf_counter()

        labels[g0[~in_region]] = OBSTACLE                      # Eq. (20)
        labels[g0[~enough]] = NOISE
    else:
        t3 = t4 = time.perf_counter()

    t5 = time.perf_counter()
    timings = {
        "prefilter_ms": 1e3 * (t1 - t0),
        "base_gpf_ms": 1e3 * (t2 - t1),
        "normals_ms": 1e3 * (t3 - t2),
        "region_growing_ms": 1e3 * (t4 - t3),
        "total_ms": 1e3 * (t5 - t0),
        "n_points": int(N),
    }
    return GreenSegResult(P, labels, base, rho, kappa, n, float(d), plane_valid,
                          timings, pixel_index=flat, image_shape=(H, W))
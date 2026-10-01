"""Offline validation of the GreenSeg core on synthetic greenhouse scenes.

Run:  python3 -m pytest -q test/test_core.py      (or python3 test/test_core.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greenseg.core import (ABOVE, GROUND, NOISE, OBSTACLE, DepthProjector,  # noqa: E402
                           GreenSegParams, greenseg, quaternion_to_rotation,
                           transform_points)

RNG = np.random.default_rng(0)


def synthetic_aisle(slope=0.03, aisle_width=1.0, spacing=0.01):
    """Returns (points, truth) in base_link. truth uses the same label ids;
    ghost structures are tagged with 10 (flat ghost) and 11 (scattered ghost)."""
    parts, truth = [], []

    # Ground of the aisle: z = slope * x + noise (5 mm)
    xs = np.arange(0.3, 3.0, spacing)
    ys = np.arange(-aisle_width / 2, aisle_width / 2, spacing)
    X, Y = np.meshgrid(xs, ys)
    Z = slope * X + RNG.normal(0, 0.005, X.shape)
    g = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    # hole under the flat ghost so that it is spatially disconnected
    parts.append(g); truth.append(np.full(len(g), GROUND))

    # Two crop rows (dense foliage volumes, 0 .. 1.2 m high)
    for y0 in (aisle_width / 2, -aisle_width / 2 - 0.3):
        n = 60000
        p = np.column_stack((RNG.uniform(0.3, 3.0, n),
                             RNG.uniform(y0, y0 + 0.3, n),
                             RNG.uniform(0.0, 1.2, n)))
        p[:, 2] += slope * p[:, 0]
        parts.append(p)
        truth.append(np.where(p[:, 2] - slope * p[:, 0] > 0.5, ABOVE, OBSTACLE))

    # Flat ghost patch (e.g. specular reflection): planar, within h_ground,
    # but floating 9 cm above the ground -> must NOT be accepted as ground.
    gx, gy = np.meshgrid(np.arange(2.0, 2.3, spacing), np.arange(-0.15, 0.15, spacing))
    gz = slope * gx + 0.09 + RNG.normal(0, 0.002, gx.shape)
    gp = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    parts.append(gp); truth.append(np.full(len(gp), 10))

    # Scattered ghost cluster touching the ground (multipath blob)
    n = 1500
    c = np.array([1.2, 0.1, slope * 1.2 + 0.05])
    sp = c + RNG.normal(0, 0.04, (n, 3))
    parts.append(sp); truth.append(np.full(n, 11))

    # Isolated sparse points (sensor noise)
    n = 200
    iso = np.column_stack((RNG.uniform(0.5, 2.8, n), RNG.uniform(-0.4, 0.4, n),
                           RNG.uniform(-0.1, 0.1, n)))
    parts.append(iso); truth.append(np.full(n, 12))

    return np.vstack(parts), np.concatenate(truth)


def report(res, truth):
    print(f"plane n={np.round(res.plane_normal, 4)} d={res.plane_offset:.4f} "
          f"valid={res.plane_valid}")
    print("timings:", {k: round(v, 1) if isinstance(v, float) else v
                       for k, v in res.timings.items()})
    names = {GROUND: "ground", OBSTACLE: "obstacle", ABOVE: "above", NOISE: "noise",
             10: "flat ghost", 11: "scatter ghost", 12: "isolated"}
    for t in np.unique(truth):
        m = truth == t
        counts = np.bincount(res.labels[m], minlength=4)
        base = np.bincount(res.base_labels[m], minlength=4)
        print(f"  truth={names[t]:13s} n={m.sum():6d} | base G/O/A/N={base} "
              f"| greenseg G/O/A/N={counts}")


def test_synthetic_aisle():
    P, truth = synthetic_aisle()
    prm = GreenSegParams(voxel_size=0.0)   # keep 1:1 with the ground truth
    # the synthetic cloud is already in base_link: skip nothing else
    keep = np.hypot(P[:, 0], P[:, 1]) >= prm.radial_min_distance
    P, truth = P[keep], truth[keep]
    res = greenseg(P, prm)
    # greenseg() re-applies the radial filter; indices are unchanged here
    assert res.points.shape[0] == P.shape[0]
    report(res, truth)

    lab = res.labels
    # Plane: slope 0.03 -> nearly vertical normal
    assert res.plane_valid and res.plane_normal[2] > 0.99
    # Real ground mostly kept as ground
    assert np.mean(lab[truth == GROUND] == GROUND) > 0.95
    # Flat ghost: accepted by the base algorithm, rejected by GreenSeg
    assert np.mean(res.base_labels[truth == 10] == GROUND) > 0.9
    assert np.mean(lab[truth == 10] == GROUND) < 0.05
    assert np.mean(lab[truth == 10] == OBSTACLE) > 0.9
    # Scattered ghost: never ground after verification
    assert np.mean(lab[truth == 11] == GROUND) < 0.25
    # Crop rows: never ground
    assert np.mean(lab[(truth == OBSTACLE) | (truth == ABOVE)] == GROUND) < 0.05


def _floor_view(w, h, fx):
    """Camera at (0.4, 0, 0.5) m in base_link, tilted 30 deg down, looking
    at a flat floor. Returns (depth_m, K, R, t)."""
    fy = fx
    cx, cy = w / 2, h / 2
    K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    # base_link -> camera_link (x fwd, y left, z up), pitch down 30 deg
    pitch = np.radians(30.0)
    R_link = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
    # camera_link -> optical frame (z fwd, x right, y down)
    R_opt = quaternion_to_rotation(-0.5, 0.5, -0.5, 0.5)
    R = R_link @ R_opt
    t = np.array([0.4, 0.0, 0.5])

    # Ray-cast the floor plane z=0
    v, u = np.mgrid[0:h, 0:w]
    rays_cam = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)), -1)
    rays_base = rays_cam @ R.T
    s = -t[2] / rays_base[..., 2]          # depth along optical z (ray z=1)
    s[(rays_base[..., 2] >= 0) | (s > 3.0)] = 0.0
    return s, K, R, t


def test_depth_projection_geometry():
    """Downsampled D435 depth (424x240). The recovered floor must lie at
    z ~ 0 in base_link and be segmented as ground up to ~2 m."""
    s, K, R, t = _floor_view(424, 240, 425.0 / 2)
    for enc, img in (("16UC1", (s * 1000).astype(np.uint16)),
                     ("32FC1", s.astype(np.float32))):
        pts = DepthProjector().project(img, K, enc, stride=1,
                                       min_depth=0.3, max_depth=3.0)
        pb = transform_points(pts, R, t)
        assert pb.shape[0] > 1000
        assert np.abs(pb[:, 2]).max() < 0.01, enc
        assert pb[:, 0].min() > 0.4, enc       # floor seen in front of camera
        res = greenseg(pb, GreenSegParams())
        assert res.plane_valid and res.plane_normal[2] > 0.999
        near = res.points[:, 0] <= 2.0
        fn = np.mean(res.labels[near] == GROUND)
        ff = np.mean(res.labels[~near] == GROUND)
        print(f"424x240 {enc}: {res.points.shape[0]} pts, ground x<=2m={fn:.3f} "
              f"x>2m={ff:.3f}, total={res.timings['total_ms']:.0f} ms")
        assert fn > 0.95


def test_full_resolution_far_field():
    """Full-resolution D435 depth (848x480): denser far field."""
    s, K, R, t = _floor_view(848, 480, 425.0)
    pts = DepthProjector().project(s.astype(np.float32), K, "32FC1",
                                   min_depth=0.3, max_depth=3.0)
    res = greenseg(transform_points(pts, R, t), GreenSegParams())
    near = res.points[:, 0] <= 2.0
    fn = np.mean(res.labels[near] == GROUND)
    ff = np.mean(res.labels[~near] == GROUND)
    print(f"848x480: {res.points.shape[0]} pts, ground x<=2m={fn:.3f} "
          f"x>2m={ff:.3f}, total={res.timings['total_ms']:.0f} ms")
    assert fn > 0.95


def test_steep_plane_rejected():
    """A 45 deg plane violates the horizontal prior: no ground at all."""
    X, Y = np.meshgrid(np.arange(0.5, 2.0, 0.01), np.arange(-0.5, 0.5, 0.01))
    P = np.column_stack((X.ravel(), Y.ravel(), X.ravel()))
    res = greenseg(P, GreenSegParams())
    assert not res.plane_valid
    assert np.sum(res.labels == GROUND) == 0


if __name__ == "__main__":
    test_synthetic_aisle()
    test_depth_projection_geometry()
    test_full_resolution_far_field()
    test_steep_plane_rejected()
    print("ALL TESTS PASSED")

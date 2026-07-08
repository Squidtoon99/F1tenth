"""Pure-Python tests for track CSV loading + boundary computation.

Runs without ROS 2 (no rclpy import).
"""

import os
import tempfile

import numpy as np

from f1tenth_rl_agent.track_io import compute_track_boundaries, load_track_csv
from _helpers import make_oval, write_track_csv


def test_load_track_csv_roundtrip():
    cl, wl, wr = make_oval(n=120)
    with tempfile.TemporaryDirectory() as d:
        path = write_track_csv(os.path.join(d, "track.csv"), cl, wl, wr)
        cl2, wl2, wr2 = load_track_csv(path)
    assert cl2.shape == (120, 2)
    assert wl2.shape == (120,)
    assert wr2.shape == (120,)
    np.testing.assert_allclose(cl2, cl, atol=1e-4)
    np.testing.assert_allclose(wl2, wl, atol=1e-4)
    np.testing.assert_allclose(wr2, wr, atol=1e-4)


def test_compute_boundaries_offset():
    cl, wl, wr = make_oval(n=200, width=1.5)
    left, right = compute_track_boundaries(cl, wl, wr)
    assert left.shape == cl.shape
    assert right.shape == cl.shape
    # boundaries should be offset from centerline by roughly the width
    d_left = np.linalg.norm(left - cl, axis=1)
    d_right = np.linalg.norm(right - cl, axis=1)
    np.testing.assert_allclose(d_left, 1.5, atol=1e-3)
    np.testing.assert_allclose(d_right, 1.5, atol=1e-3)


def test_load_iv2026_bundled_centerline():
    # Bundled asset lives at the package root: <pkg>/assets/. Assert that
    # load_track_csv reproduces the raw file contents and yields a sane track,
    # deriving expectations from the file itself (no brittle hard-coded counts).
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(pkg_root, "assets", "IV_2026_SIM_centerline.csv")
    assert os.path.exists(path), path

    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    n = raw.shape[0]
    assert n > 100  # a real closed-loop centerline, not a stub

    cl, wl, wr = load_track_csv(path)
    assert cl.shape == (n, 2)
    assert wl.shape == (n,)
    assert wr.shape == (n,)
    # Column order is x_m, y_m, w_tr_right_m, w_tr_left_m.
    np.testing.assert_allclose(cl, raw[:, :2], atol=1e-4)
    np.testing.assert_allclose(wr, raw[:, 2], atol=1e-4)
    np.testing.assert_allclose(wl, raw[:, 3], atol=1e-4)
    assert np.isfinite(cl).all()
    assert (wl > 0).all() and (wr > 0).all()

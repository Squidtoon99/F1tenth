"""Cross-check the cached/windowed Frenet projection against a brute-force one.

`frenet_projection_cached` uses a coarse nearest-vertex search (stride 10) plus a
+/-40 segment window. This test compares its proj / arc-length / tangent / lateral
error against an exhaustive all-segment projection over many random poses and
several track shapes, including deliberately uneven spacing and a sharp hairpin.
A large discrepancy means the windowing is selecting the wrong segment.
"""

from __future__ import annotations

import numpy as np
import torch

from conftest import (
    brute_force_frenet,
    build_track_state,
    make_circle_track,
    make_hairpin_track,
    make_uneven_track,
)

DEVICE = torch.device("cpu")


def _cached_frenet(real_modules, track_state, pos_xy):
    base_pos = torch.tensor(
        np.concatenate(
            [pos_xy.astype(np.float32), np.zeros((pos_xy.shape[0], 1), np.float32)],
            axis=-1,
        ),
        dtype=torch.float32,
    )
    ss = real_modules.utils.build_step_state(
        base_pos=base_pos,
        track_state=track_state,
        device=DEVICE,
        cache_id="frenet_xcheck",
    )
    return ss


def _sample_poses_near(cl: np.ndarray, rng, n=200, max_off=1.0, idx_lo=0,
                       idx_hi=None):
    hi = cl.shape[0] if idx_hi is None else idx_hi
    idx = rng.integers(idx_lo, hi, size=n)
    base = cl[idx]
    nxt = cl[(idx + 1) % cl.shape[0]]
    tang = nxt - base
    tang = tang / np.maximum(np.linalg.norm(tang, axis=-1, keepdims=True), 1e-8)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
    off = rng.uniform(-max_off, max_off, size=(n, 1)).astype(np.float32)
    return (base + off * nrm).astype(np.float32)


def _angle(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[:, 1], v[:, 0])


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _run_case(real_modules, cl, wl, wr, seed, *, proj_tol=2e-2, ey_tol=2e-2,
              head_tol=2e-2, idx_lo=0, idx_hi=None):
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    rng = np.random.default_rng(seed)
    pos = _sample_poses_near(cl, rng, idx_lo=idx_lo, idx_hi=idx_hi)

    ss = _cached_frenet(real_modules, ts, pos)
    fr = ss["frenet"]
    bf = brute_force_frenet(pos, cl)

    proj_err = np.linalg.norm(fr["proj"].numpy() - bf["proj"], axis=-1)
    ey_err = np.abs(ss["boundary"]["ey"].numpy() - bf["ey"])
    head_err = np.abs(_wrap(_angle(fr["seg_dir"].numpy()) - _angle(bf["seg_dir"])))
    L = float(fr["L"])
    ds = np.abs(ss["frenet"]["s"].numpy() - bf["s"])
    ds = np.minimum(ds, L - ds)

    # Position-level checks hold for EVERY sample: if the windowed search ever
    # picked a wrong far segment, proj/ey/s would blow up.
    assert proj_err.max() < proj_tol, (
        f"proj err {proj_err.max():.4f} "
        f"(n_bad={int((proj_err >= proj_tol).sum())})"
    )
    assert ey_err.max() < ey_tol, f"ey err {ey_err.max():.4f}"
    assert ds.max() < proj_tol + 1e-2, f"arclen err {ds.max():.4f}"

    # seg_dir is only unambiguous in a segment's interior: when the projection
    # lands on a shared vertex (best_t ~ 0 or 1) the cached and brute-force
    # implementations may legitimately pick different adjacent segments (a tie),
    # so restrict the heading comparison to interior projections.
    interior = (bf["best_t"] > 0.05) & (bf["best_t"] < 0.95)
    if interior.any():
        assert head_err[interior].max() < head_tol, (
            f"interior heading err {head_err[interior].max():.4f}"
        )


def test_frenet_circle(real_modules):
    cl, wl, wr = make_circle_track(radius=25.0, n=720)
    _run_case(real_modules, cl, wl, wr, seed=1)


# NOTE: a straight line is intentionally NOT cross-checked as a loop here. Closing
# it adds a collinear return segment that ties exactly with the forward segments,
# so float tie-breaking flips ey's sign - a fixture artifact impossible on a real
# curved loop. Straight-line ey/heading are covered analytically in
# test_observation_geometry.py instead.


def test_frenet_uneven_spacing(real_modules):
    """Uneven point density stresses the coarse(stride 10)+window(40) search."""
    cl, wl, wr = make_uneven_track(radius=20.0, n=300)
    _run_case(real_modules, cl, wl, wr, seed=3)


def test_frenet_hairpin(real_modules):
    cl, wl, wr = make_hairpin_track()
    _run_case(real_modules, cl, wl, wr, seed=4)

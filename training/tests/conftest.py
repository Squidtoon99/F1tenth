"""Shared fixtures/helpers for the observation accuracy audit tests.

These tests verify the training-side observation pipeline (`f1tenth_env`) without
spinning up a Genesis simulation. They load `f1tenth_env/utils.py`,
`f1tenth_env/observations.py` and `f1tenth_env/car.py` as standalone modules with
the few module-level `genesis` constants configured, mirroring the approach in
`ros2_deploy/f1tenth_rl_agent/test/test_obs_parity.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

gs = pytest.importorskip("genesis")

import torch  # noqa: E402


def init_genesis_headless(*, precision: str = "32") -> None:
    """Initialize Genesis without a display (CI / headless macOS)."""
    try:
        gs.utils.try_get_display_size()
    except Exception:
        import pyglet
        from genesis.vis.rasterizer import Rasterizer

        pyglet.options["headless"] = True

        def _headless_build(self):
            if self._context is None:
                return
            self.visualizer = self._context.visualizer

        Rasterizer.build = _headless_build

    if not gs._initialized:
        gs.init(backend=gs.cpu, precision=precision, logging_level="warning")


@pytest.fixture(scope="module")
def genesis_backend():
    init_genesis_headless(precision="32")
    return gs


@pytest.fixture(scope="module")
def genesis_backend_f64():
    init_genesis_headless(precision="64")
    return gs

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _configure_genesis() -> None:
    gs.tc_float = torch.float32
    gs.tc_int = torch.int32
    gs.device = torch.device("cpu")
    if getattr(gs, "EPS", None) is None:
        gs.EPS = 1e-12


def _stub_requests() -> None:
    """Stub `requests` so utils.py module-level load_tracks() returns immediately."""
    if "requests" in sys.modules:
        return

    class _Resp:
        status_code = 404

        def json(self):
            return {}

    stub = types.ModuleType("requests")
    stub.get = lambda *a, **k: _Resp()
    sys.modules["requests"] = stub


def _load_module(name: str, relpath: str):
    path = os.path.join(_REPO_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def real_modules():
    _configure_genesis()
    _stub_requests()
    utils = _load_module("audit_f1tenth_utils", "f1tenth_env/utils.py")
    observations = _load_module("audit_f1tenth_observations", "f1tenth_env/observations.py")
    car = _load_module("audit_f1tenth_car", "f1tenth_env/car.py")
    return types.SimpleNamespace(utils=utils, observations=observations, car=car)


@pytest.fixture(scope="session")
def obs_cfg() -> dict[str, Any]:
    """Mirror of config.py DEFAULT_CONFIG['obs'] (clip disabled for clean asserts)."""
    return {
        "num_obs": 380,
        "obs_scales": {"lin_vel": 1.0, "ang_vel": 1.0, "lin_acc": 1.0},
        "clip_obs": 0.0,
        "norm_clip": 10.0,
        "norm_eps": 1e-8,
        "contact_margin_m": 0.08,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        "future_track_min_lookahead_m": 5.0,
        "future_track_width": 2.2,
    }


# --- synthetic tracks ---------------------------------------------------------
def make_straight_track(length: float = 50.0, n: int = 200, w_left: float = 1.5,
                        w_right: float = 1.5):
    """Straight centerline along +x at y=0, evenly spaced over [0, length)."""
    x = np.linspace(0.0, length, n, endpoint=False).astype(np.float32)
    cl = np.stack([x, np.zeros_like(x)], axis=-1).astype(np.float32)
    wl = np.full(n, w_left, dtype=np.float32)
    wr = np.full(n, w_right, dtype=np.float32)
    return cl, wl, wr


def make_circle_track(radius: float = 25.0, n: int = 360, w_left: float = 1.5,
                      w_right: float = 1.5):
    """CCW circle centerline of given radius, centered at origin."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    cl = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=-1).astype(np.float32)
    wl = np.full(n, w_left, dtype=np.float32)
    wr = np.full(n, w_right, dtype=np.float32)
    return cl, wl, wr


def make_uneven_track(radius: float = 20.0, n: int = 300):
    """Circle with deliberately uneven point spacing (stresses coarse/window search)."""
    u = np.linspace(0.0, 1.0, n, endpoint=False).astype(np.float64)
    # warp the parameter so points bunch up on one side and spread on the other
    th = (2.0 * np.pi) * (u + 0.15 * np.sin(2.0 * np.pi * u))
    cl = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=-1).astype(np.float32)
    wl = np.full(n, 1.5, dtype=np.float32)
    wr = np.full(n, 1.5, dtype=np.float32)
    return cl, wl, wr


def make_hairpin_track():
    """Two straights joined by a 180-degree bend (sharp curvature)."""
    pts = []
    for x in np.linspace(0.0, 30.0, 120, endpoint=False):
        pts.append((x, 2.0))
    cx, cy, r = 30.0, 0.0, 2.0
    for a in np.linspace(np.pi / 2.0, -np.pi / 2.0, 60, endpoint=False):
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    for x in np.linspace(30.0, 0.0, 120, endpoint=False):
        pts.append((x, -2.0))
    cl = np.asarray(pts, dtype=np.float32)
    n = cl.shape[0]
    return cl, np.full(n, 1.0, dtype=np.float32), np.full(n, 1.0, dtype=np.float32)


def yaw_quat_wxyz(yaw) -> torch.Tensor:
    """(w, x, y, z) quaternion for a planar yaw rotation. Accepts scalar or array."""
    yaw = np.atleast_1d(np.asarray(yaw, dtype=np.float32))
    half = 0.5 * yaw
    quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
    quat[:, 0] = np.cos(half)
    quat[:, 3] = np.sin(half)
    return torch.tensor(quat, dtype=torch.float32)


def build_track_state(utils, cl, wl, wr, device=None):
    device = device or torch.device("cpu")
    return {
        "centerline": np.asarray(cl, dtype=np.float32),
        "w_tr_left": np.asarray(wl, dtype=np.float32),
        "w_tr_right": np.asarray(wr, dtype=np.float32),
        "w_tr_left_torch": torch.tensor(wl, dtype=torch.float32, device=device),
        "w_tr_right_torch": torch.tensor(wr, dtype=torch.float32, device=device),
        "track_geom_cache": {},
        "frenet_step_cache": {},
    }


# --- brute-force Frenet reference ---------------------------------------------
def _close_loop(cl: np.ndarray) -> np.ndarray:
    if np.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        return np.concatenate([cl, cl[:1]], axis=0)
    return cl


def brute_force_frenet(pos_xy: np.ndarray, centerline: np.ndarray):
    """Exhaustive nearest-segment projection over the closed centerline loop.

    Mirrors the arc-length/seg_dir conventions of utils.build_track_cache /
    frenet_projection_cached, but checks EVERY segment (no coarse search, no
    window) so it is an independent ground truth for the cached projection.

    Returns dict with arrays: best_idx, best_t, proj (B,2), seg_dir (B,2),
    s (B,), L (scalar), ey (B,).
    """
    cl = _close_loop(np.asarray(centerline, dtype=np.float64))
    c = cl[:-1]
    d = cl[1:]
    seg = d - c
    seg_len = np.maximum(np.linalg.norm(seg, axis=-1), 1e-8)
    cumlen = np.zeros_like(seg_len)
    cumlen[1:] = np.cumsum(seg_len[:-1])
    length = float(seg_len.sum())

    pos = np.asarray(pos_xy, dtype=np.float64)[:, :2]
    b = pos.shape[0]
    m = c.shape[0]

    # (b, m): projection param t of each pos onto each segment, clamped to [0,1]
    seg_len2 = (seg * seg).sum(-1)
    seg_len2 = np.maximum(seg_len2, 1e-10)
    diff = pos[:, None, :] - c[None, :, :]  # (b, m, 2)
    t = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = c[None, :, :] + t[..., None] * seg[None, :, :]  # (b, m, 2)
    dist2 = ((proj - pos[:, None, :]) ** 2).sum(-1)  # (b, m)
    best_idx = dist2.argmin(axis=1)

    ar = np.arange(b)
    best_t = t[ar, best_idx]
    best_proj = proj[ar, best_idx]
    best_seg = seg[best_idx]
    seg_dir = best_seg / np.maximum(np.linalg.norm(best_seg, axis=-1, keepdims=True), 1e-8)
    s = cumlen[best_idx] + best_t * seg_len[best_idx]

    n_hat = np.stack([-seg_dir[:, 1], seg_dir[:, 0]], axis=-1)
    ey = ((pos - best_proj) * n_hat).sum(-1)

    return {
        "best_idx": best_idx,
        "best_t": best_t,
        "proj": best_proj,
        "seg_dir": seg_dir,
        "s": s,
        "L": length,
        "ey": ey,
    }

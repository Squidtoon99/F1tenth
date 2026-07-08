"""Track CSV loading + boundary helpers shared by nodes.

Mirrors the parsing in ``f1tenth_env/utils.py`` (``load_track_state``): the CSV must
have columns ``x_m``, ``y_m``, ``w_tr_right_m`` and ``w_tr_left_m``.
"""

from __future__ import annotations

import numpy as np

REQUIRED_COLUMNS = {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"}


def load_track_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a centerline CSV.

    Returns ``(centerline[N, 2], w_tr_left[N], w_tr_right[N])`` as float32 arrays.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    if data.dtype.names is None:
        raise ValueError(f"Could not parse track csv with named columns: {path}")

    missing = REQUIRED_COLUMNS.difference(set(data.dtype.names))
    if missing:
        raise ValueError(f"Track csv missing required columns: {sorted(missing)}")

    centerline = np.stack([data["x_m"], data["y_m"]], axis=-1).astype(np.float32)
    w_tr_left = np.asarray(data["w_tr_left_m"], dtype=np.float32)
    w_tr_right = np.asarray(data["w_tr_right_m"], dtype=np.float32)
    return centerline, w_tr_left, w_tr_right


def compute_track_boundaries(
    centerline: np.ndarray,
    w_tr_left: np.ndarray,
    w_tr_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(left[N, 2], right[N, 2])`` boundary points (port of utils.py)."""
    cl = centerline.astype(np.float32)
    nxt = np.roll(cl, -1, axis=0)
    tangent = nxt - cl
    tangent_norm = np.clip(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8, None)
    tangent = tangent / tangent_norm

    normal = np.zeros_like(tangent)
    normal[:, 0] = -tangent[:, 1]
    normal[:, 1] = tangent[:, 0]

    left = cl + normal * w_tr_left[:, None]
    right = cl - normal * w_tr_right[:, None]
    return left, right

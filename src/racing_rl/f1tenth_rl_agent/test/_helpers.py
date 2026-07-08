"""Shared test helpers (synthetic track + CSV writer)."""

from __future__ import annotations

import numpy as np


def make_oval(n: int = 200, a: float = 20.0, b: float = 10.0, width: float = 1.5):
    """Return (centerline[n,2], w_tr_left[n], w_tr_right[n]) for an oval track."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False).astype(np.float32)
    cl = np.stack([a * np.cos(th), b * np.sin(th)], axis=-1).astype(np.float32)
    wl = np.full(n, width, dtype=np.float32)
    wr = np.full(n, width, dtype=np.float32)
    return cl, wl, wr


def write_track_csv(path: str, centerline, w_tr_left, w_tr_right) -> str:
    """Write a centerline CSV in f1tenth_racetracks column order."""
    header = "x_m,y_m,w_tr_right_m,w_tr_left_m"
    rows = np.column_stack(
        [centerline[:, 0], centerline[:, 1], w_tr_right, w_tr_left]
    ).astype(np.float32)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    return path

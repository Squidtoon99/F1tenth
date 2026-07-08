"""Load reference track CSVs and boundary geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from centerline_extractor import MapData
from track_geometry import reference_free_cell_fraction


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_track_io_path() -> None:
    track_io_root = repo_root() / "ros2_deploy" / "f1tenth_rl_agent"
    if str(track_io_root) not in sys.path:
        sys.path.insert(0, str(track_io_root))


def load_reference_track(reference_csv: Path):
    ensure_track_io_path()
    from f1tenth_rl_agent.track_io import compute_track_boundaries, load_track_csv

    ref_cl, ref_wl, ref_wr = load_track_csv(str(reference_csv))
    ref_left, ref_right = compute_track_boundaries(ref_cl, ref_wl, ref_wr)
    return ref_cl, ref_wl, ref_wr, ref_left, ref_right


def boundaries_from_result(centerline: np.ndarray, w_left: np.ndarray, w_right: np.ndarray):
    ensure_track_io_path()
    from f1tenth_rl_agent.track_io import compute_track_boundaries

    return compute_track_boundaries(
        centerline.astype(np.float32),
        w_left.astype(np.float32),
        w_right.astype(np.float32),
    )


def reference_on_map_free_pct(ref_cl: np.ndarray, data: MapData) -> float:
    return reference_free_cell_fraction(
        ref_cl, data.grid, data.origin_x, data.origin_y, data.resolution
    )

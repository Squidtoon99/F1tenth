"""Regression tests for SLAM map -> centerline post-processing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS = ROOT / "postprocess"
sys.path.insert(0, str(POSTPROCESS))

from centerline_extractor import extract_centerline, load_map_yaml, world_to_grid  # noqa: E402
from track_geometry import loop_length, nearest_distances  # noqa: E402
from track_reference import load_reference_track  # noqa: E402
from validate_track_alignment import STRICT_THRESHOLDS, build_alignment_report, passes  # noqa: E402

IV_MAP = Path(__file__).resolve().parents[2] / "ros2_deploy" / "assets" / "IV_2026_SIM.yaml"
IV_CENTERLINE = (
    Path(__file__).resolve().parents[2] / "ros2_deploy" / "assets" / "IV_2026_SIM_centerline.csv"
)


@pytest.mark.skipif(not IV_MAP.is_file(), reason="IV_2026_SIM map assets not present")
def test_extract_centerline_from_iv2026_map():
    data = load_map_yaml(IV_MAP)
    result = extract_centerline(data, spacing_m=0.15, half_width_fallback=1.1)
    assert result.centerline.shape[0] > 50
    assert result.w_tr_left.shape[0] == result.centerline.shape[0]
    assert np.all(result.w_tr_left > 0.3)
    assert np.all(result.w_tr_right > 0.3)


@pytest.mark.skipif(
    not IV_MAP.is_file() or not IV_CENTERLINE.is_file(),
    reason="IV_2026 assets not present",
)
def test_iv2026_known_track_matches_reference():
    data = load_map_yaml(IV_MAP)
    result = extract_centerline(data, spacing_m=0.15, half_width_fallback=1.1)
    ref_cl, _, _, ref_left, ref_right = load_reference_track(IV_CENTERLINE)
    report = build_alignment_report(result, data, ref_cl, ref_left, ref_right)
    assert passes(report, STRICT_THRESHOLDS)


@pytest.mark.skipif(
    not IV_MAP.is_file() or not IV_CENTERLINE.is_file(),
    reason="IV_2026 assets not present",
)
def test_iv2026_autonomous_extraction_still_produces_loop():
    data = load_map_yaml(IV_MAP)
    result = extract_centerline(
        data, spacing_m=0.15, half_width_fallback=1.1, prefer_known_track=False
    )
    ref_cl, _, _, _, _ = load_reference_track(IV_CENTERLINE)

    inside = 0
    for pt in result.centerline[::10]:
        row, col = world_to_grid(float(pt[0]), float(pt[1]), data)
        if 0 <= row < data.grid.shape[0] and 0 <= col < data.grid.shape[1]:
            if data.grid[row, col] == 0:
                inside += 1
    assert inside / max(len(result.centerline[::10]), 1) > 0.95

    cl_d = nearest_distances(result.centerline, ref_cl, step=5)
    ratio = loop_length(result.centerline) / loop_length(ref_cl)
    assert 0.90 <= ratio <= 1.30
    assert float(np.mean(cl_d)) < 15.0

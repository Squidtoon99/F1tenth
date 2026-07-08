"""Oschersleben map vs saved centerline alignment regression."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS = ROOT / "postprocess"
sys.path.insert(0, str(POSTPROCESS))

from centerline_extractor import extract_centerline, load_map_yaml  # noqa: E402
from track_geometry import loop_length, nearest_distances  # noqa: E402
from track_reference import load_reference_track  # noqa: E402
from validate_track_alignment import (  # noqa: E402
    STRICT_THRESHOLDS,
    evaluate_alignment,
    passes,
)

ASSETS = Path(__file__).resolve().parents[2] / "ros2_deploy" / "assets"
OSCH_MAP = ASSETS / "Oschersleben_map.yaml"
OSCH_CSV = ASSETS / "Oschersleben_centerline.csv"


@pytest.mark.skipif(not OSCH_MAP.is_file(), reason="Oschersleben map assets not present")
def test_oschersleben_extracted_path_is_in_free_space():
    report = evaluate_alignment(OSCH_MAP, OSCH_CSV)
    assert report.free_fraction >= 0.8


@pytest.mark.skipif(
    not OSCH_MAP.is_file() or not OSCH_CSV.is_file(),
    reason="Oschersleben assets not present",
)
def test_oschersleben_known_track_matches_reference():
    report = evaluate_alignment(OSCH_MAP, OSCH_CSV)
    assert passes(report, STRICT_THRESHOLDS)


@pytest.mark.skipif(
    not OSCH_MAP.is_file() or not OSCH_CSV.is_file(),
    reason="Oschersleben assets not present",
)
def test_oschersleben_autonomous_extraction_still_produces_loop():
    data = load_map_yaml(OSCH_MAP)
    result = extract_centerline(data, prefer_known_track=False)
    ref_cl, _, _, _, _ = load_reference_track(OSCH_CSV)
    cl_d = nearest_distances(result.centerline, ref_cl, step=3)
    ratio = loop_length(result.centerline) / loop_length(ref_cl)
    assert 0.95 <= ratio <= 1.05
    assert float(cl_d.mean()) < 5.0

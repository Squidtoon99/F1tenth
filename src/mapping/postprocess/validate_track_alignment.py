#!/usr/bin/env python3
"""Compare extracted map centerline/boundaries against a saved reference CSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from centerline_extractor import CenterlineResult, MapData, extract_centerline, load_map_yaml
from track_geometry import (
    best_cyclic_mean,
    centerline_drivable_fraction,
    loop_length,
    nearest_distances,
)
from track_reference import boundaries_from_result, load_reference_track, reference_on_map_free_pct


@dataclass
class AlignmentReport:
    centerline_mean_m: float
    centerline_max_m: float
    centerline_p95_m: float
    left_mean_m: float
    right_mean_m: float
    length_ratio: float
    free_fraction: float
    cyclic_mean_m: float | None


@dataclass(frozen=True)
class AlignmentThresholds:
    max_mean_m: float = 1.0
    max_p95_m: float = 2.0
    min_free_fraction: float = 0.8
    min_length_ratio: float = 0.6
    max_length_ratio: float = 1.4


STRICT_THRESHOLDS = AlignmentThresholds(
    max_mean_m=0.5,
    max_p95_m=1.0,
    min_free_fraction=0.95,
    min_length_ratio=0.95,
    max_length_ratio=1.05,
)


def build_alignment_report(
    extracted: CenterlineResult,
    data: MapData,
    ref_cl: np.ndarray,
    ref_left: np.ndarray,
    ref_right: np.ndarray,
) -> AlignmentReport:
    ext_cl = extracted.centerline
    ext_left, ext_right = boundaries_from_result(
        ext_cl, extracted.w_tr_left, extracted.w_tr_right
    )
    cl_d = nearest_distances(ext_cl, ref_cl, step=3)
    left_d = nearest_distances(ext_left, ref_left, step=5)
    right_d = nearest_distances(ext_right, ref_right, step=5)
    return AlignmentReport(
        centerline_mean_m=float(np.mean(cl_d)),
        centerline_max_m=float(np.max(cl_d)),
        centerline_p95_m=float(np.percentile(cl_d, 95)),
        left_mean_m=float(np.mean(left_d)),
        right_mean_m=float(np.mean(right_d)),
        length_ratio=loop_length(ext_cl) / max(loop_length(ref_cl), 1e-6),
        free_fraction=centerline_drivable_fraction(
            ext_cl, data.grid, data.origin_x, data.origin_y, data.resolution
        ),
        cyclic_mean_m=best_cyclic_mean(ref_cl, ext_cl),
    )


def passes(report: AlignmentReport, thresholds: AlignmentThresholds) -> bool:
    return (
        report.centerline_mean_m <= thresholds.max_mean_m
        and report.centerline_p95_m <= thresholds.max_p95_m
        and report.free_fraction >= thresholds.min_free_fraction
        and thresholds.min_length_ratio <= report.length_ratio <= thresholds.max_length_ratio
    )


def evaluate_alignment(
    map_yaml: Path,
    reference_csv: Path,
    spacing_m: float = 0.07,
    half_width_fallback: float = 1.1,
    extracted: CenterlineResult | None = None,
) -> AlignmentReport:
    data = load_map_yaml(map_yaml)
    if extracted is None:
        extracted = extract_centerline(
            data,
            spacing_m=spacing_m,
            half_width_fallback=half_width_fallback,
        )
    ref_cl, _, _, ref_left, ref_right = load_reference_track(reference_csv)
    return build_alignment_report(extracted, data, ref_cl, ref_left, ref_right)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.07)
    parser.add_argument("--half-width-fallback", type=float, default=1.1)
    parser.add_argument("--max-mean-m", type=float, default=1.0)
    parser.add_argument("--max-p95-m", type=float, default=2.0)
    parser.add_argument("--overlay-out", type=Path, default=None, help="Write comparison overlay PNG")
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    data = load_map_yaml(args.map_yaml)
    ref_cl, _, _, ref_left, ref_right = load_reference_track(args.reference_csv)
    extracted = extract_centerline(
        data,
        spacing_m=args.spacing_m,
        half_width_fallback=args.half_width_fallback,
    )
    report = build_alignment_report(extracted, data, ref_cl, ref_left, ref_right)
    ref_free = reference_on_map_free_pct(ref_cl, data)
    thresholds = AlignmentThresholds(max_mean_m=args.max_mean_m, max_p95_m=args.max_p95_m)

    if args.overlay_out is not None:
        from overlay_tracks import save_track_overlay

        ext_left, ext_right = boundaries_from_result(
            extracted.centerline, extracted.w_tr_left, extracted.w_tr_right
        )
        save_track_overlay(
            args.overlay_out,
            data,
            ref_cl,
            extracted.centerline,
            ref_left=ref_left,
            ref_right=ref_right,
            ext_left=ext_left,
            ext_right=ext_right,
        )

    print(f"Map: {args.map_yaml}")
    print(f"Reference: {args.reference_csv}")
    print(f"Reference on map free cells: {ref_free:.1f}% (frame sanity, NOT similarity)")
    print(
        f"Extracted vs ref centerline mean / p95 / max: "
        f"{report.centerline_mean_m:.3f} / {report.centerline_p95_m:.3f} / {report.centerline_max_m:.3f} m"
    )
    print(
        f"Extracted vs ref left / right boundary mean: "
        f"{report.left_mean_m:.3f} / {report.right_mean_m:.3f} m"
    )
    print(f"Length ratio (extracted / reference): {report.length_ratio:.3f}")
    print(f"Extracted points on drivable cells: {report.free_fraction * 100:.1f}%")
    if report.cyclic_mean_m is not None:
        print(f"Best cyclic alignment mean error: {report.cyclic_mean_m:.3f} m")

    ok = passes(report, thresholds)
    print("RESULT:", "PASS" if ok else "FAIL")
    if args.overlay_out is not None:
        print(f"Overlay: {args.overlay_out}")
    if args.fail_on_threshold and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

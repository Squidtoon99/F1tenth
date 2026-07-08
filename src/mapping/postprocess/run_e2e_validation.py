#!/usr/bin/env python3
"""End-to-end validation: SLAM map -> race assets -> metrics + comparison images.

When the full sim stack is unavailable, this script uses the bundled gym occupancy
grid as a stand-in for ``map_saver_cli`` output from an Oschersleben mapping run.
The post-processing and alignment checks are identical to production.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from centerline_extractor import (
    export_cleaned_map,
    extract_centerline,
    load_map_yaml,
    write_centerline_csv,
)
from overlay_tracks import (
    render_side_by_side_panels,
    render_single_track_panel,
    save_track_overlay,
)
from track_geometry import loop_length, resample_to_count
from track_reference import boundaries_from_result, load_reference_track
from track_viz import plot_width_profiles
from validate_track_alignment import build_alignment_report


@dataclass
class E2EReport:
    track: str
    timestamp_utc: str
    map_yaml: str
    reference_csv: str
    output_dir: str
    pipeline: str
    alignment: dict
    width_stats: dict
    images: dict[str, str]
    notes: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _width_stats(
    ref_wl: np.ndarray,
    ref_wr: np.ndarray,
    ext_wl: np.ndarray,
    ext_wr: np.ndarray,
) -> dict:
    count = min(len(ref_wl), len(ext_wl))
    ref_wl_r = resample_to_count(ref_wl, count)
    ref_wr_r = resample_to_count(ref_wr, count)
    ext_wl_r = resample_to_count(ext_wl, count)
    ext_wr_r = resample_to_count(ext_wr, count)
    return {
        "reference_left_mean_m": float(np.mean(ref_wl)),
        "reference_right_mean_m": float(np.mean(ref_wr)),
        "parsed_left_mean_m": float(np.mean(ext_wl)),
        "parsed_right_mean_m": float(np.mean(ext_wr)),
        "left_width_mean_abs_diff_m": float(np.mean(np.abs(ext_wl_r - ref_wl_r))),
        "right_width_mean_abs_diff_m": float(np.mean(np.abs(ext_wr_r - ref_wr_r))),
    }


def run_e2e(
    map_yaml: Path,
    reference_csv: Path,
    out_dir: Path,
    track_name: str,
    spacing_m: float = 0.07,
    half_width_fallback: float = 1.1,
) -> E2EReport:
    notes = [
        "Live SLAM loop (f1tenth_gym_ros + mapping_sim.launch.py) was not run in this "
        "environment (no Docker / slam_toolbox). The bundled gym map stands in for "
        "map_saver_cli output after a converged Oschersleben mapping session.",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_map_yaml(map_yaml)
    result = extract_centerline(
        data,
        spacing_m=spacing_m,
        half_width_fallback=half_width_fallback,
    )
    centerline_path = out_dir / f"{track_name}_centerline.csv"
    write_centerline_csv(result, centerline_path)
    export_cleaned_map(data, out_dir, track_name)

    ref_cl, ref_wl, ref_wr, ref_left, ref_right = load_reference_track(reference_csv)
    report = build_alignment_report(result, data, ref_cl, ref_left, ref_right)
    ext_left, ext_right = boundaries_from_result(
        result.centerline, result.w_tr_left, result.w_tr_right
    )

    width_stats = _width_stats(ref_wl, ref_wr, result.w_tr_left, result.w_tr_right)

    ref_panel_path = out_dir / f"{track_name}_e2e_reference_widths.png"
    parsed_panel_path = out_dir / f"{track_name}_e2e_parsed_widths.png"
    side_by_side_path = out_dir / f"{track_name}_e2e_side_by_side.png"
    overlay_path = out_dir / f"{track_name}_e2e_overlay.png"
    width_plot_path = out_dir / f"{track_name}_e2e_width_profiles.png"

    ref_panel = render_single_track_panel(
        data,
        ref_cl,
        ref_left,
        ref_right,
        center_color=(0, 200, 0),
        left_color=(255, 200, 0),
        right_color=(200, 100, 255),
        title="Reference",
    )
    parsed_panel = render_single_track_panel(
        data,
        result.centerline,
        ext_left,
        ext_right,
        center_color=(0, 0, 255),
        left_color=(0, 140, 255),
        right_color=(0, 255, 255),
        title="Parsed",
    )
    cv2.imwrite(str(ref_panel_path), ref_panel)
    cv2.imwrite(str(parsed_panel_path), parsed_panel)
    cv2.imwrite(
        str(side_by_side_path),
        render_side_by_side_panels(
            ref_panel,
            parsed_panel,
            "Actual (reference centerline + widths)",
            "Parsed (post-process output)",
        ),
    )
    save_track_overlay(
        overlay_path,
        data,
        ref_cl,
        result.centerline,
        ref_left=ref_left,
        ref_right=ref_right,
        ext_left=ext_left,
        ext_right=ext_right,
    )
    plot_width_profiles(ref_wl, ref_wr, result.w_tr_left, result.w_tr_right, width_plot_path)

    return E2EReport(
        track=track_name,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        map_yaml=str(map_yaml.resolve()),
        reference_csv=str(reference_csv.resolve()),
        output_dir=str(out_dir.resolve()),
        pipeline="load_map_yaml -> extract_centerline -> write_centerline_csv -> export_cleaned_map",
        alignment=asdict(report),
        width_stats=width_stats,
        images={
            "reference_widths": str(ref_panel_path.resolve()),
            "parsed_widths": str(parsed_panel_path.resolve()),
            "side_by_side": str(side_by_side_path.resolve()),
            "overlay": str(overlay_path.resolve()),
            "width_profiles": str(width_plot_path.resolve()),
        },
        notes=notes,
    )


def _write_markdown_report(report: E2EReport, md_path: Path) -> None:
    a = report.alignment
    w = report.width_stats
    lines = [
        f"# {report.track} end-to-end validation report",
        "",
        f"Generated: {report.timestamp_utc}",
        "",
        "## Pipeline",
        "",
        report.pipeline,
        "",
        "## Inputs",
        "",
        f"- Map: `{report.map_yaml}`",
        f"- Reference CSV: `{report.reference_csv}`",
        f"- Output dir: `{report.output_dir}`",
        "",
        "## Alignment metrics (parsed vs reference)",
        "",
        "| Metric | Value | Plan target |",
        "|---|---:|---|",
        f"| Centerline mean error (m) | {a['centerline_mean_m']:.4f} | < 0.5 |",
        f"| Centerline p95 error (m) | {a['centerline_p95_m']:.4f} | < 1.0 |",
        f"| Left boundary mean error (m) | {a['left_mean_m']:.4f} | < 0.5 |",
        f"| Right boundary mean error (m) | {a['right_mean_m']:.4f} | < 0.5 |",
        f"| Length ratio | {a['length_ratio']:.4f} | 0.95–1.05 |",
        f"| Free fraction | {a['free_fraction']:.3f} | ≥ 0.95 |",
        f"| Cyclic mean error (m) | {a.get('cyclic_mean_m', 'n/a')} | — |",
        "",
        "## Track width statistics",
        "",
        "| Stat | Reference | Parsed |",
        "|---|---:|---:|",
        f"| Mean left half-width (m) | {w['reference_left_mean_m']:.3f} | {w['parsed_left_mean_m']:.3f} |",
        f"| Mean right half-width (m) | {w['reference_right_mean_m']:.3f} | {w['parsed_right_mean_m']:.3f} |",
        f"| Mean |left width diff| (m) | — | {w['left_width_mean_abs_diff_m']:.3f} |",
        f"| Mean |right width diff| (m) | — | {w['right_width_mean_abs_diff_m']:.3f} |",
        "",
        "## Generated images",
        "",
    ]
    for label, path in report.images.items():
        lines.append(f"- **{label}**: `{path}`")
    lines.extend(["", "## Notes", ""])
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    md_path.write_text("\n".join(lines))


def main() -> None:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        default=repo / "ros2_deploy/f1tenth_rl_agent/assets/Oschersleben_map.yaml",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=repo / "ros2_deploy/assets/Oschersleben_centerline.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "ros2_mapping/output/oschersleben_e2e",
    )
    parser.add_argument("--track-name", type=str, default="Oschersleben")
    parser.add_argument("--spacing-m", type=float, default=0.07)
    parser.add_argument("--half-width-fallback", type=float, default=1.1)
    args = parser.parse_args()

    report = run_e2e(
        args.map_yaml,
        args.reference_csv,
        args.out_dir,
        args.track_name,
        spacing_m=args.spacing_m,
        half_width_fallback=args.half_width_fallback,
    )

    json_path = args.out_dir / f"{args.track_name}_e2e_report.json"
    md_path = args.out_dir / f"{args.track_name}_E2E_REPORT.md"
    json_path.write_text(json.dumps(asdict(report), indent=2))
    _write_markdown_report(report, md_path)

    a = report.alignment
    print(f"Track: {report.track}")
    print(f"Centerline mean/p95: {a['centerline_mean_m']:.4f} / {a['centerline_p95_m']:.4f} m")
    print(f"Boundary mean L/R: {a['left_mean_m']:.4f} / {a['right_mean_m']:.4f} m")
    print(f"Length ratio: {a['length_ratio']:.4f}")
    print(f"Report: {md_path}")
    for label, path in report.images.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()

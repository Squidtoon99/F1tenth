#!/usr/bin/env python3
"""Offline end-to-end mapping: simulated drive + lidar SLAM map + autonomous extraction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from centerline_extractor import (
    MapData,
    export_cleaned_map,
    extract_centerline,
    load_map_yaml,
    occupancy_grid_to_gray,
    write_centerline_csv,
)
from track_reference import load_reference_track
from track_viz import plot_disparity_overlay
from validate_track_alignment import AlignmentReport, build_alignment_report


@dataclass
class MappingSessionReport:
    iteration: int
    drive_samples: int
    mapped_free_pct: float
    mapped_known_pct: float
    alignment: dict
    map_yaml: str
    centerline_csv: str
    overlay_png: str
    issue: str = ""
    root_cause: str = ""
    fix_applied: str = ""


def _drive_poses(centerline: np.ndarray, step_m: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(centerline, axis=0, append=centerline[:1]), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    samples = np.arange(0.0, total, step_m)
    poses = np.zeros((len(samples), 2), dtype=np.float64)
    yaws = np.zeros(len(samples), dtype=np.float64)
    j = 0
    for i, s in enumerate(samples):
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        nxt = (j + 1) % len(centerline)
        poses[i] = centerline[j] * (1.0 - t) + centerline[nxt] * t
        tangent = centerline[nxt] - centerline[j]
        yaws[i] = float(math.atan2(tangent[1], tangent[0]))
    return poses, yaws


def _world_to_col_row(pt: np.ndarray, data: MapData) -> tuple[int, int]:
    col = int((pt[0] - data.origin_x) / data.resolution)
    row = int((pt[1] - data.origin_y) / data.resolution)
    return row, col


def _stamp_disk(grid: np.ndarray, row: int, col: int, value: int, radius: int) -> None:
    h, w = grid.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < h and 0 <= c < w:
                if value == 1:
                    grid[r, c] = 1
                elif grid[r, c] != 1:
                    grid[r, c] = 0


def simulate_mapping_session(
    truth_yaml: Path,
    reference_csv: Path,
    out_dir: Path,
    laps: float = 2.5,
    drive_step_m: float = 0.55,
    n_rays: int = 360,
    max_range_m: float = 10.0,
) -> tuple[MapData, Path, int]:
    truth = load_map_yaml(truth_yaml)
    ref_cl, _, _, _, _ = load_reference_track(reference_csv)
    grid = np.full(truth.grid.shape, 2, dtype=np.uint8)
    poses, yaws = _drive_poses(ref_cl, step_m=drive_step_m)
    n_poses = int(len(poses) * laps)
    step = truth.resolution
    h, w = truth.grid.shape
    angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)

    for idx in range(n_poses):
        pose = poses[idx % len(poses)]
        yaw = yaws[idx % len(poses)]
        pr, pc = _world_to_col_row(pose, truth)
        if 0 <= pr < h and 0 <= pc < w and grid[pr, pc] != 1:
            grid[pr, pc] = 0

        for da in angles:
            angle = yaw + float(da)
            dx = math.cos(angle) * step
            dy = math.sin(angle) * step
            dist = 0.0
            x, y = float(pose[0]), float(pose[1])
            while dist < max_range_m:
                x += dx
                y += dy
                dist += step
                row, col = _world_to_col_row(np.array([x, y]), truth)
                if row < 0 or col < 0 or row >= h or col >= w:
                    break
                if truth.grid[row, col] == 1:
                    _stamp_disk(grid, row, col, 1, 1)
                    break
                if grid[row, col] != 1:
                    grid[row, col] = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    track_name = f"sim_map_{truth_yaml.stem}"
    png_path = out_dir / f"{track_name}.png"
    yaml_path = out_dir / f"{track_name}.yaml"
    cv2.imwrite(str(png_path), occupancy_grid_to_gray(grid))
    meta = {
        "image": png_path.name,
        "resolution": truth.resolution,
        "origin": [truth.origin_x, truth.origin_y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.45,
        "free_thresh": 0.196,
    }
    with yaml_path.open("w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    mapped = MapData(grid=grid, resolution=truth.resolution, origin_x=truth.origin_x, origin_y=truth.origin_y)
    return mapped, yaml_path, n_poses


def _mapped_coverage(grid: np.ndarray) -> tuple[float, float]:
    return float(np.mean(grid == 0)) * 100.0, float(np.mean(grid != 2)) * 100.0


def run_iteration(
    iteration: int,
    truth_yaml: Path,
    reference_csv: Path,
    out_root: Path,
) -> tuple[MappingSessionReport, AlignmentReport]:
    iter_dir = out_root / f"iteration_{iteration}"
    mapped_data, map_yaml, n_poses = simulate_mapping_session(truth_yaml, reference_csv, iter_dir)
    free_pct, known_pct = _mapped_coverage(mapped_data.grid)

    extracted = extract_centerline(mapped_data, prefer_known_track=False)
    cl_path = iter_dir / "extracted_centerline.csv"
    write_centerline_csv(extracted, cl_path)
    export_cleaned_map(mapped_data, iter_dir, "sim_map")

    ref_cl, _, _, ref_left, ref_right = load_reference_track(reference_csv)
    report = build_alignment_report(extracted, mapped_data, ref_cl, ref_left, ref_right)
    overlay = iter_dir / "disparity_overlay.png"
    plot_disparity_overlay(overlay, mapped_data, extracted.centerline, ref_cl, ref_left, iteration, report.centerline_mean_m)

    session = MappingSessionReport(
        iteration=iteration,
        drive_samples=n_poses,
        mapped_free_pct=free_pct,
        mapped_known_pct=known_pct,
        alignment=asdict(report),
        map_yaml=str(map_yaml),
        centerline_csv=str(cl_path),
        overlay_png=str(overlay),
    )
    return session, report


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-yaml", type=Path, default=repo / "ros2_deploy/assets/Oschersleben_map.yaml")
    parser.add_argument("--reference-csv", type=Path, default=repo / "ros2_deploy/assets/Oschersleben_centerline.csv")
    parser.add_argument("--out-dir", type=Path, default=repo / "ros2_mapping/output/mapping_iterations")
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()

    session, report = run_iteration(args.iteration, args.truth_yaml, args.reference_csv, args.out_dir)
    print(json.dumps(asdict(session), indent=2))
    print(f"mean={report.centerline_mean_m:.3f} p95={report.centerline_p95_m:.3f} ratio={report.length_ratio:.3f}")


if __name__ == "__main__":
    main()

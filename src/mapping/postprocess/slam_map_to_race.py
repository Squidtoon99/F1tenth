#!/usr/bin/env python3
"""Convert a SLAM occupancy grid into F1TENTH race assets (centerline CSV + map PNG/YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

from centerline_extractor import (
    extract_centerline,
    export_cleaned_map,
    load_map_yaml,
    write_centerline_csv,
    write_raceline_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-yaml",
        type=Path,
        required=True,
        help="Path to map YAML (map_server format from slam_toolbox / map_saver).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated assets.",
    )
    parser.add_argument(
        "--track-name",
        type=str,
        default="mapped_track",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=0.07,
        help="Centerline resampling spacing in meters.",
    )
    parser.add_argument(
        "--half-width-fallback",
        type=float,
        default=1.1,
        help="Fallback track half-width when ray-cast fails.",
    )
    parser.add_argument(
        "--write-raceline",
        action="store_true",
        help="Also write a raceline CSV for f1tenth_gym_ros.",
    )
    parser.add_argument(
        "--prefer-known-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load bundled centerline when map fingerprint matches a known track.",
    )
    args = parser.parse_args()

    data = load_map_yaml(args.map_yaml)
    result = extract_centerline(
        data,
        spacing_m=args.spacing_m,
        half_width_fallback=args.half_width_fallback,
        prefer_known_track=args.prefer_known_track,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    centerline_path = args.out_dir / f"{args.track_name}_centerline.csv"
    write_centerline_csv(result, centerline_path)
    png_path, yaml_path = export_cleaned_map(data, args.out_dir, args.track_name)

    print(f"Wrote centerline ({result.centerline.shape[0]} pts) -> {centerline_path}")
    print(f"Wrote map -> {png_path}, {yaml_path}")

    if args.write_raceline:
        raceline_path = args.out_dir / f"{args.track_name}_raceline.csv"
        write_raceline_csv(result, raceline_path)
        print(f"Wrote raceline -> {raceline_path}")


if __name__ == "__main__":
    main()

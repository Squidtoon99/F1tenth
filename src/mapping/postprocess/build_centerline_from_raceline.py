#!/usr/bin/env python3
"""Convert a f1tenth_gym raceline CSV into a centerline CSV for Genesis / ROS.

Input columns (comma or semicolon): s_m, x_m, y_m, psi_rad, ...
Output columns: x_m, y_m, w_tr_right_m, w_tr_left_m

WARNING: raceline (x, y) alone may not sit in the center of the gym occupancy
map corridor. For deploy / f1tenth_gym, prefer scripts/build_centerline_from_map.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_raceline(path: Path) -> np.ndarray:
    text = path.read_text()
    delimiter = ";" if ";" in text.splitlines()[0] else ","
    data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=np.float64)
    if data.dtype.names is None or "x_m" not in data.dtype.names:
        raise ValueError(f"{path}: expected named columns including x_m, y_m")
    return data


def write_centerline(
    raceline: np.ndarray,
    out_path: Path,
    half_width_m: float = 1.1,
) -> int:
    header = "# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
    lines = [header]
    w = float(half_width_m)
    for row in raceline:
        lines.append(f"{row['x_m']}, {row['y_m']}, {w}, {w}\n")
    out_path.write_text("".join(lines))
    return len(raceline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raceline_csv", type=Path)
    parser.add_argument("centerline_csv", type=Path)
    parser.add_argument(
        "--half-width",
        type=float,
        default=1.1,
        help="Track half-width in meters (default 1.1 -> 2.2 m total)",
    )
    args = parser.parse_args()
    raceline = load_raceline(args.raceline_csv)
    n = write_centerline(raceline, args.centerline_csv, args.half_width)
    print(f"Wrote {n} points -> {args.centerline_csv}")


if __name__ == "__main__":
    main()

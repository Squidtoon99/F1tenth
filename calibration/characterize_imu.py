#!/usr/bin/env python3
"""Quantify IMU bias/noise/drift from sensor-policy calibration rosbags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python calibration/characterize_imu.py` without install.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bag_io import inventory, load_series  # noqa: E402
from imu_characterization import (  # noqa: E402
    MissingChannelsError,
    characterize_imu_bag,
    load_imu_calibration,
    write_characterization_report,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument(
        "--condition",
        choices=("static_off", "static_on", "dynamic", "unknown"),
        default="unknown",
        help="recording condition tag for the summary",
    )
    ap.add_argument(
        "--cal-yaml",
        type=str,
        default="",
        help="sensor_policy IMU calibration YAML (defaults to static-fit inference)",
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default="calibration/results/imu_characterization.json",
    )
    ap.add_argument(
        "--out-report",
        type=str,
        default="calibration/results/imu_characterization.md",
    )
    ap.add_argument(
        "--require-core",
        action="store_true",
        help="fail when /sensors/core is absent (needed for current/temp correlation)",
    )
    args = ap.parse_args(argv)

    bag_dir = Path(args.bag).expanduser().resolve()
    if not bag_dir.is_dir():
        raise SystemExit(f"bag not found: {bag_dir}")

    cal = load_imu_calibration(args.cal_yaml) if args.cal_yaml else None
    series = load_series(bag_dir)
    try:
        result = characterize_imu_bag(
            series,
            condition=args.condition,
            cal=cal,
            require_core=args.require_core,
        )
    except (MissingChannelsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    result["bag"] = inventory(bag_dir, series)
    result["cal_yaml"] = str(Path(args.cal_yaml).expanduser()) if args.cal_yaml else None

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    write_characterization_report(result, out_report)

    print(f"wrote {out_json}")
    print(f"wrote {out_report}")
    converted = result["converted_si"]
    print(
        f"  converted IMU: {converted.get('sample_count', 0)} samples "
        f"@ ~{converted.get('rate_hz', 0.0):.1f} Hz"
    )
    corr = result["telemetry_correlation"]
    print(f"  telemetry correlation: {corr.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

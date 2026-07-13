#!/usr/bin/env python3
"""Print / save Velineon 3500 + gearing seed values for sim force envelopes.

Does not talk to the car. Export live VESC mcconf separately (vesc_tool XML /
``/sensors/core`` notes) and verify pinion/spur/diff tooth counts on the chassis.
"""

from __future__ import annotations

import argparse
import json

from drivetrain_priors import (
    MOTOR_BURST_A_REF,
    MOTOR_CONT_A_REF,
    STOCK_GEAR_RATIO,
    STOCK_PINION,
    STOCK_SPUR,
    VELINEON_3500_KV,
    VELINEON_KT_NM_PER_A,
    VESC6_CONT_A,
    VESC6_PEAK_A,
    seed_force_envelope,
    seed_force_per_amp,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i-drive-max", type=float, default=10.0)
    p.add_argument("--i-brake-max", type=float, default=10.0)
    p.add_argument("--efficiency", type=float, default=0.85)
    p.add_argument("--wheel-radius", type=float, default=0.05)
    p.add_argument("--gear-ratio", type=float, default=STOCK_GEAR_RATIO)
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    fpa = seed_force_per_amp(
        gear_ratio=args.gear_ratio,
        efficiency=args.efficiency,
        wheel_radius_m=args.wheel_radius,
    )
    env = seed_force_envelope(
        i_drive_max_a=args.i_drive_max,
        i_brake_max_a=args.i_brake_max,
        force_per_amp=fpa,
    )
    payload = {
        "motor": "Traxxas Velineon 3500",
        "kv": VELINEON_3500_KV,
        "kt_nm_per_a": VELINEON_KT_NM_PER_A,
        "motor_cont_a_ref": MOTOR_CONT_A_REF,
        "motor_burst_a_ref": MOTOR_BURST_A_REF,
        "vesc": "TRAMPA VESC 6 MKV",
        "vesc_cont_a": VESC6_CONT_A,
        "vesc_peak_a": VESC6_PEAK_A,
        "stock_pinion_spur": [STOCK_PINION, STOCK_SPUR],
        "gear_ratio": args.gear_ratio,
        "efficiency": args.efficiency,
        "wheel_radius_m": args.wheel_radius,
        **env,
        "checklist": [
            "Export VESC mcconf (current limits, FOC detection, pole count, voltage)",
            "Count installed pinion/spur/diff gears (do not assume stock)",
            "Record boxed-wheel current/brake bag with /sensors/core",
            "Reject fits with poor excitation or sign inconsistency",
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

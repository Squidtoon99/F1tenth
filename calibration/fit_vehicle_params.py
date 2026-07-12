#!/usr/bin/env python3
"""Offline vehicle-parameter fitting from calibration rosbags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Allow running as `python calibration/fit_vehicle_params.py` without install.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bag_io import inventory, load_series  # noqa: E402
from fit_imu import fit_imu_static  # noqa: E402
from fit_longitudinal import (  # noqa: E402
    fit_accel_limit,
    fit_brake_current,
    fit_c_roll,
    fit_current_to_force,
    fit_f_drive_equivalent,
    fit_odom_sign_and_scale,
    fit_speed_to_erpm,
    fit_tire_mu_lower_bound,
    not_identifiable,
)
from fit_steering import (  # noqa: E402
    fit_max_steer_from_circles,
    fit_servo_map,
    fit_steer_lag,
)


def _pick(bags: dict[str, dict], *keys: str) -> dict | None:
    for k in keys:
        for name, series in bags.items():
            if k in name:
                return series
    return None


def _pick_name(bags: dict[str, dict], *keys: str) -> str | None:
    for k in keys:
        for name in bags:
            if k in name:
                return name
    return None


def fit_all(
    bag_dirs: list[Path],
    mass: float = 3.74,
    wheelbase: float = 0.325,
) -> dict:
    loaded = {p.name: load_series(p) for p in bag_dirs}
    inv = {p.name: inventory(p, loaded[p.name]) for p in bag_dirs}

    static = _pick(loaded, "static")
    steering = _pick(loaded, "steering", "steer")
    accel = _pick(loaded, "accel", "longitudinal")

    imu_fit = fit_imu_static(static["/sensors/imu/raw"]) if static else not_identifiable(
        "no_static_bag"
    )

    servo_fit = (
        fit_servo_map(steering["/ackermann_cmd"], steering["/commands/servo/position"])
        if steering
        else not_identifiable("no_steering_bag")
    )
    max_steer_fit = (
        fit_max_steer_from_circles(
            steering["/odom"], steering["/ackermann_cmd"], wheelbase=wheelbase
        )
        if steering
        else not_identifiable("no_steering_bag")
    )
    lag_fit = (
        fit_steer_lag(steering["/ackermann_cmd"], steering["/odom"])
        if steering
        else not_identifiable("no_steering_bag")
    )

    # Prefer accel bag for longitudinal; fall back to steering.
    long_bag = accel or steering
    erpm_fit = (
        fit_speed_to_erpm(long_bag["/ackermann_cmd"], long_bag["/commands/motor/speed"])
        if long_bag
        else not_identifiable("no_longitudinal_bag")
    )
    odom_fit = (
        fit_odom_sign_and_scale(long_bag["/odom"], long_bag["/ackermann_cmd"])
        if long_bag
        else not_identifiable("no_longitudinal_bag")
    )
    accel_fit = (
        fit_accel_limit(long_bag["/odom"], long_bag["/ackermann_cmd"])
        if long_bag
        else not_identifiable("no_longitudinal_bag")
    )
    fdrive_fit = (
        fit_f_drive_equivalent(long_bag["/odom"], mass=mass)
        if long_bag
        else not_identifiable("no_longitudinal_bag")
    )
    croll_fit = (
        fit_c_roll(long_bag["/odom"], long_bag["/ackermann_cmd"], mass=mass)
        if long_bag
        else not_identifiable("no_longitudinal_bag")
    )
    mu_src = accel or steering
    mu_fit = (
        fit_tire_mu_lower_bound(mu_src["/sensors/imu/raw"], mu_src["/odom"])
        if mu_src
        else not_identifiable("no_motion_bag")
    )

    # Optional Part 2 bags: any bag with nonzero current/brake commands.
    current_fit = not_identifiable("no_current_bag")
    brake_fit = not_identifiable("no_brake_bag")
    for series in loaded.values():
        if series["/commands/motor/current"].shape[0] > 0 and current_fit.get(
            "status"
        ) == "NOT_IDENTIFIABLE":
            current_fit = fit_current_to_force(
                series["/commands/motor/current"], series["/odom"], mass=mass
            )
        if series["/commands/motor/brake"].shape[0] > 0 and brake_fit.get(
            "status"
        ) == "NOT_IDENTIFIABLE":
            brake_fit = fit_brake_current(
                series["/commands/motor/brake"], series["/odom"], mass=mass
            )

    identifiability = {
        "f_brake_max": "NOT_IDENTIFIABLE",
        "motor_kt": "NOT_IDENTIFIABLE",
        "motor_i_max": "NOT_IDENTIFIABLE",
        "power_max": "NOT_IDENTIFIABLE",
        "k_drive_front": "NOT_IDENTIFIABLE",
        "susp_stiffness": "NOT_IDENTIFIABLE",
        "susp_damping": "NOT_IDENTIFIABLE",
        "anti_roll": "NOT_IDENTIFIABLE",
        "reason": (
            "bags lack /sensors/core, explicit current/brake commands, "
            "and suspension excitation"
        ),
    }

    result = {
        "bags": inv,
        "fits": {
            "imu": imu_fit,
            "servo": servo_fit,
            "max_steer": max_steer_fit,
            "steer_lag": lag_fit,
            "speed_to_erpm": erpm_fit,
            "odom": odom_fit,
            "accel_limit": accel_fit,
            "f_drive_equivalent": fdrive_fit,
            "c_roll": croll_fit,
            "tire_mu": mu_fit,
            "current_to_force": current_fit,
            "brake_current": brake_fit,
        },
        "identifiability": identifiability,
        "priors": {
            "mass": mass,
            "wheelbase": wheelbase,
            "wheel_radius": 0.05,
            "k_drive_front": 0.0,
            "roll_stiffness_front": 0.5,
        },
    }
    return result


def to_sim_yaml(result: dict) -> dict:
    """Flatten fits into a training/deploy overlay-shaped YAML dict."""
    fits = result["fits"]
    priors = result["priors"]

    def val(block, key, default=None):
        if not isinstance(block, dict):
            return default
        if block.get("status") == "NOT_IDENTIFIABLE":
            return default
        return block.get(key, default)

    imu = fits["imu"]
    yaml_out = {
        "mass": priors["mass"],
        "wheelbase": priors["wheelbase"],
        "wheel_radius": priors["wheel_radius"],
        "max_steer": val(fits["max_steer"], "max_steer", 0.33),
        "t_delta": val(fits["steer_lag"], "t_delta", 0.10),
        "throttle_mode": "speed",  # keep speed until force path is validated
        "f_drive_max": val(fits["f_drive_equivalent"], "f_drive_max", 23.0)
        if fits.get("current_to_force", {}).get("status") == "NOT_IDENTIFIABLE"
        else val(fits["current_to_force"], "f_drive_max_at_iabs", 23.0),
        "f_brake_max": val(fits.get("brake_current", {}), "f_brake_max_at_bmax"),
        "c_roll": val(fits["c_roll"], "c_roll", 0.0),
        "tire_friction": val(fits["tire_mu"], "tire_friction_lower_bound", 0.9),
        "v_eps": 0.1,
        "k_drive_front": priors["k_drive_front"],
        "torch_sim": {
            "model": "dynamic",
            "suspension_mode": "quasi_static",
            "roll_stiffness_front": priors["roll_stiffness_front"],
            "vesc_accel_limit": val(fits["accel_limit"], "vesc_accel_limit", 2.5),
        },
        "deploy_overlay": {
            "speed_to_erpm_gain": val(fits["speed_to_erpm"], "speed_to_erpm_gain"),
            "speed_to_erpm_offset": val(fits["speed_to_erpm"], "speed_to_erpm_offset", 0.0),
            "steering_angle_to_servo_gain": val(
                fits["servo"], "steering_angle_to_servo_gain"
            ),
            "steering_angle_to_servo_offset": val(
                fits["servo"], "steering_angle_to_servo_offset"
            ),
            "imu_ax_sign": val(imu, "imu_ax_sign", 1.0),
            "imu_ay_sign": val(imu, "imu_ay_sign", 1.0),
            "imu_yaw_rate_sign": val(imu, "imu_yaw_rate_sign", 1.0),
            "imu_accel_to_ms2": val(imu, "accel_scale", 1.0),
            "imu_gyro_to_rads": val(imu, "gyro_scale", 1.0),
        },
        "identifiability": result["identifiability"],
        "fit_status": {k: v.get("status") for k, v in fits.items()},
    }
    return yaml_out


def write_report(result: dict, path: Path) -> None:
    lines = ["# Calibration report", ""]
    lines.append("## Bags")
    for name, inv in result["bags"].items():
        lines.append(f"- **{name}**: duration={inv['duration_s']:.1f}s")
        for topic, info in inv["topics"].items():
            if info["count"] == 0:
                continue
            lines.append(
                f"  - `{topic}`: {info['count']} msgs @ ~{info['rate_hz']:.1f} Hz"
            )
    lines.append("")
    lines.append("## Fits")
    for name, fit in result["fits"].items():
        status = fit.get("status", "?")
        lines.append(f"### {name} — `{status}`")
        for k, v in fit.items():
            if k == "status":
                continue
            lines.append(f"- `{k}`: {v}")
        lines.append("")
    lines.append("## Not identifiable from these bags")
    for k, v in result["identifiability"].items():
        lines.append(f"- `{k}`: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bags",
        nargs="+",
        required=True,
        help="rosbag2 directories (static / steering / accel)",
    )
    ap.add_argument("--out", type=str, default="calibration/results/car01_sim.yaml")
    ap.add_argument("--report", type=str, default="calibration/results/car01_report.md")
    ap.add_argument("--mass", type=float, default=3.74)
    ap.add_argument("--wheelbase", type=float, default=0.325)
    ap.add_argument("--json", type=str, default="", help="optional full JSON dump")
    args = ap.parse_args(argv)

    bag_dirs = [Path(p).expanduser().resolve() for p in args.bags]
    for p in bag_dirs:
        if not p.is_dir():
            raise SystemExit(f"bag not found: {p}")

    result = fit_all(bag_dirs, mass=args.mass, wheelbase=args.wheelbase)
    sim_yaml = to_sim_yaml(result)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(sim_yaml, f, sort_keys=False)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(result, report)

    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"wrote {out}")
    print(f"wrote {report}")
    for name, fit in result["fits"].items():
        print(f"  {name}: {fit.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

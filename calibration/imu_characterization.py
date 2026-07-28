"""IMU bias/noise/drift analysis for sensor-policy calibration bags."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from bag_io import linear_fit, resample
from fit_imu import fit_imu_static

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
DYNAMIC_CHANNELS = ("ax", "ay", "gz")
OUTLIER_SIGMA = 3.0
ROLLING_STD_WINDOW_S = 0.5

_REPO = Path(__file__).resolve().parents[1]
_AGENT_PKG = _REPO / "src" / "racing_rl" / "f1tenth_rl_agent"


class MissingChannelsError(ValueError):
    """Raised when a bag lacks channels required for the requested analysis."""

    def __init__(self, missing: Iterable[str], available: Iterable[str]) -> None:
        missing = list(missing)
        available = sorted(available)
        msg = (
            "missing required bag topics: "
            + ", ".join(missing)
            + "; available: "
            + (", ".join(available) if available else "(none)")
        )
        super().__init__(msg)
        self.missing = missing
        self.available = available


@dataclass(frozen=True)
class ImuCalibration:
    accel_to_ms2: float = 9.80665
    gyro_to_rads: float = math.pi / 180.0
    ax_sign: float = 1.0
    ay_sign: float = 1.0
    az_sign: float = 1.0
    gx_sign: float = 1.0
    gy_sign: float = 1.0
    gz_sign: float = 1.0
    ax_bias: float = 0.0
    ay_bias: float = 0.0
    az_bias: float = 0.0
    gx_bias: float = 0.0
    gy_bias: float = 0.0
    gz_bias: float = 0.0


def _import_preprocess():
    if str(_AGENT_PKG) not in sys.path:
        sys.path.insert(0, str(_AGENT_PKG))
    from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
        ImuCalibration as AgentCal,
        RawImuSample,
        actor_imu_from_interval,
        convert_raw_imu,
    )

    return AgentCal, RawImuSample, actor_imu_from_interval, convert_raw_imu


def _to_agent_cal(cal: ImuCalibration):
    AgentCal, _, _, _ = _import_preprocess()
    return AgentCal(
        accel_to_ms2=cal.accel_to_ms2,
        gyro_to_rads=cal.gyro_to_rads,
        ax_sign=cal.ax_sign,
        ay_sign=cal.ay_sign,
        az_sign=cal.az_sign,
        gx_sign=cal.gx_sign,
        gy_sign=cal.gy_sign,
        gz_sign=cal.gz_sign,
        ax_bias=cal.ax_bias,
        ay_bias=cal.ay_bias,
        az_bias=cal.az_bias,
        gx_bias=cal.gx_bias,
        gy_bias=cal.gy_bias,
        gz_bias=cal.gz_bias,
    )


def _collect_imu_params(data: dict) -> dict:
    """Gather ``imu_*`` keys from flat or multi-node ROS parameter YAML."""
    merged: dict = {}

    def take(block: dict | None) -> None:
        if not isinstance(block, dict):
            return
        for key, value in block.items():
            if isinstance(key, str) and key.startswith("imu_"):
                merged[key] = value

    for value in data.values():
        if isinstance(value, dict):
            take(value.get("ros__parameters", value))
    # sensor_policy wins over other node blocks when both define imu_*.
    if "sensor_policy" in data and isinstance(data["sensor_policy"], dict):
        sp = data["sensor_policy"]
        take(sp.get("ros__parameters", sp))
    take(data)
    return merged


def load_imu_calibration(path: Path | str | None) -> ImuCalibration | None:
    if path is None:
        return None
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"calibration yaml must be a mapping: {path}")

    block = _collect_imu_params(data)

    def pick(key: str, default: float) -> float:
        return float(block.get(key, default))

    return ImuCalibration(
        accel_to_ms2=pick("imu_accel_to_ms2", 9.80665),
        gyro_to_rads=pick("imu_gyro_to_rads", math.pi / 180.0),
        ax_sign=pick("imu_ax_sign", 1.0),
        ay_sign=pick("imu_ay_sign", 1.0),
        az_sign=pick("imu_az_sign", 1.0),
        gx_sign=pick("imu_gx_sign", 1.0),
        gy_sign=pick("imu_gy_sign", 1.0),
        gz_sign=pick("imu_gz_sign", 1.0),
        ax_bias=pick("imu_ax_bias", 0.0),
        ay_bias=pick("imu_ay_bias", 0.0),
        az_bias=pick("imu_az_bias", 0.0),
        gx_bias=pick("imu_gx_bias", 0.0),
        gy_bias=pick("imu_gy_bias", 0.0),
        gz_bias=pick("imu_gz_bias", 0.0),
    )


def calibration_from_static_fit(fit: dict) -> ImuCalibration:
    if "accel_scale" not in fit:
        raise ValueError(
            "static fit is not identifiable; cannot build IMU calibration "
            f"({fit.get('reason', fit.get('status', 'unknown'))})"
        )
    accel_scale = float(fit["accel_scale"])
    gyro_scale = float(fit.get("gyro_scale", math.pi / 180.0))
    bias_a = fit.get("bias_accel_raw", {})
    bias_g = fit.get("bias_gyro_raw", {})
    # Keep az_bias at 0 so gravity remains in converted SI; horizontal/gyro
    # biases are the stationary raw means.
    return ImuCalibration(
        accel_to_ms2=accel_scale,
        gyro_to_rads=gyro_scale,
        ax_sign=float(fit.get("imu_ax_sign", 1.0)),
        ay_sign=float(fit.get("imu_ay_sign", 1.0)),
        az_sign=float(fit.get("imu_az_sign", 1.0)),
        gz_sign=float(fit.get("imu_yaw_rate_sign", 1.0)),
        ax_bias=float(bias_a.get("ax", 0.0)),
        ay_bias=float(bias_a.get("ay", 0.0)),
        az_bias=0.0,
        gx_bias=float(bias_g.get("gx", 0.0)),
        gy_bias=float(bias_g.get("gy", 0.0)),
        gz_bias=float(bias_g.get("gz", 0.0)),
    )


def convert_driver_imu(raw: np.ndarray, cal: ImuCalibration) -> np.ndarray:
    """Convert driver-frame IMU columns to SI using the deploy calibration."""
    if raw.size == 0:
        return raw.copy()
    agent_cal = _to_agent_cal(cal)
    _, _, _, convert_raw_imu = _import_preprocess()
    out_rows = []
    for row in raw:
        ax, ay, az, gx, gy, gz = convert_raw_imu(
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            agent_cal,
        )
        out_rows.append([float(row[0]), ax, ay, az, gx, gy, gz])
    return np.asarray(out_rows, dtype=float)


def estimate_rate_hz(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    dt = max(float(t[-1] - t[0]), 1e-9)
    return float((t.size - 1) / dt)


def per_axis_stats(imu: np.ndarray) -> dict:
    """Per-axis mean/std/outliers/drift for columns [t, ax..gz]."""
    if imu.shape[0] < 2:
        return {
            "status": "insufficient_samples",
            "sample_count": int(imu.shape[0]),
            "rate_hz": estimate_rate_hz(imu[:, 0]) if imu.size else 0.0,
            "channels": {},
        }

    t = imu[:, 0]
    out = {
        "status": "ok",
        "sample_count": int(imu.shape[0]),
        "duration_s": float(t[-1] - t[0]),
        "rate_hz": estimate_rate_hz(t),
        "channels": {},
    }
    for idx, name in enumerate(CHANNELS, start=1):
        values = imu[:, idx]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std > 1e-12:
            mask = np.abs(values - mean) > OUTLIER_SIGMA * std
        else:
            mask = np.zeros(values.shape, dtype=bool)
        slope, intercept, r2 = linear_fit(t, values)
        out["channels"][name] = {
            "mean": mean,
            "std": std,
            "outlier_count": int(np.count_nonzero(mask)),
            "outlier_fraction": float(np.mean(mask)),
            "drift_slope_per_s": slope,
            "drift_intercept": intercept,
            "drift_r2": r2,
        }
    return out


def rolling_std(t: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    if t.size == 0:
        return np.zeros(0, dtype=float)
    if t.size == 1:
        return np.zeros(1, dtype=float)
    half = max(window_s * 0.5, 1e-3)
    out = np.zeros(t.size, dtype=float)
    for i, ti in enumerate(t):
        mask = (t >= ti - half) & (t <= ti + half)
        chunk = values[mask]
        out[i] = float(np.std(chunk)) if chunk.size > 1 else 0.0
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def correlate_with_core(imu_si: np.ndarray, core: np.ndarray) -> dict:
    if imu_si.shape[0] < 10 or core.shape[0] < 10:
        return {
            "status": "insufficient_samples",
            "reason": "need at least 10 IMU and core samples",
        }

    core_on_imu = resample(core, imu_si[:, 0])
    current = np.abs(core_on_imu[:, 3])
    temp_fet = core_on_imu[:, 1]
    temp_motor = core_on_imu[:, 2]

    channel_idx = {name: i + 1 for i, name in enumerate(CHANNELS)}
    correlations = {}
    for name in DYNAMIC_CHANNELS:
        values = imu_si[:, channel_idx[name]]
        roll_std = rolling_std(imu_si[:, 0], values, ROLLING_STD_WINDOW_S)
        correlations[name] = {
            "std_vs_abs_current_motor": pearson(roll_std, current),
            "std_vs_temp_fet": pearson(roll_std, temp_fet),
            "std_vs_temp_motor": pearson(roll_std, temp_motor),
            "mean_vs_abs_current_motor": pearson(values, current),
        }

    return {
        "status": "ok",
        "window_s": ROLLING_STD_WINDOW_S,
        "mean_abs_current_motor_a": float(np.mean(current)),
        "mean_temp_fet_c": float(np.mean(temp_fet)),
        "mean_temp_motor_c": float(np.mean(temp_motor)),
        "channels": correlations,
    }


def _replay_policy_ticks(
    raw_driver: np.ndarray,
    cal: ImuCalibration,
    *,
    control_hz: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay raw driver IMU through the 10 Hz interval preprocessor."""
    _, RawImuSample, actor_imu_from_interval, _ = _import_preprocess()
    agent_cal = _to_agent_cal(cal)
    empty = np.zeros((0, 7), dtype=float)
    if raw_driver.size == 0:
        return empty, empty

    period = 1.0 / max(control_hz, 1.0)
    t0 = float(raw_driver[0, 0])
    t_end = float(raw_driver[-1, 0])
    tick_times = np.arange(t0, t_end + period, period, dtype=float)
    raw_mean_rows = []
    actor_rows = []
    sample_idx = 0
    n = raw_driver.shape[0]

    for tick in tick_times:
        interval_end = tick + period
        interval: list = []
        while sample_idx < n and float(raw_driver[sample_idx, 0]) < interval_end:
            row = raw_driver[sample_idx]
            interval.append(
                RawImuSample(
                    stamp_s=float(row[0]),
                    ax=float(row[1]),
                    ay=float(row[2]),
                    az=float(row[3]),
                    gx=float(row[4]),
                    gy=float(row[5]),
                    gz=float(row[6]),
                )
            )
            sample_idx += 1
        actor, raw_mean = actor_imu_from_interval(
            interval, agent_cal, freeze_const_channels=True
        )
        raw_mean_rows.append([tick, *raw_mean.tolist()])
        actor_rows.append([tick, *actor.tolist()])

    return np.asarray(raw_mean_rows, dtype=float), np.asarray(actor_rows, dtype=float)


def replay_policy_imu(
    raw_driver: np.ndarray,
    cal: ImuCalibration,
    *,
    control_hz: float = 10.0,
) -> np.ndarray:
    _, actor = _replay_policy_ticks(raw_driver, cal, control_hz=control_hz)
    return actor


def compare_actor_streams(
    recorded: np.ndarray, replayed: np.ndarray
) -> dict | None:
    if recorded.size == 0 or replayed.size == 0:
        return None
    replay_on_rec = resample(replayed, recorded[:, 0])
    diffs = np.abs(recorded[:, 1:] - replay_on_rec[:, 1:])
    per_axis = {}
    for i, name in enumerate(CHANNELS):
        per_axis[name] = {
            "max_abs_diff": float(np.max(diffs[:, i])),
            "mean_abs_diff": float(np.mean(diffs[:, i])),
        }
    return {
        "max_abs_diff": float(np.max(diffs)),
        "mean_abs_diff": float(np.mean(diffs)),
        "per_axis": per_axis,
    }


def proposed_sensor_policy_params(
    static_fit: dict,
    converted_stats: dict,
    *,
    current: ImuCalibration | None = None,
) -> dict:
    if current is not None:
        proposed = current
    elif "accel_scale" in static_fit:
        proposed = calibration_from_static_fit(static_fit)
    else:
        proposed = ImuCalibration()

    raw_bias = static_fit.get("bias_accel_raw", {})
    raw_gyro = static_fit.get("bias_gyro_raw", {})
    # az_bias stays 0 / configured so conversion does not null gravity.
    az_bias = float(proposed.az_bias)
    return {
        "status": static_fit.get("status", "unknown"),
        "imu_accel_to_ms2": proposed.accel_to_ms2,
        "imu_gyro_to_rads": proposed.gyro_to_rads,
        "imu_ax_sign": proposed.ax_sign,
        "imu_ay_sign": proposed.ay_sign,
        "imu_az_sign": proposed.az_sign,
        "imu_gx_sign": proposed.gx_sign,
        "imu_gy_sign": proposed.gy_sign,
        "imu_gz_sign": proposed.gz_sign,
        "imu_ax_bias": float(raw_bias.get("ax", proposed.ax_bias)),
        "imu_ay_bias": float(raw_bias.get("ay", proposed.ay_bias)),
        "imu_az_bias": az_bias,
        "imu_gx_bias": float(raw_gyro.get("gx", proposed.gx_bias)),
        "imu_gy_bias": float(raw_gyro.get("gy", proposed.gy_bias)),
        "imu_gz_bias": float(raw_gyro.get("gz", proposed.gz_bias)),
        "converted_means_ms2_or_rads": {
            name: converted_stats["channels"][name]["mean"]
            for name in CHANNELS
            if name in converted_stats.get("channels", {})
        },
    }


def _nonempty_topics(series: dict[str, np.ndarray]) -> list[str]:
    return [name for name, data in series.items() if data.shape[0] > 0]


def characterize_imu_bag(
    series: dict[str, np.ndarray],
    *,
    condition: str = "unknown",
    cal: ImuCalibration | None = None,
    require_core: bool = False,
) -> dict:
    available = _nonempty_topics(series)
    if "/sensors/imu/raw" not in series:
        raise MissingChannelsError(["/sensors/imu/raw"], available)
    raw = series["/sensors/imu/raw"]
    if raw.shape[0] == 0:
        raise MissingChannelsError(["/sensors/imu/raw"], available)

    if require_core:
        core = series.get("/sensors/core", np.zeros((0, 17)))
        if core.shape[0] == 0:
            raise MissingChannelsError(["/sensors/core"], available)

    static_fit = fit_imu_static(raw)
    if cal is None:
        effective_cal = calibration_from_static_fit(static_fit)
    else:
        effective_cal = cal
    converted = convert_driver_imu(raw, effective_cal)

    result = {
        "condition": condition,
        "available_topics": available,
        "static_fit": static_fit,
        "calibration_used": {
            "accel_to_ms2": effective_cal.accel_to_ms2,
            "gyro_to_rads": effective_cal.gyro_to_rads,
            "ax_bias": effective_cal.ax_bias,
            "ay_bias": effective_cal.ay_bias,
            "az_bias": effective_cal.az_bias,
            "gz_bias": effective_cal.gz_bias,
        },
        "driver_raw": per_axis_stats(raw),
        "converted_si": per_axis_stats(converted),
    }

    recorded_raw = series.get(
        "/sensor_policy/imu_raw_record", np.zeros((0, 7), dtype=float)
    )
    recorded_actor = series.get(
        "/sensor_policy/imu_actor_record", np.zeros((0, 7), dtype=float)
    )
    if recorded_raw.shape[0] > 0:
        result["recorded_converted_mean"] = per_axis_stats(recorded_raw)
    if recorded_actor.shape[0] > 0:
        result["recorded_actor"] = per_axis_stats(recorded_actor)

    replayed_raw_mean, replayed = _replay_policy_ticks(raw, effective_cal)
    result["replayed_raw_mean"] = per_axis_stats(replayed_raw_mean)
    result["replayed_actor"] = per_axis_stats(replayed)
    if recorded_raw.shape[0] > 0:
        result["recorded_raw_vs_replay"] = compare_actor_streams(
            recorded_raw, replayed_raw_mean
        )
    if recorded_actor.shape[0] > 0:
        result["recorded_vs_replay"] = compare_actor_streams(recorded_actor, replayed)

    core = series.get("/sensors/core", np.zeros((0, 17), dtype=float))
    if core.shape[0] > 0:
        result["telemetry_correlation"] = correlate_with_core(converted, core)
    else:
        result["telemetry_correlation"] = {
            "status": "absent",
            "reason": "bag has no /sensors/core samples",
        }

    result["proposed_sensor_policy_params"] = proposed_sensor_policy_params(
        static_fit,
        result["converted_si"],
        current=cal,
    )
    return result


def write_characterization_report(result: dict, path: Path) -> None:
    lines = [
        "# IMU characterization",
        "",
        f"- condition: `{result.get('condition', 'unknown')}`",
        f"- static fit: `{result['static_fit'].get('status', '?')}`",
        "",
        "## Converted SI statistics",
    ]
    converted = result["converted_si"]
    lines.append(
        f"- samples={converted.get('sample_count', 0)} "
        f"rate={converted.get('rate_hz', 0.0):.1f} Hz "
        f"duration={converted.get('duration_s', 0.0):.1f} s"
    )
    for name, stats in converted.get("channels", {}).items():
        lines.append(
            f"- `{name}`: mean={stats['mean']:.5f} std={stats['std']:.5f} "
            f"outliers={stats['outlier_count']} "
            f"drift={stats['drift_slope_per_s']:.2e}/s"
        )

    replay = result.get("replayed_actor", {})
    lines.extend(["", "## Replayed actor (10 Hz preprocess)", ""])
    for name, stats in replay.get("channels", {}).items():
        lines.append(
            f"- `{name}`: mean={stats['mean']:.5f} std={stats['std']:.5f}"
        )

    corr = result.get("telemetry_correlation", {})
    lines.extend(["", "## Telemetry correlation", ""])
    if corr.get("status") != "ok":
        lines.append(f"- {corr.get('status')}: {corr.get('reason', '')}")
    else:
        for name, vals in corr.get("channels", {}).items():
            lines.append(
                f"- `{name}` std vs |current|: {vals['std_vs_abs_current_motor']:.3f}; "
                f"vs temp_fet: {vals['std_vs_temp_fet']:.3f}; "
                f"vs temp_motor: {vals['std_vs_temp_motor']:.3f}"
            )

    raw_cmp = result.get("recorded_raw_vs_replay")
    if raw_cmp:
        lines.extend(
            [
                "",
                "## Recorded vs replayed converted mean",
                f"- max_abs_diff={raw_cmp['max_abs_diff']:.5f}",
                f"- mean_abs_diff={raw_cmp['mean_abs_diff']:.5f}",
            ]
        )

    cmp = result.get("recorded_vs_replay")
    if cmp:
        lines.extend(
            [
                "",
                "## Recorded vs replayed actor",
                f"- max_abs_diff={cmp['max_abs_diff']:.5f}",
                f"- mean_abs_diff={cmp['mean_abs_diff']:.5f}",
            ]
        )

    proposed = result.get("proposed_sensor_policy_params", {})
    lines.extend(["", "## Proposed sensor_policy IMU params", ""])
    for key in sorted(proposed):
        if key == "converted_means_ms2_or_rads":
            continue
        lines.append(f"- `{key}`: {proposed[key]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

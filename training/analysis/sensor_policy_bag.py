"""Rosbag observation reconstruction for sensor-policy parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
for _path in (
    _REPO,
    _REPO / "training",
    _REPO / "src/racing_rl/f1tenth_rl_agent",
    _REPO / "libs/f1tenth_contract",
    _REPO / "libs/f1tenth_policy",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from calibration.bag_io import load_sensor_policy_series  # noqa: E402
from f1tenth_policy import applied_current_fraction  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402

from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    beam_angles_rad,
    pack_actor_observation,
    pack_lidar_from_scan,
)


@dataclass(frozen=True)
class PoseFreeLidarGateResult:
    passed: bool
    supported_frame_fraction: float
    frame_median_error_p95_m: float
    passing_beam_fraction: float
    quantile_error_p95_m: float
    max_range_rate_difference: float
    dropout_run_p95_difference_beams: float
    temporal_quantile_error_m: float


@dataclass(frozen=True)
class ActorDistributionGateResult:
    passed: bool
    passing_dimension_fraction: float
    non_lidar_passing_fraction: float


PROPRIO_NAMES = (
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "vesc_speed",
    "vesc_current",
    "throttle_current",
    "throttle_pred",
    "steer_t",
    "steer_t1",
    "steer_t2",
    "steer_delta0",
    "steer_delta1",
    "steer_delta2",
)


def save_observation_npz(path: Path, **arrays) -> None:
    payload = {}
    for key, value in arrays.items():
        payload[key] = np.asarray(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _validate_actor_observations(
    real_observations: np.ndarray,
    simulated_observations: np.ndarray,
    minimum_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(real_observations, dtype=np.float64)
    simulated = np.asarray(simulated_observations, dtype=np.float64)
    expected_shape = si.NUM_OBS
    if (
        real.ndim != 2
        or simulated.ndim != 2
        or real.shape[1] != expected_shape
        or simulated.shape[1] != expected_shape
    ):
        raise ValueError(
            f"real and simulated observations must have shape (frames, {expected_shape})"
        )
    if min(real.shape[0], simulated.shape[0]) < minimum_frames:
        raise ValueError(f"at least {minimum_frames} observations are required")
    if not np.isfinite(real).all() or not np.isfinite(simulated).all():
        raise ValueError("actor observations must be finite")
    return real, simulated


def _actor_dimension_error(
    real: np.ndarray,
    simulated: np.ndarray,
    *,
    minimum_frames: int,
) -> np.ndarray:
    state_indices = (si.VESC_SPEED, si.THROTTLE_CURRENT, si.STEER_T)
    state_scale = np.asarray((0.25, 0.10, 0.10), dtype=np.float64)
    real_state = real[:, state_indices] / state_scale
    simulated_state = simulated[:, state_indices] / state_scale
    quantiles = (10.0, 50.0, 90.0)
    conditional_error = np.empty((real.shape[0], si.NUM_OBS), dtype=np.float64)
    for index, state in enumerate(real_state):
        real_error = np.max(np.abs(real_state - state[None, :]), axis=1)
        simulated_error = np.max(
            np.abs(simulated_state - state[None, :]), axis=1
        )
        real_indices = np.flatnonzero(real_error <= 1.0)
        simulated_indices = np.flatnonzero(simulated_error <= 1.0)
        if real_indices.size < minimum_frames:
            real_indices = np.argpartition(
                real_error, minimum_frames - 1
            )[:minimum_frames]
        if simulated_indices.size < minimum_frames:
            simulated_indices = np.argpartition(
                simulated_error, minimum_frames - 1
            )[:minimum_frames]
        conditional_error[index] = np.max(
            np.abs(
                np.percentile(real[real_indices], quantiles, axis=0)
                - np.percentile(
                    simulated[simulated_indices], quantiles, axis=0
                )
            ),
            axis=0,
        )
    return np.percentile(conditional_error, 95, axis=0)


def evaluate_actor_distribution_gate(
    real_observations: np.ndarray,
    simulated_observations: np.ndarray,
    *,
    lidar_tolerance_m: float = 0.50,
    non_lidar_tolerance: float = 0.10,
    minimum_frames: int = 20,
) -> ActorDistributionGateResult:
    real, simulated = _validate_actor_observations(
        real_observations, simulated_observations, minimum_frames
    )
    dimension_error = _actor_dimension_error(
        real, simulated, minimum_frames=minimum_frames
    )
    tolerance = np.full(si.NUM_OBS, non_lidar_tolerance, dtype=np.float64)
    tolerance[:si.LIDAR_DIM] = lidar_tolerance_m
    passing_dimensions = dimension_error <= tolerance
    passing_fraction = float(np.mean(passing_dimensions))
    non_lidar_fraction = float(np.mean(passing_dimensions[si.LIDAR_DIM:]))
    return ActorDistributionGateResult(
        passed=bool(passing_fraction >= 0.99 and non_lidar_fraction == 1.0),
        passing_dimension_fraction=passing_fraction,
        non_lidar_passing_fraction=non_lidar_fraction,
    )


def _longest_true_run(row: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in row:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def match_simulated_support(
    real_scans: np.ndarray,
    simulated_scans: np.ndarray,
    *,
    range_max_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    real = np.clip(
        np.nan_to_num(
            np.asarray(real_scans, dtype=np.float64),
            nan=range_max_m,
            posinf=range_max_m,
        ),
        0.0,
        range_max_m,
    )
    simulated = np.clip(
        np.nan_to_num(
            np.asarray(simulated_scans, dtype=np.float64),
            nan=range_max_m,
            posinf=range_max_m,
        ),
        0.0,
        range_max_m,
    )
    match_stride = max(1, real.shape[1] // 180)
    matched_indices = np.empty(real.shape[0], dtype=np.int64)
    frame_errors = np.empty(real.shape[0], dtype=np.float64)
    sampled_simulated = simulated[:, ::match_stride]
    sentinel = range_max_m * 0.999
    simulated_max = sampled_simulated >= sentinel
    for index, scan in enumerate(real[:, ::match_stride]):
        range_error = np.median(
            np.minimum(np.abs(sampled_simulated - scan[None, :]), 5.0),
            axis=1,
        )
        mask_error = np.mean(
            simulated_max != (scan[None, :] >= sentinel), axis=1
        )
        candidate_error = range_error + 5.0 * mask_error
        matched_indices[index] = int(np.argmin(candidate_error))
        frame_errors[index] = range_error[matched_indices[index]]
    return matched_indices, frame_errors


def diagnose_support_ray_hits(
    real_scans: np.ndarray,
    simulated_scans: np.ndarray,
    poses: np.ndarray,
    field,
    *,
    lidar_offset: tuple[float, float, float] = (0.27, 0.0, 0.0),
    range_max_m: float = 30.0,
    variant_id: int = 0,
) -> dict:
    from analysis.map_lidar import (
        RAY_HIT_MAX_RANGE,
        RAY_HIT_UNKNOWN,
        RAY_HIT_WALL,
        raycast_distance_field,
    )

    real = np.asarray(real_scans, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    matched_indices, _ = match_simulated_support(
        real, simulated_scans, range_max_m=range_max_m
    )
    angles = beam_angles_rad()
    sentinel = range_max_m * 0.999
    wall = 0
    unknown = 0
    max_range = 0
    opening_wall = 0
    opening_unknown = 0
    opening_total = 0
    for frame, pose_index in enumerate(matched_indices):
        _, kinds = raycast_distance_field(
            field,
            poses[pose_index],
            angles,
            0.06,
            range_max_m,
            lidar_offset=lidar_offset,
            variant_id=variant_id,
            return_hits=True,
        )
        wall += int(np.sum(kinds == RAY_HIT_WALL))
        unknown += int(np.sum(kinds == RAY_HIT_UNKNOWN))
        max_range += int(np.sum(kinds == RAY_HIT_MAX_RANGE))
        opening = real[frame] >= sentinel
        opening_total += int(opening.sum())
        opening_wall += int(np.sum(opening & (kinds == RAY_HIT_WALL)))
        opening_unknown += int(
            np.sum(
                opening
                & ((kinds == RAY_HIT_UNKNOWN) | (kinds == RAY_HIT_MAX_RANGE))
            )
        )
    counted = max(wall + unknown + max_range, 1)
    return {
        "matched_unique_poses": int(np.unique(matched_indices).size),
        "ray_hit_wall_fraction": wall / counted,
        "ray_hit_unknown_fraction": unknown / counted,
        "ray_hit_max_range_fraction": max_range / counted,
        "real_opening_beams": opening_total,
        "real_opening_hits_wall_fraction": (
            opening_wall / opening_total if opening_total else 0.0
        ),
        "real_opening_hits_unknown_or_max_fraction": (
            opening_unknown / opening_total if opening_total else 0.0
        ),
    }


def evaluate_pose_free_lidar_gate(
    real_scans: np.ndarray,
    simulated_scans: np.ndarray,
    *,
    range_max_m: float = 30.0,
    quantile_tolerance_m: float = 0.50,
    temporal_tolerance_m: float = 0.50,
    max_range_rate_tolerance: float = 0.05,
    dropout_run_tolerance_beams: float = 16.0,
    frame_median_tolerance_m: float = 0.50,
    minimum_frames: int = 20,
) -> PoseFreeLidarGateResult:
    real = np.asarray(real_scans, dtype=np.float64)
    simulated = np.asarray(simulated_scans, dtype=np.float64)
    if real.ndim != 2 or simulated.ndim != 2 or real.shape[1] != simulated.shape[1]:
        raise ValueError("real and simulated scans must be 2-D with equal beam counts")
    if min(real.shape[0], simulated.shape[0]) < minimum_frames:
        raise ValueError(f"at least {minimum_frames} scans are required")
    real = np.clip(np.nan_to_num(real, nan=range_max_m, posinf=range_max_m), 0.0, range_max_m)
    simulated = np.clip(
        np.nan_to_num(simulated, nan=range_max_m, posinf=range_max_m),
        0.0,
        range_max_m,
    )
    matched_indices, frame_errors = match_simulated_support(
        real, simulated, range_max_m=range_max_m
    )
    matched_simulated = simulated[matched_indices]
    supported_fraction = float(np.mean(frame_errors <= frame_median_tolerance_m))

    quantiles = (10.0, 50.0, 90.0)
    real_quantiles = np.percentile(real, quantiles, axis=0)
    simulated_quantiles = np.percentile(matched_simulated, quantiles, axis=0)
    beam_error = np.max(np.abs(real_quantiles - simulated_quantiles), axis=0)
    passing_beams = beam_error <= quantile_tolerance_m
    passing_fraction = float(np.mean(passing_beams))

    sentinel = range_max_m * 0.999
    real_max_rate = float(np.mean(real >= sentinel))
    simulated_max_rate = float(np.mean(matched_simulated >= sentinel))
    max_rate_difference = abs(real_max_rate - simulated_max_rate)
    real_runs = np.asarray([_longest_true_run(row >= sentinel) for row in real])
    simulated_runs = np.asarray(
        [_longest_true_run(row >= sentinel) for row in matched_simulated]
    )
    run_difference = abs(
        float(np.percentile(real_runs, 95))
        - float(np.percentile(simulated_runs, 95))
    )

    real_temporal = np.median(np.abs(np.diff(real, axis=0)), axis=1)
    simulated_temporal = np.median(
        np.abs(np.diff(matched_simulated, axis=0)), axis=1
    )
    temporal_error = float(
        np.max(
            np.abs(
                np.percentile(real_temporal, quantiles)
                - np.percentile(simulated_temporal, quantiles)
            )
        )
    )
    passed = (
        supported_fraction >= 0.95
        and passing_fraction >= 0.95
        and max_rate_difference <= max_range_rate_tolerance
        and run_difference <= dropout_run_tolerance_beams
        and temporal_error <= temporal_tolerance_m
    )
    return PoseFreeLidarGateResult(
        passed=bool(passed),
        supported_frame_fraction=supported_fraction,
        frame_median_error_p95_m=float(np.percentile(frame_errors, 95)),
        passing_beam_fraction=passing_fraction,
        quantile_error_p95_m=float(np.percentile(beam_error, 95)),
        max_range_rate_difference=max_rate_difference,
        dropout_run_p95_difference_beams=run_difference,
        temporal_quantile_error_m=temporal_error,
    )


def _failing_beam_sectors(failing_beams: np.ndarray) -> list[dict[str, float | int]]:
    if failing_beams.size == 0:
        return []
    angles = np.degrees(beam_angles_rad())
    sectors: list[dict[str, float | int]] = []
    start = int(failing_beams[0])
    previous = start
    for beam in failing_beams[1:]:
        beam_index = int(beam)
        if beam_index != previous + 1:
            sectors.append(
                {
                    "start_beam": start,
                    "end_beam": previous,
                    "start_deg": float(angles[start]),
                    "end_deg": float(angles[previous]),
                    "n_beams": previous - start + 1,
                }
            )
            start = beam_index
        previous = beam_index
    sectors.append(
        {
            "start_beam": start,
            "end_beam": previous,
            "start_deg": float(angles[start]),
            "end_deg": float(angles[previous]),
            "n_beams": previous - start + 1,
        }
    )
    return sectors


def diagnose_actor_parity(
    real_observations: np.ndarray,
    simulated_observations: np.ndarray,
    *,
    lidar_tolerance_m: float = 0.50,
    non_lidar_tolerance: float = 0.10,
    minimum_frames: int = 20,
) -> dict:
    real, simulated = _validate_actor_observations(
        real_observations, simulated_observations, minimum_frames
    )
    dimension_error = _actor_dimension_error(
        real, simulated, minimum_frames=minimum_frames
    )
    lidar = evaluate_pose_free_lidar_gate(
        real[:, : si.LIDAR_DIM],
        simulated[:, : si.LIDAR_DIM],
        quantile_tolerance_m=lidar_tolerance_m,
        minimum_frames=minimum_frames,
    )
    quantiles = (10.0, 50.0, 90.0)
    real_lidar_q = np.percentile(real[:, : si.LIDAR_DIM], quantiles, axis=0)
    simulated_lidar_q = np.percentile(
        simulated[:, : si.LIDAR_DIM], quantiles, axis=0
    )
    beam_error = np.max(np.abs(real_lidar_q - simulated_lidar_q), axis=0)
    failing_beams = np.flatnonzero(beam_error > lidar_tolerance_m)
    longest_run = 0
    current = 0
    previous = -2
    for beam in failing_beams:
        if beam == previous + 1:
            current += 1
        else:
            current = 1
        longest_run = max(longest_run, current)
        previous = int(beam)
    residual = real_lidar_q[1] - simulated_lidar_q[1]
    residual_mean = float(np.mean(residual[failing_beams])) if failing_beams.size else 0.0
    if residual_mean < -1.0e-6:
        direction = "real_shorter"
    elif residual_mean > 1.0e-6:
        direction = "real_longer"
    else:
        direction = "mixed"
    failing_proprio = [
        name
        for name, error in zip(PROPRIO_NAMES, dimension_error[si.LIDAR_DIM :])
        if error > non_lidar_tolerance
    ]
    proprio_errors = {
        name: float(error)
        for name, error in zip(PROPRIO_NAMES, dimension_error[si.LIDAR_DIM :])
    }

    def _range(values: np.ndarray) -> dict[str, float]:
        return {
            "real_min": float(np.min(real[:, values])),
            "real_max": float(np.max(real[:, values])),
            "simulated_min": float(np.min(simulated[:, values])),
            "simulated_max": float(np.max(simulated[:, values])),
        }

    return {
        "failing_proprio": failing_proprio,
        "proprio_errors": proprio_errors,
        "failing_beam_count": int(failing_beams.size),
        "longest_failing_beam_run": int(longest_run),
        "failing_beam_sectors": _failing_beam_sectors(failing_beams),
        "lidar_residual_direction": direction,
        "lidar_supported_frame_fraction": lidar.supported_frame_fraction,
        "operating_state": {
            "speed": _range(si.VESC_SPEED),
            "throttle_current": _range(si.THROTTLE_CURRENT),
            "steer_t": _range(si.STEER_T),
        },
    }


def build_parity_report(
    real_observations: np.ndarray,
    simulated_observations: np.ndarray,
    *,
    bag: str,
    checkpoint_sha256: str | None = None,
    config: str | None = None,
) -> dict:
    actor = evaluate_actor_distribution_gate(
        real_observations, simulated_observations
    )
    lidar = evaluate_pose_free_lidar_gate(
        np.asarray(real_observations)[:, : si.LIDAR_DIM],
        np.asarray(simulated_observations)[:, : si.LIDAR_DIM],
    )
    report = {
        "bag": bag,
        "real_frames": int(np.asarray(real_observations).shape[0]),
        "simulated_frames": int(np.asarray(simulated_observations).shape[0]),
        "actor": asdict(actor),
        "lidar": asdict(lidar),
        "diagnostics": diagnose_actor_parity(
            real_observations, simulated_observations
        ),
        "passed": bool(actor.passed and lidar.passed),
    }
    if checkpoint_sha256 is not None:
        report["checkpoint_sha256"] = checkpoint_sha256
    if config is not None:
        report["config"] = config
    return report


def write_parity_report(path: Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def load_sensor_params(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    for node_name in ("sensor_racer", "sensor_policy"):
        node = data.get(node_name, {})
        params = node.get("ros__parameters")
        if isinstance(params, dict):
            return params
    raise ValueError(f"{path}: no sensor_racer ros__parameters")


def imu_calibration(params: dict) -> ImuCalibration:
    names = ("ax", "ay", "az", "gx", "gy", "gz")
    values = {
        "accel_to_ms2": float(params["imu_accel_to_ms2"]),
        "gyro_to_rads": float(params["imu_gyro_to_rads"]),
    }
    for name in names:
        values[f"{name}_sign"] = float(params[f"imu_{name}_sign"])
        values[f"{name}_bias"] = float(params[f"imu_{name}_bias"])
    return ImuCalibration(**values)


def _previous_index(times: np.ndarray, query: float) -> int:
    index = int(np.searchsorted(times, query, side="right") - 1)
    if index < 0:
        raise ValueError(f"No sample at or before synchronized time {query:.9f}")
    return index


def synchronize_observations(
    series: dict[str, np.ndarray],
    params: dict,
) -> dict[str, np.ndarray | str]:
    diagnostics = series["/sensor_racer/diagnostics"]
    desired = series["/rl/actuator/desired"]
    if diagnostics.shape[1] <= 1 + si.DIAG_GRU_RESET:
        raise ValueError("Bag diagnostics do not contain the required legacy fields")
    valid_indices = np.flatnonzero(diagnostics[:, 1 + si.DIAG_VALID_TICK] == 1.0)
    if valid_indices.size != desired.shape[0]:
        raise ValueError(
            f"valid diagnostic ticks={valid_indices.size} but desired commands="
            f"{desired.shape[0]}"
        )
    required = ("/scan", "/sensors/imu/raw", "/odom", "/rl/actuator/applied")
    for topic in required:
        if series[topic].shape[0] == 0:
            raise ValueError(f"Bag is missing required topic {topic}")

    scans = series["/scan"]
    imu = series["/sensors/imu/raw"]
    odom = series["/odom"]
    applied = series["/rl/actuator/applied"]
    cal = imu_calibration(params)
    observations = np.zeros((valid_indices.size, si.NUM_OBS), dtype=np.float32)
    lidar = np.zeros((valid_indices.size, si.LIDAR_DIM), dtype=np.float32)
    applied_rows = np.zeros((valid_indices.size, applied.shape[1]), dtype=float)
    frame_times = diagnostics[valid_indices, 0].copy()
    reset = diagnostics[valid_indices, 1 + si.DIAG_GRU_RESET].astype(bool)
    scan_age = np.zeros(valid_indices.size, dtype=float)
    imu_count = np.zeros(valid_indices.size, dtype=np.int64)
    executed_steer = np.zeros(si.STEER_HISTORY, dtype=np.float32)
    previous_applied_long = 0.0

    for frame, diag_index in enumerate(valid_indices):
        t = float(diagnostics[diag_index, 0])
        interval_start = (
            float(diagnostics[diag_index - 1, 0]) if diag_index > 0 else -np.inf
        )
        scan_index = int(np.argmin(np.abs(scans[:, 1] - desired[frame, 2])))
        odom_index = _previous_index(odom[:, 0], t)
        diagnostic_source = int(
            diagnostics[diag_index, 1 + si.DIAG_APPLIED_SOURCE]
        )
        matching_applied = np.flatnonzero(
            (applied[:, 0] <= t) & (applied[:, 8] == diagnostic_source)
        )
        if matching_applied.size == 0:
            raise ValueError(
                f"No applied source={diagnostic_source} sample before frame {frame}"
            )
        applied_index = int(matching_applied[-1])
        applied_row = applied[applied_index]
        applied_rows[frame] = applied_row
        scan_age[frame] = t - scans[scan_index, 0]

        interval = imu[(imu[:, 0] > interval_start) & (imu[:, 0] <= t)]
        imu_count[frame] = interval.shape[0]
        samples = [
            RawImuSample(
                stamp_s=float(row[0]),
                ax=float(row[1]),
                ay=float(row[2]),
                az=float(row[3]),
                gx=float(row[4]),
                gy=float(row[5]),
                gz=float(row[6]),
            )
            for row in interval
        ]
        actor_imu, _ = actor_imu_from_interval(samples, cal, freeze_const_channels=True)
        packed_lidar = pack_lidar_from_scan(
            float(scans[scan_index, 2]),
            float(scans[scan_index, 3]),
            scans[scan_index, 6:],
            scan_range_min=float(scans[scan_index, 4]),
        )
        lidar[frame] = packed_lidar

        if reset[frame]:
            executed_steer.fill(0.0)
            previous_applied_long = 0.0
        applied_long = float(applied_row[6])
        applied_steer = float(applied_row[7]) * float(
            params.get("max_steer", si.MAX_STEER_RAD)
        )
        vesc_current = applied_current_fraction(
            float(applied_row[3] - applied_row[4]),
            float(params["i_drive_max_a"]),
            float(params["i_brake_max_a"]),
        )
        steer_view = np.array(
            [
                applied_steer,
                executed_steer[0],
                executed_steer[1],
                executed_steer[2],
            ],
            dtype=np.float32,
        )
        pack_actor_observation(
            packed_lidar,
            actor_imu,
            float(params["twist_vx_sign"]) * float(odom[odom_index, 1]),
            vesc_current,
            applied_long,
            previous_applied_long,
            steer_view,
            out=observations[frame],
        )
        executed_steer[3] = executed_steer[2]
        executed_steer[2] = executed_steer[1]
        executed_steer[1] = executed_steer[0]
        executed_steer[0] = applied_steer
        previous_applied_long = applied_long

    return {
        "observations": observations,
        "lidar": lidar,
        "frame_times": frame_times,
        "reset": reset,
        "diagnostics": diagnostics[valid_indices, 1:],
        "desired": desired,
        "applied": applied_rows,
        "scan_age_s": scan_age,
        "imu_count": imu_count,
        "observation_source": "reconstructed_from_sensor_topics",
    }


@torch.no_grad()
def policy_actions(
    actor,
    normalizer,
    observations: np.ndarray,
    reset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = actor.initial_hidden(1, device=torch.device("cpu"), dtype=torch.float32)
    actions = []
    hidden_before = []
    hidden_after = []
    for obs, do_reset in zip(observations, reset):
        if do_reset:
            hidden.zero_()
        hidden_before.append(float(torch.linalg.vector_norm(hidden).item()))
        normalized = normalizer.normalize(torch.from_numpy(obs).unsqueeze(0))
        action, _, hidden = actor.step(
            normalized,
            hidden,
            reset_mask=None,
            deterministic=True,
            with_logprob=False,
        )
        actions.append(action.squeeze(0).cpu().numpy())
        hidden_after.append(float(torch.linalg.vector_norm(hidden).item()))
    return (
        np.asarray(actions, dtype=np.float32),
        np.asarray(hidden_before),
        np.asarray(hidden_after),
    )


def collect_simulated_actor_observations(
    *,
    checkpoint: Path,
    config_path: Path,
    num_envs: int = 64,
    num_steps: int = 600,
    sample_stride: int = 2,
    seed: int = 42,
    device: str = "cuda:0",
) -> dict[str, np.ndarray]:
    import copy

    import torch

    from evaluation import load_sensor_actor_bundle
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt
    from standalone_trainer import DEFAULT_CONFIG, _deep_merge, build_env_cfg

    checkpoint = Path(checkpoint)
    config_path = Path(config_path)
    patch = json.loads(config_path.read_text())
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    _deep_merge(cfg, patch)
    torch_device = torch.device(device)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch_device,
        eps=1e-12,
    )
    env = F1tenthEnv(
        num_envs=num_envs,
        env_cfg=build_env_cfg(
            cfg,
            launch_strategy="uniform_jittered",
            launch_strategy_data={"num_cars": num_envs},
        ),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    actor, normalizer, _, _ = load_sensor_actor_bundle(
        checkpoint,
        torch_device,
        expected_actor_obs_dim=int(cfg["obs"]["num_actor_obs"]),
        expected_action_dim=int(cfg["env"]["num_actions"]),
        expected_layout_version=int(cfg["obs"]["actor_layout_version"]),
        norm_eps=float(cfg["obs"]["norm_eps"]),
        norm_clip=float(cfg["obs"]["norm_clip"]),
        expected_critic_obs_dim=int(cfg["obs"]["num_obs"]),
        require_obs_norm=True,
    )
    obs, _ = env.reset(seed=seed, with_sensors=True)
    hidden = actor.initial_hidden(
        num_envs, device=torch_device, dtype=torch.float32
    )
    collected = []
    poses = []
    clip = float(cfg["env"]["clip_actions"])
    control_interval = int(cfg["env"]["control_interval"])
    with torch.no_grad():
        for step in range(num_steps):
            actor_obs = obs["actor"].to(torch.float32)
            if step % sample_stride == 0:
                collected.append(actor_obs.cpu().numpy())
                state = env.read_state()
                quat = state["base_quat"].cpu().numpy()
                yaw = 2.0 * np.arctan2(quat[:, 3], quat[:, 0])
                poses.append(
                    np.stack(
                        [
                            state["base_pos"][:, 0].cpu().numpy(),
                            state["base_pos"][:, 1].cpu().numpy(),
                            yaw,
                        ],
                        axis=1,
                    )
                )
            actions, _, hidden = actor.step(
                normalizer.normalize(actor_obs),
                hidden,
                reset_mask=None,
                deterministic=True,
                with_logprob=False,
            )
            obs, _, done, _ = env.step(
                actions.clamp(-clip, clip),
                n_steps=control_interval,
                with_sensors=True,
            )
            done = done.to(device=hidden.device, dtype=torch.bool)
            hidden[done] = 0
    env.close()
    observations = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
    pose_array = np.concatenate(poses, axis=0).astype(np.float32, copy=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "observations": observations,
        "poses": pose_array,
        "checkpoint_sha256": np.asarray(digest),
        "config": np.asarray(str(config_path)),
        "seed": np.asarray(seed),
        "num_envs": np.asarray(num_envs),
        "num_steps": np.asarray(num_steps),
        "sample_stride": np.asarray(sample_stride),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument("--simulated", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sim-out-npz", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    real = None
    if args.bag is not None:
        if args.params is None:
            parser.error("--params is required with --bag")
        sync = synchronize_observations(
            load_sensor_policy_series(args.bag), load_sensor_params(args.params)
        )
        real = sync["observations"]
        if args.out_npz is not None:
            save_observation_npz(
                args.out_npz,
                **{key: value for key, value in sync.items() if isinstance(value, np.ndarray)},
                bag=np.asarray(args.bag.name),
                observation_source=np.asarray(sync["observation_source"]),
            )
        print(json.dumps({
            "frames": int(real.shape[0]),
            "observation_source": sync["observation_source"],
            "mean_scan_age_s": float(np.mean(sync["scan_age_s"])),
        }, indent=2))
    elif args.out_npz is not None:
        real = np.load(args.out_npz)["observations"]

    simulated = None
    sim_meta = {}
    sim_poses = None
    if args.checkpoint is not None:
        if args.config is None:
            parser.error("--config is required with --checkpoint")
        captured = collect_simulated_actor_observations(
            checkpoint=args.checkpoint,
            config_path=args.config,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
        )
        simulated = captured["observations"]
        sim_poses = captured["poses"]
        sim_meta = {
            "checkpoint_sha256": str(captured["checkpoint_sha256"]),
            "config": str(captured["config"]),
        }
        if args.sim_out_npz is not None:
            save_observation_npz(args.sim_out_npz, **captured)
    elif args.simulated is not None:
        loaded = np.load(args.simulated)
        simulated = loaded["observations"]
        if "poses" in loaded.files:
            sim_poses = loaded["poses"]
        if "checkpoint_sha256" in loaded.files:
            sim_meta["checkpoint_sha256"] = str(loaded["checkpoint_sha256"])
        if "config" in loaded.files:
            sim_meta["config"] = str(loaded["config"])

    if args.report is not None:
        if real is None or simulated is None:
            parser.error(
                "--report requires reconstructed real and simulated observations"
            )
        bag_name = args.bag.name if args.bag is not None else "unknown"
        report = build_parity_report(
            real,
            simulated,
            bag=bag_name,
            **sim_meta,
        )
        config_path = args.config
        if config_path is None and "config" in sim_meta:
            config_path = Path(str(sim_meta["config"]))
        if sim_poses is not None and config_path is not None:
            patch = json.loads(Path(config_path).read_text())
            map_yaml = patch.get("sensor", {}).get("lidar_map_yaml")
            if map_yaml:
                from f1tenth_env.utils import load_occupancy_distance_field

                yaml_path = Path(map_yaml)
                if not yaml_path.is_absolute():
                    training_dir = Path(config_path).resolve().parents[1]
                    candidates = (
                        training_dir / map_yaml,
                        _REPO / "training" / map_yaml,
                        Path(config_path).resolve().parent / map_yaml,
                    )
                    yaml_path = next(
                        (path for path in candidates if path.exists()),
                        candidates[0],
                    )
                sensor = patch.get("sensor", {})
                field = load_occupancy_distance_field(
                    yaml_path,
                    expected_sha256=sensor.get("lidar_map_sha256"),
                    variant_count=int(sensor.get("lidar_map_variant_count", 1)),
                    max_wall_offset_m=float(
                        sensor.get("lidar_wall_offset_max_m", 0.0)
                    ),
                    variant_seed=int(sensor.get("lidar_map_variant_seed", 0)),
                )
                report["diagnostics"]["support_ray_hits"] = diagnose_support_ray_hits(
                    real[:, : si.LIDAR_DIM],
                    simulated[:, : si.LIDAR_DIM],
                    sim_poses,
                    field,
                )
        write_parity_report(args.report, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

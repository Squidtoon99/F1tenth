"""Rosbag observation reconstruction for sensor-policy parity."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
for _path in (
    _REPO,
    _REPO / "src/racing_rl/f1tenth_rl_agent",
    _REPO / "libs/f1tenth_contract",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from calibration.bag_io import load_sensor_policy_series  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402

from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    ImuCalibration,
    RawImuSample,
    actor_imu_from_interval,
    pack_actor_observation,
    pack_lidar_from_scan,
)


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
        applied_steer = float(applied_row[7])
        vesc_current = float(applied_row[3] - applied_row[4])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()
    sync = synchronize_observations(
        load_sensor_policy_series(args.bag), load_sensor_params(args.params)
    )
    print(json.dumps({
        "frames": int(sync["observations"].shape[0]),
        "observation_source": sync["observation_source"],
        "mean_scan_age_s": float(np.mean(sync["scan_age_s"])),
    }, indent=2))


if __name__ == "__main__":
    main()

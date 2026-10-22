"""CPU tests for IRL Gigaflow packing + frozen-condition inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.sensor_preprocessing import ImuCalibration, RawImuSample

_REPO_SRC = None
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "gigaflow" / "src"
    if (_candidate / "gigaflow_f1tenth").is_dir():
        _REPO_SRC = _candidate
        sys.path.insert(0, str(_candidate))
        break

gigaflow_f1tenth = pytest.importorskip("gigaflow_f1tenth")

from f1tenth_rl_agent.gigaflow_inference_runtime import (  # noqa: E402
    FOLDED_CONDITION_DIM,
    FoldedLidarGruAdapter,
    GigaflowInferenceRuntime,
)
from f1tenth_rl_agent.policy_model import (  # noqa: E402
    SquashedGaussianLidarGRUActor,
    load_folded_sensor_actor,
)
from gigaflow_f1tenth.artifacts import (  # noqa: E402
    build_actor_artifact_payload,
    load_irl_actor,
)
from gigaflow_f1tenth.config import load_config  # noqa: E402
from gigaflow_f1tenth.model import build_actor  # noqa: E402


def _smoke_config():
    assert _REPO_SRC is not None
    return load_config(_REPO_SRC.parent / "configs" / "smoke.yaml")


def _write_baked(tmp_path: Path) -> Path:
    cfg = _smoke_config()
    actor = build_actor(cfg)
    path = tmp_path / "gigaflow_policy.pt"
    torch.save(build_actor_artifact_payload(cfg, actor), path)
    return path


def test_gigaflow_runtime_actions_in_unit_interval(tmp_path):
    path = _write_baked(tmp_path)
    actor, condition, _ = load_irl_actor(str(path), device="cpu")
    runtime = GigaflowInferenceRuntime(actor, condition, torch.device("cpu"))
    scan = [1.0] * si.LIDAR_DIM
    imu = [
        RawImuSample(stamp_s=0.0, ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)
    ]
    raw, _, _, finite = runtime.pack_observation(
        scan_angle_min=si.LIDAR_ANGLE_MIN,
        scan_angle_increment=si.LIDAR_ANGLE_INCREMENT,
        scan_ranges=scan,
        scan_range_min=si.LIDAR_RANGE_MIN,
        imu_interval=imu,
        imu_cal=ImuCalibration(),
        speed=1.0,
        applied_long=0.1,
        applied_steer_rad=0.0,
        vesc_current=0.0,
    )
    assert finite
    assert raw.shape == (si.NUM_OBS,)
    action, _ = runtime.infer_host_obs()
    assert action.shape == (si.NUM_ACTIONS,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0 + 1e-5
    again, _ = runtime.infer_host_obs()
    # Frozen condition: same obs buffer, recurrent hidden advances.
    assert again.shape == action.shape


def test_folded_lidar_cnn_gru_runtime_272d_gru(tmp_path):
    actor = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=torch.nn.ReLU,
        act_limit=si.ACT_LIMIT,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    assert int(actor.gru.input_size) == 272
    payload = {
        "actor": actor.state_dict(),
        "obs_norm": {
            "mean": torch.zeros(si.NUM_OBS),
            "var": torch.ones(si.NUM_OBS),
            "count": torch.tensor(8.0),
        },
        "actor_architecture": dict(actor.actor_architecture),
        "control_hz": 20.0,
        "steering_delta_max_rad": float(np.pi / 120.0),
        "steering_action_mode": "delta",
        "longitudinal_mode": "force",
    }
    path = tmp_path / "folded.pt"
    torch.save(payload, path)
    loaded, obs_norm, _ = load_folded_sensor_actor(str(path), torch.device("cpu"))
    assert int(loaded.gru.input_size) == 272
    dummy = torch.zeros(1, FOLDED_CONDITION_DIM)
    runtime = GigaflowInferenceRuntime(
        FoldedLidarGruAdapter(loaded, obs_norm), dummy, torch.device("cpu")
    )
    scan = [1.0] * si.LIDAR_DIM
    imu = [
        RawImuSample(stamp_s=0.0, ax=0.0, ay=0.0, az=1.0, gx=0.0, gy=0.0, gz=0.0)
    ]
    raw, _, _, finite = runtime.pack_observation(
        scan_angle_min=si.LIDAR_ANGLE_MIN,
        scan_angle_increment=si.LIDAR_ANGLE_INCREMENT,
        scan_ranges=scan,
        scan_range_min=si.LIDAR_RANGE_MIN,
        imu_interval=imu,
        imu_cal=ImuCalibration(),
        speed=1.0,
        applied_long=0.1,
        applied_steer_rad=0.0,
        vesc_current=0.0,
    )
    assert finite
    assert raw.shape == (si.NUM_OBS,)
    action, _ = runtime.infer_host_obs()
    assert action.shape == (si.NUM_ACTIONS,)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0 + 1e-5

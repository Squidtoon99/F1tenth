"""Tests for preallocated sensor inference runtime."""

from __future__ import annotations

import gc
import os
import subprocess
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from f1tenth_rl_agent import sensor_interfaces as si
from f1tenth_rl_agent.policy_model import (
    SquashedGaussianLidarGRUActor,
    load_sensor_actor,
    load_sensor_obs_norm,
)
from f1tenth_rl_agent.sensor_inference_runtime import SensorInferenceRuntime


def _write_sensor_checkpoint(path: str) -> None:
    actor = SquashedGaussianLidarGRUActor(
        obs_dim=si.NUM_OBS,
        act_dim=si.NUM_ACTIONS,
        hidden_sizes=si.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=si.ACT_LIMIT,
        gru_hidden_dim=si.GRU_HIDDEN_DIM,
    )
    payload = {
        "actor": actor.state_dict(),
        "obs_norm": {
            "mean": torch.zeros(si.NUM_OBS),
            "var": torch.ones(si.NUM_OBS),
            "count": torch.tensor(1.0),
        },
        "obs_dim": si.NUM_OBS,
        "actor_obs_dim": si.NUM_OBS,
        "critic_obs_dim": 392,
        "actor_layout_version": si.ACTOR_LAYOUT_VERSION,
        "action_dim": si.NUM_ACTIONS,
        "policy_format_version": si.POLICY_FORMAT_VERSION,
        "observation_preprocessing_version": si.OBS_PREPROCESSING_VERSION,
        "artifact_scope": si.ARTIFACT_SCOPE_SIM_TRAINING,
        "actor_architecture": actor.actor_architecture,
        "longitudinal_mode": "force",
        "steering_action_mode": si.STEERING_ACTION_MODE,
        "steering_delta_max_rad": float(si.STEERING_DELTA_MAX_RAD),
        "control_hz": si.CONTROL_HZ,
    }
    torch.save(payload, path)


def _reference_step(actor, normalizer, obs_np, hidden, device):
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).reshape(1, -1)
    with torch.inference_mode():
        normed = normalizer.normalize(obs_t)
        action, _, next_hidden = actor.step(
            normed,
            hidden,
            reset_mask=None,
            deterministic=True,
            with_logprob=False,
        )
    return (
        action.squeeze(0).cpu().numpy().astype(np.float32),
        next_hidden.detach().clone(),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no cuda"
            ),
        ),
    ],
)
def test_runtime_matches_reference_step(device):
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "policy.pt")
        _write_sensor_checkpoint(ckpt)
        dev = torch.device(device)
        actor = load_sensor_actor(ckpt, "actor", dev)
        norm = load_sensor_obs_norm(ckpt, dev, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
        runtime = SensorInferenceRuntime(actor, norm, dev, warmup_iters=2)

        rng = np.random.default_rng(0)
        hidden_ref = actor.initial_hidden(1, device=dev, dtype=torch.float32)
        for _ in range(8):
            obs = rng.standard_normal(si.NUM_OBS, dtype=np.float32)
            expected_action, hidden_ref = _reference_step(
                actor, norm, obs, hidden_ref, dev
            )
            actual_action, _ = runtime.infer_from_host_obs(obs)
            assert np.allclose(expected_action, actual_action, atol=1e-5, rtol=1e-5)
            assert np.allclose(
                hidden_ref.cpu().numpy(),
                runtime.hidden.detach().cpu().numpy(),
                atol=1e-5,
                rtol=1e-5,
            )


def test_runtime_reset_zeros_hidden():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "policy.pt")
        _write_sensor_checkpoint(ckpt)
        dev = torch.device("cpu")
        actor = load_sensor_actor(ckpt, "actor", dev)
        norm = load_sensor_obs_norm(ckpt, dev, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
        runtime = SensorInferenceRuntime(actor, norm, dev, warmup_iters=1)
        obs = np.ones(si.NUM_OBS, dtype=np.float32)
        runtime.infer_from_host_obs(obs)
        assert float(runtime.hidden.abs().sum()) > 0.0
        runtime.reset_hidden()
        assert float(runtime.hidden.abs().sum()) == 0.0


def test_runtime_reuses_host_obs_buffer():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "policy.pt")
        _write_sensor_checkpoint(ckpt)
        dev = torch.device("cpu")
        actor = load_sensor_actor(ckpt, "actor", dev)
        norm = load_sensor_obs_norm(ckpt, dev, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
        runtime = SensorInferenceRuntime(actor, norm, dev, warmup_iters=0)
        buf = runtime.host_obs_buffer
        buf.fill(3.0)
        runtime.infer_host_obs()
        assert buf is runtime.host_obs_buffer


def test_runtime_hot_path_allocation_stable():
    if not torch.cuda.is_available():
        pytest.skip("cuda required for allocation stability check")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "policy.pt")
        _write_sensor_checkpoint(ckpt)
        dev = torch.device("cuda")
        actor = load_sensor_actor(ckpt, "actor", dev)
        norm = load_sensor_obs_norm(ckpt, dev, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
        runtime = SensorInferenceRuntime(actor, norm, dev, warmup_iters=3)
        obs = runtime.host_obs_buffer
        obs.fill(0.1)

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
        before = torch.cuda.memory_allocated(dev)
        for _ in range(200):
            runtime.infer_host_obs()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(dev)
        assert after <= before + 4096


def _trainer_using_gpu() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", "standalone_trainer.py"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


@pytest.mark.skipif(
    _trainer_using_gpu(),
    reason="standalone_trainer occupies GPU; stop service for cross-backend parity",
)
def test_format4_checkpoint_cpu_cuda_parity():
    ckpt = os.environ.get("SENSOR_POLICY_CKPT")
    if not ckpt or not os.path.isfile(ckpt):
        pytest.skip("set SENSOR_POLICY_CKPT to a format-4 artifact")

    rng = np.random.default_rng(0)
    obs_list = [rng.standard_normal(si.NUM_OBS, dtype=np.float32) for _ in range(6)]

    cpu_actor = load_sensor_actor(ckpt, "actor", torch.device("cpu"))
    cpu_norm = load_sensor_obs_norm(
        ckpt, torch.device("cpu"), si.OBS_NORM_EPS, si.OBS_NORM_CLIP
    )
    cpu_runtime = SensorInferenceRuntime(cpu_actor, cpu_norm, torch.device("cpu"))

    if not torch.cuda.is_available():
        pytest.skip("cuda required for cross-backend parity")

    cuda_actor = load_sensor_actor(ckpt, "actor", torch.device("cuda"))
    cuda_norm = load_sensor_obs_norm(
        ckpt, torch.device("cuda"), si.OBS_NORM_EPS, si.OBS_NORM_CLIP
    )
    cuda_runtime = SensorInferenceRuntime(
        cuda_actor, cuda_norm, torch.device("cuda"), warmup_iters=3
    )

    for obs in obs_list:
        cpu_action, _ = cpu_runtime.infer_from_host_obs(obs)
        cuda_action, _ = cuda_runtime.infer_from_host_obs(obs)
        assert np.allclose(cpu_action, cuda_action, atol=1e-5, rtol=1e-5)
        assert np.allclose(
            cpu_runtime.hidden.numpy(),
            cuda_runtime.hidden.detach().cpu().numpy(),
            atol=2e-3,
            rtol=2e-3,
        )

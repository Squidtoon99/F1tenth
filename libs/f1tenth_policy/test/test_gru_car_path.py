"""Car-path GRU step() must load old and cuDNN-trained artifacts identically."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from f1tenth_policy import ACTOR_OBS_DIM, make_actor
from f1tenth_policy.actor import SquashedGaussianLidarGRUActor
from f1tenth_policy.artifact import load_sensor_artifact, validate_sensor_policy_artifact

_OVERNIGHT = Path(
    "/home/ubuntu/projects/F1tenth-overnight/outputs/runs/"
    "overnight-20261012/checkpoints/policy_317440000.pt"
)
_ENTROPY_DIR = Path(
    "/home/ubuntu/projects/F1tenth-entropy-ab-runs/outputs/runs/"
    "entropy-ab-A/checkpoints"
)


def _cpu() -> torch.device:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return torch.device("cpu")


def _tiny_actor(seed: int = 0) -> SquashedGaussianLidarGRUActor:
    torch.manual_seed(seed)
    return make_actor(ACTOR_OBS_DIM, 2, [32, 32], activation=nn.ReLU)


def _step_loop(actor, obs, hidden):
    actions = []
    h = hidden
    for t in range(obs.shape[1]):
        action, _, h = actor.step(obs[:, t], h, deterministic=True, with_logprob=False)
        actions.append(action)
    return torch.stack(actions, dim=1), h


def test_step_matches_gru_cell_loop_on_cpu():
    device = _cpu()
    actor = _tiny_actor().to(device).eval()
    obs = torch.randn(2, 5, ACTOR_OBS_DIM, device=device)
    hidden = actor.initial_hidden(2, device=device, dtype=obs.dtype)
    with torch.inference_mode():
        stepped, h_step = _step_loop(actor, obs, hidden.clone())
        h = hidden.clone()
        for t in range(obs.shape[1]):
            features, _ = actor.encode(obs[:, t])
            h = actor.gru_cell(features, h)
    assert stepped.shape == (2, 5, 2)
    assert torch.allclose(h_step, h, atol=1e-5, rtol=1e-5)


def test_segment_gru_sequence_matches_step_loop():
    device = _cpu()
    actor = _tiny_actor(1).to(device).eval()
    obs = torch.randn(3, 8, ACTOR_OBS_DIM, device=device)
    hidden = actor.initial_hidden(3, device=device, dtype=obs.dtype)
    reset = torch.zeros(3, 8, device=device)
    reset[:, 0] = 1.0
    reset[1, 4] = 1.0
    with torch.inference_mode():
        seq_actions, _, seq_h = actor.forward_sequence(
            obs, hidden.clone(), reset_mask=reset, deterministic=True
        )
        step_actions, step_h = _step_loop_with_reset(actor, obs, hidden.clone(), reset)
    assert torch.allclose(seq_actions, step_actions, atol=1e-5, rtol=1e-5)
    assert torch.allclose(seq_h, step_h, atol=1e-5, rtol=1e-5)


def _step_loop_with_reset(actor, obs, hidden, reset):
    actions = []
    h = hidden
    for t in range(obs.shape[1]):
        action, _, h = actor.step(
            obs[:, t],
            h,
            reset_mask=reset[:, t],
            deterministic=True,
            with_logprob=False,
        )
        actions.append(action)
    return torch.stack(actions, dim=1), h


def _load_and_step(path: Path, obs: torch.Tensor, device: torch.device):
    payload = load_sensor_artifact(str(path), map_location=device)
    validate_sensor_policy_artifact(
        payload, expected_architecture=payload["actor_architecture"]
    )
    actor = make_actor(
        ACTOR_OBS_DIM,
        2,
        list(payload["actor_architecture"]["hidden_layers"]),
        activation=nn.ReLU,
    )
    actor.load_state_dict(payload["actor"])
    actor.to(device).eval()
    hidden = actor.initial_hidden(1, device=device, dtype=torch.float32)
    with torch.inference_mode():
        action, _, hidden = actor.step(
            obs, hidden, deterministic=True, with_logprob=False
        )
        action2, _, _ = actor.step(obs, hidden, deterministic=True, with_logprob=False)
    return actor, action, action2


def test_overnight_and_entropy_checkpoints_load_and_step_on_cpu():
    device = _cpu()
    obs = torch.zeros(1, ACTOR_OBS_DIM, device=device)
    if _OVERNIGHT.is_file():
        actor, action, action2 = _load_and_step(_OVERNIGHT, obs, device)
        assert action.shape == (1, 2)
        assert torch.isfinite(action).all()
        assert torch.isfinite(action2).all()
        assert float(action.abs().max()) <= 1.0 + 1e-5
        assert actor.gru.weight_ih_l0.shape[0] == 3 * actor.gru_hidden_dim
    else:
        pytest.skip(f"overnight checkpoint missing: {_OVERNIGHT}")

    entropy_ckpts = sorted(_ENTROPY_DIR.glob("policy_*.pt")) if _ENTROPY_DIR.is_dir() else []
    if not entropy_ckpts:
        pytest.skip(f"entropy-ab checkpoint missing under {_ENTROPY_DIR}")
    latest = max(entropy_ckpts, key=lambda p: p.stat().st_mtime)
    actor_b, action_b, action_b2 = _load_and_step(latest, obs, device)
    assert action_b.shape == (1, 2)
    assert torch.isfinite(action_b).all()
    assert torch.isfinite(action_b2).all()
    assert actor_b.gru.weight_ih_l0.shape == actor.gru.weight_ih_l0.shape

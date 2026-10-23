"""Unit tests for on-car KL-clipped PPO (SOURCE_RL ticks only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_SRC = None
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "gigaflow" / "src"
    if (_candidate / "gigaflow_f1tenth").is_dir():
        _REPO_SRC = _candidate
        sys.path.insert(0, str(_candidate))
        break

pytest.importorskip("gigaflow_f1tenth")

from f1tenth_rl_agent.online_ppo import (  # noqa: E402
    OnlinePPOUpdater,
    actor_supports_online_ppo,
)
from gigaflow_f1tenth.artifacts import (  # noqa: E402
    bake_irl_artifact,
    build_actor_artifact_payload,
    load_irl_actor,
)
from gigaflow_f1tenth.config import load_config  # noqa: E402
from gigaflow_f1tenth.model import build_actor  # noqa: E402


def _smoke_actor_and_condition(tmp_path: Path):
    assert _REPO_SRC is not None
    cfg = load_config(_REPO_SRC.parent / "configs" / "courtyard_ppo.yaml")
    src = tmp_path / "actor.pt"
    baked = tmp_path / "baked.pt"
    torch.save(build_actor_artifact_payload(cfg, build_actor(cfg)), src)
    bake_irl_artifact(str(src), str(baked))
    actor, condition, payload = load_irl_actor(str(baked), device="cpu")
    return actor, condition, payload


def test_online_ppo_updates_actor_and_writes_latest_checkpoint(tmp_path):
    actor, condition, payload = _smoke_actor_and_condition(tmp_path)
    assert actor_supports_online_ppo(actor)
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    updater = OnlinePPOUpdater(
        actor,
        condition,
        rollout_steps=4,
        lr=3.0e-5,
        kl_coef=0.1,
        kl_clip=10.0,
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "online_ppo"),
        artifact_payload=payload,
    )
    rng = np.random.default_rng(0)
    stats = None
    for step in range(4):
        obs = rng.normal(size=(1097,)).astype(np.float32)
        action = rng.uniform(-0.5, 0.5, size=(2,)).astype(np.float32)
        stats = updater.record(obs, action, logp=-1.0, reward=float(step))
    assert stats is not None
    assert stats["skipped"] == pytest.approx(0.0)
    assert stats["steps"] == pytest.approx(4.0)
    latest = tmp_path / "online_ppo" / "actor_latest.pt"
    assert latest.is_file()
    changed = any(
        not torch.allclose(before[key], value)
        for key, value in actor.state_dict().items()
    )
    assert changed
    loaded, loaded_cond, _ = load_irl_actor(str(latest), device="cpu")
    assert actor_supports_online_ppo(loaded)
    assert loaded_cond.numel() == condition.numel()


def _assert_cpu_training_copies(updater: OnlinePPOUpdater, actor: torch.nn.Module) -> None:
    assert updater.actor is actor
    assert next(updater._train_actor.parameters()).device.type == "cpu"
    assert next(updater._base_actor.parameters()).device.type == "cpu"
    assert updater._train_actor is not actor
    assert updater._base_actor is not actor
    assert updater._train_condition.device.type == "cpu"


def _fill_rollout(updater: OnlinePPOUpdater, steps: int, seed: int) -> dict[str, float] | None:
    rng = np.random.default_rng(seed)
    stats = None
    for step in range(steps):
        obs = rng.normal(size=(1097,)).astype(np.float32)
        action = rng.uniform(-0.5, 0.5, size=(2,)).astype(np.float32)
        stats = updater.record(obs, action, logp=-1.0, reward=float(step))
    return stats


def test_online_ppo_trains_on_cpu_copy_and_syncs_inference_actor(tmp_path):
    actor, condition, payload = _smoke_actor_and_condition(tmp_path)
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    updater = OnlinePPOUpdater(
        actor,
        condition,
        rollout_steps=4,
        lr=3.0e-5,
        kl_coef=0.1,
        kl_clip=10.0,
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "online_ppo"),
        artifact_payload=payload,
    )
    _assert_cpu_training_copies(updater, actor)
    stats = _fill_rollout(updater, steps=4, seed=2)
    assert stats is not None
    assert stats["skipped"] == pytest.approx(0.0)
    for key, value in actor.state_dict().items():
        assert torch.allclose(value, updater._train_actor.state_dict()[key])
        assert torch.allclose(value, updater._base_actor.state_dict()[key])
    changed = any(
        not torch.allclose(before[key], value)
        for key, value in actor.state_dict().items()
    )
    assert changed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_online_ppo_keeps_inference_on_cuda_and_updates_on_cpu(tmp_path):
    actor, condition, payload = _smoke_actor_and_condition(tmp_path)
    device = torch.device("cuda")
    actor = actor.to(device)
    condition = condition.to(device)
    before = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    updater = OnlinePPOUpdater(
        actor,
        condition,
        rollout_steps=4,
        lr=3.0e-5,
        kl_coef=0.1,
        kl_clip=10.0,
        device=device,
        output_dir=str(tmp_path / "online_ppo"),
        artifact_payload=payload,
    )
    _assert_cpu_training_copies(updater, actor)
    assert next(actor.parameters()).is_cuda
    stats = _fill_rollout(updater, steps=4, seed=3)
    assert stats is not None
    assert stats["skipped"] == pytest.approx(0.0)
    assert next(actor.parameters()).is_cuda
    assert next(updater._train_actor.parameters()).device.type == "cpu"
    for key, value in actor.state_dict().items():
        assert value.device.type == "cuda"
        assert torch.allclose(value.cpu(), updater._train_actor.state_dict()[key].cpu())
    changed = any(
        not torch.allclose(before[key], value.cpu())
        for key, value in actor.state_dict().items()
    )
    assert changed


def test_online_ppo_sequence_logp_uses_batched_evaluate_actions_sequence():
    import inspect

    from f1tenth_rl_agent.online_ppo import OnlinePPOUpdater

    source = inspect.getsource(OnlinePPOUpdater._sequence_logp)
    assert "evaluate_actions_sequence" in source
    assert "for t in range" not in source


def test_online_ppo_discards_partial_rollout_on_interrupt(tmp_path):
    actor, condition, payload = _smoke_actor_and_condition(tmp_path)
    updater = OnlinePPOUpdater(
        actor,
        condition,
        rollout_steps=4,
        lr=3.0e-5,
        kl_coef=0.1,
        kl_clip=10.0,
        device=torch.device("cpu"),
        output_dir=str(tmp_path / "online_ppo"),
        artifact_payload=payload,
    )
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(1097,)).astype(np.float32)
    action = np.zeros(2, dtype=np.float32)
    assert updater.record(obs, action, logp=-0.5, reward=1.0) is None
    assert updater.buffered_steps == 1
    updater.discard()
    assert updater.buffered_steps == 0
    assert updater.record(obs, action, logp=-0.5, reward=1.0) is None
    assert updater.buffered_steps == 1
    assert not (tmp_path / "online_ppo" / "actor_latest.pt").exists()

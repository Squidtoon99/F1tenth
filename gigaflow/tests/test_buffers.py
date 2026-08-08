"""Focused compact rollout buffer tests."""

from __future__ import annotations

from pathlib import Path

import torch

from gigaflow_f1tenth.buffers import (
    DEFAULT_AGENT_STATE_DIM,
    allocate_rollout_buffer,
    buffer_shapes,
)
from gigaflow_f1tenth.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_buffer_shapes_include_state_dim():
    cfg = load_config(SMOKE)
    shapes = buffer_shapes(cfg)
    assert shapes.num_slots == 8
    assert shapes.rollout_length == cfg.ppo.rollout_length
    assert shapes.state_dim == DEFAULT_AGENT_STATE_DIM


def test_allocate_store_finalize_roundtrip():
    cfg = load_config(SMOKE)
    buf = allocate_rollout_buffer(cfg, "cpu", state_dim=8)
    shapes = buf.shapes()
    t = shapes.rollout_length
    s = shapes.num_slots
    torch.manual_seed(0)
    hidden = torch.randn(s, shapes.gru_hidden_dim)
    buf.set_rollout_start_hidden(hidden)
    buf.store_step(0, track_id=torch.arange(s, dtype=torch.int32))
    for step in range(t):
        buf.store_step(
            step,
            state=torch.randn(s, 8),
            actions=torch.randn(s, shapes.action_dim),
            rewards=torch.randn(s),
            valid=torch.ones(s, dtype=torch.bool),
            done=torch.zeros(s, dtype=torch.bool),
            timeout=torch.zeros(s, dtype=torch.bool),
            reset_mask=torch.zeros(s, dtype=torch.bool),
            condition=torch.randn(s, shapes.condition_dim),
            sensor_noise_seed=torch.arange(s, dtype=torch.int64) + step,
            episode_id=torch.zeros(s, dtype=torch.int32) + step,
            episode_step=torch.arange(s, dtype=torch.int32),
            old_logp=torch.randn(s),
            obs_digest=torch.arange(s, dtype=torch.int64),
            pre_tanh=torch.randn(s, shapes.action_dim),
        )
    batch = buf.finalize()
    assert batch.actions.shape == (t, s, shapes.action_dim)
    assert batch.old_logp.shape == (t, s)
    assert batch.condition.shape == (t, s, shapes.condition_dim)
    assert batch.obs_digest.shape == (t, s)
    assert batch.episode_id.shape == (t, s)
    assert batch.episode_step.shape == (t, s)
    assert batch.pre_tanh.shape == (t, s, shapes.action_dim)
    assert batch.gru_start.shape == (s, shapes.gru_hidden_dim)
    assert torch.equal(batch.gru_start, hidden)
    assert batch.state.shape[-1] == 8
    # Observations are replayed from compact state, never stored per rollout.
    assert not hasattr(batch, "sensor_obs")


def test_finalize_requires_full_rollout():
    cfg = load_config(SMOKE)
    buf = allocate_rollout_buffer(cfg, "cpu")
    try:
        buf.finalize()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "incomplete" in str(exc)


def test_reset_clears_filled_flags():
    cfg = load_config(SMOKE)
    buf = allocate_rollout_buffer(cfg, "cpu", state_dim=4)
    s = buf.shapes().num_slots
    buf.store_step(
        0,
        state=torch.ones(s, 4),
        actions=torch.zeros(s, 2),
        rewards=torch.zeros(s),
        valid=torch.ones(s, dtype=torch.bool),
        done=torch.zeros(s, dtype=torch.bool),
        timeout=torch.zeros(s, dtype=torch.bool),
        reset_mask=torch.zeros(s, dtype=torch.bool),
        sensor_noise_seed=torch.zeros(s, dtype=torch.int64),
        old_logp=torch.zeros(s),
    )
    buf.reset()
    try:
        buf.finalize()
        assert False, "expected RuntimeError after reset"
    except RuntimeError:
        pass

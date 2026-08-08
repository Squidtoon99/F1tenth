"""Focused tests for the conditioned CNN-GRU actor and lean ablations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    build_actor_artifact_payload,
    validate_actor_artifact,
)
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.model import (
    architecture_metadata,
    build_actor,
    actor_architecture_from_module,
)
from gigaflow_f1tenth.normalization import SensorNormalizer
from gigaflow_f1tenth.rewards import sample_private_styles, styles_to_condition_batch

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _batch(cfg, n=4, seed=0):
    rng = np.random.default_rng(seed)
    obs = torch.as_tensor(rng.random((n, 1097), dtype=np.float32))
    styles = sample_private_styles(cfg, n, rng)
    cond = torch.as_tensor(styles_to_condition_batch(styles))
    return obs, cond


def test_gru_actor_shapes_and_actions():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    obs, cond = _batch(cfg, 5)
    h0 = actor.initial_hidden(5, "cpu")
    out = actor.forward(obs, cond, h0, deterministic=True)
    assert out.actions.shape == (5, 2)
    assert out.log_prob.shape == (5,)
    assert out.entropy.shape == (5,)
    assert out.hidden.shape == (5, cfg.agents.gru_hidden_dim)
    assert torch.isfinite(out.actions).all()
    assert out.actions.abs().max() <= 1.0 + 1e-5
    meta = architecture_metadata(cfg)
    assert meta["sensor_obs_dim"] == 1097
    assert meta["action_dim"] == 2
    assert meta["mlp_sizes"] == list(cfg.agents.actor_mlp_sizes)


def test_gru_partial_reset_clears_only_marked_rows():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    obs, cond = _batch(cfg, 3, seed=2)
    h = actor.initial_hidden(3, "cpu")
    out1 = actor.forward(obs, cond, h, deterministic=True)
    reset = torch.tensor([False, True, False])
    out2 = actor.forward(obs, cond, out1.hidden, reset_mask=reset, deterministic=True)
    # Non-reset rows should differ from a fresh hidden; reset row matches fresh.
    fresh = actor.forward(
        obs, cond, actor.initial_hidden(3, "cpu"), deterministic=True
    )
    assert torch.allclose(out2.hidden[1], fresh.hidden[1], atol=1e-5)
    assert not torch.allclose(out2.hidden[0], fresh.hidden[0], atol=1e-4)


def test_partition_invariance_batch_split():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    actor.eval()
    obs, cond = _batch(cfg, 6, seed=3)
    h = actor.initial_hidden(6, "cpu")
    # Stochastic path is partition-invariant under the erfinv sampler; check
    # the deterministic path for exact batch-split equality.
    full_d = actor.forward(obs, cond, h, deterministic=True)
    left = actor.forward(obs[:3], cond[:3], h[:3], deterministic=True)
    right = actor.forward(obs[3:], cond[3:], h[3:], deterministic=True)
    assert torch.allclose(full_d.actions[:3], left.actions, atol=1e-5)
    assert torch.allclose(full_d.actions[3:], right.actions, atol=1e-5)
    stoch = actor.forward(obs, cond, h, deterministic=False)
    assert stoch.actions.shape == (6, 2)


def test_feedforward_and_frame_stack_ablations():
    cfg = load_config(SMOKE)
    ff = build_actor(cfg, variant="feedforward")
    obs, cond = _batch(cfg, 2, seed=4)
    out = ff.forward(obs, cond, ff.initial_hidden(2, "cpu"), deterministic=True)
    assert out.actions.shape == (2, 2)

    stack = build_actor(cfg, variant="frame_stack", frame_stack=4)
    stacked = obs.unsqueeze(1).repeat(1, 4, 1)
    out_s = stack.forward(
        stacked, cond, stack.initial_hidden(2, "cpu"), deterministic=True
    )
    assert out_s.actions.shape == (2, 2)
    arch = actor_architecture_from_module(stack)
    assert arch["frame_stack"] == 4
    assert arch["recurrent"] is False


def test_evaluate_actions_sequence_matches_step_logprob():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    obs, cond = _batch(cfg, 2, seed=11)
    h = actor.initial_hidden(2, "cpu")
    out = actor.forward(obs, cond, h, deterministic=True)
    logp, ent, h2 = actor.evaluate_actions_sequence(
        obs.unsqueeze(1),
        cond,
        h,
        out.actions.unsqueeze(1),
        reset_mask=torch.zeros(2, 1, dtype=torch.bool),
    )
    assert logp.shape == (2, 1)
    assert ent.shape == (2, 1)
    assert h2.shape == (2, cfg.agents.gru_hidden_dim)
    assert torch.allclose(logp[:, 0], out.log_prob, atol=1e-4)


def test_build_ppo_accepts_conditioned_actor():
    from gigaflow_f1tenth import critic, ppo

    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    value_critic = critic.build_critic(cfg)
    learner = ppo.build_ppo(cfg, actor, value_critic, device="cpu")
    assert learner.actor is not None


def test_artifact_payload_roundtrip(tmp_path):
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    payload = build_actor_artifact_payload(cfg, actor)
    validate_actor_artifact(payload)
    assert payload["format_version"] == ARTIFACT_FORMAT_VERSION
    assert "critic" not in payload
    assert payload["actor_architecture"]["sensor_obs_dim"] == 1097
    assert len(payload["condition_schema"]["field_names"]) == 10
    path = tmp_path / "actor.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_actor_artifact(loaded)
    actor2 = build_actor(cfg)
    actor2.load_state_dict(loaded["actor_state_dict"])
    obs, cond = _batch(cfg, 1, seed=9)
    a1 = actor.forward(obs, cond, actor.initial_hidden(1, "cpu"), deterministic=True)
    a2 = actor2.forward(obs, cond, actor2.initial_hidden(1, "cpu"), deterministic=True)
    assert torch.allclose(a1.actions, a2.actions, atol=1e-6)


def test_artifact_roundtrip_with_trained_normalizer_matches_inference(tmp_path):
    """D3a gate: non-empty normalizer stats survive export, and inference is
    identical before and after, since the actor applies normalization itself."""
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    rng = np.random.default_rng(5)
    for _ in range(4):
        raw = torch.as_tensor(
            rng.normal(loc=4.0, scale=6.0, size=(32, 1097)).astype(np.float32)
        )
        actor.sensor_normalizer.update(raw)
    assert float(actor.sensor_normalizer.count.item()) == 128.0

    obs, cond = _batch(cfg, 3, seed=13)
    before = actor.forward(
        obs, cond, actor.initial_hidden(3, "cpu"), deterministic=True
    )

    payload = build_actor_artifact_payload(cfg, actor)
    validate_actor_artifact(payload)
    normalizer = payload["sensor_normalizer"]
    assert normalizer["count"] == 128.0
    assert normalizer["mean"].shape == (1097,)

    path = tmp_path / "actor_normalized.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_actor_artifact(loaded)

    actor2 = build_actor(cfg)
    actor2.load_state_dict(loaded["actor_state_dict"])
    assert torch.allclose(actor2.sensor_normalizer.mean, actor.sensor_normalizer.mean)
    after = actor2.forward(
        obs, cond, actor2.initial_hidden(3, "cpu"), deterministic=True
    )
    assert torch.allclose(before.actions, after.actions, atol=1e-6)
    # Sanity: normalization actually changed behavior vs. an untrained normalizer.
    fresh = build_actor(cfg)
    fresh.load_state_dict(actor.state_dict())
    fresh.sensor_normalizer.load_dict(SensorNormalizer(1097).to_dict())
    unnormalized = fresh.forward(
        obs, cond, fresh.initial_hidden(3, "cpu"), deterministic=True
    )
    assert not torch.allclose(before.actions, unnormalized.actions, atol=1e-6)


def test_validate_actor_artifact_rejects_empty_sensor_normalizer():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    payload = build_actor_artifact_payload(cfg, actor, sensor_normalizer={})
    with pytest.raises(ValueError, match="sensor_normalizer"):
        validate_actor_artifact(payload)


def test_validate_actor_artifact_rejects_mismatched_normalizer_dim():
    cfg = load_config(SMOKE)
    actor = build_actor(cfg)
    bad = SensorNormalizer(dim=42).to_dict()
    payload = build_actor_artifact_payload(cfg, actor, sensor_normalizer=bad)
    with pytest.raises(ValueError, match="sensor_normalizer"):
        validate_actor_artifact(payload)

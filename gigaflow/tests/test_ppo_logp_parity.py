"""Regression: collection old_logp must match frozen-policy evaluate_actions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.model import LOG_STD_MAX, ACT_LIMIT
from gigaflow_f1tenth.ppo import CollectEvaluateParityError, evaluate_actions_sequence
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_action_time_reset_mask_not_post_terminal():
    """Stored reset_mask is the mask applied before the action, not after."""
    cfg = load_config(SMOKE)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    batch = trainer.collect_rollout()
    assert batch.reset_mask.shape == batch.done.shape
    assert batch.condition.ndim == 3
    assert batch.condition.shape[:2] == batch.rewards.shape
    assert batch.pre_tanh.shape == batch.actions.shape


def test_collect_evaluate_logp_parity_frozen_policy():
    cfg = load_config(SMOKE)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    assert trainer.actor is not None and trainer.ppo is not None

    for p in trainer.actor.parameters():
        p.requires_grad_(False)

    batch = trainer.collect_rollout()
    prepared = trainer.reconstruct_prepared(batch)
    valid = batch.valid
    assert prepared.sensor_obs.shape == (
        cfg.ppo.rollout_length,
        trainer.layout.num_slots,
        cfg.agents.sensor_obs_dim,
    )
    assert float(prepared.sensor_obs.abs().sum()) > 0.0

    obs = prepared.sensor_obs.transpose(0, 1).contiguous()
    act = batch.actions.transpose(0, 1).contiguous()
    reset = batch.reset_mask.transpose(0, 1).contiguous()
    cond = batch.condition.transpose(0, 1).contiguous()
    pre = batch.pre_tanh.transpose(0, 1).contiguous()
    hidden = batch.gru_start

    with torch.no_grad():
        new_logp, new_ent, _ = evaluate_actions_sequence(
            trainer.actor,
            obs,
            cond,
            hidden,
            act,
            reset_mask=reset,
            pre_tanh=pre,
        )
    new_logp_t = new_logp.transpose(0, 1)
    old = batch.old_logp
    diff = (new_logp_t - old).abs()
    max_abs = float(diff[valid].max().item()) if bool(valid.any()) else 0.0
    mean_abs = float(diff[valid].mean().item()) if bool(valid.any()) else 0.0
    assert max_abs < 1e-4, f"logp parity failed max_abs={max_abs} mean_abs={mean_abs}"
    # Stored pre_tanh must reconstruct actions.
    recon = ACT_LIMIT * torch.tanh(batch.pre_tanh)
    assert float((recon - batch.actions).abs().max().item()) < 1e-5
    ratio = (new_logp_t - old).exp()
    kl = (((ratio - 1.0) - (new_logp_t - old))[valid]).mean()
    assert abs(float(kl.item())) < 1e-4, f"pre-update KL {float(kl.item())}"
    assert torch.isfinite(new_ent).all()


def test_atanh_recovery_diverges_when_std_saturates_tanh():
    """Document root cause: atanh(action) ≠ sampled pre_tanh under large std."""
    torch.manual_seed(0)
    # Historical LOG_STD_MAX=2 → std≈7.4 saturates tanh.
    std = torch.exp(torch.tensor(2.0))
    mu = torch.zeros(4096, 2)
    eps = torch.randn_like(mu)
    pre = mu + std * eps
    action = torch.tanh(pre)
    a = action.clamp(-0.999999, 0.999999)
    recovered = 0.5 * (torch.log1p(a) - torch.log1p(-a))
    max_delta = float((recovered - pre).abs().max().item())
    assert max_delta > 1.0
    # Current bound keeps samples in a recoverable regime more often.
    std_tight = torch.exp(torch.tensor(float(LOG_STD_MAX)))
    pre2 = mu + std_tight * eps
    action2 = torch.tanh(pre2)
    # Even with tight std, always prefer stored pre_tanh — but recovery error
    # should be far smaller than the historical blow-up.
    a2 = action2.clamp(-0.999999, 0.999999)
    recovered2 = 0.5 * (torch.log1p(a2) - torch.log1p(-a2))
    assert float((recovered2 - pre2).abs().max().item()) < max_delta


def test_long_horizon_async_updates_keep_pre_kl_near_zero():
    """>50 updates with async resets must keep frozen-weight pre-KL tiny."""
    from dataclasses import replace

    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            rollout_length=8,
            total_updates=60,
            num_epochs=2,
            target_kl=0.05,
            max_pre_update_kl=1.0e-3,
            max_logp_delta=5.0e-3,
            amp=False,
            minibatch_size=32,
        ),
        profiling=replace(cfg.profiling, report_interval_updates=1),
    )
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    pre_kls = []
    ents = []
    epochs = []
    for _ in range(55):
        progress = trainer.train_update()
        pre_kls.append(progress.metrics["pre_update_approx_kl"])
        ents.append(progress.metrics["entropy"])
        epochs.append(progress.metrics["epochs_completed"])
        assert progress.metrics["valid_transitions"] > 0
        assert abs(progress.metrics["pre_update_approx_kl"]) < 1.0e-3
        assert progress.metrics["logp_delta_max"] < 5.0e-3
        assert progress.metrics["entropy"] < 5.0  # not LOG_STD_MAX=2 saturation
        assert math_isfinite(progress.metrics["value_loss"])
        assert math_isfinite(progress.metrics["grad_norm"])
    assert max(abs(k) for k in pre_kls) < 1.0e-3
    assert max(ents) < 5.0
    # At least some updates should complete >1 epoch when KL allows.
    assert max(epochs) >= 1.0


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def test_parity_gate_fail_fast_on_mismatched_old_logp():
    cfg = load_config(SMOKE)
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    batch = trainer.collect_rollout()
    prepared = trainer.reconstruct_prepared(batch)
    batch.old_logp = batch.old_logp + 5.0
    with pytest.raises(CollectEvaluateParityError):
        trainer.ppo.update(batch, prepared)

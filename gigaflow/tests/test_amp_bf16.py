"""BF16 autocast: frozen logp parity and finite PPO gradients (CUDA)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.ppo import AMP_DTYPE, evaluate_actions_sequence
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_dtype_is_bfloat16_not_float16():
    assert AMP_DTYPE == torch.bfloat16
    assert AMP_DTYPE != torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_update_finite_grads_and_frozen_logp_parity():
    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        worlds=replace(cfg.worlds, num_worlds=2, max_agents_per_world=2, device="cuda"),
        ppo=replace(
            cfg.ppo,
            rollout_length=8,
            num_epochs=2,
            minibatch_size=32,
            amp=True,
            target_kl=0.0,
            total_updates=4,
        ),
        profiling=replace(cfg.profiling, estimate_bytes_budget=8_000_000_000),
    )
    trainer = build_trainer(cfg, device="cuda", run_dir=None)
    trainer.setup()
    assert trainer.ppo is not None and trainer.actor is not None
    assert trainer.ppo.amp is True
    assert trainer.ppo.amp_dtype == torch.bfloat16
    assert trainer.ppo.scaler.is_enabled() is False

    # Frozen-weight collect/evaluate parity stays FP32 on the actor path.
    for p in trainer.actor.parameters():
        p.requires_grad_(False)
    batch = trainer.collect_rollout()
    prepared = trainer.reconstruct_prepared(batch)
    obs = prepared.sensor_obs.transpose(0, 1).contiguous()
    act = batch.actions.transpose(0, 1).contiguous()
    reset = batch.reset_mask.transpose(0, 1).contiguous()
    cond = batch.condition.transpose(0, 1).contiguous()
    pre = batch.pre_tanh.transpose(0, 1).contiguous()
    with torch.no_grad():
        new_logp, _, _ = evaluate_actions_sequence(
            trainer.actor,
            obs,
            cond,
            batch.gru_start,
            act,
            reset_mask=reset,
            pre_tanh=pre,
        )
    new_logp_t = new_logp.transpose(0, 1).float()
    valid = batch.valid
    diff = (new_logp_t - batch.old_logp).abs()
    max_abs = float(diff[valid].max().item()) if bool(valid.any()) else 0.0
    assert max_abs < 1e-4, f"FP32 actor logp parity max_abs={max_abs}"

    for p in trainer.actor.parameters():
        p.requires_grad_(True)
    progress = trainer.train_update()
    assert abs(progress.metrics["pre_update_approx_kl"]) < 1.0e-3
    assert progress.metrics["logp_delta_max"] < 5.0e-3
    assert progress.metrics["grad_norm"] == progress.metrics["grad_norm"]
    assert abs(progress.metrics["grad_norm"]) != float("inf")
    assert progress.metrics["policy_loss"] == progress.metrics["policy_loss"]
    assert progress.metrics["value_loss"] == progress.metrics["value_loss"]
    assert progress.metrics["epochs_completed"] >= 1.0

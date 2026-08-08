"""Regression: training must async-respawn and keep nonzero valid transitions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def test_training_sim_defaults_to_async_respawn():
    cfg = load_config(SMOKE)
    assert cfg.evaluation.sync_no_respawn is True
    assert cfg.agents.async_respawn is True
    sim = build_simulator(cfg, make_synthetic_oval_atlas(max_agents=2), "cpu")
    assert sim.sync_no_respawn is False


def test_collect_keeps_population_across_updates():
    cfg = load_config(SMOKE)
    # Short but multi-update gate on CPU Warp.
    trainer = build_trainer(cfg, device="cpu", run_dir=None)
    trainer.setup()
    assert trainer.sim.sync_no_respawn is False

    reports = []
    for _ in range(3):
        batch = trainer.collect_rollout()
        n_valid = int(batch.valid.sum().item())
        active_end = int((trainer.sim.buffers.torch_arrays.active > 0).sum().item())
        prepared = trainer.reconstruct_prepared(batch)
        stats = trainer.ppo.update(batch, prepared)
        reports.append(
            {
                "valid": n_valid,
                "active_end": active_end,
                "epochs": int(stats.epochs_completed),
                "policy_loss": float(stats.policy_loss),
                "value_loss": float(stats.value_loss),
                "entropy": float(stats.entropy),
            }
        )

    assert all(r["valid"] > 0 for r in reports), reports
    assert all(r["active_end"] > 0 for r in reports), reports
    assert all(r["epochs"] > 0 for r in reports), reports
    # Learning signal present (losses need not be nonzero every step under early KL stop,
    # but entropy should stay finite/nonzero with a live population).
    assert all(r["entropy"] > 0.0 for r in reports), reports
    assert all(
        torch.isfinite(torch.tensor(r["policy_loss"]))
        and torch.isfinite(torch.tensor(r["value_loss"]))
        for r in reports
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Warp gate")
def test_cuda_reduced_scale_learning_signal():
    cfg_path = ROOT / "configs" / "gpu_smoke.yaml"
    if not cfg_path.exists():
        pytest.skip("gpu_smoke.yaml missing")
    cfg = load_config(cfg_path)
    trainer = build_trainer(cfg, device="cuda", run_dir=None)
    trainer.setup()
    total = 0
    for _ in range(2):
        progress = trainer.train_update()
        total = progress.transitions
        assert progress.metrics["valid_transitions"] > 0
        assert progress.metrics["active_end"] > 0
        assert progress.metrics["epochs_completed"] > 0
        assert progress.metrics["entropy"] > 0.0
        assert abs(progress.metrics["policy_loss"]) + abs(
            progress.metrics["value_loss"]
        ) > 0.0
        assert progress.metrics["transitions_per_s"] > 0.0
        assert progress.metrics["grad_norm"] > 0.0
        assert progress.metrics["grad_norm"] < float("inf")
        assert progress.metrics["filter_eta"] < float("inf")
    assert total > 50

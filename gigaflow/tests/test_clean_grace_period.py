"""Clean grace-period config: quality/KL signals must not stop training."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.evaluation import (
    BehaviorGateState,
    EvalMetrics,
    EvalReport,
)
from gigaflow_f1tenth.trainer import _consume_behavioral_eval
from gigaflow_f1tenth.wandb_log import WandbSession

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "configs" / "clean_grace_rtx4080.yaml"


def _catastrophic_reports() -> list[EvalReport]:
    common = {
        "clean_overtakes": 0.0,
        "stall_rate": 0.0,
        "return_mean": 0.0,
    }
    return [
        EvalReport(
            suite="solo",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=120.0,
                completion_rate=0.0,
                progress_rate_mps=1.5,
                collision_per_km=0.1,
                oob_per_km=35.0,
                **common,
            ),
        ),
        EvalReport(
            suite="head_to_head",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=120.0,
                completion_rate=0.0,
                progress_rate_mps=1.5,
                collision_per_km=0.1,
                oob_per_km=35.0,
                **common,
            ),
        ),
        EvalReport(
            suite="dense",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=120.0,
                completion_rate=0.0,
                progress_rate_mps=1.5,
                collision_per_km=0.1,
                oob_per_km=35.0,
                **common,
            ),
        ),
    ]


def test_clean_grace_config_disables_quality_stops():
    cfg = load_config(CLEAN)
    assert cfg.ppo.target_kl == 0.0
    assert cfg.ppo.actor_kl_stop_enabled is False
    assert cfg.evaluation.quality_stop_enabled is False
    assert cfg.wandb.resume == "never"
    assert cfg.ppo.learning_rate == pytest.approx(1.0e-4)
    assert cfg.ppo.minibatch_size == 2048
    assert cfg.worlds.num_worlds == 48


def test_behavioral_catastrophic_does_not_stop_when_quality_disabled(tmp_path):
    cfg = load_config(CLEAN)
    import json

    from gigaflow_f1tenth.async_cpu_eval import eval_output_dir

    state = BehaviorGateState()
    state.feasible = True
    reports = _catastrophic_reports()
    out_dir = eval_output_dir(tmp_path, 500)
    out_dir.mkdir(parents=True)
    payload = {
        "reports": [
            {
                "suite": report.suite,
                "seed": report.seed,
                "metrics": {
                    "lap_time_s": report.metrics.lap_time_s,
                    "completion_rate": report.metrics.completion_rate,
                    "progress_rate_mps": report.metrics.progress_rate_mps,
                    "collision_per_km": report.metrics.collision_per_km,
                    "oob_per_km": report.metrics.oob_per_km,
                    "clean_overtakes": report.metrics.clean_overtakes,
                    "stall_rate": report.metrics.stall_rate,
                    "return_mean": report.metrics.return_mean,
                },
                "extras": {},
            }
            for report in reports
        ]
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    stop = _consume_behavioral_eval(
        cfg,
        state,
        {"state": "completed", "step": 500},
        session=WandbSession(cfg.wandb, experiment=cfg, run_dir=tmp_path),
        run_dir=tmp_path,
        train_step=500,
    )
    assert stop is None
    assert state.consecutive_failures >= 1


def test_kl_rollback_signal_does_not_stop_when_quality_disabled():
    cfg = load_config(CLEAN)
    progress_metrics = {
        "rollback_stop_requested": 1.0,
        "early_stopped": 1.0,
        "actor_rollback_count": 3.0,
    }
    would_stop = (
        cfg.evaluation.quality_stop_enabled
        and bool(progress_metrics.get("rollback_stop_requested", 0.0))
    )
    assert not would_stop
    assert cfg.ppo.actor_kl_stop_enabled is False
    assert cfg.ppo.target_kl == 0.0


def test_clean_grace_filter_candidates_one_knob_each():
    ref = load_config(CLEAN)
    eta005 = load_config(ROOT / "configs" / "clean_grace_filter_eta005.yaml")
    off = load_config(ROOT / "configs" / "clean_grace_filter_off.yaml")
    assert ref.ppo.adaptive_filter_eta_scale == pytest.approx(0.01)
    assert eta005.ppo.adaptive_filter_eta_scale == pytest.approx(0.005)
    assert eta005.ppo.adaptive_filter_enabled is True
    assert off.ppo.adaptive_filter_enabled is False
    assert off.ablations.adaptive_filter_enabled is False
    for cfg in (ref, eta005, off):
        assert cfg.ppo.target_kl == 0.0
        assert cfg.ppo.actor_kl_stop_enabled is False
        assert cfg.evaluation.quality_stop_enabled is False


def test_lower_eta_scale_retains_more_transitions():
    import torch

    from gigaflow_f1tenth.ppo import AdaptiveFilterState, adaptive_advantage_keep_mask

    adv = torch.tensor([[0.05, 1.0, 3.0, 8.0]])
    valid = torch.ones_like(adv, dtype=torch.bool)
    state = AdaptiveFilterState(
        ewma_max_abs_adv=8.0, beta=0.25, eta_scale=0.01, initialized=True
    )
    keep_ref, _, eta_ref = adaptive_advantage_keep_mask(adv, valid, state)
    state_lo = AdaptiveFilterState(
        ewma_max_abs_adv=8.0, beta=0.25, eta_scale=0.005, initialized=True
    )
    keep_lo, _, eta_lo = adaptive_advantage_keep_mask(adv, valid, state_lo)
    assert eta_lo < eta_ref
    assert int(keep_lo.sum()) >= int(keep_ref.sum())
    keep_off, _, _ = adaptive_advantage_keep_mask(
        adv, valid, state, enabled=False
    )
    assert int(keep_off.sum()) == int(valid.sum())

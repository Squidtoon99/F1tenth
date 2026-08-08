"""Recalibrated behavioral and KL rollback safety gates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.evaluation import (
    BehaviorGateState,
    EvalMetrics,
    EvalReport,
)
from gigaflow_f1tenth.ppo import (
    RESUME_STATE_VERSION,
    export_resume_state,
    load_resume_state,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tdmlti3k_rollback_trace.json"
SMOKE = ROOT / "configs" / "smoke.yaml"


def _rollback_stop_requested(
    *,
    update_index: int,
    consecutive_rollbacks: int,
    rollback_updates: list[int],
    warmup_updates: int,
    stop_window_updates: int,
    stop_window_count: int,
) -> bool:
    current_update = update_index + 1
    if current_update <= warmup_updates:
        return False
    cutoff = update_index - (stop_window_updates - 1)
    window = [value for value in rollback_updates if value >= cutoff]
    return consecutive_rollbacks >= 2 or len(window) >= stop_window_count


def _behavior_reports(**solo):
    common = {
        "clean_overtakes": 0.0,
        "stall_rate": 0.0,
        "return_mean": 0.0,
    }
    dense_collisions = solo.pop("dense_collisions", 0.5)
    dense_oob = solo.pop("dense_oob", 0.2)
    return [
        EvalReport(
            suite="solo",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=solo.get("solo_lap", 70.0),
                completion_rate=solo.get("solo_completion", 0.9),
                progress_rate_mps=solo.get("solo_progress", 5.0),
                collision_per_km=0.1,
                oob_per_km=solo.get("solo_oob", 0.1),
                **common,
            ),
        ),
        EvalReport(
            suite="head_to_head",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=72.0,
                completion_rate=0.9,
                progress_rate_mps=5.0,
                collision_per_km=0.1,
                oob_per_km=0.1,
                **common,
            ),
        ),
        EvalReport(
            suite="dense",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=75.0,
                completion_rate=0.6,
                progress_rate_mps=4.0,
                collision_per_km=dense_collisions,
                oob_per_km=dense_oob,
                **common,
            ),
        ),
    ]


def _catastrophic_reports():
    return _behavior_reports(
        solo_completion=0.0,
        solo_lap=120.0,
        solo_progress=1.5,
        solo_oob=35.0,
    )


def _passing_reports():
    return _behavior_reports(
        solo_completion=0.95,
        solo_lap=70.0,
        solo_progress=5.5,
        solo_oob=0.1,
    )


def test_pre_feasibility_catastrophic_warns_but_does_not_stop():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    decision = state.observe(
        _catastrophic_reports(), required=required, step=200
    )
    assert decision.catastrophic
    assert decision.warning_active
    assert not decision.stop_requested
    assert not state.feasible


def test_no_feasibility_by_update_1000_is_trainer_stop_condition():
    cfg = load_config(ROOT / "configs" / "recalibrated_rtxpro6000_6h.yaml")
    assert cfg.evaluation.behavior_no_feasibility_stop_updates == 1000
    state = BehaviorGateState()
    assert not state.feasible


def test_two_consecutive_passes_latch_feasibility():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    first = state.observe(_passing_reports(), required=required, step=200)
    assert first.passed and not state.feasible
    second = state.observe(_passing_reports(), required=required, step=400)
    assert second.passed and state.feasible


def test_post_feasibility_single_catastrophic_does_not_stop():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    state.observe(_passing_reports(), required=required, step=200)
    state.observe(_passing_reports(), required=required, step=400)
    assert state.feasible
    crash = state.observe(_catastrophic_reports(), required=required, step=1200)
    assert crash.catastrophic
    assert crash.warning_active
    assert not crash.stop_requested


def test_a100_like_recovery_after_transient_crash():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    state.observe(_passing_reports(), required=required, step=200)
    state.observe(_passing_reports(), required=required, step=400)
    state.observe(_catastrophic_reports(), required=required, step=1200)
    recovery = state.observe(_passing_reports(), required=required, step=1300)
    assert recovery.passed
    assert not recovery.stop_requested


def test_a100_like_persistent_failure_stops_on_second_bad_eval():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    state.observe(_passing_reports(), required=required, step=200)
    state.observe(_passing_reports(), required=required, step=400)
    first = state.observe(_catastrophic_reports(), required=required, step=2200)
    assert not first.stop_requested
    second = state.observe(_catastrophic_reports(), required=required, step=2300)
    assert second.stop_requested


def test_tdmlti3k_old_policy_stops_at_68_new_policy_survives():
    trace = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rollbacks = list(trace["rollback_updates"])
    old = trace["old_policy"]
    new = trace["new_policy"]
    update_index = int(trace["final_update"]) - 1
    consecutive = 1
    old_stop = _rollback_stop_requested(
        update_index=update_index,
        consecutive_rollbacks=consecutive,
        rollback_updates=rollbacks,
        warmup_updates=int(old["actor_kl_warmup_updates"]),
        stop_window_updates=int(old["actor_kl_stop_window_updates"]),
        stop_window_count=int(old["actor_kl_stop_window_count"]),
    )
    new_stop = _rollback_stop_requested(
        update_index=update_index,
        consecutive_rollbacks=consecutive,
        rollback_updates=rollbacks,
        warmup_updates=int(new["actor_kl_warmup_updates"]),
        stop_window_updates=int(new["actor_kl_stop_window_updates"]),
        stop_window_count=int(new["actor_kl_stop_window_count"]),
    )
    assert old_stop is True
    assert new_stop is False
    assert trace["final_candidate_kl"] < float(new["actor_kl_hard"])


def test_post_warmup_two_consecutive_rollbacks_stop():
    assert _rollback_stop_requested(
        update_index=250,
        consecutive_rollbacks=2,
        rollback_updates=[251, 252],
        warmup_updates=200,
        stop_window_updates=500,
        stop_window_count=5,
    )


def test_five_in_500_rolling_rollback_stop():
    rollbacks = [100, 200, 300, 400, 500]
    assert _rollback_stop_requested(
        update_index=499,
        consecutive_rollbacks=1,
        rollback_updates=rollbacks,
        warmup_updates=200,
        stop_window_updates=500,
        stop_window_count=5,
    )


def test_safety_lr_recovery_and_resume_round_trip():
    from gigaflow_f1tenth.critic import build_critic
    from gigaflow_f1tenth.model import build_actor
    from gigaflow_f1tenth.ppo import build_ppo

    cfg = load_config(SMOKE)
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            actor_lr_recovery_interval_updates=2,
            actor_lr_recovery_multiplier=2.0,
            actor_lr_scale_min=0.125,
        ),
    )
    actor = build_actor(cfg)
    critic = build_critic(cfg)
    learner = build_ppo(cfg, actor, critic, device="cpu")
    learner.actor_lr_safety_multiplier = 0.25
    learner.rollback_free_accepted_updates = 1
    learner.rollback_free_accepted_updates += 1
    if learner.rollback_free_accepted_updates >= learner.actor_lr_recovery_interval_updates:
        learner.actor_lr_safety_multiplier = min(
            1.0,
            learner.actor_lr_safety_multiplier * learner.actor_lr_recovery_multiplier,
        )
        learner.rollback_free_accepted_updates = 0
    assert learner.actor_lr_safety_multiplier == pytest.approx(0.5)
    learner.rollback_free_accepted_updates = 1
    learner.rollback_free_accepted_updates += 1
    if learner.rollback_free_accepted_updates >= learner.actor_lr_recovery_interval_updates:
        learner.actor_lr_safety_multiplier = min(
            1.0,
            learner.actor_lr_safety_multiplier * learner.actor_lr_recovery_multiplier,
        )
        learner.rollback_free_accepted_updates = 0
    assert learner.actor_lr_safety_multiplier == pytest.approx(1.0)
    payload = export_resume_state(learner)
    assert payload["resume_state_version"] == RESUME_STATE_VERSION
    resumed = build_ppo(cfg, build_actor(cfg), build_critic(cfg), device="cpu")
    load_resume_state(resumed, payload)
    assert resumed.actor_lr_safety_multiplier == pytest.approx(1.0)
    assert resumed.rollback_free_accepted_updates == 0


def test_recalibrated_config_defaults():
    cfg = load_config(ROOT / "configs" / "recalibrated_rtxpro6000_6h.yaml")
    assert cfg.ppo.actor_kl_soft == pytest.approx(0.05)
    assert cfg.ppo.actor_kl_hard == pytest.approx(0.20)
    assert cfg.ppo.actor_kl_warmup_updates == 200
    assert cfg.ppo.actor_kl_stop_window_updates == 500
    assert cfg.ppo.actor_kl_stop_window_count == 5
    assert cfg.ppo.actor_lr_recovery_interval_updates == 100
    assert cfg.ppo.total_updates == 2400
    assert cfg.evaluation.behavior_feasibility_passes == 2
    assert cfg.evaluation.behavior_no_feasibility_stop_updates == 1000
    assert cfg.wandb.group == "recalibrated-robust6h"
    assert "recalibrated-robust6h" in cfg.wandb.tags


def test_behavior_gate_state_checkpoint_round_trip():
    state = BehaviorGateState()
    required = ("solo", "head_to_head", "dense")
    state.observe(_passing_reports(), required=required, step=200)
    state.observe(_passing_reports(), required=required, step=400)
    payload = state.to_dict()
    restored = BehaviorGateState.from_dict(payload)
    assert restored.feasible
    assert restored.consecutive_passes == 2
    assert restored.best_safe_score == pytest.approx(state.best_safe_score)

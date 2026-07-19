"""Deterministic tests for trainer step diagnostics and logging contracts."""

from __future__ import annotations

import math

import pytest
import torch

from standalone_trainer import (
    RunningStats,
    TRAINING_SUMMARY_REWARD_TAIL,
    accumulate_completed_episode_lifespans,
    accumulate_step_diagnostics,
    training_summary_reward_tail_args,
)


def _accumulate_once(diag: RunningStats, extras: dict, *, num_envs: int = 4) -> None:
    accumulate_step_diagnostics(
        diag,
        reward=torch.zeros(num_envs),
        actions=torch.zeros(num_envs, 2),
        obs=torch.zeros(num_envs, 10),
        extras=extras,
    )


def test_accumulate_step_diagnostics_wall_and_oob_events():
    diag = RunningStats()
    extras = {
        "rewards": {
            "terms": {
                "oob_penalty": torch.tensor([-0.1, 0.0, -0.2, 0.0]),
                "wall_penalty": torch.tensor([-0.01, -0.05, 0.0, -0.02]),
                "wall_impact": torch.tensor([0.0, -25.0, 0.0, 0.0]),
            }
        },
        "metrics": {
            "oob_mask": torch.tensor([1.0, 0.0, 1.0, 0.0]),
            "wall_contact": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        },
        "termination": {},
    }
    _accumulate_once(diag, extras)

    assert diag.total("metric/wall_contact_count") == 2.0
    assert diag.total("metric/wall_impact_events") == 1.0
    assert diag.mean("metric/wall_contact") == pytest.approx(0.5)
    assert diag.mean("reward_term/oob_penalty_when_oob") == pytest.approx(-0.15)
    assert diag.mean("reward_term/wall_penalty_when_contact") == pytest.approx(-0.035)
    assert diag.mean("reward_term/wall_impact_when_event") == pytest.approx(-25.0)


def test_accumulate_step_diagnostics_zero_events_yields_zero_counts_and_nan_means():
    diag = RunningStats()
    extras = {
        "rewards": {
            "terms": {
                "oob_penalty": torch.zeros(2),
                "wall_penalty": torch.zeros(2),
                "wall_impact": torch.zeros(2),
            }
        },
        "metrics": {
            "oob_mask": torch.zeros(2),
            "wall_contact": torch.zeros(2),
        },
        "termination": {},
    }
    _accumulate_once(diag, extras, num_envs=2)

    assert diag.total("metric/wall_contact_count") == 0.0
    assert diag.total("metric/wall_impact_events") == 0.0
    assert diag.mean("metric/wall_contact") == pytest.approx(0.0)
    assert math.isnan(diag.mean("reward_term/oob_penalty_when_oob"))
    assert math.isnan(diag.mean("reward_term/wall_penalty_when_contact"))
    assert math.isnan(diag.mean("reward_term/wall_impact_when_event"))


def test_completed_episode_lifespan_accumulates_exact_reset_safe_seconds():
    stats = RunningStats()
    accumulate_completed_episode_lifespans(
        stats,
        completed_episode_steps=torch.tensor([20, 999, 50], dtype=torch.int32),
        done=torch.tensor([True, False, True]),
        control_dt=0.05,
    )
    accumulate_completed_episode_lifespans(
        stats,
        completed_episode_steps=torch.tensor([777, 10, 888], dtype=torch.int32),
        done=torch.tensor([False, True, False]),
        control_dt=0.05,
    )
    assert stats.mean("episode/lifespan_s") == pytest.approx(4.0 / 3.0)


def test_training_summary_log_format_contract():
    diag = RunningStats()
    args = training_summary_reward_tail_args(
        mean_policy_loss=0.1,
        mean_critic_loss=0.2,
        mean_ep_reward=0.3,
        diag=diag,
        ep_count=0,
    )
    rendered = TRAINING_SUMMARY_REWARD_TAIL % args
    assert "episode_lifespan=nan" in rendered
    assert "(n=0)" in rendered

    accumulate_completed_episode_lifespans(
        diag,
        completed_episode_steps=torch.tensor([10], dtype=torch.int32),
        done=torch.tensor([True]),
        control_dt=0.05,
    )
    args_with_life = training_summary_reward_tail_args(
        mean_policy_loss=0.1,
        mean_critic_loss=0.2,
        mean_ep_reward=1.5,
        diag=diag,
        ep_count=1,
    )
    rendered_with_life = TRAINING_SUMMARY_REWARD_TAIL % args_with_life
    assert "episode_lifespan=0.500s" in rendered_with_life
    assert "(n=1)" in rendered_with_life

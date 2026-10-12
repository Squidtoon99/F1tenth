"""Deterministic tests for trainer step diagnostics and logging contracts."""

from __future__ import annotations

import math

import pytest
import torch

from standalone_trainer import (
    RunningStats,
    STEP_DIAGNOSTIC_STRIDE,
    TRAINING_SUMMARY_REWARD_TAIL,
    accumulate_completed_episode_lifespans,
    accumulate_step_diagnostics,
    accumulate_tick_diagnostics,
    training_summary_reward_tail_args,
    update_episode_lap_counts,
)


def test_step_diagnostic_stride_samples_every_eighth_tick():
    assert STEP_DIAGNOSTIC_STRIDE == 8


def _seeded_eval_tick_streams(seed: int = 0):
    """Build a fixed-seed tick stream with episode ends off the stride-8 grid."""
    rng = torch.Generator().manual_seed(seed)
    ticks, num_envs = 40, 6
    names = ["flying", "standing"]
    counts = [3, 3]
    reward = torch.randn(ticks, num_envs, generator=rng)
    laps_completed = (torch.rand(ticks, num_envs, generator=rng) < 0.25).to(
        torch.float32
    )
    done = torch.zeros(ticks, num_envs, dtype=torch.bool)
    # 1-indexed ticks match the trainer (vector_ticks += 1 before fold).
    off_stride = (1, 3, 7, 9, 14, 15, 17, 23, 25, 31, 33, 39)
    on_stride = (8, 16, 24, 32)
    for i, tick in enumerate(off_stride):
        done[tick - 1, i % num_envs] = True
    for i, tick in enumerate(on_stride):
        done[tick - 1, (i + 2) % num_envs] = True
    crash_on_done = (torch.rand(ticks, num_envs, generator=rng) < 0.4) & done
    oob = crash_on_done.clone()
    oob[ticks // 2 :] = False
    collision = crash_on_done & ~oob
    lap_count = torch.zeros(ticks, num_envs, dtype=torch.int32)
    live_laps = torch.zeros(num_envs, dtype=torch.int32)
    for t in range(ticks):
        live_laps = live_laps + laps_completed[t].to(torch.int32)
        lap_count[t] = torch.where(done[t], torch.zeros_like(live_laps), live_laps)
        live_laps = torch.where(done[t], torch.zeros_like(live_laps), live_laps)
    return {
        "names": names,
        "counts": counts,
        "reward": reward,
        "laps_completed": laps_completed,
        "done": done,
        "oob": oob,
        "collision": collision,
        "lap_count": lap_count,
    }


def _fold_eval_ticks(streams: dict, stride: int) -> RunningStats:
    diag = RunningStats()
    num_envs = streams["reward"].shape[1]
    episode_rewards = torch.zeros(num_envs)
    episode_laps = torch.zeros(num_envs)
    zeros_act = torch.zeros(num_envs, 2)
    zeros_obs = torch.zeros(num_envs, 4)
    for t in range(streams["reward"].shape[0]):
        extras = {
            "metrics": {
                "lap_count": streams["lap_count"][t],
                "laps_completed": streams["laps_completed"][t],
            },
            "termination": {
                "out_of_bounds": streams["oob"][t],
                "not_moving": torch.zeros(num_envs, dtype=torch.bool),
                "invalid_state": torch.zeros(num_envs, dtype=torch.bool),
                "collision": streams["collision"][t],
            },
            "rewards": {"terms": {}},
        }
        episode_rewards = episode_rewards + streams["reward"][t]
        episode_laps = update_episode_lap_counts(
            episode_laps, streams["lap_count"][t], streams["done"][t]
        )
        accumulate_tick_diagnostics(
            diag,
            vector_ticks=t + 1,
            stride=stride,
            scenario_names=streams["names"],
            scenario_counts=streams["counts"],
            episode_rewards=episode_rewards,
            episode_laps=episode_laps,
            done=streams["done"][t],
            extras=extras,
            reward=streams["reward"][t],
            actions=zeros_act,
            obs=zeros_obs,
        )
        not_done = 1.0 - streams["done"][t].to(episode_rewards.dtype)
        episode_rewards = episode_rewards * not_done
        episode_laps = episode_laps * not_done
    return diag


def test_episode_eval_metrics_match_between_stride_1_and_8():
    streams = _seeded_eval_tick_streams(seed=0)
    tick_ids = torch.arange(1, streams["done"].shape[0] + 1)
    assert int(streams["done"][tick_ids % 8 != 0].sum()) > 0
    expected_episodes = int(streams["done"].sum())
    expected_laps = float(streams["laps_completed"].sum())

    stride_1 = _fold_eval_ticks(streams, stride=1)
    stride_8 = _fold_eval_ticks(streams, stride=8)

    for name in streams["names"]:
        prefix = f"scn/{name}"
        assert stride_1.total(f"{prefix}/episodes") == stride_8.total(
            f"{prefix}/episodes"
        )
        assert stride_1.mean(f"{prefix}/episode_return") == pytest.approx(
            stride_8.mean(f"{prefix}/episode_return")
        )
        assert stride_1.mean(f"{prefix}/lap_completion") == pytest.approx(
            stride_8.mean(f"{prefix}/lap_completion")
        )
        assert stride_1.mean(f"{prefix}/crash") == pytest.approx(
            stride_8.mean(f"{prefix}/crash")
        )
    assert stride_1.total("scn/flying/episodes") + stride_1.total(
        "scn/standing/episodes"
    ) == expected_episodes
    assert stride_8.total("metric/laps_completed") == pytest.approx(expected_laps)
    assert stride_1.total("metric/laps_completed") == stride_8.total(
        "metric/laps_completed"
    )


def _accumulate_once(diag: RunningStats, extras: dict, *, num_envs: int = 4) -> None:
    accumulate_step_diagnostics(
        diag,
        reward=torch.zeros(num_envs),
        actions=torch.zeros(num_envs, 2),
        obs=torch.zeros(num_envs, 10),
        extras=extras,
    )


def test_accumulate_step_diagnostics_wall_contact_events():
    diag = RunningStats()
    extras = {
        "rewards": {
            "terms": {
                "wall_contact": torch.tensor([0.0, -4.0, 0.0, 0.0]),
            }
        },
        "metrics": {
            "wall_contact": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        },
        "termination": {},
    }
    _accumulate_once(diag, extras)

    assert diag.total("metric/wall_contact_events") == 2.0
    assert diag.mean("reward_term/wall_contact_when_event") == pytest.approx(-2.0)


def test_accumulate_step_diagnostics_boundary_contact_and_oob_impact_events():
    diag = RunningStats()
    extras = {
        "rewards": {
            "terms": {
                "boundary_contact": torch.tensor([0.0, -4.0, 0.0, -4.0]),
                "oob_impact": torch.tensor([0.0, 0.0, -1.62, 0.0]),
            }
        },
        "metrics": {},
        "termination": {},
    }
    _accumulate_once(diag, extras)

    assert diag.total("metric/boundary_contact_events") == 2.0
    assert diag.mean("reward_term/boundary_contact_when_event") == pytest.approx(-4.0)
    assert diag.total("metric/oob_impact_events") == 1.0
    assert diag.mean("reward_term/oob_impact_when_event") == pytest.approx(-1.62)


def test_accumulate_step_diagnostics_zero_events_yields_zero_counts_and_nan_means():
    diag = RunningStats()
    extras = {
        "rewards": {
            "terms": {
                "wall_contact": torch.zeros(2),
            }
        },
        "metrics": {
            "wall_contact": torch.zeros(2),
        },
        "termination": {},
    }
    _accumulate_once(diag, extras, num_envs=2)

    assert diag.total("metric/wall_contact_events") == 0.0
    assert math.isnan(diag.mean("reward_term/wall_contact_when_event"))


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

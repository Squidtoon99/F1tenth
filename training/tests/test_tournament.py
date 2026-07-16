"""Focused tests for tournament bracket/seeding/race helper logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TRAINING_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = TRAINING_DIR / "analysis"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from tournament import (  # noqa: E402
    SEED_METRIC_MEAN,
    SEED_METRIC_MIN,
    bracket_config_fingerprint,
    decide_match_winner,
    decide_winner,
    hard_collision_fault,
    infer_seed_metric,
    load_seed_rows,
    match_id,
    opponent_race_obs_dim,
    pair_avoid_rematch,
    pin_actions,
    seed_candidates_from_rows,
    seed_sort_key,
    select_bye,
    validate_resume_config,
)


def test_opponent_race_obs_dim_expands_base_config():
    cfg = {"num_obs": 384, "enable_opponent_obs": False, "opponent_obs_dim": 6}
    assert opponent_race_obs_dim(cfg) == 390


def test_opponent_race_obs_dim_keeps_expanded_config():
    cfg = {"num_obs": 390, "enable_opponent_obs": True, "opponent_obs_dim": 6}
    assert opponent_race_obs_dim(cfg) == 390


def test_seed_sort_key_prefers_min_then_crashes():
    fast = {"min_s": 60.0, "crashes": 5, "mean_s": 62.0, "transitions": 100}
    faster = {"min": 58.0, "crashes": 10, "mean": 59.0, "transitions": 200}
    assert seed_sort_key(faster, SEED_METRIC_MIN) < seed_sort_key(fast, SEED_METRIC_MIN)


def test_seed_sort_key_mean_metric_prefers_mean():
    better_mean = {"min": 51.0, "mean": 52.0, "crashes": 0, "transitions": 100}
    worse_mean = {"min": 50.0, "mean": 53.0, "crashes": 0, "transitions": 200}
    assert seed_sort_key(better_mean, SEED_METRIC_MEAN) < seed_sort_key(
        worse_mean, SEED_METRIC_MEAN
    )


def test_infer_seed_metric_legacy_lap_bench():
    rows = [{"checkpoint": "a.pt", "min_s": 60.0, "mean_s": 62.0}]
    assert infer_seed_metric(rows) == SEED_METRIC_MIN


def test_infer_seed_metric_lap_timing_selector():
    rows = [{"checkpoint": "a.pt", "reference_mode": "min-crash", "mean": 52.0}]
    assert infer_seed_metric(rows) == SEED_METRIC_MEAN


def test_infer_seed_metric_explicit_column():
    rows = [{"checkpoint": "a.pt", "seed_metric": "mean", "mean": 52.0}]
    assert infer_seed_metric(rows) == SEED_METRIC_MEAN


def test_seed_candidates_preserve_explicit_seed_rank(tmp_path):
    rows = [
        {
            "seed_rank": 1, "seed_metric": "mean",
            "checkpoint": "policy_1766400000.pt", "n_laps": 103,
            "min": 52.0, "mean": 52.185, "crashes": 5,
            "reference_mode": "min-crash",
        },
        {
            "seed_rank": 2, "seed_metric": "mean",
            "checkpoint": "policy_1607680000.pt", "n_laps": 96,
            "min": 51.95, "mean": 52.190, "crashes": 9,
            "reference_mode": "min-crash",
        },
        {
            "seed_rank": 3, "seed_metric": "mean",
            "checkpoint": "policy_1612800000.pt", "n_laps": 103,
            "min": 52.0, "mean": 52.195, "crashes": 5,
            "reference_mode": "min-crash",
        },
        {
            "seed_rank": 4, "seed_metric": "mean",
            "checkpoint": "policy_1751040000.pt", "n_laps": 103,
            "min": 52.0, "mean": 52.226, "crashes": 5,
            "reference_mode": "min-crash",
        },
    ]
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for row in rows:
        (ckpt_dir / row["checkpoint"]).write_bytes(b"x")
    got = seed_candidates_from_rows(rows, 4, ckpt_dir)
    assert got == [
        "policy_1766400000.pt",
        "policy_1607680000.pt",
        "policy_1612800000.pt",
        "policy_1751040000.pt",
    ]


def test_seed_candidates_mean_metric_without_explicit_rank(tmp_path):
    rows = [
        {"checkpoint": "slow_min.pt", "n_laps": 25, "min": 50.0, "mean": 53.0},
        {"checkpoint": "fast_mean.pt", "n_laps": 25, "min": 52.0, "mean": 51.0},
        {"checkpoint": "mid.pt", "n_laps": 25, "min": 51.0, "mean": 52.0},
    ]
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for row in rows:
        (ckpt_dir / row["checkpoint"]).write_bytes(b"x")
    got = seed_candidates_from_rows(rows, 2, ckpt_dir)
    assert got == ["fast_mean.pt", "mid.pt"]


def test_seed_candidates_from_rows_validates_missing(tmp_path):
    rows = [
        {"checkpoint": "policy_a.pt", "n_laps": 3, "min": 70.0, "mean": 71.0},
        {"checkpoint": "policy_b.pt", "n_laps": 2, "min_s": 68.0, "mean_s": 69.0},
    ]
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "policy_a.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="policy_b.pt"):
        seed_candidates_from_rows(rows, 2, ckpt_dir)


def test_seed_candidates_from_rows_orders_and_limits(tmp_path):
    rows = [
        {"checkpoint": "slow.pt", "n_laps": 2, "min_s": 80.0, "mean_s": 82.0,
         "crashes": 0},
        {"checkpoint": "fast.pt", "n_laps": 5, "min_s": 60.0, "mean_s": 62.0,
         "crashes": 1},
        {"checkpoint": "mid.pt", "n_laps": 4, "min_s": 65.0, "mean_s": 67.0,
         "crashes": 0},
    ]
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ("slow.pt", "fast.pt", "mid.pt"):
        (ckpt_dir / name).write_bytes(b"x")
    got = seed_candidates_from_rows(rows, 2, ckpt_dir)
    assert got == ["fast.pt", "mid.pt"]


def test_decide_winner_no_car_a_bias_on_progress_tie():
    winner, reason = decide_winner(
        laps_a=3, laps_b=3, prog_a=2.1, prog_b=2.5,
        crashes_a=0, crashes_b=0, collisions_a=0, collisions_b=0,
        target_laps=10, capped=True, seed_rank_a=0, seed_rank_b=1,
    )
    assert winner == "b"
    assert reason == "tie_break_progress"


def test_decide_winner_uses_seed_rank_after_penalties():
    winner, reason = decide_winner(
        laps_a=4, laps_b=4, prog_a=1.0, prog_b=1.0,
        crashes_a=2, crashes_b=2, collisions_a=1, collisions_b=1,
        target_laps=10, capped=True, seed_rank_a=3, seed_rank_b=1,
    )
    assert winner == "b"
    assert reason == "tie_break_seed"


def _leg(
    *,
    winner: str,
    laps_a: int = 2,
    laps_b: int = 1,
    progress_a: float = 2.0,
    progress_b: float = 1.5,
    crashes_a: int = 0,
    crashes_b: int = 0,
    collisions_a: int = 0,
    collisions_b: int = 0,
    leg: int = 1,
    sim_side_positive_a: bool = True,
) -> dict:
    return {
        "leg": leg,
        "sim_side_positive_a": sim_side_positive_a,
        "winner": winner,
        "finish_reason": "laps",
        "laps_a": laps_a,
        "laps_b": laps_b,
        "progress_a": progress_a,
        "progress_b": progress_b,
        "crashes_a": crashes_a,
        "crashes_b": crashes_b,
        "collisions_a": collisions_a,
        "collisions_b": collisions_b,
        "steps": 100,
        "race_time_s": 10.0,
        "capped": False,
    }


def test_decide_match_winner_more_leg_wins():
    leg1 = _leg(winner="a", leg=1, sim_side_positive_a=True)
    leg2 = _leg(winner="a", laps_a=2, laps_b=1, progress_a=2.0, progress_b=1.5,
                leg=2, sim_side_positive_a=False)
    winner, reason = decide_match_winner(leg1, leg2, seed_rank_a=0, seed_rank_b=1)
    assert winner == "a"
    assert reason == "legs_won"


def test_decide_match_winner_split_breaks_on_aggregate_laps():
    leg1 = _leg(winner="a", laps_a=3, laps_b=1, progress_a=3.0, progress_b=1.0,
                leg=1, sim_side_positive_a=True)
    leg2 = _leg(winner="b", laps_a=1, laps_b=2, progress_a=1.0, progress_b=2.5,
                leg=2, sim_side_positive_a=False)
    winner, reason = decide_match_winner(leg1, leg2, seed_rank_a=0, seed_rank_b=1)
    assert winner == "a"
    assert reason == "tie_break_agg_laps"


def test_decide_match_winner_seat_invariant_under_label_swap():
    leg1 = _leg(winner="a", laps_a=2, laps_b=1, progress_a=2.1, progress_b=1.8,
                leg=1, sim_side_positive_a=True)
    leg2 = _leg(winner="a", laps_a=2, laps_b=1, progress_a=2.0, progress_b=1.7,
                leg=2, sim_side_positive_a=False)
    winner_ab, _ = decide_match_winner(leg1, leg2, seed_rank_a=0, seed_rank_b=1)

    swapped1 = {
        **leg1,
        "winner": "b" if leg1["winner"] == "a" else "a",
        "laps_a": leg1["laps_b"],
        "laps_b": leg1["laps_a"],
        "progress_a": leg1["progress_b"],
        "progress_b": leg1["progress_a"],
        "crashes_a": leg1["crashes_b"],
        "crashes_b": leg1["crashes_a"],
        "collisions_a": leg1["collisions_b"],
        "collisions_b": leg1["collisions_a"],
        "sim_side_positive_a": not leg1["sim_side_positive_a"],
    }
    swapped2 = {
        **leg2,
        "winner": "b" if leg2["winner"] == "a" else "a",
        "laps_a": leg2["laps_b"],
        "laps_b": leg2["laps_a"],
        "progress_a": leg2["progress_b"],
        "progress_b": leg2["progress_a"],
        "crashes_a": leg2["crashes_b"],
        "crashes_b": leg2["crashes_a"],
        "collisions_a": leg2["collisions_b"],
        "collisions_b": leg2["collisions_a"],
        "sim_side_positive_a": not leg2["sim_side_positive_a"],
    }
    winner_ba, _ = decide_match_winner(
        swapped1, swapped2, seed_rank_a=1, seed_rank_b=0,
    )
    assert winner_ab == "a"
    assert winner_ba == "b"


def test_match_id_includes_rematch_suffix():
    assert match_id(2, 1, "a.pt", "b.pt") == "r002_m001__a__vs__b"
    assert match_id(2, 1, "a.pt", "b.pt", rematch=2) == (
        "r002_m001__a__vs__b__rematch2"
    )


def test_pair_cross_tier_when_one_player_per_loss_bucket():
    players = ["a", "b"]
    losses = {"a": 0, "b": 2}
    seed_rank = {"a": 0, "b": 1}
    pairs = pair_avoid_rematch(players, losses, seed_rank, {"a": None, "b": None})
    assert pairs == [("a", "b")]


def test_pair_avoid_rematch_swaps_to_skip_immediate_rematch():
    players = ["a", "b", "c", "d"]
    losses = {m: 0 for m in players}
    seed_rank = {m: i for i, m in enumerate(players)}
    last = {"a": "b", "b": "a", "c": None, "d": None}
    pairs = pair_avoid_rematch(players, losses, seed_rank, last)
    assert ("a", "b") not in pairs
    assert len(pairs) == 2


def test_select_bye_prefers_fewest_prior_byes():
    players = ["top", "mid", "low"]
    losses = {m: 0 for m in players}
    seed_rank = {"top": 0, "mid": 1, "low": 2}
    bye_counts = {"top": 2, "mid": 0, "low": 1}
    assert select_bye(players, losses, seed_rank, bye_counts) == "mid"


def test_pin_actions_zeros_held_rows():
    actions = torch.ones(2, 2)
    hold = torch.tensor([2, 0], dtype=torch.int32)
    pinned = pin_actions(actions, hold)
    assert torch.all(pinned[0] == 0)
    assert torch.all(pinned[1] == 1)


def test_hard_collision_fault_attributes_trailing_active_car():
    hard = torch.tensor([True])
    sim_prog = torch.tensor([1.0])
    opp_prog = torch.tensor([2.0])
    sim_active = torch.tensor([True])
    opp_active = torch.tensor([True])
    sim_fault, opp_fault = hard_collision_fault(
        hard, sim_prog, opp_prog, sim_active, opp_active,
    )
    assert bool(sim_fault[0])
    assert not bool(opp_fault[0])


def test_hard_collision_fault_blames_moving_car_when_other_frozen():
    hard = torch.tensor([True])
    sim_prog = torch.tensor([1.0])
    opp_prog = torch.tensor([2.0])
    sim_active = torch.tensor([True])
    opp_active = torch.tensor([False])
    sim_fault, opp_fault = hard_collision_fault(
        hard, sim_prog, opp_prog, sim_active, opp_active,
    )
    assert bool(sim_fault[0])
    assert not bool(opp_fault[0])


def test_validate_resume_config_rejects_seed_mismatch():
    saved = bracket_config_fingerprint(
        track="Austin", laps=10, freeze_s=2.0, seed=0, candidates=4,
        elim_losses=3, seeds=["a.pt"], seed_csv=None, checkpoints_dir="/ckpt",
    )
    current = dict(saved)
    current["seeds"] = ["b.pt"]
    with pytest.raises(ValueError, match="seed list"):
        validate_resume_config(saved, current)


def test_load_seed_rows_reads_lap_bench_csv():
    csv_path = TRAINING_DIR / "outputs" / "lap_bench" / "lap_times.csv"
    if not csv_path.exists():
        pytest.skip("lap_bench CSV not available")
    rows = load_seed_rows(csv_path)
    assert rows
    assert "checkpoint" in rows[0]
    assert "min_s" in rows[0] or "min" in rows[0]

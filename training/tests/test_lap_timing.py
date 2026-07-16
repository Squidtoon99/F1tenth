"""Focused tests for lap-timing crash classification and tournament seeding."""

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

from lap_timing import (  # noqa: E402
    BENCHMARK_VERSION,
    LAP_TIMING_SPAWN_POLICY,
    NoZeroCrashReferenceError,
    REFERENCE_MODE_MIN_CRASH,
    REFERENCE_MODE_ZERO_CRASH,
    apply_domain_randomization_setting,
    apply_lap_timing_spawn_policy,
    benchmark_config_fingerprint,
    classify_step_terminations,
    default_equivalent_tolerance_s,
    is_valid_cached_result,
    lap_timing_config_sha,
    load_lap_timing_config,
    select_tournament_seeds,
)


def _term_extras(**flags: list[bool]) -> dict:
    tensors = {
        key: torch.tensor(vals, dtype=torch.float32)
        for key, vals in flags.items()
    }
    return {"termination": tensors}


def test_classify_step_terminations_counts_crash_not_timeout():
    done = [True, True, False, True]
    extras = _term_extras(
        time_out=[1.0, 0.0, 0.0, 0.0],
        out_of_bounds=[0.0, 1.0, 0.0, 0.0],
        not_moving=[0.0, 0.0, 0.0, 1.0],
        invalid_state=[0.0, 0.0, 0.0, 0.0],
    )
    crashes, timeouts, breakdown, _ = classify_step_terminations(done, extras)
    assert crashes == 2
    assert timeouts == 1
    assert breakdown["out_of_bounds"] == 1
    assert breakdown["not_moving"] == 1


def test_classify_step_terminations_ignores_collision_when_absent():
    done = [True]
    extras = _term_extras(
        time_out=[0.0],
        out_of_bounds=[0.0],
        collision=[1.0],
    )
    crashes, timeouts, breakdown, _ = classify_step_terminations(done, extras)
    assert crashes == 1
    assert timeouts == 0
    assert breakdown["collision"] == 1


def test_classify_step_terminations_timeout_with_crash_flag_counts_crash():
    done = [True]
    extras = _term_extras(
        time_out=[1.0],
        out_of_bounds=[1.0],
    )
    crashes, timeouts, breakdown, _ = classify_step_terminations(done, extras)
    assert crashes == 1
    assert timeouts == 0
    assert breakdown["out_of_bounds"] == 1


def _row(
    name: str,
    *,
    mean: float,
    min_lap: float = 60.0,
    n_laps: int = 25,
    crashes: int = 0,
    transitions: int = 0,
) -> dict:
    return {
        "checkpoint": name,
        "transitions": transitions,
        "n_laps": n_laps,
        "min": min_lap,
        "mean": mean,
        "crashes": crashes,
    }


def test_select_tournament_seeds_faster_crashing_model_included():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, transitions=100),
        _row("fast_crash.pt", mean=61.0, crashes=3, transitions=200),
        _row("slow.pt", mean=70.0, crashes=0, transitions=300),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    names = [r["checkpoint"] for r in out["selected"]]
    assert "ref.pt" in names
    assert "fast_crash.pt" in names
    assert "slow.pt" not in names
    assert out["reference_checkpoint"] == "ref.pt"
    assert out["cutoff_mean_s"] == pytest.approx(62.05)


def test_select_tournament_seeds_slower_crashing_model_excluded():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, transitions=100),
        _row("slow_crash.pt", mean=63.0, crashes=5, transitions=200),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    names = [r["checkpoint"] for r in out["selected"]]
    assert names == ["ref.pt"]


def test_select_tournament_seeds_tolerance_equality():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, transitions=100),
        _row("tie.pt", mean=62.05, crashes=1, transitions=200),
        _row("over.pt", mean=62.06, crashes=0, transitions=300),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    names = [r["checkpoint"] for r in out["selected"]]
    assert "ref.pt" in names
    assert "tie.pt" in names
    assert "over.pt" not in names


def test_select_tournament_seeds_min_laps_gate():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, n_laps=25, transitions=100),
        _row("few_laps.pt", mean=60.0, crashes=0, n_laps=10, transitions=200),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    names = [r["checkpoint"] for r in out["selected"]]
    assert names == ["ref.pt"]


def test_select_tournament_seeds_no_zero_crash_raises():
    rows = [
        _row("a.pt", mean=60.0, crashes=2, transitions=100),
        _row("b.pt", mean=61.0, crashes=1, transitions=200),
    ]
    with pytest.raises(NoZeroCrashReferenceError):
        select_tournament_seeds(
            rows, min_laps=20, equivalent_tolerance_s=0.05,
            reference_mode=REFERENCE_MODE_ZERO_CRASH,
        )


def test_select_min_crash_reference_from_fastest_in_tier():
    rows = [
        _row("slow_min.pt", mean=60.0, crashes=2, transitions=100),
        _row("fast_min.pt", mean=58.0, min_lap=59.0, crashes=2, transitions=200),
        _row("faster_more_crash.pt", mean=57.0, crashes=5, transitions=300),
        _row("within.pt", mean=58.04, crashes=10, transitions=400),
        _row("outside.pt", mean=58.10, crashes=2, transitions=500),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_MIN_CRASH,
    )
    assert out["reference_mode"] == REFERENCE_MODE_MIN_CRASH
    assert out["min_crash_count"] == 2
    assert out["reference_checkpoint"] == "fast_min.pt"
    assert out["cutoff_mean_s"] == pytest.approx(58.05)
    names = [r["checkpoint"] for r in out["selected"]]
    assert names == ["faster_more_crash.pt", "fast_min.pt", "within.pt"]
    assert "slow_min.pt" not in names
    assert "outside.pt" not in names


def test_select_min_crash_reference_tiebreaks_min_then_transitions():
    rows = [
        _row("a.pt", mean=58.0, min_lap=59.0, crashes=3, transitions=300),
        _row("b.pt", mean=58.0, min_lap=57.0, crashes=3, transitions=200),
        _row("c.pt", mean=58.0, min_lap=57.0, crashes=3, transitions=100),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_MIN_CRASH,
    )
    assert out["reference_checkpoint"] == "c.pt"


def test_default_reference_mode_is_zero_crash():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, transitions=100),
        _row("other.pt", mean=63.0, crashes=1, transitions=200),
    ]
    out = select_tournament_seeds(rows, min_laps=20, equivalent_tolerance_s=0.05)
    assert out["reference_mode"] == REFERENCE_MODE_ZERO_CRASH
    assert out["min_crash_count"] == 0


def test_select_tournament_seeds_sort_order():
    rows = [
        _row("ref.pt", mean=62.0, min_lap=61.0, crashes=0, transitions=100),
        _row("same_mean_more_crash.pt", mean=62.0, min_lap=60.0, crashes=2, transitions=200),
        _row("same_mean_zero_crash.pt", mean=62.0, min_lap=60.5, crashes=0, transitions=300),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    names = [r["checkpoint"] for r in out["selected"]]
    assert names.index("same_mean_zero_crash.pt") < names.index("same_mean_more_crash.pt")


def test_select_tournament_seeds_writes_seed_rank_and_metric():
    rows = [
        _row("ref.pt", mean=62.0, crashes=0, transitions=100),
        _row("other.pt", mean=62.02, crashes=1, transitions=200),
    ]
    out = select_tournament_seeds(
        rows, min_laps=20, equivalent_tolerance_s=0.05,
        reference_mode=REFERENCE_MODE_ZERO_CRASH,
    )
    assert out["selected"][0]["seed_rank"] == 1
    assert out["selected"][0]["seed_metric"] == "mean"
    assert out["selected"][1]["seed_rank"] == 2


def test_default_equivalent_tolerance_s_from_run_config(tmp_path):
    cfg = {
        "config": {
            "env": {"sim_dt": 0.005, "control_interval": 10},
        }
    }
    (tmp_path / "config.json").write_text(
        __import__("json").dumps(cfg),
    )
    assert default_equivalent_tolerance_s(tmp_path) == pytest.approx(0.05)


def test_apply_domain_randomization_setting_disables_dr():
    cfg = {"env": {"domain_randomization": {"enabled": True, "mass_scale": [0.9, 1.1]}}}
    apply_domain_randomization_setting(cfg, enabled=False)
    assert cfg["env"]["domain_randomization"]["enabled"] is False
    assert cfg["env"]["domain_randomization"]["mass_scale"] == [0.9, 1.1]


def test_apply_lap_timing_spawn_policy_sets_centerline_env_keys():
    env_cfg = {"reset_yaw_jitter_rad": 0.2}
    apply_lap_timing_spawn_policy(env_cfg)
    assert env_cfg["reset_lateral_offset_m"] == 0.0
    assert env_cfg["reset_yaw_jitter_rad"] == 0.0


def test_benchmark_fingerprint_includes_centerline_spawn_policy():
    fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
    )
    assert fp["version"] == BENCHMARK_VERSION
    assert fp["spawn_policy"] == LAP_TIMING_SPAWN_POLICY


def test_is_valid_cached_result_rejects_stale_spawn_policy():
    fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
    )
    stale = {
        "benchmark_config": {**fp, "version": 2, "spawn_policy": "uniform_jittered"},
        "n_laps": 30,
        "mean": 60.0,
    }
    assert not is_valid_cached_result(stale, fp)


def test_lap_timing_spawn_policy_forces_centerline():
    env_cfg = {
        "reset_lateral_offset_m": 0.75,
        "reset_yaw_jitter_rad": 0.3,
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": 8},
    }
    apply_lap_timing_spawn_policy(env_cfg)
    assert env_cfg["reset_lateral_offset_m"] == 0.0
    assert env_cfg["reset_yaw_jitter_rad"] == 0.0


def test_benchmark_fingerprint_distinguishes_nominal_from_dr():
    dr_fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
    )
    nominal_fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=False,
    )
    assert dr_fp != nominal_fp
    assert dr_fp["domain_randomization_enabled"] is True
    assert nominal_fp["domain_randomization_enabled"] is False


def test_is_valid_cached_result_rejects_dr_result_for_nominal_fingerprint():
    dr_fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
    )
    nominal_fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=False,
    )
    dr_result = {"benchmark_config": dr_fp, "n_laps": 30, "mean": 60.0}
    assert is_valid_cached_result(dr_result, dr_fp)
    assert not is_valid_cached_result(dr_result, nominal_fp)


def test_load_lap_timing_config_override(tmp_path):
    cfg_path = tmp_path / "canonical.json"
    cfg_path.write_text(
        __import__("json").dumps({"config": {"env": {"track": "Canon"}, "obs": {}}}),
    )
    ckpt = tmp_path / "checkpoints" / "policy_1.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    cfg, source = load_lap_timing_config(ckpt, config=str(cfg_path))
    assert source == str(cfg_path.resolve())
    assert cfg["env"]["track"] == "Canon"


def test_benchmark_fingerprint_includes_config_override(tmp_path):
    cfg_path = tmp_path / "canonical.json"
    cfg_path.write_text(
        __import__("json").dumps({"config": {"env": {"track": "Austin"}, "obs": {}}}),
    )
    sha = lap_timing_config_sha(
        __import__("json").loads(cfg_path.read_text())["config"], "Austin",
    )
    fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
        config_source=str(cfg_path), config_sha=sha,
    )
    assert fp["config_source"] == str(cfg_path)
    assert fp["config_sha"] == sha


def test_is_valid_cached_result_rejects_stale_config():
    fp = benchmark_config_fingerprint(
        track="Austin", num_envs=32, steps=4500, seed=0,
        device="cpu", precision="32", domain_randomization_enabled=True,
    )
    stale = {"benchmark_config": {**fp, "steps": 2500}, "n_laps": 30, "mean": 60.0}
    assert not is_valid_cached_result(stale, fp)
    fresh = {"benchmark_config": fp, "n_laps": 30, "mean": 60.0}
    assert is_valid_cached_result(fresh, fp)

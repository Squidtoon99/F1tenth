"""Tests for the 1v1 passing reward (rewards.reward_passing).

Verifies smoothness (bounded step-to-step change, including across the
start/finish line), reset-safety (the opponent progress delta is exactly zero on
the reset step, so resets never inject a spike), and sign correctness. The
passing reward is ``passing_k * (ego_ds - opp_ds)`` gated to opponents within
``[-passing_gate_behind_m, +passing_gate_ahead_m]`` on the centerline, built
from per-step arc-length deltas, so it is smooth and reset-safe by construction.

``rewards.py`` uses package-relative imports, so we register a small
``f1tenth_env`` package shim pointing the relative ``.car`` / ``.utils`` at the
already-loaded audit modules, then load ``rewards.py`` under that package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def rewards_mod(real_modules):
    pkg_name = "f1tenth_env_under_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, "f1tenth_env")]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.car"] = real_modules.car
    sys.modules[f"{pkg_name}.utils"] = real_modules.utils

    path = os.path.join(_REPO_ROOT, "f1tenth_env", "rewards.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.rewards", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.rewards"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _step_state(opp_s, ego_ds, L=100.0, ego_s=5.0):
    return {
        "frenet": {
            "L": torch.tensor(L, dtype=torch.float32),
            "s": torch.tensor([ego_s], dtype=torch.float32),
        },
        "progress_ds": torch.tensor([ego_ds], dtype=torch.float32),
        "opp_s": torch.tensor([opp_s], dtype=torch.float32),
    }


def _cfg(passing_k=5.0, ahead_m=40.0, behind_m=20.0):
    return {
        "passing_k": passing_k,
        "passing_gate_ahead_m": ahead_m,
        "passing_gate_behind_m": behind_m,
        "progress_max_step_frac": 0.05,
    }


# --- sign correctness ---------------------------------------------------------
def test_sign_ego_gaining_is_positive(rewards_mod):
    rs = {}
    cfg = _cfg()
    # establish baselines
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    # ego advances 0.5, opponent only 0.1 -> ego gains -> positive (raw component)
    ss = _step_state(10.1, 0.5)
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.5 - 0.1, abs=1e-5)
    assert r[0].item() > 0


def test_sign_ego_losing_is_negative(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.6, 0.1)  # opponent advances 0.6, ego only 0.1
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.1 - 0.6, abs=1e-5)
    assert r[0].item() < 0


def test_gate_active_when_prev_in_window_cur_out(rewards_mod):
    """max(gate_prev, gate_cur): an opponent that was in range last step keeps the
    passing term active this step, then deactivates once both states are out."""
    rs = {}
    cfg = _cfg()
    # step A: opponent in window (gap ~5) -> prev gate set active
    rewards_mod.reward_passing(_step_state(10.0, 0.3), cfg, rs, torch.tensor([5]))
    # step B: opponent leaves the +40 ahead gate this step, but prev was in-window
    b = rewards_mod.reward_passing(_step_state(46.0, 0.3), cfg, rs, torch.tensor([6]))
    assert b[0].item() != 0.0
    # step C: opponent stays out of window; prev now out -> gate inactive
    c = rewards_mod.reward_passing(_step_state(46.0, 0.3), cfg, rs, torch.tensor([7]))
    assert c[0].item() == pytest.approx(0.0, abs=1e-6)


def test_static_relative_position_is_zero(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(10.0, 0.3), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.3, 0.3)  # both advance 0.3
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.0, abs=1e-5)


# --- reset safety -------------------------------------------------------------
def test_opp_delta_zero_on_reset(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.ensure_opp_progress_delta(_step_state(10.0, 0.0), torch.tensor([5]), cfg, rs)
    ss = _step_state(10.3, 0.0)
    rewards_mod.ensure_opp_progress_delta(ss, torch.tensor([6]), cfg, rs)
    assert ss["opp_progress_ds"][0].item() == pytest.approx(0.3, abs=1e-5)
    # reset: episode step counter drops below previous -> delta forced to 0
    ss_reset = _step_state(2.0, 0.0)
    rewards_mod.ensure_opp_progress_delta(ss_reset, torch.tensor([1]), cfg, rs)
    assert ss_reset["opp_progress_ds"][0].item() == 0.0


# --- continuity sweep ---------------------------------------------------------
def test_continuity_sweep_bounded(rewards_mod):
    rs = {}
    cfg = _cfg()
    L = 100.0
    opp_s = 0.0
    step = 0
    rewards_mod.reward_passing(_step_state(opp_s, 0.0, L), cfg, rs, torch.tensor([step]))
    prev_r = None
    max_jump = 0.0
    for _ in range(400):
        step += 1
        opp_s = (opp_s + 0.25) % L  # advances past the start/finish line repeatedly
        ss = _step_state(opp_s, 0.25, L)
        r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([step]))[0].item()
        assert abs(r) < 5.0  # bounded (no full-lap spike at the wrap)
        if prev_r is not None:
            max_jump = max(max_jump, abs(r - prev_r))
        prev_r = r
    # ego and opponent both advance 0.25/step -> passing ~ 0 with tiny variation
    assert max_jump < 0.5


# --- raw magnitude ------------------------------------------------------------
def test_raw_component_is_relative_arclength(rewards_mod):
    """The raw component equals ego_ds - opp_ds (the single coefficient is applied
    by compute_rewards): ego advancing 0.4 past a static opponent -> 0.4."""
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.0, 0.4)  # opponent static, ego advances 0.4
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.4, abs=1e-5)


# --- locality gate ------------------------------------------------------------
def test_gate_opponent_far_ahead_is_zero(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(90.0, 0.0, L=1000.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(90.5, 0.5, L=1000.0)  # gap=85.5 > 40, ego gaining
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.0, abs=1e-5)


def test_gate_opponent_far_behind_is_zero(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(
        _step_state(1.0, 0.0, L=1000.0, ego_s=50.0), cfg, rs, torch.tensor([5])
    )
    ss = _step_state(1.5, 0.5, L=1000.0, ego_s=50.0)  # gap=-48.5 < -20
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.0, abs=1e-5)


def test_gate_opponent_in_window_passes_through(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.1, 0.5)  # gap=5.1, within [-20, +40]
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(0.5 - 0.1, abs=1e-5)


# --- overtake-completed bonus -------------------------------------------------
def _ovc(bonus_k=1.0, gap_m=5.0):
    return {"overtake_bonus_k": bonus_k, "overtake_gap_m": gap_m}


def test_overtake_fires_on_close_pass(rewards_mod):
    rs = {}
    cfg = _ovc()
    # step1: opponent 2 m ahead (establishes prev sign)
    r0 = rewards_mod.reward_overtake(_step_state(7.0, 0.3), cfg, rs, torch.tensor([5]))
    assert r0[0].item() == 0.0
    # step2: opponent now 1 m behind, still close -> completed pass -> +bonus_k
    r1 = rewards_mod.reward_overtake(_step_state(4.0, 0.3), cfg, rs, torch.tensor([6]))
    assert r1[0].item() == pytest.approx(1.0, abs=1e-6)


def test_overtake_no_fire_on_lap_wrap(rewards_mod):
    rs = {}
    cfg = _ovc()
    # step1: opponent just ahead (gap +1)
    rewards_mod.reward_overtake(_step_state(6.0, 0.3), cfg, rs, torch.tensor([5]))
    # step2: sign flips to behind but the wrapped gap is 45 m -> not a real pass
    r = rewards_mod.reward_overtake(_step_state(60.0, 0.3), cfg, rs, torch.tensor([6]))
    assert r[0].item() == 0.0


def test_overtake_no_fire_on_reset(rewards_mod):
    rs = {}
    cfg = _ovc()
    rewards_mod.reward_overtake(_step_state(7.0, 0.3), cfg, rs, torch.tensor([6]))
    # step counter drops (episode reset): even a sign flip must not fire
    r = rewards_mod.reward_overtake(_step_state(4.0, 0.3), cfg, rs, torch.tensor([1]))
    assert r[0].item() == 0.0


def test_overtake_no_fire_when_being_passed(rewards_mod):
    rs = {}
    cfg = _ovc()
    # step1: opponent behind; step2: opponent moves ahead (we are being passed)
    rewards_mod.reward_overtake(_step_state(4.0, 0.3), cfg, rs, torch.tensor([5]))
    r = rewards_mod.reward_overtake(_step_state(6.0, 0.3), cfg, rs, torch.tensor([6]))
    assert r[0].item() == 0.0

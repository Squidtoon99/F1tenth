"""Tests for the 1v1 passing reward (rewards.reward_passing).

Verifies smoothness (bounded step-to-step change, including across the
start/finish line), reset-safety (the opponent progress delta is exactly zero on
the reset step, so resets never inject a spike), and sign correctness. The
passing reward is ``passing_k * (ego_ds - opp_ds)`` built from per-step
arc-length deltas, so it is smooth and reset-safe by construction.

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


def _step_state(opp_s, ego_ds, L=100.0):
    return {
        "frenet": {"L": torch.tensor(L, dtype=torch.float32)},
        "progress_ds": torch.tensor([ego_ds], dtype=torch.float32),
        "opp_s": torch.tensor([opp_s], dtype=torch.float32),
    }


def _cfg(passing_k=5.0):
    return {"passing_k": passing_k, "progress_max_step_frac": 0.05}


# --- sign correctness ---------------------------------------------------------
def test_sign_ego_gaining_is_positive(rewards_mod):
    rs = {}
    cfg = _cfg()
    # establish baselines
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    # ego advances 0.5, opponent only 0.1 -> ego gains -> positive
    ss = _step_state(10.1, 0.5)
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(5.0 * (0.5 - 0.1), abs=1e-5)
    assert r[0].item() > 0


def test_sign_ego_losing_is_negative(rewards_mod):
    rs = {}
    cfg = _cfg()
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.6, 0.1)  # opponent advances 0.6, ego only 0.1
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(5.0 * (0.1 - 0.6), abs=1e-5)
    assert r[0].item() < 0


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


# --- scale / dominance --------------------------------------------------------
def test_passing_scale_comparable_to_progress(rewards_mod):
    """With passing_k == progress gain, ego advancing past a static opponent yields
    a passing reward of the same scale as the raw progress term (k * ds)."""
    rs = {}
    cfg = _cfg(passing_k=5.0)
    rewards_mod.reward_passing(_step_state(10.0, 0.0), cfg, rs, torch.tensor([5]))
    ss = _step_state(10.0, 0.4)  # opponent static, ego advances 0.4
    r = rewards_mod.reward_passing(ss, cfg, rs, torch.tensor([6]))
    assert r[0].item() == pytest.approx(5.0 * 0.4, abs=1e-5)

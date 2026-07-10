"""Unit tests for the additive, deadzoned combined-slip penalty."""

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
    pkg_name = "f1tenth_env_slip_reward_test"
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


def _slip(ratio, angle):
    """Build an (N,8) slip tensor from per-wheel ratio/angle (broadcast scalars)."""
    r = torch.full((1, 4), float(ratio))
    a = torch.full((1, 4), float(angle))
    return torch.cat([r, a], dim=-1)


def test_zero_slip_is_zero_penalty(rewards_mod):
    ss = {"tyre_slip": torch.zeros(1, 8)}
    r = rewards_mod.reward_tyre_slip_penalty(ss, {})
    assert r.item() == pytest.approx(0.0)


def test_additive_over_channels_no_deadzone(rewards_mod):
    """With deadzones 0 and unit angle weight, penalty = -(sum|ratio| + sum|angle|)."""
    ss = {"tyre_slip": _slip(0.2, 0.1)}
    cfg = {"slip_deadzone_ratio": 0.0, "slip_deadzone_angle": 0.0, "slip_angle_weight": 1.0}
    r = rewards_mod.reward_tyre_slip_penalty(ss, cfg)
    assert r.item() == pytest.approx(-(4 * 0.2 + 4 * 0.1))


def test_ratio_is_clamped_to_one(rewards_mod):
    """Slip ratio magnitude is capped at 1 so a spinning wheel can't dominate."""
    ss = {"tyre_slip": _slip(5.0, 0.0)}
    r = rewards_mod.reward_tyre_slip_penalty(ss, {"slip_angle_weight": 1.0})
    assert r.item() == pytest.approx(-4.0)


def test_deadzone_leaves_controlled_slip_unpenalized(rewards_mod):
    """Slip strictly inside both deadzones incurs no penalty (the controlled regime)."""
    ss = {"tyre_slip": _slip(0.08, 0.04)}
    cfg = {"slip_deadzone_ratio": 0.1, "slip_deadzone_angle": 0.05, "slip_angle_weight": 1.0}
    r = rewards_mod.reward_tyre_slip_penalty(ss, cfg)
    assert r.item() == pytest.approx(0.0)


def test_deadzone_penalizes_only_excess(rewards_mod):
    """Beyond the deadzone only the excess is penalized (relu behaviour)."""
    ss = {"tyre_slip": _slip(0.3, 0.2)}
    cfg = {"slip_deadzone_ratio": 0.1, "slip_deadzone_angle": 0.05, "slip_angle_weight": 2.0}
    r = rewards_mod.reward_tyre_slip_penalty(ss, cfg)
    expected = -(4 * (0.3 - 0.1) + 2.0 * 4 * (0.2 - 0.05))
    assert r.item() == pytest.approx(expected)


def test_angle_weight_scales_lateral_channel(rewards_mod):
    ss = {"tyre_slip": _slip(0.0, 0.1)}
    base = rewards_mod.reward_tyre_slip_penalty(ss, {"slip_angle_weight": 1.0})
    heavy = rewards_mod.reward_tyre_slip_penalty(ss, {"slip_angle_weight": 3.0})
    assert heavy.item() == pytest.approx(3.0 * base.item())

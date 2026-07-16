"""Unit tests for the GT Sophy tyre-slip penalty (product form with cadence)."""

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
    r = rewards_mod.reward_tyre_slip_penalty(ss, {"control_dt": 0.05})
    assert r.item() == pytest.approx(0.0)


def test_product_over_channels_with_cadence(rewards_mod):
    """Representative: -cadence * sum_i min(|ratio_i|,1) * |angle_i|.

    Four wheels at ratio 0.2, angle 0.1 -> 4 * 0.2 * 0.1 = 0.08; cadence 0.5 -> -0.04.
    """
    ss = {"tyre_slip": _slip(0.2, 0.1)}
    r = rewards_mod.reward_tyre_slip_penalty(ss, {"control_dt": 0.05})
    assert r.item() == pytest.approx(-0.5 * (4 * 0.2 * 0.1))


def test_ratio_is_clamped_to_one(rewards_mod):
    """Boundary: slip ratio magnitude is capped at 1 so a spinning wheel cannot
    dominate (4 * min(5,1) * 0.1 = 0.4; cadence 0.5 -> -0.2)."""
    ss = {"tyre_slip": _slip(5.0, 0.1)}
    r = rewards_mod.reward_tyre_slip_penalty(ss, {"control_dt": 0.05})
    assert r.item() == pytest.approx(-0.5 * (4 * 1.0 * 0.1))


def test_pure_slip_channels_do_not_penalize(rewards_mod):
    """Product form: pure wheelspin (angle 0) or pure drift (ratio 0) -> no penalty."""
    spin = {"tyre_slip": _slip(0.5, 0.0)}
    drift = {"tyre_slip": _slip(0.0, 0.3)}
    assert rewards_mod.reward_tyre_slip_penalty(
        spin, {"control_dt": 0.05}
    ).item() == pytest.approx(0.0)
    assert rewards_mod.reward_tyre_slip_penalty(
        drift, {"control_dt": 0.05}
    ).item() == pytest.approx(0.0)

"""Pure-torch tests for the GT Sophy any-collision penalty (rewards.reward_collision).

The penalty is ``-collision_k * c`` where ``c`` is the binary car-to-car overlap
indicator. These tests call ``reward_collision`` directly with a synthetic
``car_collision`` mask, so no Genesis simulation is needed.

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
    pkg_name = "f1tenth_env_collision_under_test"
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


def _step_state(collision_mask):
    return {
        "progress_ds": torch.zeros(len(collision_mask), dtype=torch.float32),
        "car_collision": torch.tensor(collision_mask, dtype=torch.bool),
    }


def test_no_opponent_returns_zero(rewards_mod):
    """No car_collision key (1v0 path) -> zero penalty, shaped like progress_ds."""
    ss = {"progress_ds": torch.zeros(3, dtype=torch.float32)}
    r = rewards_mod.reward_collision(ss, {"collision_k": 5.0})
    assert torch.equal(r, torch.zeros(3))


def test_penalty_negative_on_contact(rewards_mod):
    ss = _step_state([True, False])
    r = rewards_mod.reward_collision(ss, {"collision_k": 5.0})
    assert r[0].item() == pytest.approx(-5.0, abs=1e-6)
    assert r[1].item() == 0.0


def test_penalty_zero_when_separated(rewards_mod):
    ss = _step_state([False, False])
    r = rewards_mod.reward_collision(ss, {"collision_k": 5.0})
    assert torch.equal(r, torch.zeros(2))


def test_penalty_scales_with_collision_k(rewards_mod):
    ss = _step_state([True])
    r = rewards_mod.reward_collision(ss, {"collision_k": 12.5})
    assert r[0].item() == pytest.approx(-12.5, abs=1e-6)


def test_default_collision_k(rewards_mod):
    ss = _step_state([True])
    r = rewards_mod.reward_collision(ss, {})
    assert r[0].item() == pytest.approx(-5.0, abs=1e-6)

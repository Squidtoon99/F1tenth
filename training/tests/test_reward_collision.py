"""Pure-torch tests for the GT Sophy any-collision penalty (rewards.reward_collision).

The raw component is ``Rc = -cadence * c`` where ``c`` is the binary car-to-car
overlap indicator and ``cadence = control_dt / 0.1``. These tests call
``reward_collision`` directly with a synthetic ``car_collision`` mask, so no
Genesis simulation is needed.

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
    r = rewards_mod.reward_collision(ss, {"control_dt": 0.05})
    assert torch.equal(r, torch.zeros(3))


def test_any_contact_penalty_with_cadence(rewards_mod):
    """Representative: contact -> -cadence, separated -> 0 (cadence 0.5 at 20 Hz)."""
    ss = _step_state([True, False])
    r = rewards_mod.reward_collision(ss, {"control_dt": 0.05})
    assert r[0].item() == pytest.approx(-0.5, abs=1e-6)
    assert r[1].item() == 0.0


def test_cadence_tracks_control_dt(rewards_mod):
    """Boundary: at the Sophy 10 Hz rate (control_dt=0.1) cadence is exactly 1."""
    ss = _step_state([True])
    r = rewards_mod.reward_collision(ss, {"control_dt": 0.1})
    assert r[0].item() == pytest.approx(-1.0, abs=1e-6)


def test_default_cadence_is_sophy_10hz(rewards_mod):
    ss = _step_state([True])
    r = rewards_mod.reward_collision(ss, {})
    assert r[0].item() == pytest.approx(-1.0, abs=1e-6)

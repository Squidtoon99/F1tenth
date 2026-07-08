"""Pure-torch tests for the GT Sophy rear-end penalty (rewards.reward_rear_end).

The penalty is ``-rear_end_k * c * 1(opp ahead) * ||v_ego - v_opp||^2`` where ``c``
is the binary car-to-car overlap indicator. These tests call ``reward_rear_end``
directly with a synthetic step_state, so no Genesis simulation is needed.

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
    pkg_name = "f1tenth_env_rear_end_under_test"
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


def _step_state(*, collision, ego_s, opp_s, ego_vel, opp_vel, length=100.0):
    n = len(collision)
    return {
        "progress_ds": torch.zeros(n, dtype=torch.float32),
        "car_collision": torch.tensor(collision, dtype=torch.bool),
        "opp_s": torch.tensor(opp_s, dtype=torch.float32),
        "opp_vel_world": torch.tensor(opp_vel, dtype=torch.float32),
        "ego_vel_world": torch.tensor(ego_vel, dtype=torch.float32),
        "frenet": {"s": torch.tensor(ego_s, dtype=torch.float32), "L": length},
    }


def test_no_opponent_returns_zero(rewards_mod):
    """No car_collision key (1v0 path) -> zero penalty, shaped like progress_ds."""
    ss = {"progress_ds": torch.zeros(3, dtype=torch.float32)}
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert torch.equal(r, torch.zeros(3))


def test_missing_velocity_returns_zero(rewards_mod):
    """car_collision present but no velocity/arc-length state -> zeros."""
    ss = {
        "progress_ds": torch.zeros(2, dtype=torch.float32),
        "car_collision": torch.tensor([True, False]),
    }
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert torch.equal(r, torch.zeros(2))


def test_penalty_negative_when_opp_ahead_and_closing(rewards_mod):
    ss = _step_state(
        collision=[True],
        ego_s=[10.0],
        opp_s=[20.0],          # gap +10 -> opponent ahead
        ego_vel=[[5.0, 0.0]],
        opp_vel=[[2.0, 0.0]],  # rel velocity [3, 0] -> closing_sq = 9
    )
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert r[0].item() == pytest.approx(-45.0, abs=1e-5)


def test_zero_when_opponent_behind(rewards_mod):
    ss = _step_state(
        collision=[True],
        ego_s=[20.0],
        opp_s=[10.0],          # gap -10 -> opponent behind
        ego_vel=[[5.0, 0.0]],
        opp_vel=[[2.0, 0.0]],
    )
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert r[0].item() == 0.0


def test_zero_when_no_closing_speed(rewards_mod):
    """Touching an opponent ahead while matched in velocity -> no rear-end energy."""
    ss = _step_state(
        collision=[True],
        ego_s=[10.0],
        opp_s=[20.0],
        ego_vel=[[4.0, 0.0]],
        opp_vel=[[4.0, 0.0]],  # rel velocity 0
    )
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert r[0].item() == pytest.approx(0.0, abs=1e-6)


def test_zero_when_not_colliding(rewards_mod):
    ss = _step_state(
        collision=[False],
        ego_s=[10.0],
        opp_s=[20.0],
        ego_vel=[[5.0, 0.0]],
        opp_vel=[[0.0, 0.0]],
    )
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 5.0})
    assert r[0].item() == 0.0


def test_penalty_scales_with_rear_end_k(rewards_mod):
    kwargs = dict(
        collision=[True],
        ego_s=[10.0],
        opp_s=[20.0],
        ego_vel=[[5.0, 0.0]],
        opp_vel=[[2.0, 0.0]],  # closing_sq = 9
    )
    r1 = rewards_mod.reward_rear_end(_step_state(**kwargs), {"rear_end_k": 1.0})
    r2 = rewards_mod.reward_rear_end(_step_state(**kwargs), {"rear_end_k": 10.0})
    assert r1[0].item() == pytest.approx(-9.0, abs=1e-5)
    assert r2[0].item() == pytest.approx(-90.0, abs=1e-5)


def test_default_rear_end_k(rewards_mod):
    ss = _step_state(
        collision=[True],
        ego_s=[10.0],
        opp_s=[20.0],
        ego_vel=[[5.0, 0.0]],
        opp_vel=[[2.0, 0.0]],  # closing_sq = 9
    )
    r = rewards_mod.reward_rear_end(ss, {})
    assert r[0].item() == pytest.approx(-45.0, abs=1e-5)


def test_wraps_around_start_finish_seam(rewards_mod):
    """Opponent just past the start/finish line is still 'ahead' via wrap."""
    ss = _step_state(
        collision=[True],
        ego_s=[98.0],
        opp_s=[2.0],           # raw gap -96 -> wraps to +4 -> ahead
        ego_vel=[[6.0, 0.0]],
        opp_vel=[[1.0, 0.0]],  # closing_sq = 25
        length=100.0,
    )
    r = rewards_mod.reward_rear_end(ss, {"rear_end_k": 2.0})
    assert r[0].item() == pytest.approx(-50.0, abs=1e-5)

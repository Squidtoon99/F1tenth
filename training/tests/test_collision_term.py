"""Tests for 1v1 collision termination (terminations.collision_mask) and the
guarantee that the existing 1v0 termination outputs are unchanged.

``terminations.py`` uses package-relative imports, so (as in test_passing_reward)
we register a small package shim pointing ``.utils`` at the loaded audit module.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# F110 envelope defaults (match config.py).
_CAR_LENGTH = 0.46
_CAR_WIDTH = 0.30


@pytest.fixture(scope="session")
def term_mod(real_modules):
    pkg_name = "f1tenth_env_term_under_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, "f1tenth_env")]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.utils"] = real_modules.utils

    path = os.path.join(_REPO_ROOT, "f1tenth_env", "terminations.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.terminations", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.terminations"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _collision(
    term_mod,
    ego_xy,
    opp_xy,
    ego_yaw=0.0,
    car_length=_CAR_LENGTH,
    car_width=_CAR_WIDTH,
    margin=0.0,
):
    ego = torch.tensor([ego_xy], dtype=torch.float32)
    opp = torch.tensor([opp_xy], dtype=torch.float32)
    yaw = torch.tensor([ego_yaw], dtype=torch.float32)
    return term_mod.collision_mask(
        ego, opp, yaw, car_length, car_width, margin
    ).item()


# --- anisotropic ego-frame collision -----------------------------------------
def test_collision_rear_end_within_longitudinal(term_mod):
    """Directly behind/ahead within long_thresh and ~0 lateral -> collision."""
    assert _collision(term_mod, (0.0, 0.0), (0.3, 0.0)) is True
    assert _collision(term_mod, (0.0, 0.0), (-0.3, 0.0)) is True


def test_collision_beyond_longitudinal(term_mod):
    """Directly behind beyond long_thresh -> no collision."""
    assert _collision(term_mod, (0.0, 0.0), (0.5, 0.0)) is False
    assert _collision(term_mod, (0.0, 0.0), (-0.5, 0.0)) is False


def test_collision_clean_side_by_side_pass(term_mod):
    """Side-by-side with lateral separation >= car_width -> no collision."""
    assert _collision(term_mod, (0.0, 0.0), (0.0, 0.35)) is False


def test_collision_side_by_side_lateral_overlap(term_mod):
    """Lateral overlap with small longitudinal offset -> collision."""
    assert _collision(term_mod, (0.0, 0.0), (0.0, 0.2)) is True


def test_collision_rotated_ego_longitudinal(term_mod):
    """Opponent offset along ego heading after yaw=90deg -> longitudinal."""
    yaw = math.pi / 2
    # Opponent 0.3 m ahead in ego frame (north in world when ego faces north).
    assert _collision(term_mod, (0.0, 0.0), (0.0, 0.3), ego_yaw=yaw) is True
    # Opponent 0.5 m ahead -> beyond long_thresh.
    assert _collision(term_mod, (0.0, 0.0), (0.0, 0.5), ego_yaw=yaw) is False
    # Opponent 0.35 m to ego's left (west in world) -> lateral, no collision.
    assert _collision(term_mod, (0.0, 0.0), (-0.35, 0.0), ego_yaw=yaw) is False


def test_collision_batched_mixed(term_mod):
    """Vectorized batch: mix of colliding and non-colliding envs."""
    ego = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]], dtype=torch.float32
    )
    opp = torch.tensor(
        [[0.3, 0.0], [0.0, 0.35], [0.0, 0.2], [10.0, 10.0]], dtype=torch.float32
    )
    yaw = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    mask = term_mod.collision_mask(
        ego, opp, yaw, _CAR_LENGTH, _CAR_WIDTH, 0.0
    )
    assert mask.tolist() == [True, False, True, False]


def test_collision_margin_expands_thresholds(term_mod):
    """Optional margin widens both longitudinal and lateral thresholds."""
    # 0.48 m longitudinal: just outside default long_thresh (0.46).
    assert _collision(term_mod, (0.0, 0.0), (0.48, 0.0), margin=0.0) is False
    assert _collision(term_mod, (0.0, 0.0), (0.48, 0.0), margin=0.05) is True


# --- 1v0 no-regression --------------------------------------------------------
def test_existing_terminations_have_no_collision_key(term_mod):
    """compute_terminations (the shared 1v0/1v1 core) never adds a collision key;
    collision is layered on by the env only when an opponent exists."""
    n = 4
    boundary = {
        "ey": torch.zeros(n),
        "w_l_s": torch.full((n,), 1.5),
        "w_r_s": torch.full((n,), 1.5),
        "boundary_dist": torch.full((n,), 1.5),
    }
    step_state = {
        "boundary": boundary,
        "frenet": {"seg_dir": torch.tensor([[1.0, 0.0]] * n)},
        "progress_ds": torch.full((n,), 0.1),
    }
    term_params = {
        "term_oob_margin_m": 0.15,
        "term_oob_max_consecutive": 10,
        "term_speed_threshold": 0.2,
        "term_not_moving_min_ds": 1e-3,
        "not_moving_steps_threshold": 20,
        "term_heading_error_rad": 3.0,
    }
    term_state = {
        "oob_consecutive_buf": torch.zeros(n, dtype=torch.int32),
        "not_moving_steps_buf": torch.zeros(n, dtype=torch.int32),
    }
    base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
    reset, extras, _ = term_mod.compute_terminations(
        step_state=step_state,
        episode_steps_buf=torch.zeros(n, dtype=torch.int32),
        max_episode_steps=250,
        base_pos=torch.zeros(n, 3),
        base_quat=base_quat,
        base_lin_vel=torch.tensor([[5.0, 0.0, 0.0]] * n),
        base_ang_vel=torch.zeros(n, 3),
        term_state=term_state,
        term_params=term_params,
    )
    assert "collision" not in extras
    assert not bool(reset.any())  # moving, on-track, not timed out -> no termination

"""Unit tests for lateral centerline penalty and rejoin progress masking."""

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
    pkg_name = "f1tenth_env_lateral_test"
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


def test_lateral_penalty_scales_with_ey_squared(rewards_mod):
    ss = {"boundary": {"ey": torch.tensor([0.0, 1.0, -2.0])}}
    cfg = {"lateral_k": 0.5}
    r = rewards_mod.reward_lateral(ss, cfg)
    assert r[0].item() == pytest.approx(0.0)
    assert r[1].item() == pytest.approx(-0.5)
    assert r[2].item() == pytest.approx(-2.0)


def test_rejoin_step_zeros_progress(rewards_mod, real_modules):
    """First on-track step after an off-track excursion must not credit progress."""
    utils = real_modules.utils
    w_l = torch.tensor([2.0])
    w_r = torch.tensor([2.0])
    # ey=0 -> on track at reward margin 0.2
    boundary_on = {
        "ey": torch.tensor([0.0]),
        "w_l_s": w_l,
        "w_r_s": w_r,
        "boundary_dist": torch.tensor([2.0]),
    }
    # ey beyond left width -> off track
    boundary_off = {
        "ey": torch.tensor([2.5]),
        "w_l_s": w_l,
        "w_r_s": w_r,
        "boundary_dist": torch.tensor([-0.5]),
    }

    reward_cfg = {
        "progress_k_fwd": 5.0,
        "progress_k_back": 5.0,
        "progress_max_lateral_m": 1.0,
        "oob_margin_m": 0.2,
        "oob_k": 0.3,
        "lateral_k": 0.5,
        "global_reward_scale": 1.0,
        "reward_scales": {
            "progress": 1.0,
            "lateral": 1.0,
            "oob_penalty": 0.0,
            "tyre_slip_penalty": 0.0,
            "smoothness": 0.0,
        },
    }
    reward_state = rewards_mod.init_reward_state(reward_cfg["reward_scales"], 1, torch.device("cpu"))

    base = {
        "frenet": {
            "pos": torch.zeros(1, 2),
            "proj": torch.zeros(1, 2),
            "L": torch.tensor(100.0),
            "seg_dir": torch.tensor([[1.0, 0.0]]),
        },
        "base_lin_vel": torch.zeros(1, 3),
        "progress_ds": torch.tensor([1.0]),
        "wheel_state": {
            "motion_link_vel": torch.zeros(1, 4, 3),
            "dof_vel": torch.zeros(1, 4),
        },
    }

    off_state = {**base, "boundary": boundary_off}
    _, _ = rewards_mod.compute_rewards(
        off_state, reward_cfg, reward_state, torch.tensor([1]), torch.tensor([0])
    )

    rejoin_state = {**base, "boundary": boundary_on, "progress_ds": torch.tensor([5.0])}
    reward_buf, _ = rewards_mod.compute_rewards(
        rejoin_state, reward_cfg, reward_state, torch.tensor([2]), torch.tensor([0])
    )
    terms = reward_state["last_reward_terms"]
    assert terms["progress"][0].item() == pytest.approx(0.0)
    assert reward_buf[0].item() < 5.0  # no large progress spike on rejoin

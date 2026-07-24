"""Unit tests for canonical Lee wall-contact reward and progress masking."""

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


@pytest.mark.parametrize("control_dt", [0.05, 0.1, 0.2])
def test_wall_contact_reward_is_linear_speed_without_dt_scaling(
    rewards_mod, control_dt
):
    ss = {
        "wall_contact": torch.tensor([True, True, True, False]),
        "base_lin_vel": torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
    }
    r = rewards_mod.reward_wall_contact(
        ss, {"wall_contact_coefficient": 20.0, "control_dt": control_dt}
    )
    assert r.tolist() == pytest.approx([-20.0, -40.0, -100.0, 0.0])


def test_wall_contact_masks_progress_and_clear_step_credits(rewards_mod, real_modules):
    del real_modules
    w_l = torch.tensor([2.0])
    w_r = torch.tensor([2.0])
    boundary_on = {
        "ey": torch.tensor([0.0]),
        "w_l_s": w_l,
        "w_r_s": w_r,
        "boundary_dist": torch.tensor([2.0]),
    }
    boundary_off = {
        "ey": torch.tensor([2.5]),
        "w_l_s": w_l,
        "w_r_s": w_r,
        "boundary_dist": torch.tensor([-0.5]),
    }

    reward_cfg = {
        "progress_max_lateral_m": 1.0,
        "wall_contact_coefficient": 20.0,
        "control_dt": 0.05,
        "global_reward_scale": 1.0,
        "reward_scales": {"progress": 1.0},
    }
    reward_state = rewards_mod.init_reward_state(
        reward_cfg["reward_scales"], 1, torch.device("cpu")
    )

    base = {
        "frenet": {
            "pos": torch.zeros(1, 2),
            "proj": torch.zeros(1, 2),
            "L": torch.tensor(100.0),
            "seg_dir": torch.tensor([[1.0, 0.0]]),
        },
        "base_lin_vel": torch.zeros(1, 3),
        "progress_ds": torch.tensor([5.0]),
    }

    off_state = {
        **base,
        "boundary": boundary_off,
        "wall_contact": torch.tensor([True]),
    }
    rewards_mod.compute_rewards(
        off_state, reward_cfg, reward_state, torch.tensor([1]), torch.tensor([0])
    )
    assert reward_state["last_reward_terms"]["progress"][0].item() == pytest.approx(0.0)

    rejoin_state = {
        **base,
        "boundary": boundary_on,
        "wall_contact": torch.tensor([False]),
    }
    rewards_mod.compute_rewards(
        rejoin_state, reward_cfg, reward_state, torch.tensor([2]), torch.tensor([0])
    )
    assert reward_state["last_reward_terms"]["progress"][0].item() == pytest.approx(5.0)

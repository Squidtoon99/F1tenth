"""Wall-contact reward and impact termination (Torch reward/termination seam).

Covers the balanced calibration:
  Rw        = scale_w * (-control_dt * (3.6 * speed)^2) while footprint contacts
              the actual corridor boundary (margin 0)
  R_impact  = scale_i * (-v_normal^2) on first contact only
  terminate when inward normal speed >= wall_impact_term_speed_mps (default 4)

Rsoc (off-course) stays unchanged and is semantically separate from wall contact.
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
    pkg_name = "f1tenth_env_wall_reward_test"
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


def _boundary(*, ey, w=2.0, half=0.148):
    ey_t = torch.as_tensor(ey, dtype=torch.float32)
    n = ey_t.numel()
    w_t = torch.full((n,), float(w), dtype=torch.float32)
    return {
        "ey": ey_t,
        "w_l_s": w_t.clone(),
        "w_r_s": w_t.clone(),
        "boundary_dist": w_t - ey_t.abs(),
        "oob_half_extent_m": torch.full((n,), float(half), dtype=torch.float32),
    }


def _cfg(**overrides):
    cfg = {
        "control_dt": 0.05,
        "oob_margin_m": 0.2,
        "lateral_k": 0.0,
        "global_reward_scale": 1.0,
        "wall_impact_term_speed_mps": 4.0,
        "reward_scales": {
            "progress": 0.0,
            "lateral": 0.0,
            "oob_penalty": 0.01,
            "tyre_slip_penalty": 0.0,
            "smoothness": 0.0,
            "wall_penalty": 0.01,
            "wall_impact": 1.0,
        },
    }
    cfg.update(overrides)
    if "reward_scales" in overrides:
        scales = dict(cfg["reward_scales"])
        scales.update(overrides["reward_scales"])
        cfg["reward_scales"] = scales
    return cfg


def _step_state(
    *,
    ey,
    vel_xy,
    half=0.148,
    w=2.0,
    progress_ds=0.0,
):
    ey_t = torch.as_tensor(ey, dtype=torch.float32).reshape(-1)
    n = ey_t.numel()
    vel = torch.as_tensor(vel_xy, dtype=torch.float32).reshape(n, 2)
    # Track along +x; left normal is +y.
    seg_dir = torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(n, 2).clone()
    return {
        "progress_ds": torch.full((n,), float(progress_ds), dtype=torch.float32),
        "boundary": _boundary(ey=ey_t, w=w, half=half),
        "base_lin_vel": torch.cat(
            [vel, torch.zeros(n, 1)], dim=-1
        ),
        "ego_vel_world": torch.cat(
            [vel, torch.zeros(n, 1)], dim=-1
        ),
        "frenet": {
            "s": torch.zeros(n),
            "L": torch.tensor(100.0),
            "seg_dir": seg_dir,
            "pos": torch.stack([torch.zeros(n), ey_t], dim=-1),
            "proj": torch.zeros(n, 2),
        },
        "tyre_slip": torch.zeros(n, 8),
    }


def test_wall_contact_at_actual_boundary_not_reward_margin(rewards_mod):
    """Footprint inside the 0.2 m reward margin but short of the wall is not contact."""
    half = 0.148
    w = 2.0
    # Off-course (margin 0.2) but not yet at the actual wall.
    ey_near = (w - 0.2) - half + 0.01  # just past reward margin
    ey_wall = w - half  # footprint exactly at left wall
    ss = _step_state(ey=[ey_near, ey_wall], vel_xy=[[0.0, 0.0], [0.0, 0.0]], half=half, w=w)
    contact, _, _ = rewards_mod.wall_contact_from_boundary(ss["boundary"])
    assert not bool(contact[0])
    assert bool(contact[1])


def test_5ms_head_on_impact_and_continuous_rw_exact(rewards_mod):
    """5 m/s into the wall: impact -25 and Rw = 0.01 * (-dt * (3.6*5)^2).

    Wall contact at the actual boundary is also off-course under the 0.2 m reward
    margin, so Rsoc applies in parallel and stays semantically separate.
    """
    half = 0.148
    w = 2.0
    ey = w - half
    ss = _step_state(ey=[ey], vel_xy=[[0.0, 5.0]], half=half, w=w)
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    reward, _ = rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    terms = state["last_reward_terms"]
    expected_rate = 0.01 * (-0.05 * (3.6 * 5.0) ** 2)
    expected_impact = -1.0 * (5.0 ** 2)
    assert terms["wall_penalty"][0].item() == pytest.approx(expected_rate, abs=1e-5)
    assert terms["wall_impact"][0].item() == pytest.approx(expected_impact, abs=1e-5)
    assert terms["oob_penalty"][0].item() == pytest.approx(expected_rate, abs=1e-5)
    assert reward[0].item() == pytest.approx(
        expected_rate + expected_rate + expected_impact, abs=1e-4
    )


def test_first_contact_only_charges_impact(rewards_mod):
    half = 0.148
    w = 2.0
    ey = w - half
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    ss = _step_state(ey=[ey], vel_xy=[[0.0, 5.0]], half=half, w=w)
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    first = state["last_reward_terms"]["wall_impact"][0].item()
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([2], dtype=torch.int32), torch.tensor([0])
    )
    second = state["last_reward_terms"]["wall_impact"][0].item()
    assert first == pytest.approx(-25.0, abs=1e-5)
    assert second == pytest.approx(0.0, abs=1e-6)


def test_persistent_rw_while_contact_continues(rewards_mod):
    half = 0.148
    w = 2.0
    ey = w - half
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    ss = _step_state(ey=[ey], vel_xy=[[3.0, 1.0]], half=half, w=w)
    speed = (3.0 ** 2 + 1.0 ** 2) ** 0.5
    expected_rw = 0.01 * (-0.05 * (3.6 * speed) ** 2)
    for step in (1, 2, 3):
        rewards_mod.compute_rewards(
            ss, cfg, state, torch.tensor([step], dtype=torch.int32), torch.tensor([0])
        )
        assert state["last_reward_terms"]["wall_penalty"][0].item() == pytest.approx(
            expected_rw, abs=1e-5
        )


def test_glancing_low_normal_speed_does_not_terminate(rewards_mod):
    half = 0.148
    w = 2.0
    ey = w - half
    # Mostly along-track with a small inward normal component.
    ss = _step_state(ey=[ey], vel_xy=[[5.0, 1.0]], half=half, w=w)
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    assert state["last_reward_terms"]["wall_impact"][0].item() == pytest.approx(
        -1.0, abs=1e-5
    )
    assert not bool(state["wall_impact_done"][0])


def test_terminate_when_normal_speed_at_least_4(rewards_mod):
    half = 0.148
    w = 2.0
    ey = w - half
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 2, torch.device("cpu"))
    ss = _step_state(
        ey=[ey, ey],
        vel_xy=[[0.0, 3.9], [0.0, 4.0]],
        half=half,
        w=w,
    )
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1, 1], dtype=torch.int32), torch.tensor([0, 0])
    )
    assert not bool(state["wall_impact_done"][0])
    assert bool(state["wall_impact_done"][1])


def test_rsoc_preserved_and_separate_from_wall(rewards_mod):
    """Off-course Rsoc still fires inside the margin; wall only at the boundary."""
    half = 0.148
    w = 2.0
    ey_off = (w - 0.2) - half + 0.05  # off-course, not at wall
    ss = _step_state(ey=[ey_off], vel_xy=[[5.0, 0.0]], half=half, w=w)
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    terms = state["last_reward_terms"]
    expected_rsoc = 0.01 * (-0.05 * (3.6 * 5.0) ** 2)
    assert terms["oob_penalty"][0].item() == pytest.approx(expected_rsoc, abs=1e-5)
    assert terms["wall_penalty"][0].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["wall_impact"][0].item() == pytest.approx(0.0, abs=1e-6)


def test_reset_clears_wall_contact_latch(rewards_mod):
    half = 0.148
    w = 2.0
    ey = w - half
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    ss = _step_state(ey=[ey], vel_xy=[[0.0, 5.0]], half=half, w=w)
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([5], dtype=torch.int32), torch.tensor([0])
    )
    assert state["last_reward_terms"]["wall_impact"][0].item() == pytest.approx(-25.0)
    # Episode step counter drops -> reset; contact latch must clear so impact
    # can fire again on the next first contact.
    rewards_mod.sync_progress_state_for_resets(
        state, ss, torch.tensor([0], dtype=torch.int32), torch.tensor([True])
    )
    rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    assert state["last_reward_terms"]["wall_impact"][0].item() == pytest.approx(-25.0)


def test_terminal_impact_reward_available_before_reset_clear(rewards_mod):
    """Impact reward is present on the terminating step prior to latch clear."""
    half = 0.148
    w = 2.0
    ey = w - half
    cfg = _cfg()
    state = rewards_mod.init_reward_state(cfg["reward_scales"], 1, torch.device("cpu"))
    ss = _step_state(ey=[ey], vel_xy=[[0.0, 5.0]], half=half, w=w)
    reward, _ = rewards_mod.compute_rewards(
        ss, cfg, state, torch.tensor([1], dtype=torch.int32), torch.tensor([0])
    )
    assert bool(state["wall_impact_done"][0])
    assert state["last_reward_terms"]["wall_impact"][0].item() == pytest.approx(-25.0)
    assert reward[0].item() < -25.0 + 1e-3
    # Clearing after the terminal transition must not rewrite the emitted terms.
    emitted = state["last_reward_terms"]["wall_impact"].clone()
    rewards_mod.sync_progress_state_for_resets(
        state, ss, torch.tensor([0], dtype=torch.int32), torch.tensor([True])
    )
    assert torch.equal(state["last_reward_terms"]["wall_impact"], emitted)

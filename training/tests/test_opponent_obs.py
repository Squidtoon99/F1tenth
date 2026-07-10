"""Known-answer tests for the 1v1 opponent-relative observation block.

The block (`observations.obs_opponent`) is pure geometry, so every value has a
closed form. These tests pin the ego-frame rotation convention, the signed
track-gap wrap, and the symmetric self/other property used to build the
opponent's own egocentric observation. Masking is applied by the env caller.
No Genesis sim is started.
"""

from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import (
    build_track_state,
    make_straight_track,
    yaw_quat_wxyz,
)
from f1tenth_env.env import F1tenthEnv

DEVICE = torch.device("cpu")

_RANGE_MASK_L = 100.0
_RANGE_MASK_AHEAD_M = 40.0
_RANGE_MASK_BEHIND_M = 20.0


def _range_mask_env():
    env = SimpleNamespace(
        obs_cfg={
            "opp_obs_ahead_m": _RANGE_MASK_AHEAD_M,
            "opp_obs_behind_m": _RANGE_MASK_BEHIND_M,
        }
    )
    env._wrapped_track_gap = MethodType(F1tenthEnv._wrapped_track_gap, env)
    return env


def _apply_range_mask(block, s_self, s_other, track_len=_RANGE_MASK_L):
    env = _range_mask_env()
    s_self_t = torch.as_tensor(s_self, dtype=torch.float32).reshape(-1)
    s_other_t = torch.as_tensor(s_other, dtype=torch.float32).reshape(-1)
    track_len_t = torch.as_tensor(track_len, dtype=torch.float32).reshape(-1)
    return F1tenthEnv._apply_opponent_range_mask(
        env, block, s_self_t, s_other_t, track_len_t
    )


def _sample_opponent_block():
    return torch.tensor([[1.0, -2.0, 3.0, -4.0, 0.5, 0.37]], dtype=torch.float32)


def _agent(pos_xy, yaw, vel_xy, s, ey, L):
    def t(x):
        return torch.tensor(np.atleast_1d(np.asarray(x, np.float32)), dtype=torch.float32)

    pos = torch.tensor(np.atleast_2d(np.asarray(pos_xy, np.float32)), dtype=torch.float32)
    vel = torch.tensor(np.atleast_2d(np.asarray(vel_xy, np.float32)), dtype=torch.float32)
    return {
        "pos_xy": pos,
        "yaw": t(yaw),
        "vel_xy": vel,
        "s": t(s),
        "ey": t(ey),
        "L": t(L),
    }


def _opp_cfg():
    return {"enable_opponent_obs": True, "opponent_obs_dim": 6}


# --- relative-position geometry ----------------------------------------------
def test_opponent_directly_ahead(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([5.0, 0.0], 0.0, [0.0, 0.0], 5.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[0].item() == pytest.approx(5.0, abs=1e-5)
    assert blk[1].item() == pytest.approx(0.0, abs=1e-5)


def test_opponent_directly_behind(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 10.0, 0.0, 100.0)
    other = _agent([-3.0, 0.0], 0.0, [0.0, 0.0], 7.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[0].item() == pytest.approx(-3.0, abs=1e-5)
    assert blk[1].item() == pytest.approx(0.0, abs=1e-5)


def test_opponent_to_the_left(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([0.0, 2.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[0].item() == pytest.approx(0.0, abs=1e-5)
    assert blk[1].item() == pytest.approx(2.0, abs=1e-5)


def test_relative_position_rotates_into_self_frame(real_modules):
    """Self yaw = +90deg, opponent ahead in world +x -> ego frame (0, -gap)."""
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], math.pi / 2.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([4.0, 0.0], math.pi / 2.0, [0.0, 0.0], 4.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[0].item() == pytest.approx(0.0, abs=1e-5)
    assert blk[1].item() == pytest.approx(-4.0, abs=1e-5)


def test_relative_velocity_in_self_frame(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [1.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([5.0, 0.0], 0.0, [3.0, 0.0], 5.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    # relative velocity = other - self = (2, 0) in world, yaw 0 -> same in ego frame
    assert blk[2].item() == pytest.approx(2.0, abs=1e-5)
    assert blk[3].item() == pytest.approx(0.0, abs=1e-5)


# --- signed track gap + wrap --------------------------------------------------
def test_track_gap_simple(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 10.0, 0.0, 100.0)
    other = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 15.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    # gap = (15 - 10) wrapped, normalized by L/2 = 50 -> 5/50 = 0.1, opponent ahead
    assert blk[4].item() == pytest.approx(0.1, abs=1e-5)


def test_track_gap_wraps_start_finish(real_modules):
    """Opponent just past the line is slightly AHEAD, not a near-full-lap behind."""
    obs_opponent = real_modules.observations.obs_opponent
    L = 100.0
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 99.0, 0.0, L)
    other = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 1.0, 0.0, L)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    # raw gap = -98 -> wrapped +2 -> normalized 2/50 = 0.04 (small + => just ahead)
    assert blk[4].item() == pytest.approx(0.04, abs=1e-5)
    assert abs(blk[4].item()) < 0.5


def test_other_lateral_offset_passthrough(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([5.0, 0.0], 0.0, [0.0, 0.0], 5.0, 0.37, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[5].item() == pytest.approx(0.37, abs=1e-6)


# --- symmetry -----------------------------------------------------------------
def test_gap_antisymmetry(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    a = _agent([0.0, 0.0], 0.3, [0.0, 0.0], 12.0, 0.1, 100.0)
    b = _agent([4.0, 1.0], -0.2, [0.0, 0.0], 20.0, -0.2, 100.0)
    blk_ab = obs_opponent(a, b, _opp_cfg())[0]
    blk_ba = obs_opponent(b, a, _opp_cfg())[0]
    assert blk_ab[4].item() == pytest.approx(-blk_ba[4].item(), abs=1e-5)


# --- env range mask (F1tenthEnv._apply_opponent_range_mask) ------------------
def test_range_mask_keeps_block_when_opponent_in_window():
    block = _sample_opponent_block()
    out_ahead = _apply_range_mask(block.clone(), s_self=0.0, s_other=5.0)
    out_behind = _apply_range_mask(block.clone(), s_self=25.0, s_other=20.0)
    assert torch.allclose(out_ahead, block)
    assert torch.allclose(out_behind, block)


def test_range_mask_zeros_when_opponent_too_far_ahead():
    block = _sample_opponent_block()
    out = _apply_range_mask(block, s_self=0.0, s_other=50.0)
    assert torch.equal(out, torch.zeros_like(block))


def test_range_mask_zeros_when_opponent_too_far_behind():
    block = _sample_opponent_block()
    out = _apply_range_mask(block, s_self=50.0, s_other=25.0)
    assert torch.equal(out, torch.zeros_like(block))


def test_range_mask_wrap_around_near_finish_line():
    block = _sample_opponent_block()
    # Raw s_opp - s_self = -98 m would be masked without wrap; wrapped gap = +2 m.
    out = _apply_range_mask(block, s_self=99.0, s_other=1.0)
    assert torch.allclose(out, block)


def test_range_mask_batch_masks_rows_independently():
    block = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        ],
        dtype=torch.float32,
    )
    s_self = [0.0, 0.0, 50.0, 99.0]
    s_other = [5.0, 50.0, 25.0, 1.0]
    expected = block.clone()
    expected[1] = 0.0
    expected[2] = 0.0
    out = _apply_range_mask(block, s_self, s_other)
    assert torch.equal(out, expected)


# --- finiteness over a random batch ------------------------------------------
def test_block_finite_random_batch(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    g = torch.Generator().manual_seed(0)
    n = 256
    me = {
        "pos_xy": torch.rand(n, 2, generator=g) * 40 - 20,
        "yaw": (torch.rand(n, generator=g) * 2 - 1) * math.pi,
        "vel_xy": torch.rand(n, 2, generator=g) * 10 - 5,
        "s": torch.rand(n, generator=g) * 100,
        "ey": torch.rand(n, generator=g) * 2 - 1,
        "L": torch.full((n,), 100.0),
    }
    other = {
        "pos_xy": torch.rand(n, 2, generator=g) * 40 - 20,
        "yaw": (torch.rand(n, generator=g) * 2 - 1) * math.pi,
        "vel_xy": torch.rand(n, 2, generator=g) * 10 - 5,
        "s": torch.rand(n, generator=g) * 100,
        "ey": torch.rand(n, generator=g) * 2 - 1,
        "L": torch.full((n,), 100.0),
    }
    blk = obs_opponent(me, other, _opp_cfg())
    assert blk.shape == (n, 6)
    assert torch.isfinite(blk).all()
    assert (blk[:, 4].abs() <= 1.0 + 1e-5).all()  # normalized gap in [-1, 1]


# --- build_observation: shape, append, sentinel ------------------------------
def _ego_step_state(real_modules, track_state, pos_xy):
    utils = real_modules.utils
    pos = np.atleast_2d(np.asarray(pos_xy, np.float32))
    base_pos = torch.tensor(
        np.concatenate([pos, np.zeros((pos.shape[0], 1), np.float32)], axis=-1),
        dtype=torch.float32,
    )
    episode_steps = torch.zeros(base_pos.shape[0], dtype=torch.int32)
    ss = utils.build_step_state(
        base_pos=base_pos,
        episode_steps_buf=episode_steps,
        track_state=track_state,
        device=DEVICE,
        cache_id="geom",
    )
    ss["tyre_slip"] = torch.zeros((base_pos.shape[0], 8), dtype=torch.float32)
    ss["tyre_load"] = torch.ones((base_pos.shape[0], 4), dtype=torch.float32)
    return base_pos, ss


def _build_obs_cfg(obs_cfg, *, enabled, k=6):
    cfg = dict(obs_cfg)
    cfg["enable_opponent_obs"] = enabled
    cfg["opponent_obs_dim"] = k
    cfg["num_obs"] = 384 + k if enabled else 384
    return cfg


def test_build_observation_appends_block_when_enabled(real_modules, obs_cfg):
    obs = real_modules.observations
    cl, wl, wr = make_straight_track(length=60.0, n=240)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _ego_step_state(real_modules, ts, [10.0, 0.0])
    n = base_pos.shape[0]
    quat = yaw_quat_wxyz(0.0)
    block = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    cfg = _build_obs_cfg(obs_cfg, enabled=True)
    out = obs.build_observation(
        num_obs=cfg["num_obs"],
        num_envs=n,
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        base_lin_acc=torch.zeros(n, 3),
        last_actions=torch.zeros(n, 2),
        base_pos=base_pos,
        base_quat=quat,
        obs_cfg=cfg,
        step_state=ss,
        device=DEVICE,
        opponent_block=block,
    )
    assert out.shape == (n, 390)
    assert torch.allclose(out[:, 384:], block)


def test_build_observation_sentinel_when_block_none(real_modules, obs_cfg):
    obs = real_modules.observations
    cl, wl, wr = make_straight_track(length=60.0, n=240)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _ego_step_state(real_modules, ts, [10.0, 0.0])
    n = base_pos.shape[0]
    cfg = _build_obs_cfg(obs_cfg, enabled=True)
    out = obs.build_observation(
        num_obs=cfg["num_obs"],
        num_envs=n,
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        base_lin_acc=torch.zeros(n, 3),
        last_actions=torch.zeros(n, 2),
        base_pos=base_pos,
        base_quat=yaw_quat_wxyz(0.0),
        obs_cfg=cfg,
        step_state=ss,
        device=DEVICE,
        opponent_block=None,
    )
    assert out.shape == (n, 390)
    assert torch.allclose(out[:, 384:], torch.zeros(n, 6))


def test_build_observation_unchanged_when_disabled(real_modules, obs_cfg):
    obs = real_modules.observations
    cl, wl, wr = make_straight_track(length=60.0, n=240)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _ego_step_state(real_modules, ts, [10.0, 0.0])
    n = base_pos.shape[0]
    cfg = _build_obs_cfg(obs_cfg, enabled=False)
    out = obs.build_observation(
        num_obs=384,
        num_envs=n,
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        base_lin_acc=torch.zeros(n, 3),
        last_actions=torch.zeros(n, 2),
        base_pos=base_pos,
        base_quat=yaw_quat_wxyz(0.0),
        obs_cfg=cfg,
        step_state=ss,
        device=DEVICE,
    )
    assert out.shape == (n, 384)

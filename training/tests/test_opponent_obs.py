"""Known-answer tests for the 1v1 opponent-relative observation block.

The block (`observations.obs_opponent`) is pure geometry, so every value has a
closed form. These tests pin the ego-frame rotation convention, the signed
track-gap wrap, the presence sentinel, and the symmetric self/other property used
to build the opponent's own egocentric observation. No Genesis sim is started.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import (
    build_track_state,
    make_straight_track,
    yaw_quat_wxyz,
)

DEVICE = torch.device("cpu")


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
    return {"enable_opponent_obs": True, "opponent_obs_dim": 7}


# --- relative-position geometry ----------------------------------------------
def test_opponent_directly_ahead(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([5.0, 0.0], 0.0, [0.0, 0.0], 5.0, 0.0, 100.0)
    blk = obs_opponent(me, other, _opp_cfg())[0]
    assert blk[0].item() == pytest.approx(5.0, abs=1e-5)
    assert blk[1].item() == pytest.approx(0.0, abs=1e-5)
    assert blk[6].item() == 1.0  # presence


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


# --- presence sentinel --------------------------------------------------------
def test_presence_sentinel_zeros_block(real_modules):
    obs_opponent = real_modules.observations.obs_opponent
    me = _agent([0.0, 0.0], 0.0, [0.0, 0.0], 0.0, 0.0, 100.0)
    other = _agent([5.0, 2.0], 0.0, [3.0, 1.0], 5.0, 0.4, 100.0)
    present = torch.zeros(1)
    blk = obs_opponent(me, other, _opp_cfg(), present=present)[0]
    assert torch.allclose(blk, torch.zeros(7), atol=0.0)


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
    assert blk.shape == (n, 7)
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
    return base_pos, ss


def _build_obs_cfg(obs_cfg, *, enabled, k=7):
    cfg = dict(obs_cfg)
    cfg["enable_opponent_obs"] = enabled
    cfg["opponent_obs_dim"] = k
    cfg["num_obs"] = 380 + k if enabled else 380
    return cfg


def test_build_observation_appends_block_when_enabled(real_modules, obs_cfg):
    obs = real_modules.observations
    cl, wl, wr = make_straight_track(length=60.0, n=240)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _ego_step_state(real_modules, ts, [10.0, 0.0])
    n = base_pos.shape[0]
    quat = yaw_quat_wxyz(0.0)
    block = torch.arange(7, dtype=torch.float32).reshape(1, 7)

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
    assert out.shape == (n, 387)
    assert torch.allclose(out[:, 380:], block)


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
    assert out.shape == (n, 387)
    assert torch.allclose(out[:, 380:], torch.zeros(n, 7))


def test_build_observation_unchanged_when_disabled(real_modules, obs_cfg):
    obs = real_modules.observations
    cl, wl, wr = make_straight_track(length=60.0, n=240)
    ts = build_track_state(real_modules.utils, cl, wl, wr)
    base_pos, ss = _ego_step_state(real_modules, ts, [10.0, 0.0])
    n = base_pos.shape[0]
    cfg = _build_obs_cfg(obs_cfg, enabled=False)
    out = obs.build_observation(
        num_obs=380,
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
    assert out.shape == (n, 380)

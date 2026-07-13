"""Pure-Python tests for the vendored actor + checkpoint loading."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.policy_model import (
    SquashedGaussianMLPActor,
    load_actor,
    load_obs_norm,
)


def _make_actor():
    return SquashedGaussianMLPActor(
        obs_dim=ifc.NUM_OBS,
        act_dim=ifc.NUM_ACTIONS,
        hidden_sizes=ifc.HIDDEN_LAYERS,
        activation=nn.ReLU,
        act_limit=ifc.ACT_LIMIT,
    )


def _ckpt_meta(**extra):
    meta = {
        "step": 10,
        "policy_format_version": ifc.POLICY_FORMAT_VERSION,
        "longitudinal_mode": "force",
        "control_hz": ifc.CONTROL_HZ,
    }
    meta.update(extra)
    return meta


def test_forward_shape_and_range():
    actor = _make_actor()
    actor.eval()
    obs = torch.zeros(4, ifc.NUM_OBS)
    with torch.no_grad():
        action, logp = actor(obs, deterministic=True, with_logprob=False)
    assert action.shape == (4, ifc.NUM_ACTIONS)
    assert logp is None
    assert bool((action.abs() <= ifc.ACT_LIMIT + 1e-5).all())


def test_load_checkpoint_actor_key():
    actor = _make_actor()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({**_ckpt_meta(), "actor": actor.state_dict()}, path)
        loaded = load_actor(
            checkpoint_path=path,
            obs_dim=ifc.NUM_OBS,
            act_dim=ifc.NUM_ACTIONS,
            hidden_sizes=ifc.HIDDEN_LAYERS,
            act_limit=ifc.ACT_LIMIT,
            state_dict_key="actor",
            device=torch.device("cpu"),
        )
    obs = torch.randn(2, ifc.NUM_OBS)
    with torch.no_grad():
        a0, _ = actor(obs, deterministic=True, with_logprob=False)
        a1, _ = loaded(obs, deterministic=True, with_logprob=False)
    assert torch.allclose(a0, a1, atol=1e-6)


def test_reject_raw_state_dict():
    actor = _make_actor()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "raw.pt")
        torch.save(actor.state_dict(), path)
        with pytest.raises(ValueError, match="policy_format_version"):
            load_actor(
                checkpoint_path=path,
                obs_dim=ifc.NUM_OBS,
                act_dim=ifc.NUM_ACTIONS,
                hidden_sizes=ifc.HIDDEN_LAYERS,
                act_limit=ifc.ACT_LIMIT,
                state_dict_key="actor",
                device=torch.device("cpu"),
            )


def test_reject_old_policy_format():
    actor = _make_actor()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        torch.save(
            {
                "step": 1,
                "actor": actor.state_dict(),
                "policy_format_version": 1,
                "longitudinal_mode": "speed",
            },
            path,
        )
        with pytest.raises(ValueError, match="policy_format_version"):
            load_actor(
                checkpoint_path=path,
                obs_dim=ifc.NUM_OBS,
                act_dim=ifc.NUM_ACTIONS,
                hidden_sizes=ifc.HIDDEN_LAYERS,
                act_limit=ifc.ACT_LIMIT,
                state_dict_key="actor",
                device=torch.device("cpu"),
            )


def test_load_obs_norm_applies_training_standardization():
    mean = torch.arange(ifc.NUM_OBS, dtype=torch.float32)
    var = torch.full((ifc.NUM_OBS,), 4.0, dtype=torch.float32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(
            {
                **_ckpt_meta(step=1),
                "actor": _make_actor().state_dict(),
                "obs_norm": {"mean": mean, "var": var, "count": 1000.0},
            },
            path,
        )
        norm = load_obs_norm(
            checkpoint_path=path,
            device=torch.device("cpu"),
            eps=ifc.OBS_NORM_EPS,
            clip=ifc.OBS_NORM_CLIP,
        )
    assert norm is not None
    obs = mean.clone().unsqueeze(0) + 2.0  # one std above the saved mean
    out = norm.normalize(obs)
    # (x - mean) / sqrt(var + eps) = 2 / sqrt(4) = 1.0, within the clip.
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_load_obs_norm_rejects_raw():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "raw.pt")
        torch.save(_make_actor().state_dict(), path)
        with pytest.raises(ValueError, match="policy_format_version"):
            load_obs_norm(
                checkpoint_path=path,
                device=torch.device("cpu"),
                eps=ifc.OBS_NORM_EPS,
                clip=ifc.OBS_NORM_CLIP,
            )

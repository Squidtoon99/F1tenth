"""Genesis-free tests for PolicyOpponent.load_snapshot hot-swap."""

from __future__ import annotations

import torch
import torch.nn as nn

from f1tenth_env.opponents import OpponentContext, PolicyOpponent
from qrsac import SquashedGaussianMLPActor

DEVICE = torch.device("cpu")
OBS_DIM = 387
ACT_DIM = 2


def _make_actor(seed: int) -> SquashedGaussianMLPActor:
    torch.manual_seed(seed)
    return SquashedGaussianMLPActor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(DEVICE)


def _reference_action(
    actor: SquashedGaussianMLPActor,
    obs: torch.Tensor,
    obs_mean: torch.Tensor | None,
    obs_var: torch.Tensor | None,
    norm_eps: float = 1e-8,
    norm_clip: float = 10.0,
) -> torch.Tensor:
    x = obs.to(DEVICE, dtype=torch.float32)
    if obs_mean is not None and obs_var is not None:
        x = (x - obs_mean) / torch.sqrt(obs_var + norm_eps)
        x = torch.clamp(x, -norm_clip, norm_clip)
    with torch.no_grad():
        action, _ = actor(x, deterministic=True, with_logprob=False)
    return torch.clamp(action, -1.0, 1.0)


def _ctx(obs: torch.Tensor) -> OpponentContext:
    return OpponentContext(
        step_state={},
        opp_pos=torch.zeros(1, 3),
        opp_vel=torch.zeros(1, 3),
        opp_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        opp_last_actions=torch.zeros(1, ACT_DIM),
        env_cfg={},
        device=DEVICE,
        opp_obs=obs,
    )


def test_load_snapshot_matches_actor_a():
    actor_a = _make_actor(1)
    actor_b = _make_actor(2)
    obs = torch.randn(3, OBS_DIM)

    opponent = PolicyOpponent(actor=actor_a, device=DEVICE)
    out_a = opponent.act(_ctx(obs))
    expected_a = _reference_action(actor_a, obs, None, None)
    assert torch.allclose(out_a, expected_a, atol=1e-6)


def test_load_snapshot_switches_to_actor_b():
    actor_a = _make_actor(10)
    actor_b = _make_actor(20)
    obs = torch.randn(4, OBS_DIM)

    opponent = PolicyOpponent(actor=actor_a, device=DEVICE)
    opponent.load_snapshot(actor_b.state_dict(), torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

    for key, param in opponent.actor.state_dict().items():
        assert torch.equal(param, actor_b.state_dict()[key])

    out_b = opponent.act(_ctx(obs))
    expected_b = _reference_action(
        actor_b, obs, torch.zeros(OBS_DIM), torch.ones(OBS_DIM)
    )
    assert torch.allclose(out_b, expected_b, atol=1e-6)


def test_load_snapshot_applies_obs_norm():
    actor = _make_actor(3)
    obs = torch.randn(2, OBS_DIM)
    mean = torch.linspace(-0.5, 0.5, OBS_DIM)
    var = torch.linspace(0.5, 1.5, OBS_DIM)

    opponent = PolicyOpponent(
        actor=actor,
        device=DEVICE,
        obs_mean=torch.zeros(OBS_DIM),
        obs_var=torch.ones(OBS_DIM),
    )
    opponent.load_snapshot(actor.state_dict(), mean, var)

    out = opponent.act(_ctx(obs))
    expected = _reference_action(actor, obs, mean, var)
    assert torch.allclose(out, expected, atol=1e-6)

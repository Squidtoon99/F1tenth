"""Genesis-free tests for MixedOpponentController routing, sampling, and refresh."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from f1tenth_env.opponents import MixedOpponentController, OpponentContext, PolicyOpponent
from qrsac import SquashedGaussianMLPActor

DEVICE = torch.device("cpu")
OBS_DIM = 387
ACT_DIM = 2


class _ConstController:
    """Stub opponent that returns a constant action and records reset masks."""

    requires_observation = False

    def __init__(self, value: float):
        self.value = value
        self.reset_masks: list[torch.Tensor] = []

    def act(self, ctx: OpponentContext) -> torch.Tensor:
        n = ctx.opp_pos.shape[0]
        return torch.full((n, ACT_DIM), self.value, dtype=torch.float32)

    def reset(self, mask: torch.Tensor) -> None:
        self.reset_masks.append(mask.clone())


def _ctx(num_envs: int) -> OpponentContext:
    return OpponentContext(
        step_state={},
        opp_pos=torch.zeros(num_envs, 3),
        opp_vel=torch.zeros(num_envs, 3),
        opp_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs),
        opp_last_actions=torch.zeros(num_envs, ACT_DIM),
        env_cfg={},
        device=DEVICE,
        opp_obs=torch.zeros(num_envs, OBS_DIM),
    )


def _make_actor(seed: int) -> SquashedGaussianMLPActor:
    torch.manual_seed(seed)
    return SquashedGaussianMLPActor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(DEVICE)


def test_act_routes_per_row_by_mode_buf():
    scripted = _ConstController(0.25)
    policy = _ConstController(-0.75)
    ctrl = MixedOpponentController(scripted, policy, scripted_weight=0.5, policy_weight=0.5)

    ctrl.mode_buf = torch.tensor([True, False, True, False])  # True -> policy
    out = ctrl.act(_ctx(4))

    assert torch.equal(out[:, 0], torch.tensor([-0.75, 0.25, -0.75, 0.25]))


def test_reset_sampling_matches_weights():
    scripted = _ConstController(0.0)
    policy = _ConstController(1.0)
    ctrl = MixedOpponentController(scripted, policy, scripted_weight=0.2, policy_weight=0.8)

    torch.manual_seed(0)
    n = 20_000
    ctrl.reset(torch.ones(n, dtype=torch.bool))
    policy_frac = ctrl.mode_buf.float().mean().item()

    assert policy_frac == pytest.approx(0.8, abs=0.02)


def test_reset_only_touches_masked_rows():
    scripted = _ConstController(0.0)
    policy = _ConstController(1.0)
    ctrl = MixedOpponentController(scripted, policy, scripted_weight=0.5, policy_weight=0.5)

    ctrl.mode_buf = torch.tensor([True, True, False, False])
    mask = torch.tensor([False, True, False, True])
    ctrl.reset(mask)

    # Unmasked rows (0 and 2) keep their prior mode.
    assert bool(ctrl.mode_buf[0]) is True
    assert bool(ctrl.mode_buf[2]) is False
    # Inner controllers receive the reset mask too.
    assert torch.equal(scripted.reset_masks[-1], mask)
    assert torch.equal(policy.reset_masks[-1], mask)


def test_invalid_weights_raise():
    scripted = _ConstController(0.0)
    policy = _ConstController(1.0)
    with pytest.raises(ValueError):
        MixedOpponentController(scripted, policy, scripted_weight=0.0, policy_weight=0.0)


def test_load_snapshot_forwards_to_inner_policy():
    actor_a = _make_actor(1)
    actor_b = _make_actor(2)
    policy = PolicyOpponent(actor=actor_a, device=DEVICE)
    scripted = _ConstController(0.0)
    ctrl = MixedOpponentController(scripted, policy, scripted_weight=0.5, policy_weight=0.5)

    ctrl.load_snapshot(actor_b.state_dict(), torch.zeros(OBS_DIM), torch.ones(OBS_DIM))

    for key, param in ctrl.policy.actor.state_dict().items():
        assert torch.equal(param, actor_b.state_dict()[key])

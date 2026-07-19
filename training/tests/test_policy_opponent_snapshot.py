"""Genesis-free tests for PolicyOpponent.load_snapshot hot-swap."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from f1tenth_env.opponents import OpponentContext, PolicyOpponent
from qrsac import Models, QRSACTrainer, QuantileCritic, SquashedGaussianMLPActor

DEVICE = torch.device("cpu")
OBS_DIM = 390
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


def test_load_snapshot_rejects_mismatched_obs_dim():
    actor = _make_actor(4)
    opponent = PolicyOpponent(actor=actor, device=DEVICE)
    torch.manual_seed(5)
    wrong = SquashedGaussianMLPActor(
        obs_dim=OBS_DIM + 1,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    )
    with pytest.raises(ValueError, match="expected"):
        opponent.load_snapshot(
            wrong.state_dict(), torch.zeros(OBS_DIM + 1), torch.ones(OBS_DIM + 1)
        )
    with pytest.raises(ValueError, match="expected actor obs dim"):
        opponent.load_snapshot(
            actor.state_dict(), torch.zeros(OBS_DIM + 1), torch.ones(OBS_DIM + 1)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_compiled_opponent_reload_accepts_plain_state_dict():
    """After torch.compile, snapshot keys stay unprefixed (``_orig_mod`` load)."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    actor_a = SquashedGaussianMLPActor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(device)
    actor_b = SquashedGaussianMLPActor(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(device)
    opponent = PolicyOpponent(actor=actor_a, device=device)
    zeros = torch.zeros(OBS_DIM, device=device)
    ones = torch.ones(OBS_DIM, device=device)
    opponent.load_snapshot(actor_a.state_dict(), zeros, ones)
    assert opponent._actor_compiled
    opponent.load_snapshot(actor_b.state_dict(), zeros, ones)
    obs = torch.randn(2, OBS_DIM, device=device)
    out = opponent.act_observation(obs)
    assert out.shape == (2, ACT_DIM)
    assert torch.isfinite(out).all()
    assert not out.is_inference()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_reduce_overhead_opponent_warmup_allows_trainer_updates():
    """Self-play opponent CUDA graphs must not poison QRSAC reduce-overhead.

    Regression for: RuntimeError: Inplace update to inference tensor outside
    InferenceMode (first trainer.update after PolicyOpponent reduce-overhead
    warmup under inference_mode).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    actor_dim, critic_dim = 1093, 390
    batch_size, num_envs = 1024, 512

    actor = SquashedGaussianMLPActor(
        actor_dim, ACT_DIM, [256, 256], nn.ReLU, 1.0
    ).to(device)
    critic = QuantileCritic(critic_dim, ACT_DIM, [256, 256], 32).to(device)
    models = Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )
    trainer = QRSACTrainer(
        models,
        device,
        n_step=7,
        alpha=0.01,
        compile=True,
        compile_mode="reduce-overhead",
    )

    opp_actor = SquashedGaussianMLPActor(
        actor_dim, ACT_DIM, [256, 256], nn.ReLU, 1.0
    ).to(device)
    opp_actor.load_state_dict(
        {key: value.detach().clone() for key, value in actor.state_dict().items()}
    )
    opponent = PolicyOpponent(actor=opp_actor, device=device)
    zeros = torch.zeros(actor_dim, device=device)
    ones = torch.ones(actor_dim, device=device)
    opponent.load_snapshot(opp_actor.state_dict(), zeros, ones)
    assert opponent._actor_compiled

    obs = torch.randn(num_envs, actor_dim, device=device)
    for _ in range(20):
        torch.compiler.cudagraph_mark_step_begin()
        action = opponent.act_observation(obs)
        assert not action.is_inference()

    batch = {
        "actor_obs": torch.randn(batch_size, actor_dim, device=device),
        "critic_obs": torch.randn(batch_size, critic_dim, device=device),
        "action": torch.rand(batch_size, ACT_DIM, device=device) * 2.0 - 1.0,
        "reward": torch.randn(batch_size, device=device),
        "next_actor_obs": torch.randn(batch_size, actor_dim, device=device),
        "next_critic_obs": torch.randn(batch_size, critic_dim, device=device),
        "done": torch.zeros(batch_size, device=device),
    }
    losses = trainer.update(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    losses = trainer.update(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)

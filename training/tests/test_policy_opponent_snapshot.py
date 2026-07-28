"""Genesis-free tests for PolicyOpponent.load_snapshot hot-swap (GRU actor)."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.opponents import OpponentContext, PolicyOpponent, make_opponent
from f1tenth_env.sensors import ACTOR_OBS_DIM
from f1tenth_policy.layout import ACTOR_ARCHITECTURE_NAME
from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from standalone_trainer import (
    actor_architecture_from_module,
    architectures_match,
    build_env_cfg,
    save_policy_artifact,
    ObsNormalizer,
)

DEVICE = torch.device("cpu")
OBS_DIM = ACTOR_OBS_DIM
CRITIC_DIM = int(DEFAULT_CONFIG["obs"]["num_obs"])
ACT_DIM = 2


def _make_actor(seed: int, obs_dim: int = OBS_DIM) -> nn.Module:
    torch.manual_seed(seed)
    return make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
        obs_dim=obs_dim,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=32,
    ).to(DEVICE)


def _reference_action(
    actor: nn.Module,
    obs: torch.Tensor,
    obs_mean: torch.Tensor | None,
    obs_var: torch.Tensor | None,
    norm_eps: float = 1e-8,
    norm_clip: float = 10.0,
) -> torch.Tensor:
    x = obs.to(DEVICE, dtype=torch.float32)
    if obs_mean is not None and obs_var is not None:
        x = (x - obs_mean.to(DEVICE)) / torch.sqrt(obs_var.to(DEVICE) + norm_eps)
        x = torch.clamp(x, -norm_clip, norm_clip)
    with torch.no_grad():
        hidden = actor.initial_hidden(x.shape[0], device=DEVICE, dtype=torch.float32)
        action, _, _ = actor.step(
            x, hidden, reset_mask=None, deterministic=True, with_logprob=False
        )
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
    assert torch.allclose(out_a, expected_a, atol=1e-5)


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
    assert torch.allclose(out_b, expected_b, atol=1e-5)


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
    assert torch.allclose(out, expected, atol=1e-5)


def test_load_snapshot_rejects_mismatched_obs_dim():
    actor = _make_actor(4)
    opponent = PolicyOpponent(actor=actor, device=DEVICE)
    with pytest.raises(ValueError, match="expected actor obs dim"):
        opponent.load_snapshot(
            actor.state_dict(), torch.zeros(OBS_DIM + 1), torch.ones(OBS_DIM + 1)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_compiled_opponent_reload_accepts_plain_state_dict():
    """After torch.compile, snapshot keys stay unprefixed (``_orig_mod`` load)."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    actor_a = make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(device)
    actor_b = make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
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
    """Opponent CUDA graphs must not poison QRSAC reduce-overhead updates."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    actor_dim, critic_dim = OBS_DIM, CRITIC_DIM
    batch_size, num_envs = 64, 32

    actor = make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
        obs_dim=actor_dim,
        act_dim=ACT_DIM,
        hidden_sizes=[64, 64],
        activation=nn.ReLU,
        act_limit=1.0,
    ).to(device)
    critic = QuantileCritic(critic_dim, ACT_DIM, [64, 64], 8).to(device)
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

    opp_actor = make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
        obs_dim=actor_dim,
        act_dim=ACT_DIM,
        hidden_sizes=[64, 64],
        activation=nn.ReLU,
        act_limit=1.0,
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
    for _ in range(5):
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


def test_make_policy_opponent_gru_matches_snapshot(tmp_path):
    """Policy opponent factory must honor lidar_cnn_gru for fixed champions."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["actor_type"] = ACTOR_ARCHITECTURE_NAME
    cfg["model"]["actor_hidden_layers"] = [64, 64]
    cfg["model"]["lidar_pool_bins"] = 32
    cfg["env"]["opponent_strategy"] = "policy"

    env_cfg = build_env_cfg(cfg)
    obs_cfg = cfg["obs"]
    opponent = make_opponent(env_cfg, obs_cfg, DEVICE)
    assert isinstance(opponent, PolicyOpponent)
    assert opponent.actor_architecture["name"] == ACTOR_ARCHITECTURE_NAME
    assert opponent.obs_dim == ACTOR_OBS_DIM

    torch.manual_seed(7)
    gru_actor = make_actor(
        actor_type=ACTOR_ARCHITECTURE_NAME,
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[64, 64],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=32,
    )
    snap_arch = actor_architecture_from_module(gru_actor)
    assert architectures_match(opponent.actor_architecture, snap_arch)

    mean = torch.zeros(ACTOR_OBS_DIM)
    var = torch.ones(ACTOR_OBS_DIM)
    opponent.load_snapshot(
        gru_actor.state_dict(),
        mean,
        var,
        actor_architecture=snap_arch,
    )

    models = Models(
        actor=gru_actor,
        critic1=QuantileCritic(CRITIC_DIM, ACT_DIM, [64, 64], 4),
        critic2=QuantileCritic(CRITIC_DIM, ACT_DIM, [64, 64], 4),
        critic1_target=QuantileCritic(CRITIC_DIM, ACT_DIM, [64, 64], 4),
        critic2_target=QuantileCritic(CRITIC_DIM, ACT_DIM, [64, 64], 4),
    )
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, DEVICE)
    ckpt = save_policy_artifact(models, 100, tmp_path, normalizer, cfg)
    bootstrap = make_opponent(
        {**env_cfg, "opponent_ckpt": str(ckpt)},
        obs_cfg,
        DEVICE,
    )
    assert isinstance(bootstrap, PolicyOpponent)
    assert bootstrap.actor_architecture["name"] == ACTOR_ARCHITECTURE_NAME
    assert architectures_match(bootstrap.actor_architecture, snap_arch)

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from f1tenth_policy.actor import LOG_STD_MAX, LOG_STD_MIN
from ppo import PPOTrainer, annealed_entropy_coef
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import (
    ObsNormalizer,
    build_ppo_models,
    validate_algorithm_config,
    validate_config_patch,
)

CRITIC_DIM = 24
ACT_DIM = 2
_GAUSSIAN_OFFSET = 0.5 * (1.0 + math.log(2.0 * math.pi))


def _small_actor(**kwargs):
    defaults = dict(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    defaults.update(kwargs)
    return make_actor(**defaults)


def _force_log_std(actor, value):
    with torch.no_grad():
        actor.log_std_layer.weight.zero_()
        actor.log_std_layer.bias.fill_(float(value))


def _tiny_cfg(**ppo_overrides):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["model"]["actor_hidden_layers"] = [32, 32]
    cfg["model"]["critic_hidden_layers"] = [32, 32]
    cfg["model"]["lidar_pool_bins"] = 16
    cfg["model"]["lidar_aug_enabled"] = False
    cfg["ppo"].update(ppo_overrides)
    return cfg


def test_annealed_entropy_coef_holds_start_when_horizon_is_zero():
    assert annealed_entropy_coef(0.01, 0.0, 0, 10_000_000) == pytest.approx(0.01)
    assert annealed_entropy_coef(0.01, 0.0, -1, 10_000_000) == pytest.approx(0.01)


def test_annealed_entropy_coef_is_linear_then_holds_end():
    start, end, horizon = 0.01, 0.0, 32_000_000
    assert annealed_entropy_coef(start, end, horizon, 0) == pytest.approx(0.01)
    assert annealed_entropy_coef(start, end, horizon, 16_000_000) == pytest.approx(
        0.005
    )
    assert annealed_entropy_coef(start, end, horizon, 32_000_000) == pytest.approx(0.0)
    assert annealed_entropy_coef(start, end, horizon, 64_000_000) == pytest.approx(0.0)


def test_default_ppo_entropy_config_is_a_no_op_anneal():
    ppo = DEFAULT_CONFIG["ppo"]
    assert ppo["entropy_coef"] == pytest.approx(0.01)
    assert ppo["entropy_coef_end"] == pytest.approx(0.01)
    assert ppo["entropy_coef_horizon_transitions"] == 0
    assert ppo["log_std_min"] == pytest.approx(LOG_STD_MIN)
    assert ppo["log_std_max"] == pytest.approx(LOG_STD_MAX)
    assert ppo["entropy_from_squashed"] is False
    assert annealed_entropy_coef(
        ppo["entropy_coef"],
        ppo["entropy_coef_end"],
        ppo["entropy_coef_horizon_transitions"],
        320_000_000,
    ) == pytest.approx(0.01)


def test_validate_accepts_zero_entropy_coef():
    cfg = _tiny_cfg(entropy_coef=0.0, entropy_coef_end=0.0)
    validate_algorithm_config(cfg, num_envs=8)
    validate_config_patch({"ppo": {"entropy_coef": 0.0, "entropy_coef_end": 0.0}})


def test_validate_rejects_negative_entropy_coef():
    cfg = _tiny_cfg(entropy_coef=-0.01)
    with pytest.raises(ValueError, match="entropy_coef"):
        validate_algorithm_config(cfg, num_envs=8)


def test_validate_rejects_inverted_log_std_bounds():
    cfg = _tiny_cfg(log_std_min=1.0, log_std_max=0.0)
    with pytest.raises(ValueError, match="log_std_min"):
        validate_algorithm_config(cfg, num_envs=8)


def test_actor_clamps_log_std_to_configured_max():
    torch.manual_seed(0)
    actor = _small_actor(log_std_max=0.0)
    _force_log_std(actor, 10.0)
    obs = torch.zeros(4, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(4)
    actions = torch.zeros(4, 1, ACT_DIM)
    logp, _, entropy = actor.evaluate_actions_sequence(
        obs.unsqueeze(1), hidden, actions
    )
    mean_log_std = float(entropy.mean().detach()) / ACT_DIM - _GAUSSIAN_OFFSET
    assert mean_log_std == pytest.approx(0.0, abs=1e-5)
    assert actor.log_std_max == 0.0
    assert float(logp.mean().detach()) < 10.0


def test_actor_default_log_std_max_still_allows_sigma_near_e2():
    torch.manual_seed(0)
    actor = _small_actor()
    _force_log_std(actor, 10.0)
    obs = torch.zeros(4, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(4)
    actions = torch.zeros(4, 1, ACT_DIM)
    _, _, entropy = actor.evaluate_actions_sequence(
        obs.unsqueeze(1), hidden, actions
    )
    mean_log_std = float(entropy.mean().detach()) / ACT_DIM - _GAUSSIAN_OFFSET
    assert mean_log_std == pytest.approx(float(LOG_STD_MAX), abs=1e-5)


def test_squashed_entropy_is_below_gaussian_when_std_is_large():
    torch.manual_seed(0)
    actor = _small_actor()
    _force_log_std(actor, 2.0)
    obs = torch.zeros(8, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(8)
    with torch.no_grad():
        action, logp, _, pre_tanh = actor.step(
            obs, hidden, return_pre_tanh=True
        )
    gaussian = actor.evaluate_actions_sequence(
        obs.unsqueeze(1),
        hidden,
        action.unsqueeze(1),
        pre_tanh_actions=pre_tanh.unsqueeze(1),
    )[2]
    squashed = -logp
    assert float(gaussian.mean().detach()) > float(squashed.mean().detach())
    assert float(gaussian.mean().detach()) == pytest.approx(
        ACT_DIM * (_GAUSSIAN_OFFSET + 2.0), abs=1e-4
    )


def _rollout_once(trainer, num_envs=4):
    actor_obs = torch.zeros(num_envs, ACTOR_OBS_DIM)
    critic_obs = torch.zeros(num_envs, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    reward = torch.zeros(num_envs)
    done = torch.zeros(num_envs, dtype=torch.bool)
    metrics = None
    for _ in range(trainer.rollout_steps):
        trainer.act(actor_obs, critic_obs)
        metrics = trainer.observe(actor_obs, critic_obs, reward, done)
    assert metrics is not None
    return metrics


def test_ppo_uses_gaussian_entropy_by_default():
    torch.manual_seed(0)
    actor = _small_actor()
    _force_log_std(actor, 0.0)
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=2,
        num_epochs=1,
        env_minibatch_size=4,
        action_clip=1.0,
        entropy_coef=0.01,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    metrics = _rollout_once(trainer)
    expected = ACT_DIM * _GAUSSIAN_OFFSET
    assert metrics["policy_entropy"] == pytest.approx(expected, abs=0.15)
    assert metrics["policy_gaussian_entropy"] == pytest.approx(
        metrics["policy_entropy"], abs=1e-5
    )
    assert metrics["policy_log_std"] == pytest.approx(0.0, abs=0.1)
    assert metrics["entropy_coef"] == pytest.approx(0.01)


def test_ppo_can_bonus_squashed_entropy_instead_of_gaussian():
    torch.manual_seed(1)
    actor = _small_actor()
    _force_log_std(actor, 2.0)
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=2,
        num_epochs=1,
        env_minibatch_size=4,
        action_clip=1.0,
        entropy_coef=0.01,
        entropy_from_squashed=True,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    metrics = _rollout_once(trainer)
    assert metrics["policy_entropy"] == pytest.approx(
        metrics["negative_log_prob"], abs=1e-5
    )
    assert metrics["policy_gaussian_entropy"] > metrics["policy_entropy"]
    assert metrics["policy_log_std"] == pytest.approx(2.0, abs=0.15)


def test_build_ppo_models_wires_entropy_and_log_std_config():
    cfg = _tiny_cfg(
        entropy_coef=0.0,
        entropy_from_squashed=True,
        log_std_max=0.0,
    )
    device = torch.device("cpu")
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, device)
    critic_norm = ObsNormalizer(int(cfg["obs"]["num_obs"]), device)
    models, trainer = build_ppo_models(
        cfg, device, actor_norm, critic_norm, num_envs=8
    )
    assert trainer.entropy_coef == pytest.approx(0.0)
    assert trainer.entropy_from_squashed is True
    assert models.actor.log_std_max == pytest.approx(0.0)
    assert models.actor.log_std_min == pytest.approx(LOG_STD_MIN)

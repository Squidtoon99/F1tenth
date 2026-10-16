from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_policy.layout import (
    ACTION_CHUNK_POLICY_FORMAT_VERSION,
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    NUM_ACTIONS,
    actor_action_dim,
    executed_plant_action,
)
from ppo import PPOTrainer
from qrsac import make_actor
from standalone_trainer import (
    build_config,
    cfg_action_horizon,
    cfg_actor_action_dim,
    collect_sensor_step,
    make_policy_network,
    parse_args,
    validate_action_chunk_config,
    validate_algorithm_config,
    validate_config_patch,
)

CRITIC_DIM = 392
HIDDEN = [32, 32]


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return build_config(args, patch=patch or {}, explicit=explicit)


def test_default_action_horizon_is_off():
    assert DEFAULT_CONFIG["env"]["action_horizon"] == 1
    assert DEFAULT_CONFIG["env"]["num_actions"] == NUM_ACTIONS == 2
    assert DEFAULT_CONFIG["policy_format_version"] == 4
    assert cfg_action_horizon(DEFAULT_CONFIG) == 1
    assert cfg_actor_action_dim(DEFAULT_CONFIG) == 2


def test_action_chunk_arm_config_turns_h4_on():
    path = Path(__file__).resolve().parents[1] / (
        "configs/austin_e2e_ppo_baseline_v6_action_chunk.json"
    )
    patch = json.loads(path.read_text(encoding="utf-8"))
    validate_config_patch(patch)
    cfg = _resolve(["--algorithm", "ppo"], patch=patch)
    assert cfg["env"]["action_horizon"] == 4
    assert cfg["env"]["num_actions"] == 2
    assert cfg["policy_format_version"] == ACTION_CHUNK_POLICY_FORMAT_VERSION
    assert cfg_action_horizon(cfg) == 4
    assert cfg_actor_action_dim(cfg) == 8
    control_dt = cfg["env"]["sim_dt"] * cfg["env"]["control_interval"]
    assert control_dt == pytest.approx(1.0 / CONTROL_HZ)
    cfg["model"]["actor_hidden_layers"] = [32, 32]
    cfg["model"]["critic_hidden_layers"] = [32, 32]
    actor = make_policy_network(cfg)
    assert actor.act_dim == 8
    assert actor.action_horizon == 4
    validate_algorithm_config(cfg, num_envs=32)


def test_action_horizon_rejected_for_qrsac_and_residual():
    cfg = {
        "algorithm": "qrsac",
        "policy_format_version": 5,
        "env": {"num_actions": 2, "action_horizon": 4},
        "ppo": {"residual_base_enabled": False},
    }
    with pytest.raises(ValueError, match="requires algorithm='ppo'"):
        validate_action_chunk_config(cfg)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["residual_base_enabled"] = True
    with pytest.raises(ValueError, match="residual_base_enabled"):
        validate_action_chunk_config(cfg)
    cfg["ppo"]["residual_base_enabled"] = False
    cfg["policy_format_version"] = 4
    with pytest.raises(ValueError, match="policy_format_version"):
        validate_action_chunk_config(cfg)


def test_ppo_logprob_covers_chunk_and_plant_gets_first_tick():
    torch.manual_seed(0)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=actor_action_dim(4),
        hidden_sizes=HIDDEN,
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
    )
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=2,
        num_epochs=1,
        env_minibatch_size=4,
        action_clip=1.0,
        advantage_filter_enabled=False,
    )
    num_envs = 4
    actor_obs = torch.randn(num_envs, ACTOR_OBS_DIM)
    critic_obs = torch.zeros(num_envs, CRITIC_DIM)
    trainer.initialize(actor_obs, critic_obs)
    chunk = trainer.act(actor_obs, critic_obs)
    assert chunk.shape == (num_envs, 8)
    plant = executed_plant_action(chunk)
    assert plant.shape == (num_envs, 2)
    assert torch.equal(plant, chunk[:, :2])
    assert not torch.equal(plant, chunk[:, 2:4])
    logp = trainer._rollout.old_logp[0]
    assert logp.shape == (num_envs,)
    logp_eval, _, entropy = actor.evaluate_actions_sequence(
        actor_obs.unsqueeze(1),
        trainer._rollout.rollout_start_hidden,
        chunk.unsqueeze(1),
        reset_mask=trainer._rollout.reset_masks[0].unsqueeze(1),
        pre_tanh_actions=trainer._rollout.pre_tanh_actions[0].unsqueeze(1),
    )
    assert torch.allclose(logp, logp_eval.squeeze(1), atol=1e-5)
    assert entropy.shape == (num_envs, 1)


def test_collect_sensor_step_slices_plant_command():
    captured = {}

    class _Env:
        def step(self, actions, n_steps=None, with_sensors=False):
            captured["actions"] = actions
            n = actions.shape[0]
            raw = {
                "actor": torch.zeros(n, ACTOR_OBS_DIM),
                "frenet": torch.zeros(n, CRITIC_DIM),
            }
            return raw, torch.zeros(n), torch.zeros(n, dtype=torch.bool), {}

    chunk = torch.arange(8, dtype=torch.float32).repeat(2, 1)
    collect_sensor_step([_Env()], chunk, [2], 10)
    assert captured["actions"].shape == (2, 2)
    assert torch.equal(captured["actions"], chunk[:, :2])

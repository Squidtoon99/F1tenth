from __future__ import annotations

import json
import math
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
    apply_scenario_patch,
    build_config,
    cfg_action_horizon,
    cfg_actor_action_dim,
    collect_sensor_reset,
    collect_sensor_step,
    make_policy_network,
    parse_args,
    unpack_sensor_observations,
    validate_action_chunk_config,
    validate_algorithm_config,
    validate_config_patch,
)
from test_warp_env import _make_env

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
    assert cfg["env"]["opponent_strategy"] == "scripted"
    assert cfg["schedule"]["total_transitions"] == 256000000
    assert cfg["reward"]["reward_scales"]["tal"] == pytest.approx(0.0)
    assert cfg["env"]["reset_speed_min_mps"] == pytest.approx(0.0)
    assert cfg["env"]["reset_speed_max_mps"] == pytest.approx(5.0)
    assert cfg["env"]["track"] == "Austin"
    iv = next(s for s in cfg["scenarios"] if s["name"] == "solo_iv_2026")
    merged_iv = apply_scenario_patch(cfg, iv, str(path.parent))
    assert merged_iv["env"]["reset_speed_min_mps"] == pytest.approx(1.0)
    assert merged_iv["env"]["reset_speed_max_mps"] == pytest.approx(3.0)
    assert cfg["ppo"].get("caps_spatial_coef", 0.0) == pytest.approx(0.0)
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
    env = _make_env(num_envs=2)
    try:
        collect_sensor_reset([env])
        chunk = torch.arange(8, dtype=torch.float32).repeat(2, 1)
        actor_obs, critic_obs, reward, done, extras = collect_sensor_step(
            [env], chunk, [2], env.control_interval
        )
        del extras
        assert env.last_actions.shape == (2, 2)
        assert torch.allclose(env.last_actions.cpu(), chunk[:, :2])
        assert actor_obs.shape == (2, ACTOR_OBS_DIM)
        assert critic_obs.shape[0] == 2
        assert reward.shape == (2,)
        assert done.shape == (2,)
    finally:
        env.close()


def test_receding_horizon_four_ticks_are_not_tail_playback():
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
    env = _make_env(num_envs=2)
    try:
        raw, _ = env.reset(seed=0, with_sensors=True)
        actor_obs, _critic = unpack_sensor_observations(raw)
        hidden = actor.initial_hidden(
            2, device=actor_obs.device, dtype=torch.float32
        )
        chunks = []
        prefixes = []
        for tick in range(4):
            chunk, _, hidden = actor.step(
                actor_obs, hidden, deterministic=True
            )
            plant = executed_plant_action(chunk)
            assert plant.shape == (2, 2)
            chunks.append(chunk.detach().clone())
            prefixes.append(plant.detach().clone())
            raw, _reward, _done, _extras = env.step(
                plant, n_steps=env.control_interval, with_sensors=True
            )
            actor_obs, _critic = unpack_sensor_observations(raw)
        chunk0 = chunks[0]
        for tick in range(4):
            assert torch.equal(prefixes[tick], chunks[tick][:, :2])
            if tick == 0:
                continue
            tail = chunk0[:, 2 * tick : 2 * tick + 2]
            assert not torch.equal(prefixes[tick], tail)
    finally:
        env.close()


def test_cpu_smoke_one_update_few_env_steps():
    path = Path(__file__).resolve().parents[1] / (
        "configs/austin_e2e_ppo_baseline_v6_action_chunk.json"
    )
    patch = json.loads(path.read_text(encoding="utf-8"))
    cfg = _resolve(["--algorithm", "ppo"], patch=patch)
    assert cfg["env"]["action_horizon"] == 4
    assert cfg["env"]["num_actions"] == 2
    control_dt = cfg["env"]["sim_dt"] * cfg["env"]["control_interval"]
    assert control_dt == pytest.approx(0.05)
    cfg["model"]["actor_hidden_layers"] = [32, 32]
    cfg["model"]["critic_hidden_layers"] = [32, 32]
    actor = make_policy_network(cfg)
    assert actor.act_dim == 8
    env = _make_env(num_envs=2)
    try:
        assert env.dt * env.control_interval == pytest.approx(0.05)
        raw, _ = env.reset(seed=0, with_sensors=True)
        actor_obs, critic_obs = unpack_sensor_observations(raw)
        trainer = PPOTrainer(
            actor,
            int(critic_obs.shape[-1]),
            device="cpu",
            value_hidden_sizes=(32, 32),
            rollout_steps=4,
            num_epochs=1,
            env_minibatch_size=2,
            action_clip=1.0,
            advantage_filter_enabled=False,
        )
        trainer.initialize(actor_obs, critic_obs)
        metrics = None
        for _ in range(4):
            chunk = trainer.act(actor_obs, critic_obs)
            assert chunk.shape == (2, 8)
            plant = executed_plant_action(chunk)
            assert plant.shape == (2, 2)
            raw, reward, done, _extras = env.step(
                plant, n_steps=env.control_interval, with_sensors=True
            )
            actor_obs, critic_obs = unpack_sensor_observations(raw)
            metrics = trainer.observe(
                actor_obs, critic_obs, reward.to(torch.float32), done
            )
        assert metrics is not None
        assert math.isfinite(float(metrics["policy_loss"]))
        assert math.isfinite(float(metrics["value_loss"]))
        clip = float(metrics["policy_clip_fraction"])
        assert 0.0 <= clip <= 1.0
    finally:
        env.close()

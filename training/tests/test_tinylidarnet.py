from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, control_dt_s
from f1tenth_env import F1tenthEnv
from f1tenth_policy.actor import TinyLidarNetEncoder, actor_from_architecture
from f1tenth_policy.artifact import (
    build_sensor_artifact_payload,
    load_sensor_artifact,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    NUM_ACTIONS,
    TINYLIDARNET_ARCHITECTURE_NAME,
    executed_plant_action,
)
from f1tenth_policy.normalizer import ObsNormalizer
from ppo import PPOTrainer
from standalone_trainer import (
    build_config,
    build_env_cfg,
    make_policy_network,
    parse_args,
    unpack_sensor_observations,
    validate_algorithm_config,
    validate_config_patch,
    validate_model_architecture,
)

HIDDEN = [32, 32]
ARM_JSON = Path(__file__).resolve().parents[1] / (
    "configs/austin_e2e_ppo_baseline_v6_tinylidarnet.json"
)


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return build_config(args, patch=patch or {}, explicit=explicit)


def _arm_patch():
    return json.loads(ARM_JSON.read_text(encoding="utf-8"))


def _arm_cfg(*, shrink_heads=True):
    cfg = _resolve(["--algorithm", "ppo"], patch=_arm_patch())
    if shrink_heads:
        cfg["model"]["actor_hidden_layers"] = HIDDEN
        cfg["model"]["critic_hidden_layers"] = HIDDEN
    return cfg


def _make_env(cfg: dict, num_envs: int = 2) -> F1tenthEnv:
    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": num_envs},
    )
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def test_encoder_has_no_adaptive_pool_on_raw_beams():
    encoder = TinyLidarNetEncoder(lidar_dim=LIDAR_DIM, projection_dim=256)
    assert not any(
        isinstance(mod, nn.AdaptiveAvgPool1d) for mod in encoder.modules()
    )
    lidar = torch.rand(2, LIDAR_DIM)
    features, conv = encoder.forward_features(lidar)
    assert features.shape == (2, 256)
    assert conv.shape[0] == 2
    assert conv.shape[-1] != 32
    n_params = sum(int(p.numel()) for p in encoder.parameters())
    assert 150_000 < n_params < 300_000


def test_arm_config_keeps_gru_ppo_and_20hz():
    patch = _arm_patch()
    validate_config_patch(patch)
    assert patch["schedule"]["total_transitions"] == 256_000_000
    assert patch["env"]["opponent_strategy"] == "scripted"
    assert "lidar_pool_bins" not in patch.get("model", {})
    cfg = _resolve(["--algorithm", "ppo"], patch=patch)
    assert cfg["algorithm"] == "ppo"
    assert cfg["model"]["actor_type"] == TINYLIDARNET_ARCHITECTURE_NAME
    assert cfg["schedule"]["total_transitions"] == 256_000_000
    assert cfg["env"]["opponent_strategy"] == "scripted"
    dt = control_dt_s(cfg["env"])
    assert dt == pytest.approx(1.0 / CONTROL_HZ)
    assert cfg["env"]["action_horizon"] == 1
    assert cfg["env"]["num_actions"] == NUM_ACTIONS == 2
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    cfg["model"]["critic_hidden_layers"] = HIDDEN
    validate_model_architecture(cfg)
    validate_algorithm_config(cfg, num_envs=32)
    actor = make_policy_network(cfg)
    assert actor.actor_architecture["name"] == TINYLIDARNET_ARCHITECTURE_NAME
    assert actor.actor_architecture["pool"] == "none"
    assert "pool_bins" not in actor.actor_architecture
    assert actor.gru_hidden_dim == GRU_HIDDEN_DIM
    assert isinstance(actor.encoder, TinyLidarNetEncoder)
    assert isinstance(actor.gru, nn.GRU)
    obs = torch.rand(3, ACTOR_OBS_DIM)
    hidden = actor.initial_hidden(3)
    action, logp, next_h = actor.step(obs, hidden, with_logprob=True)
    assert action.shape == (3, NUM_ACTIONS)
    assert next_h.shape == hidden.shape
    restored = actor_from_architecture(actor.actor_architecture)
    restored.load_state_dict(actor.state_dict())
    assert restored.actor_architecture["encoder"] == "tinylidarnet"


def test_tinylidarnet_skips_pool_bins_validation():
    cfg = _arm_cfg()
    cfg["model"].pop("lidar_pool_bins", None)
    validate_model_architecture(cfg)
    actor = make_policy_network(cfg)
    assert actor.actor_architecture["pool"] == "none"
    assert "pool_bins" not in actor.actor_architecture
    assert not any(
        isinstance(mod, nn.AdaptiveAvgPool1d) for mod in actor.modules()
    )


def test_default_lidar_cnn_gru_still_requires_allowed_pool():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    cfg["model"]["critic_hidden_layers"] = HIDDEN
    assert cfg["model"]["actor_type"] == ACTOR_ARCHITECTURE_NAME
    missing = copy.deepcopy(cfg)
    missing["model"].pop("lidar_pool_bins", None)
    with pytest.raises(ValueError, match="lidar_pool_bins"):
        validate_model_architecture(missing)
    none = copy.deepcopy(cfg)
    none["model"]["lidar_pool_bins"] = None
    with pytest.raises(ValueError, match="lidar_pool_bins"):
        validate_model_architecture(none)
    bad = copy.deepcopy(cfg)
    bad["model"]["lidar_pool_bins"] = 8
    with pytest.raises(ValueError, match="lidar_pool_bins"):
        validate_model_architecture(bad)


def test_default_encoder_still_pools_to_32():
    assert DEFAULT_CONFIG["model"]["actor_type"] == ACTOR_ARCHITECTURE_NAME
    assert DEFAULT_CONFIG["model"]["lidar_pool_bins"] == 32
    cfg = dict(DEFAULT_CONFIG)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    actor = make_policy_network(cfg)
    assert any(
        isinstance(mod, nn.AdaptiveAvgPool1d) for mod in actor.encoder.modules()
    )
    assert actor.encoder.pool_bins == 32


def test_cpu_smoke_one_update_few_env_steps(warp_runtime):
    del warp_runtime
    cfg = _arm_cfg()
    assert cfg["model"]["actor_type"] == TINYLIDARNET_ARCHITECTURE_NAME
    assert control_dt_s(cfg["env"]) == pytest.approx(0.05)
    assert cfg["env"]["action_horizon"] == 1
    assert cfg["env"]["num_actions"] == 2
    actor = make_policy_network(cfg)
    assert isinstance(actor.encoder, TinyLidarNetEncoder)
    assert isinstance(actor.gru, nn.GRU)
    assert actor.act_dim == 2
    assert not any(
        isinstance(mod, nn.AdaptiveAvgPool1d) for mod in actor.modules()
    )
    num_envs = 2
    trainer = PPOTrainer(
        actor,
        int(cfg["obs"]["num_obs"]),
        device="cpu",
        value_hidden_sizes=tuple(HIDDEN),
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=num_envs,
        action_clip=1.0,
        advantage_filter_enabled=False,
        lidar_aug_enabled=False,
    )
    env = _make_env(cfg, num_envs=num_envs)
    try:
        raw, _extras = env.reset(seed=0, with_sensors=True)
        actor_obs, critic_obs = unpack_sensor_observations(raw)
        trainer.initialize(actor_obs, critic_obs)
        metrics = None
        for _ in range(trainer.rollout_steps):
            actions = trainer.act(actor_obs, critic_obs)
            assert actions.shape == (num_envs, NUM_ACTIONS)
            raw, reward, done, _extras = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
            actor_obs, critic_obs = unpack_sensor_observations(raw)
            metrics = trainer.observe(actor_obs, critic_obs, reward, done)
        assert metrics is not None
        assert math.isfinite(float(metrics["policy_loss"]))
        assert math.isfinite(float(metrics["value_loss"]))
    finally:
        env.close()


def test_sensor_runtime_loads_tinylidarnet(tmp_path):
    cfg = _arm_cfg()
    actor = make_policy_network(cfg)
    actor.eval()
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, device=torch.device("cpu"))
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=actor.actor_architecture,
        env_transitions=0,
        f_drive_max=26.5,
        f_brake_max=23.1,
        control_hz=CONTROL_HZ,
        training_protocol={"algorithm": "ppo"},
    )
    ckpt = tmp_path / "tinylidarnet.pt"
    torch.save(payload, ckpt)
    loaded = load_sensor_artifact(str(ckpt))
    validate_sensor_policy_artifact(
        loaded, expected_architecture=actor.actor_architecture
    )
    restored = actor_from_architecture(loaded["actor_architecture"])
    restored.load_state_dict(loaded["actor"])
    restored.eval()
    assert isinstance(restored.encoder, TinyLidarNetEncoder)
    assert restored.actor_architecture["pool"] == "none"
    assert "pool_bins" not in restored.actor_architecture
    obs = torch.rand(1, ACTOR_OBS_DIM)
    hidden = restored.initial_hidden(1)
    action, _, _ = restored.step(
        obs, hidden, deterministic=True, with_logprob=False
    )
    plant = executed_plant_action(action)
    assert action.shape == (1, NUM_ACTIONS)
    assert plant.shape == (1, 2)

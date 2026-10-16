from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG, control_dt_s
from f1tenth_policy.actor import TinyLidarNetEncoder, actor_from_architecture
from f1tenth_policy.layout import (
    ACTOR_ARCHITECTURE_NAME,
    ACTOR_OBS_DIM,
    CONTROL_HZ,
    GRU_HIDDEN_DIM,
    LIDAR_DIM,
    NUM_ACTIONS,
    TINYLIDARNET_ARCHITECTURE_NAME,
)
from standalone_trainer import (
    build_config,
    make_policy_network,
    parse_args,
    validate_algorithm_config,
    validate_config_patch,
    validate_model_architecture,
)

HIDDEN = [32, 32]


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return build_config(args, patch=patch or {}, explicit=explicit)


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
    path = Path(__file__).resolve().parents[1] / (
        "configs/austin_e2e_ppo_baseline_v6_tinylidarnet.json"
    )
    patch = json.loads(path.read_text(encoding="utf-8"))
    validate_config_patch(patch)
    cfg = _resolve(["--algorithm", "ppo"], patch=patch)
    assert cfg["algorithm"] == "ppo"
    assert cfg["model"]["actor_type"] == TINYLIDARNET_ARCHITECTURE_NAME
    dt = control_dt_s(cfg["env"])
    assert dt == pytest.approx(1.0 / CONTROL_HZ)
    assert cfg["env"]["action_horizon"] == 1
    cfg["model"]["actor_hidden_layers"] = HIDDEN
    cfg["model"]["critic_hidden_layers"] = HIDDEN
    validate_model_architecture(cfg)
    validate_algorithm_config(cfg, num_envs=32)
    actor = make_policy_network(cfg)
    assert actor.actor_architecture["name"] == TINYLIDARNET_ARCHITECTURE_NAME
    assert actor.actor_architecture["pool"] == "none"
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

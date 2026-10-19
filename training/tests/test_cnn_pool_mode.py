from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_policy import (
    ACTOR_OBS_DIM,
    CNN_CONV_OUT_BINS,
    LIDAR_DIM,
    NUM_ACTIONS,
    ObsNormalizer,
    actor_from_architecture,
    build_sensor_artifact_payload,
    make_actor,
    normalize_lidar_pool_mode,
    validate_sensor_policy_artifact,
)
from f1tenth_policy.actor import CNN_CONV_CHANNELS, LidarCNNEncoder
from standalone_trainer import (
    build_env_cfg,
    make_policy_network,
    validate_model_architecture,
)


@pytest.mark.parametrize(
    ("raw_mode", "canonical"),
    [
        (None, "adaptive_avg_1d"),
        ("adaptive_avg_1d", "adaptive_avg_1d"),
        ("avg", "adaptive_avg_1d"),
        ("AVG", "adaptive_avg_1d"),
        ("adaptive_max_1d", "adaptive_max_1d"),
        ("max", "adaptive_max_1d"),
        ("none", "none"),
        ("NONE", "none"),
    ],
)
def test_normalize_lidar_pool_mode(raw_mode, canonical):
    assert normalize_lidar_pool_mode(raw_mode) == canonical


def test_normalize_lidar_pool_mode_rejects_invalid():
    with pytest.raises(ValueError, match="lidar_pool_mode"):
        normalize_lidar_pool_mode("l2_pool")
    with pytest.raises(ValueError, match="lidar_pool_mode"):
        normalize_lidar_pool_mode(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("pool_mode", "pool_bins", "proj_dim", "expected_pool_cls", "expected_bins"),
    [
        ("avg", 32, 256, nn.AdaptiveAvgPool1d, 32),
        ("adaptive_avg_1d", 64, 512, nn.AdaptiveAvgPool1d, 64),
        ("max", 32, 256, nn.AdaptiveMaxPool1d, 32),
        ("adaptive_max_1d", 64, 512, nn.AdaptiveMaxPool1d, 64),
        ("none", 91, 512, nn.Identity, CNN_CONV_OUT_BINS),
        ("none", 32, 256, nn.Identity, CNN_CONV_OUT_BINS),
    ],
)
def test_lidar_cnn_encoder_pool_modes_and_512d_projection(
    pool_mode,
    pool_bins,
    proj_dim,
    expected_pool_cls,
    expected_bins,
):
    encoder = LidarCNNEncoder(
        lidar_dim=LIDAR_DIM,
        pool_bins=pool_bins,
        projection_dim=proj_dim,
        pool_mode=pool_mode,
    )
    assert isinstance(encoder.pool, expected_pool_cls)
    assert encoder.pool_bins == expected_bins
    assert encoder.projection.in_features == CNN_CONV_CHANNELS[-1] * expected_bins
    assert encoder.projection.out_features == proj_dim

    lidar = torch.randn(4, LIDAR_DIM)
    features, pooled = encoder.forward_features(lidar)
    assert features.shape == (4, proj_dim)
    assert pooled.shape == (4, CNN_CONV_CHANNELS[-1], expected_bins)


@pytest.mark.parametrize(
    ("pool_mode", "pool_bins", "proj_dim", "canonical_mode", "expected_bins"),
    [
        ("adaptive_avg_1d", 32, 256, "adaptive_avg_1d", 32),
        ("max", 64, 512, "adaptive_max_1d", 64),
        ("none", 91, 512, "none", 91),
    ],
)
def test_actor_architecture_and_artifact_roundtrip(
    pool_mode,
    pool_bins,
    proj_dim,
    canonical_mode,
    expected_bins,
):
    torch.manual_seed(11)
    actor = make_actor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=NUM_ACTIONS,
        hidden_sizes=[64, 64],
        activation=nn.ReLU,
        lidar_pool_bins=pool_bins,
        lidar_pool_mode=pool_mode,
        lidar_projection_dim=proj_dim,
    )
    arch = actor.actor_architecture
    assert arch["pool"] == canonical_mode
    assert arch["pool_bins"] == expected_bins
    assert arch["projection_dim"] == proj_dim
    assert actor.gru.input_size == proj_dim + (ACTOR_OBS_DIM - LIDAR_DIM)

    restored = actor_from_architecture(arch)
    restored.load_state_dict(actor.state_dict(), strict=True)
    assert restored.actor_architecture == arch
    assert type(restored.encoder.pool) is type(actor.encoder.pool)

    obs = torch.randn(3, ACTOR_OBS_DIM)
    h0 = actor.initial_hidden(3)
    act1, _, h1 = actor.step(obs, h0, deterministic=True)
    act2, _, h2 = restored.step(obs, h0, deterministic=True)
    assert torch.allclose(act1, act2, atol=1e-6)
    assert torch.allclose(h1, h2, atol=1e-6)

    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(obs)
    payload = build_sensor_artifact_payload(
        actor_state_dict=actor.state_dict(),
        obs_norm=normalizer.state_dict(),
        actor_architecture=arch,
        env_transitions=512,
    )
    validate_sensor_policy_artifact(payload, expected_architecture=arch)


def test_standalone_trainer_make_policy_network_with_pool_mode():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["actor_hidden_layers"] = [64, 64]
    cfg["model"]["critic_hidden_layers"] = [64, 64]
    cfg["model"]["lidar_pool_bins"] = 64
    cfg["model"]["lidar_pool_mode"] = "max"
    cfg["model"]["lidar_projection_dim"] = 512

    validate_model_architecture(cfg)
    actor = make_policy_network(cfg)
    assert isinstance(actor.encoder.pool, nn.AdaptiveMaxPool1d)
    assert actor.encoder.pool_bins == 64
    assert actor.encoder.projection_dim == 512
    assert actor.actor_architecture["pool"] == "adaptive_max_1d"

    env_cfg = build_env_cfg(cfg)
    assert env_cfg["lidar_pool_bins"] == 64
    assert env_cfg["lidar_pool_mode"] == "adaptive_max_1d"
    assert env_cfg["lidar_projection_dim"] == 512

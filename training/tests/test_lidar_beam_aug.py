"""Deterministic tests for active LiDAR beam-shift augmentation."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from f1tenth_env.sensors import ACTOR_LIDAR_DIM, ACTOR_OBS_DIM, ACTOR_PROPRIO_DIM
from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import (
    ObsNormalizer,
    augment_actor_lidar_beam_shift,
    validate_model_architecture,
)
from config import DEFAULT_CONFIG

BURN_IN = 4
TRAIN_LEN = 4
N_STEP = 3
SEQ_LEN = BURN_IN + TRAIN_LEN + N_STEP
CRITIC_DIM = 24
ACT_DIM = 2


def _window(num_seq: int = 3, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(num_seq, SEQ_LEN, ACTOR_OBS_DIM)


def test_zero_max_shift_is_identity():
    obs = _window()
    out = augment_actor_lidar_beam_shift(obs, max_shift_beams=0)
    assert out is obs


def test_nonzero_shift_changes_lidar_preserves_proprio_and_sequence():
    obs = _window(num_seq=2, seed=1)
    # Distinct beams so a shift is observable.
    obs[..., :ACTOR_LIDAR_DIM] = torch.arange(ACTOR_LIDAR_DIM, dtype=torch.float32).view(
        1, 1, ACTOR_LIDAR_DIM
    )
    obs[..., ACTOR_LIDAR_DIM:] = torch.arange(
        ACTOR_PROPRIO_DIM, dtype=torch.float32
    ).view(1, 1, ACTOR_PROPRIO_DIM)
    shifts = torch.tensor([2, -3], dtype=torch.long)
    out = augment_actor_lidar_beam_shift(obs, max_shift_beams=4, shifts=shifts)

    assert out.shape == obs.shape
    # Proprio untouched.
    assert torch.equal(out[..., ACTOR_LIDAR_DIM:], obs[..., ACTOR_LIDAR_DIM:])
    # Same shift applied to every frame in each sequence (temporal consistency).
    assert torch.equal(out[0, 0, :ACTOR_LIDAR_DIM], out[0, -1, :ACTOR_LIDAR_DIM])
    assert torch.equal(out[1, 0, :ACTOR_LIDAR_DIM], out[1, -1, :ACTOR_LIDAR_DIM])
    # Positive shift: index 0 reads former beam 2.
    assert float(out[0, 0, 0]) == 2.0
    assert float(out[0, 0, 1]) == 3.0
    # LiDAR changed for both sequences.
    assert not torch.equal(out[0, :, :ACTOR_LIDAR_DIM], obs[0, :, :ACTOR_LIDAR_DIM])
    assert not torch.equal(out[1, :, :ACTOR_LIDAR_DIM], obs[1, :, :ACTOR_LIDAR_DIM])


def test_reflected_padding_at_edges():
    obs = torch.zeros(1, 2, ACTOR_OBS_DIM)
    # beams: 0,1,2,..., so left reflect of pad=2 is [2,1 | 0,1,2,...]
    obs[..., :ACTOR_LIDAR_DIM] = torch.arange(ACTOR_LIDAR_DIM, dtype=torch.float32)
    out = augment_actor_lidar_beam_shift(
        obs, max_shift_beams=2, shifts=torch.tensor([-2])
    )
    # start index in padded = max_shift + shift = 0 → first values are reflected.
    padded = F.pad(obs[..., :ACTOR_LIDAR_DIM], (2, 2), mode="reflect")
    assert torch.equal(out[0, 0, :ACTOR_LIDAR_DIM], padded[0, 0, 0:ACTOR_LIDAR_DIM])
    assert float(out[0, 0, 0]) == 2.0
    assert float(out[0, 0, 1]) == 1.0
    assert float(out[0, 0, 2]) == 0.0


def test_seeded_shifts_are_deterministic():
    obs = _window(num_seq=8, seed=2)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = augment_actor_lidar_beam_shift(obs, max_shift_beams=4, generator=g1)
    b = augment_actor_lidar_beam_shift(obs, max_shift_beams=4, generator=g2)
    assert torch.equal(a, b)


def test_bootstrap_slice_shares_augmented_normalized_window():
    obs = _window(num_seq=2, seed=3)
    shifts = torch.tensor([1, -1], dtype=torch.long)
    aug = augment_actor_lidar_beam_shift(obs, max_shift_beams=4, shifts=shifts)
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(aug.reshape(-1, ACTOR_OBS_DIM))
    normalized = normalizer.normalize(aug)
    boot_lo = BURN_IN + N_STEP
    boot_hi = BURN_IN + TRAIN_LEN + N_STEP
    bootstrap = normalized[:, boot_lo:boot_hi]
    assert torch.equal(bootstrap, normalized[:, boot_lo:boot_hi])
    assert torch.isfinite(bootstrap).all()


def test_sequence_update_with_aug_has_finite_gradients():
    torch.manual_seed(0)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=GRU_HIDDEN_DIM,
    )
    critic = QuantileCritic(CRITIC_DIM, ACT_DIM, [32, 32], 4)
    models = Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )
    trainer = QRSACTrainer(
        models,
        torch.device("cpu"),
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        n_step=N_STEP,
        alpha=0.01,
    )
    num_seq = 2
    actor_obs = _window(num_seq=num_seq, seed=4)
    actor_obs = augment_actor_lidar_beam_shift(
        actor_obs, max_shift_beams=4, shifts=torch.tensor([2, -2])
    )
    normalizer = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    normalizer.update(actor_obs.reshape(-1, ACTOR_OBS_DIM))
    normalized = normalizer.normalize(actor_obs)
    boot_lo = BURN_IN + N_STEP
    boot_hi = BURN_IN + TRAIN_LEN + N_STEP
    batch = {
        "actor_obs": normalized,
        "critic_obs": torch.randn(num_seq, SEQ_LEN, CRITIC_DIM),
        "action": torch.rand(num_seq, SEQ_LEN, ACT_DIM) * 2 - 1,
        "reset": torch.zeros(num_seq, SEQ_LEN),
        "hidden": torch.zeros(num_seq, GRU_HIDDEN_DIM),
        "n_step_reward": torch.randn(num_seq, TRAIN_LEN),
        "n_step_done": torch.zeros(num_seq, TRAIN_LEN),
        "bootstrap_actor_obs": normalized[:, boot_lo:boot_hi],
        "bootstrap_critic_obs": torch.randn(num_seq, TRAIN_LEN, CRITIC_DIM),
    }
    losses = trainer.update_from_sequences(batch)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.critic_loss)
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in models.actor.parameters()
        if p.requires_grad
    )


def test_config_validation_rejects_bad_lidar_aug():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["lidar_aug_max_shift_beams"] = -1
    with pytest.raises(ValueError, match="lidar_aug_max_shift_beams"):
        validate_model_architecture(cfg)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["lidar_aug_max_shift_beams"] = ACTOR_LIDAR_DIM
    with pytest.raises(ValueError, match="must be <"):
        validate_model_architecture(cfg)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["model"]["lidar_aug_enabled"] = "yes"
    with pytest.raises(ValueError, match="lidar_aug_enabled"):
        validate_model_architecture(cfg)


def test_default_config_enables_lidar_aug_with_shift_4():
    assert DEFAULT_CONFIG["model"]["lidar_aug_enabled"] is True
    assert DEFAULT_CONFIG["model"]["lidar_aug_max_shift_beams"] == 4

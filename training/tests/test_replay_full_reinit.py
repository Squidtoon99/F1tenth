"""Real-module tests for Lee et al. 2025 replay-full network reinitialization."""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from config import DEFAULT_CONFIG
from f1tenth_env.sensors import ACTOR_OBS_DIM
from f1tenth_policy import ObsNormalizer
from fixed_opponents import ChampionEntry, FixedChampionManager
from qrsac import Models, QRSACTrainer, QuantileCritic, make_actor
from qrsac.replay import TrajectoryReplayBuffer
from standalone_trainer import (
    initial_training_protocol_state,
    maybe_replay_full_reinit,
    save_policy_artifact,
)

CRITIC_DIM = 24
ACT_DIM = 2
BURN_IN = 4
TRAIN_LEN = 4
N_STEP = 3
HIDDEN_DIM = 32


def _gru_models(seed: int = 0) -> Models:
    torch.manual_seed(seed)
    actor = make_actor(
        actor_type="lidar_cnn_gru",
        obs_dim=ACTOR_OBS_DIM,
        act_dim=ACT_DIM,
        hidden_sizes=[32, 32],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_pool_bins=16,
        gru_hidden_dim=HIDDEN_DIM,
    )
    critic = QuantileCritic(CRITIC_DIM, ACT_DIM, [32, 32], 4)
    return Models(
        actor=actor,
        critic1=critic,
        critic2=copy.deepcopy(critic),
        critic1_target=copy.deepcopy(critic),
        critic2_target=copy.deepcopy(critic),
    )


def _sequence_batch(num_seq: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    seq_len = BURN_IN + TRAIN_LEN + N_STEP
    actor_obs = torch.randn(num_seq, seq_len, ACTOR_OBS_DIM)
    critic_obs = torch.randn(num_seq, seq_len, CRITIC_DIM)
    boot_lo = BURN_IN + N_STEP
    boot_hi = BURN_IN + TRAIN_LEN + N_STEP
    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "action": torch.rand(num_seq, seq_len, ACT_DIM) * 2 - 1,
        "reset": torch.zeros(num_seq, seq_len),
        "hidden": torch.randn(num_seq, HIDDEN_DIM),
        "n_step_reward": torch.randn(num_seq, TRAIN_LEN),
        "n_step_done": torch.zeros(num_seq, TRAIN_LEN),
        "bootstrap_actor_obs": actor_obs[:, boot_lo:boot_hi],
        "bootstrap_critic_obs": critic_obs[:, boot_lo:boot_hi],
    }


def test_default_learning_rates_match_paper_and_actor():
    models = _gru_models()
    trainer = QRSACTrainer(
        models,
        torch.device("cpu"),
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        n_step=N_STEP,
    )
    assert trainer.actor_lr == 2.5e-5
    assert trainer.critic_lr == 2.5e-5
    assert DEFAULT_CONFIG["model"]["actor_lr"] == 2.5e-5


def test_maybe_replay_full_reinit_preserves_champion_and_replay(tmp_path):
    num_envs = 2
    capacity = num_envs * 64
    buffer = TrajectoryReplayBuffer(
        capacity=capacity,
        actor_obs_dim=ACTOR_OBS_DIM,
        critic_obs_dim=CRITIC_DIM,
        act_dim=ACT_DIM,
        num_envs=num_envs,
        device=torch.device("cpu"),
        n_step=N_STEP,
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        checkpoint_interval=4,
        hidden_dim=HIDDEN_DIM,
    )
    models = _gru_models(seed=4)
    trainer = QRSACTrainer(
        models,
        torch.device("cpu"),
        burn_in=BURN_IN,
        train_len=TRAIN_LEN,
        n_step=N_STEP,
    )
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    actor_norm.update(torch.randn(8, ACTOR_OBS_DIM))
    mean_a = actor_norm.mean.clone()

    hidden = torch.randn(num_envs, HIDDEN_DIM)
    for _ in range(buffer.steps_per_env):
        buffer.add(
            torch.randn(num_envs, ACTOR_OBS_DIM),
            torch.randn(num_envs, CRITIC_DIM),
            torch.zeros(num_envs, ACT_DIM),
            torch.ones(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            hidden=hidden,
        )
    replay_actor_before = buffer.actor_obs.clone()

    entry = ChampionEntry(
        checkpoint="/tmp/champ.pt",
        weight=1.0,
        transitions=999,
        actor={k: v.detach().cpu().clone() for k, v in models.actor.state_dict().items()},
        mean=actor_norm.mean.detach().cpu().clone(),
        var=actor_norm.var.detach().cpu().clone(),
        actor_architecture=dict(models.actor.actor_architecture),
    )
    mgr = FixedChampionManager([entry], torch.tensor([1.0]), seed=42)

    class _Env:
        def __init__(self, n_envs: int):
            self.num_envs = n_envs
            self.refreshes = 0
            self.device = torch.device("cpu")

        def refresh_opponent_pool(self, entries):
            self.refreshes += 1

        def set_opponent_resample_callback(self, _cb):
            return None

        def assign_opponent_policies(self, _mask, _indices):
            return None

    env = _Env(num_envs)
    learner_hidden = torch.randn(num_envs, HIDDEN_DIM)
    protocol = initial_training_protocol_state()
    did = maybe_replay_full_reinit(
        enabled=True,
        protocol_state=protocol,
        buffer=buffer,
        trainer=trainer,
        models=models,
        actor_normalizer=actor_norm,
        env_transitions=10_000,
        learner_hidden=learner_hidden,
        champion_mgr=mgr,
        env=env,
        log=logging.getLogger("test_reinit"),
    )
    assert did is True
    assert protocol["replay_full_reinit_done"] is True
    assert torch.count_nonzero(learner_hidden).item() == 0
    assert env.refreshes == 1
    assert torch.equal(buffer.actor_obs, replay_actor_before)
    assert torch.equal(actor_norm.mean, mean_a)

    did_again = maybe_replay_full_reinit(
        enabled=True,
        protocol_state=protocol,
        buffer=buffer,
        trainer=trainer,
        models=models,
        actor_normalizer=actor_norm,
        env_transitions=20_000,
        learner_hidden=learner_hidden,
        champion_mgr=mgr,
        env=env,
        log=logging.getLogger("test_reinit"),
    )
    assert did_again is False

    path = save_policy_artifact(
        models, 10_000, tmp_path, actor_norm, DEFAULT_CONFIG, protocol_state=protocol
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["training_protocol"]["replay_full_reinit_done"] is True


def test_cold_start_protocol_state_is_unset():
    state = initial_training_protocol_state()
    assert state == {
        "algorithm": "qrsac",
        "replay_full_reinit_done": False,
        "replay_full_reinit_count": 0,
        "replay_full_reinit_transitions": None,
    }

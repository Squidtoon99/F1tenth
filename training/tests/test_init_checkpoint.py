"""--init-checkpoint loads weights and normalizers, not optimizer or step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from f1tenth_env.sensors import ACTOR_OBS_DIM
from ppo import PPOTrainer
from qrsac import make_actor
from qrsac.spinningup.core import GRU_HIDDEN_DIM
from standalone_trainer import (
    ActorModels,
    ObsNormalizer,
    load_init_checkpoint,
    parse_args,
    save_resume_state,
)

CRITIC_DIM = 24
ACT_DIM = 2
_TRAINING_ROOT = Path(__file__).resolve().parents[1]


def _bundle(seed: int):
    torch.manual_seed(seed)
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
    trainer = PPOTrainer(
        actor,
        CRITIC_DIM,
        device="cpu",
        value_hidden_sizes=(32, 32),
        rollout_steps=4,
        num_epochs=1,
        env_minibatch_size=8,
        action_clip=1.0,
        compile=False,
        advantage_filter_enabled=False,
    )
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, torch.device("cpu"))
    critic_norm = ObsNormalizer(CRITIC_DIM, torch.device("cpu"))
    models = ActorModels(actor=actor)
    return models, trainer, actor_norm, critic_norm


def _has_adam_moments(optimizer) -> bool:
    return any("exp_avg" in state for state in optimizer.state.values())


def _fill_optimizer(trainer, actor_norm, critic_norm):
    actor_obs = torch.randn(8, ACTOR_OBS_DIM)
    critic_obs = torch.randn(8, CRITIC_DIM)
    actor_norm.update(actor_obs)
    critic_norm.update(critic_obs)
    loss = trainer.value_critic(critic_obs).sum()
    loss = loss + trainer.actor(actor_obs, deterministic=True, with_logprob=False)[0].sum()
    loss.backward()
    trainer.actor_optimizer.step()
    trainer.value_optimizer.step()
    trainer.actor_optimizer.zero_grad()
    trainer.value_optimizer.zero_grad()


def _clone_params(module):
    return [p.detach().cpu().clone() for p in module.parameters()]


def test_init_checkpoint_loads_weights_and_normalizers_not_optimizer(tmp_path):
    src_models, src_trainer, src_actor_norm, src_critic_norm = _bundle(0)
    _fill_optimizer(src_trainer, src_actor_norm, src_critic_norm)
    assert _has_adam_moments(src_trainer.actor_optimizer)
    assert _has_adam_moments(src_trainer.value_optimizer)
    assert src_actor_norm.count > src_actor_norm.eps
    assert src_critic_norm.count > src_critic_norm.eps

    ckpt = tmp_path / "resume_state.pt"
    save_resume_state(
        ckpt,
        algorithm="ppo",
        models=src_models,
        trainer=src_trainer,
        actor_normalizer=src_actor_norm,
        critic_normalizer=src_critic_norm,
        env_transitions=131072,
        gradient_updates=2,
        vector_ticks=512,
        sampled_replay_rows=0,
        device=torch.device("cpu"),
    )
    src_actor = _clone_params(src_models.actor)
    src_critic = _clone_params(src_trainer.value_critic)
    src_actor_mean = src_actor_norm.mean.clone()
    src_actor_var = src_actor_norm.var.clone()
    src_actor_count = float(src_actor_norm.count)
    src_critic_mean = src_critic_norm.mean.clone()
    src_critic_var = src_critic_norm.var.clone()
    src_critic_count = float(src_critic_norm.count)

    dst_models, dst_trainer, dst_actor_norm, dst_critic_norm = _bundle(1)
    assert not torch.equal(src_actor[0], _clone_params(dst_models.actor)[0])
    assert not _has_adam_moments(dst_trainer.actor_optimizer)

    load_init_checkpoint(
        str(ckpt),
        algorithm="ppo",
        models=dst_models,
        trainer=dst_trainer,
        actor_normalizer=dst_actor_norm,
        critic_normalizer=dst_critic_norm,
        device=torch.device("cpu"),
    )

    for got, expected in zip(_clone_params(dst_models.actor), src_actor):
        assert torch.equal(got, expected)
    for got, expected in zip(_clone_params(dst_trainer.value_critic), src_critic):
        assert torch.equal(got, expected)
    assert torch.equal(dst_actor_norm.mean, src_actor_mean)
    assert torch.equal(dst_actor_norm.var, src_actor_var)
    assert dst_actor_norm.count == pytest.approx(src_actor_count)
    assert torch.equal(dst_critic_norm.mean, src_critic_mean)
    assert torch.equal(dst_critic_norm.var, src_critic_var)
    assert dst_critic_norm.count == pytest.approx(src_critic_count)
    assert not _has_adam_moments(dst_trainer.actor_optimizer)
    assert not _has_adam_moments(dst_trainer.value_optimizer)


def test_init_checkpoint_rejects_policy_artifact(tmp_path):
    path = tmp_path / "policy_1.pt"
    torch.save({"format": "sensor_policy", "actor": {}}, path)
    models, trainer, actor_norm, critic_norm = _bundle(2)
    with pytest.raises(ValueError, match="resume_state"):
        load_init_checkpoint(
            str(path),
            algorithm="ppo",
            models=models,
            trainer=trainer,
            actor_normalizer=actor_norm,
            critic_normalizer=critic_norm,
            device=torch.device("cpu"),
        )


def test_parse_args_accepts_init_checkpoint():
    args, _explicit = parse_args(["--init-checkpoint", "/tmp/resume_state.pt"])
    assert args.init_checkpoint == "/tmp/resume_state.pt"
    assert args.init_ckpt is None
    assert args.resume_ckpt is None


def test_finetune_config_matches_courtyard_multi_scenario():
    root = _TRAINING_ROOT / "configs"
    base = json.loads((root / "courtyard_2_e2e_ppo_multi_scenario.json").read_text())
    finetune = json.loads((root / "courtyard_2_e2e_ppo_finetune.json").read_text())
    assert finetune == base

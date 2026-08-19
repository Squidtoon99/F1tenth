"""Tests for PPO training resume checkpoints."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from f1tenth_policy import ObsNormalizer, SquashedGaussianLidarGRUActor
from config import DEFAULT_CONFIG
from ppo import PPOTrainer
from standalone_trainer import (
    load_init_ckpt,
    load_resume_state,
    prune_policy_artifacts,
    resume_state_path,
    save_policy_artifact,
    save_resume_state,
    initial_training_protocol_state,
)

LIDAR_DIM = 64
PROPRIO_DIM = 3
ACTOR_OBS_DIM = LIDAR_DIM + PROPRIO_DIM
CRITIC_OBS_DIM = 7


def _actor(seed=0):
    torch.manual_seed(seed)
    return SquashedGaussianLidarGRUActor(
        obs_dim=ACTOR_OBS_DIM,
        act_dim=2,
        hidden_sizes=[8],
        activation=nn.ReLU,
        act_limit=1.0,
        lidar_dim=LIDAR_DIM,
        proprio_dim=PROPRIO_DIM,
        pool_bins=16,
        projection_dim=8,
        gru_hidden_dim=8,
    )


def _cfg():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["algorithm"] = "ppo"
    cfg["obs"]["num_actor_obs"] = ACTOR_OBS_DIM
    cfg["obs"]["num_obs"] = CRITIC_OBS_DIM
    return cfg


def _build(device, num_envs=4):
    actor = _actor()
    actor_norm = ObsNormalizer(ACTOR_OBS_DIM, device)
    critic_norm = ObsNormalizer(CRITIC_OBS_DIM, device)
    trainer = PPOTrainer(
        actor,
        CRITIC_OBS_DIM,
        actor_norm,
        critic_norm,
        device,
        value_hidden_sizes=(16,),
        rollout_steps=4,
        num_epochs=2,
        env_minibatch_size=2,
        actor_lr=1e-3,
        value_lr=1e-3,
        max_grad_norm=1e-6,
        action_clip=1.0,
        compile=False,
    )
    models = SimpleNamespace(actor=actor)
    return _cfg(), models, trainer, actor_norm, critic_norm


def _run_rollout(trainer, device, num_envs=4, steps=4):
    actor_obs = torch.randn(num_envs, ACTOR_OBS_DIM, device=device)
    critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, device=device)
    trainer.initialize(actor_obs, critic_obs)
    for _ in range(steps):
        trainer.act(actor_obs, critic_obs)
        next_actor = torch.randn(num_envs, ACTOR_OBS_DIM, device=device)
        next_critic = torch.randn(num_envs, CRITIC_OBS_DIM, device=device)
        reward = torch.randn(num_envs, device=device)
        done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        trainer.observe(next_actor, next_critic, reward, done)
        actor_obs, critic_obs = next_actor, next_critic


def test_save_and_load_resume_restores_optimizer_and_counters(tmp_path):
    device = torch.device("cpu")
    cfg, models, trainer, actor_norm, critic_norm = _build(device)
    _run_rollout(trainer, device)
    actor_before = {k: v.clone() for k, v in models.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in trainer.value_critic.state_dict().items()}
    actor_opt_before = trainer.actor_optimizer.state_dict()["state"][0]["exp_avg"].clone()
    path = resume_state_path(tmp_path)
    save_resume_state(
        path,
        algorithm="ppo",
        models=models,
        trainer=trainer,
        actor_normalizer=actor_norm,
        critic_normalizer=critic_norm,
        env_transitions=8192,
        gradient_updates=17,
        vector_ticks=2048,
        sampled_replay_rows=999,
        device=device,
    )
    _, models2, trainer2, actor_norm2, critic_norm2 = _build(device)
    counters = load_resume_state(
        str(path),
        algorithm="ppo",
        models=models2,
        trainer=trainer2,
        actor_normalizer=actor_norm2,
        critic_normalizer=critic_norm2,
        device=device,
    )
    assert counters["env_transitions"] == 8192
    assert counters["gradient_updates"] == 17
    assert counters["vector_ticks"] == 2048
    assert counters["sampled_replay_rows"] == 999
    for key, tensor in actor_before.items():
        assert torch.allclose(models2.actor.state_dict()[key], tensor)
    for key, tensor in critic_before.items():
        assert torch.allclose(trainer2.value_critic.state_dict()[key], tensor)
    restored = trainer2.actor_optimizer.state_dict()["state"][0]["exp_avg"]
    assert torch.allclose(restored, actor_opt_before)


def test_atomic_resume_write_leaves_valid_checkpoint(tmp_path):
    device = torch.device("cpu")
    _, models, trainer, actor_norm, critic_norm = _build(device)
    path = resume_state_path(tmp_path)
    save_resume_state(
        path,
        algorithm="ppo",
        models=models,
        trainer=trainer,
        actor_normalizer=actor_norm,
        critic_normalizer=critic_norm,
        env_transitions=1,
        gradient_updates=0,
        vector_ticks=0,
        sampled_replay_rows=0,
        device=device,
    )
    payload = torch.load(path, weights_only=False)
    assert payload["format"] == "training_resume"
    assert not path.with_name(path.name + ".tmp").exists()


def test_warm_start_still_only_loads_actor_and_actor_norm(tmp_path):
    device = torch.device("cpu")
    cfg, models, trainer, actor_norm, critic_norm = _build(device)
    _run_rollout(trainer, device)
    critic_before = {k: v.clone() for k, v in trainer.value_critic.state_dict().items()}
    policy = save_policy_artifact(
        models,
        1234,
        tmp_path,
        actor_norm,
        cfg,
        protocol_state=initial_training_protocol_state("ppo"),
    )
    actor2 = _actor(seed=1)
    actor_norm2 = ObsNormalizer(ACTOR_OBS_DIM, device)
    critic_norm2 = ObsNormalizer(CRITIC_OBS_DIM, device)
    trainer2 = PPOTrainer(
        actor2, CRITIC_OBS_DIM, actor_norm2, critic_norm2, device,
        value_hidden_sizes=(16,), rollout_steps=4, num_epochs=2,
        env_minibatch_size=2, action_clip=1.0, compile=False,
    )
    models2 = SimpleNamespace(actor=actor2)
    init_t = load_init_ckpt(
        models2,
        actor_norm2,
        str(policy),
        device,
        expected_layout_version=int(cfg["obs"]["actor_layout_version"]),
        expected_steering_delta_max_rad=float(cfg["env"]["steering_delta_max_rad"]),
    )
    assert init_t == 1234
    for key, tensor in critic_before.items():
        assert not torch.allclose(trainer2.value_critic.state_dict()[key], tensor)


def test_prune_policy_artifacts_keeps_most_recent(tmp_path):
    device = torch.device("cpu")
    cfg, models, trainer, actor_norm, critic_norm = _build(device)
    for t in (100, 200, 300, 400):
        save_policy_artifact(
            models,
            t,
            tmp_path,
            actor_norm,
            cfg,
            protocol_state=initial_training_protocol_state("ppo"),
        )
    prune_policy_artifacts(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("policy_*.pt"))
    assert remaining == ["policy_300.pt", "policy_400.pt"]


def test_kill_and_resume_continues_counters(tmp_path):
    trainer_dir = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "killrun"
    run_dir.mkdir()
    resume = run_dir / "resume_state.pt"
    cfg_patch = tmp_path / "patch.json"
    cfg_patch.write_text(json.dumps({
        "algorithm": "ppo",
        "ppo": {"rollout_steps": 4, "env_minibatches": 2, "epochs": 1},
        "schedule": {
            "total_transitions": 200000,
            "log_interval_transitions": 4096,
            "export_interval_transitions": 4096,
            "eval_interval_transitions": 0,
        },
    }))
    base = [
        sys.executable,
        "standalone_trainer.py",
        "--algorithm", "ppo",
        "--num-envs", "32",
        "--device", "cpu",
        "--no-wandb",
        "--no-compile",
        "--seed", "0",
        "--config", str(cfg_patch),
        "--run-dir", str(run_dir),
        "--run-id", "killrun",
        "--reuse-run-dir",
    ]
    proc = subprocess.Popen(base, cwd=str(trainer_dir), env=os.environ.copy())
    deadline = time.time() + 300
    while time.time() < deadline:
        if resume.is_file():
            break
        time.sleep(0.5)
    assert resume.is_file(), "resume checkpoint never written"
    payload1 = torch.load(resume, weights_only=False)
    t1 = payload1["env_transitions"]
    g1 = payload1["gradient_updates"]
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=30)
    (run_dir / "run.lock").unlink(missing_ok=True)
    proc2 = subprocess.Popen(
        base + ["--resume-ckpt", str(resume)],
        cwd=str(trainer_dir),
        env=os.environ.copy(),
    )
    deadline = time.time() + 300
    payload2 = payload1
    while time.time() < deadline:
        if resume.is_file():
            payload2 = torch.load(resume, weights_only=False)
            if payload2["env_transitions"] > t1:
                break
        time.sleep(0.5)
    os.kill(proc2.pid, signal.SIGKILL)
    proc2.wait(timeout=30)
    assert payload2["env_transitions"] > t1
    assert payload2["gradient_updates"] >= g1

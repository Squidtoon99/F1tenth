"""Periodic policy artifacts must all remain on disk after a short training run."""

from __future__ import annotations

import json
import sys

from run_layout import checkpoint_dir
from standalone_trainer import main, resume_state_path


def test_short_training_loop_keeps_every_policy_checkpoint(tmp_path, monkeypatch):
    cfg_path = tmp_path / "keep_all.json"
    cfg_path.write_text(
        json.dumps(
            {
                "algorithm": "ppo",
                "model": {
                    "actor_hidden_layers": [32, 32],
                    "critic_hidden_layers": [32, 32],
                    "lidar_aug_enabled": False,
                },
                "ppo": {
                    "rollout_steps": 2,
                    "epochs": 1,
                    "env_minibatches": 1,
                },
                "schedule": {
                    "total_transitions": 16,
                    "log_interval_transitions": 1_000_000,
                    "export_interval_transitions": 4,
                    "eval_interval_transitions": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "standalone_trainer.py",
            "--config",
            str(cfg_path),
            "--algorithm",
            "ppo",
            "--num-envs",
            "2",
            "--total-transitions",
            "16",
            "--device",
            "cpu",
            "--no-compile",
            "--no-wandb",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "keep-all-ckpts",
            "--opponent",
            "none",
            "--seed",
            "0",
        ],
    )

    main()

    ckpt_dir = checkpoint_dir(run_dir)
    names = sorted(p.name for p in ckpt_dir.glob("policy_*.pt"))
    assert names == [
        "policy_4.pt",
        "policy_8.pt",
        "policy_12.pt",
        "policy_16.pt",
    ]
    for name in names:
        assert (ckpt_dir / name).stat().st_size > 0
    resume = resume_state_path(run_dir)
    assert resume.is_file()
    assert resume.stat().st_size > 0

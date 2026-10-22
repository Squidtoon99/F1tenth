"""Courtyard online PPO CLI, IRL condition pin, and installer flag smoke."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.artifacts import bake_irl_artifact, build_actor_artifact_payload
from gigaflow_f1tenth.cli import (
    COURTYARD_CENTERLINE,
    COURTYARD_CONFIG,
    build_parser,
    main,
    prepare_courtyard_atlas,
)
from gigaflow_f1tenth.config import SENSOR_OBS_DIM, load_config
from gigaflow_f1tenth.critic import TRACK_PREVIEW_DIM
from gigaflow_f1tenth.model import actor_shapes, build_actor
from gigaflow_f1tenth.trainer import run_training

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "deploy" / "scripts" / "install_irl_images.sh"
RUN_ENTRY = ROOT / "deploy" / "docker" / "run_courtyard_ppo.sh"
DOCKERFILE = ROOT / "deploy" / "docker" / "Dockerfile.courtyard_trainer"


def _baked_actor(tmp_path: Path) -> Path:
    cfg = load_config(COURTYARD_CONFIG)
    src = tmp_path / "actor.pt"
    dest = tmp_path / "baked.pt"
    torch.save(build_actor_artifact_payload(cfg, build_actor(cfg)), src)
    bake_irl_artifact(str(src), str(dest))
    return dest


def test_courtyard_config_hardcodes_irl_and_1097d():
    cfg = load_config(COURTYARD_CONFIG)
    assert cfg.agents.sensor_obs_dim == SENSOR_OBS_DIM == 1097
    assert cfg.agents.lidar_dim == 1081
    assert cfg.agents.proprio_dim == 16
    assert cfg.ablations.reward_randomization is False
    assert cfg.reward_conditioning.randomize_styles is False
    assert cfg.reward_conditioning.vmax_scale_min == cfg.reward_conditioning.vmax_scale_max == 1.0
    assert cfg.evaluation.conservative_deployment_style == "centered_high_collision"
    assert cfg.evaluation.quality_stop_enabled is False
    assert COURTYARD_CENTERLINE.is_file()


def test_courtyard_actor_has_no_track_geometry_inputs():
    cfg = load_config(COURTYARD_CONFIG)
    shapes = actor_shapes(cfg)
    assert shapes.sensor_obs_dim == 1097
    assert shapes.lidar_dim + shapes.proprio_dim == 1097
    assert TRACK_PREVIEW_DIM > 0
    assert shapes.sensor_obs_dim != TRACK_PREVIEW_DIM
    params = inspect.signature(build_actor(cfg).forward).parameters
    assert "sensor_obs" in params
    assert "private_condition" in params
    assert not any("track" in name for name in params)


def test_prepare_courtyard_atlas_is_courtyard_2_only(tmp_path):
    cfg = load_config(COURTYARD_CONFIG)
    atlas = prepare_courtyard_atlas(cfg, tmp_path / "cache")
    names = [e.name for e in atlas.manifest()]
    assert names == ["courtyard_2"]


def test_train_courtyard_cli_help(capsys):
    try:
        main(["train-courtyard", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "courtyard_2" in out or "actor_005700" in out


def test_train_courtyard_cli_has_no_resume_from():
    ns = build_parser().parse_args(["train-courtyard"])
    assert not hasattr(ns, "resume_from")
    assert ns.checkpoint.name == "actor_005700.pt"
    assert ns.run_dir.as_posix() == "/opt/f1tenth/outputs/courtyard_ppo"


def test_train_courtyard_rejects_randomized_styles(tmp_path, capsys):
    rc = main(
        [
            "train-courtyard",
            "--config",
            str(ROOT / "gigaflow" / "configs" / "smoke.yaml"),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--run-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert rc == 1
    assert "hardcoded IRL condition" in capsys.readouterr().out


def test_run_training_rejects_resume_plus_init_actor(tmp_path):
    cfg = load_config(COURTYARD_CONFIG)
    with pytest.raises(ValueError, match="init_actor_from"):
        run_training(
            cfg,
            1,
            resume_from=tmp_path / "ckpt.pt",
            init_actor_from=tmp_path / "actor.pt",
        )


def test_train_courtyard_rejects_missing_checkpoint(tmp_path, capsys):
    missing = tmp_path / "missing.pt"
    rc = main(
        [
            "train-courtyard",
            "--checkpoint",
            str(missing),
            "--run-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert rc == 1
    assert "checkpoint not found" in capsys.readouterr().out


def test_train_courtyard_smoke_from_baked_actor(tmp_path):
    ckpt = _baked_actor(tmp_path)
    rc = main(
        [
            "train-courtyard",
            "--checkpoint",
            str(ckpt),
            "--run-dir",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert rc == 0
    report = tmp_path / "out" / "train_courtyard.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["track"] == "courtyard_2"
    assert payload["sensor_obs_dim"] == 1097
    assert payload["deployment_style"] == "centered_high_collision"
    assert payload["reward_randomization"] is False
    assert payload["update_index"] >= 1
    assert (tmp_path / "out" / "actor_final.pt").is_file()


def test_install_irl_images_courtyard_trainer_flag():
    proc = subprocess.run(
        ["bash", str(INSTALLER), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--courtyard-trainer" in proc.stdout
    text = INSTALLER.read_text(encoding="utf-8")
    assert "f1tenth-courtyard-trainer" in text
    assert "/opt/f1tenth/rollback/courtyard_trainer/run_train.sh" in text
    assert "Dockerfile.gigaflow" not in text


def test_install_irl_images_courtyard_trainer_missing_tar():
    missing = "/tmp/f1tenth-courtyard-trainer-missing.tar.gz"
    proc = subprocess.run(
        ["bash", str(INSTALLER), "user@gpu-box", "-", "--courtyard-trainer", missing],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "not found" in (proc.stdout + proc.stderr)


def test_run_courtyard_ppo_entrypoint_requires_checkpoint(tmp_path):
    proc = subprocess.run(
        ["bash", str(RUN_ENTRY)],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "CHECKPOINT": str(tmp_path / "missing.pt"),
            "RUN_DIR": str(tmp_path / "out"),
            "DEVICE": "cpu",
        },
    )
    assert proc.returncode == 1
    assert "checkpoint not found" in proc.stderr


def test_courtyard_trainer_scripts_pass_bash_n():
    scripts = (
        INSTALLER,
        RUN_ENTRY,
        ROOT / "deploy" / "scripts" / "build_image.sh",
        ROOT / "deploy" / "scripts" / "snapshot.sh",
    )
    for script in scripts:
        proc = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{script}: {proc.stderr}"


def test_courtyard_dockerfile_is_separate_training_image():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "Dockerfile.gigaflow" in text
    assert "nvidia/cuda" in text
    assert "train-courtyard" in text
    assert "actor_005700.pt" in text
    assert "/opt/f1tenth/outputs/courtyard_ppo" in text
    gigaflow_df = (ROOT / "deploy" / "docker" / "Dockerfile.gigaflow").read_text(
        encoding="utf-8"
    )
    assert "train-courtyard" not in gigaflow_df
    assert "courtyard_ppo" not in gigaflow_df

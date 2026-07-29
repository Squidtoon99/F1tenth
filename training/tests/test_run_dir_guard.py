"""Tests for run directory attach guards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from run_layout import config_snapshot_path, run_lock_path, run_log_path
from standalone_trainer import (
    assert_run_dir_exclusive,
    build_config,
    config_provenance,
    fingerprint_from_snapshot,
    load_config_patch,
    parse_args,
    run_identity_fingerprint,
    write_run_lock,
)


def _noise_argv(run_dir: str) -> list[str]:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        "--config",
        f"{repo}/training/outputs/experiments/long-horizon-2b-sensors/"
        "progab-control-noise-1b-a001.json",
        "--num-envs",
        "1024",
        "--total-transitions",
        "1000000000",
        "--opponent",
        "policy",
        "--self-play",
        "--device",
        "cpu",
        "--seed",
        "42",
        "--run-id",
        "guard-test-a001",
        "--run-dir",
        run_dir,
        "--no-wandb",
    ]


def _warm_argv(run_dir: str) -> list[str]:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return [
        "--config",
        f"{repo}/training/outputs/experiments/long-horizon-2b-sensors/"
        "progab-control-a001.json",
        "--init-ckpt",
        f"{repo}/training/outputs/runs/pcplus2b-a001/checkpoints/"
        "policy_1443840000.pt",
        "--num-envs",
        "1024",
        "--total-transitions",
        "300000000",
        "--opponent",
        "policy",
        "--self-play",
        "--device",
        "cpu",
        "--seed",
        "42",
        "--run-id",
        "guard-test-a001",
        "--run-dir",
        run_dir,
        "--no-wandb",
    ]


def _write_config_snapshot(run_dir: Path, args) -> None:
    patch, patch_meta = load_config_patch(args.config)
    cfg = build_config(args, patch=patch, explicit=set())
    payload = {
        "run_id": args.run_id,
        "args": {
            "config": args.config,
            "total_transitions": args.total_transitions,
            "init_ckpt": args.init_ckpt,
            "seed": args.seed,
            "num_envs": args.num_envs,
        },
        "config": cfg,
        "config_provenance": config_provenance(patch_meta, set()),
    }
    config_snapshot_path(run_dir).write_text(json.dumps(payload), encoding="utf-8")


def test_run_dir_refuses_conflicting_config(tmp_path):
    run_dir = tmp_path / "shared-run"
    run_dir.mkdir()
    noise_args, _ = parse_args(_noise_argv(str(run_dir)))
    warm_args, _ = parse_args(_warm_argv(str(run_dir)))

    _write_config_snapshot(run_dir, noise_args)
    run_log_path(run_dir).write_text("partial run\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        assert_run_dir_exclusive(run_dir, warm_args)


def test_run_dir_refuses_live_lock(tmp_path):
    run_dir = tmp_path / "locked-run"
    run_dir.mkdir()
    noise_args, _ = parse_args(_noise_argv(str(run_dir)))
    warm_args, _ = parse_args(_warm_argv(str(run_dir)))

    write_run_lock(run_dir, noise_args)
    lock = json.loads(run_lock_path(run_dir).read_text(encoding="utf-8"))
    lock["pid"] = os.getpid()
    run_lock_path(run_dir).write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(SystemExit):
        assert_run_dir_exclusive(run_dir, warm_args)


def test_second_trainer_cannot_attach_to_live_run_dir(tmp_path):
    run_dir = tmp_path / "live-run"
    run_dir.mkdir()
    noise_args, _ = parse_args(_noise_argv(str(run_dir)))
    warm_args, _ = parse_args(_warm_argv(str(run_dir)))

    write_run_lock(run_dir, noise_args)
    lock = json.loads(run_lock_path(run_dir).read_text(encoding="utf-8"))
    lock["pid"] = os.getpid()
    run_lock_path(run_dir).write_text(json.dumps(lock), encoding="utf-8")

    _write_config_snapshot(run_dir, noise_args)
    run_log_path(run_dir).write_text(
        "[2026-07-29 15:56:21] standalone_trainer INFO: Run id: live-run\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "standalone_trainer.py",
            *_warm_argv(str(run_dir)),
            "--total-transitions",
            "1000",
            "--device",
            "cpu",
            "--no-wandb",
            "--no-compile",
        ],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "TRAINING_SKIP_DIRTY_TREE_GUARD": "1",
        },
    )
    assert proc.returncode != 0
    assert "ERROR:" in proc.stderr
    assert "refusing" in proc.stderr.lower() or "locked" in proc.stderr.lower()


def test_fingerprint_from_snapshot_matches_args():
    run_dir = "/tmp/example"
    args, _ = parse_args(_noise_argv(run_dir))
    fp_args = run_identity_fingerprint(args)
    snapshot = {"args": {"config": args.config, "total_transitions": args.total_transitions,
                         "init_ckpt": args.init_ckpt, "seed": args.seed,
                         "num_envs": args.num_envs}}
    assert fingerprint_from_snapshot(snapshot) == fp_args

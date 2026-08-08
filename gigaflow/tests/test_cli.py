"""CLI smoke tests for validate-config and prepare-tracks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from gigaflow_f1tenth.cli import main

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tracks" / "oval_centerline.csv"


def test_validate_config_cli(capsys):
    rc = main(["validate-config", "--config", str(SMOKE)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["config_version"] == 1
    assert payload["estimate_memory_bytes"] <= payload["budget_bytes"]


def test_validate_config_wandb_cli_overrides(capsys):
    rc = main(
        [
            "validate-config",
            "--config",
            str(SMOKE),
            "--dump",
            "--wandb",
            "--wandb-mode",
            "offline",
            "--wandb-project",
            "cli-override",
            "--wandb-tag",
            "t1",
            "--wandb-run-id",
            "rid",
            "--wandb-resume",
            "must",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["wandb_enabled"] is True
    assert payload["wandb_mode"] == "offline"
    wb = payload["config"]["wandb"]
    assert wb["project"] == "cli-override"
    assert wb["tags"] == ["t1"]
    assert wb["run_id"] == "rid"
    assert wb["resume"] == "must"


def test_prepare_tracks_cli_local_fixture(tmp_path, capsys):
    local = tmp_path / "local"
    local.mkdir()
    shutil.copy(FIXTURE, local / "oval_centerline.csv")
    rc = main(
        [
            "prepare-tracks",
            "--config",
            str(SMOKE),
            "--cache-dir",
            str(tmp_path),
            "--skip-download",
            "--lut-resolution",
            "0.5",
            "--edt-resolution",
            "0.25",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["num_tracks"] == 1
    assert payload["track_names"] == ["oval"]

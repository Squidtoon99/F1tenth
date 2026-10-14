"""Dry-run tests for the Discord model-release announcer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import announce_model_release as announce

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": "stdmax-v6",
                "config": {"env": {"track": "IV_2026_SIM"}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.log").write_text(
        "[2026-07-25 10:00:00] trainer INFO: transitions/s=100.0\n"
        "[2026-07-25 12:05:30] trainer INFO: transitions/s=84210.5\n",
        encoding="utf-8",
    )
    (run_dir / "follow_eval.json").write_text(
        json.dumps({"running": 6, "cars": 8}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "overlay_follow.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (run_dir / "reward.png").write_bytes(_PNG)
    (run_dir / "laps.png").write_bytes(_PNG)
    checkpoint = tmp_path / "policy_1200000.pt"
    torch.save(
        {
            "env_transitions": 1200000,
            "policy_format_version": 4,
            "actor_obs_dim": 1097,
            "actor": {},
        },
        checkpoint,
    )
    return checkpoint, run_dir


def test_parse_run_log_duration_and_last_throughput():
    parsed = announce.parse_run_log(
        "[2026-07-25 10:00:00] transitions/s=100.0\n"
        "[2026-07-25 12:05:30] transitions/s=84210.5\n"
    )
    assert parsed["training_time"] == "2h 05m 30s"
    assert parsed["throughput"] == "84210.5 transitions/s"


def test_embed_includes_full_lap_and_summary():
    embed = announce.build_embed(
        {
            "track": "IV_2026_SIM",
            "training_time": "2h 05m 30s",
            "throughput": "84210.5 transitions/s",
            "transitions": "1200000",
            "policy": "v4, obs 1097",
            "eval": "running 6/8 at 60 s",
            "wandb": "https://wandb.ai/entity/f1tenth-genesis/runs/abc",
            "full_lap": "median 12.4s",
        },
        "wider steering margin",
        None,
    )
    assert embed["description"] == "wider steering margin"
    values = {field["name"]: field["value"] for field in embed["fields"]}
    assert values["Full lap"] == "median 12.4s"
    assert values["Track"] == "IV_2026_SIM"
    assert values["Policy"] == "v4, obs 1097"


def test_payload_orders_video_charts_and_checkpoint():
    payload = announce.build_payload(
        {"title": "F1TENTH policy release"},
        ["overlay_follow.mp4", "reward.png", "laps.png", "policy_1200000.pt"],
    )
    names = [item["filename"] for item in payload["attachments"]]
    assert names == [
        "overlay_follow.mp4",
        "reward.png",
        "laps.png",
        "policy_1200000.pt",
    ]
    assert payload["embeds"][0]["image"]["url"] == "attachment://reward.png"
    assert payload["embeds"][1]["image"]["url"] == "attachment://laps.png"


def test_dry_run_prints_fields_and_does_not_open_a_socket(tmp_path, monkeypatch, capsys):
    checkpoint, run_dir = _fixture(tmp_path)

    def boom(*_args, **_kwargs):
        raise AssertionError("socket opened")

    monkeypatch.setattr(announce.urllib.request, "urlopen", boom)
    monkeypatch.setenv(
        "DISCORD_MODEL_RELEASE_WEBHOOK",
        "https://discord.com/api/webhooks/9/super-secret-token",
    )
    announce.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--summary",
            "dry run",
            "--full-lap",
            "median 12.4s",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "track: IV_2026_SIM" in out
    assert "training time: 2h 05m 30s" in out
    assert "throughput: 84210.5 transitions/s" in out
    assert "transitions: 1200000" in out
    assert "policy format: v4, obs 1097" in out
    assert "running 6/8" in out
    assert "summary: dry run" in out
    assert "full lap: median 12.4s" in out
    assert "overlay_follow.mp4" in out
    assert "reward.png" in out
    assert "laps.png" in out
    assert "policy_1200000.pt" in out
    assert "dry-run: not posting" in out
    assert "super-secret-token" not in out
    assert "W&B history was not available" not in out


def test_dry_run_exits_without_webhook(tmp_path, monkeypatch, capsys):
    checkpoint, run_dir = _fixture(tmp_path)
    monkeypatch.delenv("DISCORD_MODEL_RELEASE_WEBHOOK", raising=False)
    announce.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--run-dir",
            str(run_dir),
            "--summary",
            "dry run",
            "--dry-run",
        ]
    )
    assert "dry-run: not posting" in capsys.readouterr().out


def test_refuses_oversize_before_posting(tmp_path, monkeypatch):
    checkpoint, run_dir = _fixture(tmp_path)
    monkeypatch.setattr(announce, "MAX_ATTACHMENT_BYTES", 10)

    def boom(*_args, **_kwargs):
        raise AssertionError("posted")

    monkeypatch.setattr(announce.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="limit"):
        announce.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--run-dir",
                str(run_dir),
                "--summary",
                "too big",
            ]
        )


def test_refuses_gpu(tmp_path):
    with pytest.raises(SystemExit, match="CPU"):
        announce.main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--summary",
                "x",
                "--device",
                "cuda",
            ]
        )


def test_render_calls_recovered_script(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, check):
        seen["cmd"] = cmd
        assert check is True
        return None

    monkeypatch.setattr(announce.subprocess, "run", fake_run)
    announce.render_follow_video(tmp_path / "a.npz", tmp_path / "a.mp4")
    assert seen["cmd"][1].endswith("render_follow.py")
    assert "--npz" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--width") + 1] == "1120"
    assert seen["cmd"][seen["cmd"].index("--height") + 1] == "700"


def test_missing_charts_are_noted():
    embed = announce.build_embed(
        {
            "track": "t",
            "training_time": "1s",
            "throughput": "1 transitions/s",
            "transitions": "1",
            "policy": "v4, obs 1097",
            "eval": "running 8/8 at 60 s",
            "wandb": "not linked",
        },
        "summary",
        "W&B history was not available, so this release has no training charts.",
    )
    assert "no training charts" in embed["description"]

"""Garage API against a real Postgres database."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

os.environ["DATABASE_URL"] = "postgresql://garage:garage@127.0.0.1/garage_test"
os.environ.pop("TIGRIS_ACCESS_KEY_ID", None)
os.environ.pop("TIGRIS_SECRET_ACCESS_KEY", None)
os.environ.pop("TIGRIS_BUCKET", None)

from app import create_app  # noqa: E402
from db import connect, reset_db  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "web"


def setup_module():
    try:
        reset_db()
    except Exception as exc:
        pytest.skip(f"garage_test postgres is not running: {exc}")


def client():
    return create_app().test_client()


def test_overview_stats():
    body = client().get("/api/overview").get_json()
    labels = [stat["label"] for stat in body["stats"]]
    assert labels == ["Best flying lap", "Completion", "Crashes", "Overtakes"]
    lap = body["stats"][0]
    assert lap["value"] == pytest.approx(55.8)
    assert lap["delta"] > 0
    assert "progress-reward" in body["best"]["name"]
    assert body["best"]["lap"] == pytest.approx(55.8)
    assert body["scenarios"] == [{"name": "Austin", "fraction": 1.0}]
    assert body["episode"]["marks"]
    assert "spatialcaps-v2-B" in body["live"]


def test_tree_attempts():
    payload = client().get("/api/tree").get_json()
    nodes = payload["nodes"]
    by_slug = {node["slug"]: node for node in nodes}
    assert "entropy-ab-austin-C" in by_slug
    assert "opponents-v6-B" in by_slug
    assert "tinylidarnet-v6-B" in by_slug
    assert "actionchunk-notal-v6-B" in by_slug
    for slug in ("spatialcaps-v2-B", "simv2-progress-s42", "gigaflow-v2",
                 "sophy-multi-v2-B"):
        assert slug in by_slug
        assert by_slug[slug]["verdict"] in {"running", "stopped", "fail", "pass"}
    parent = by_slug["opponents-notal-v6-B"]["id"]
    assert by_slug["opponents-notal-v6-B-s43"]["parent_id"] == parent
    assert by_slug["opponents-notal-v6-B-s43"]["verdict"] == "fail"
    assert by_slug["opponents-notal-fastopp-v6-B"]["verdict"] == "stopped"
    assert by_slug["actionchunk-notal-v6-B"]["lap"] == pytest.approx(55.8)
    assert "progress-reward" in by_slug["opponents-notal-v6-B"]["name"]
    names = " ".join(node["name"] for node in nodes)
    assert "TAL" not in names
    assert "notal" not in names.lower()
    assert payload["live"] == [
        "spatialcaps-v2-B", "simv2-progress-s42", "gigaflow-v2",
        "sophy-multi-v2-B",
    ]
    assert "110 / lap" in payload["score_formula"]
    assert by_slug["actionchunk-notal-v6-B"]["score"] is not None


def test_run_preview_is_video_first():
    http = client()
    runs = http.get("/api/runs").get_json()["runs"]
    row = next(item for item in runs if item["wandb_run_id"] == "actionchunk-notal-v6-B")
    preview = http.get(f"/api/runs/{row['id']}/preview").get_json()
    assert preview["run"]["id"] == row["id"]
    assert "progress-reward" in preview["run"]["name"]
    assert "TAL" not in preview["run"]["name"]
    assert "series" not in preview
    assert "evals" not in preview
    assert "score_formula" not in preview
    assert preview["videos"]


def test_run_evals_videos_policies():
    http = client()
    runs = http.get("/api/runs").get_json()["runs"]
    notal = next(row for row in runs if row["wandb_run_id"] == "opponents-notal-v6-B")
    detail = http.get(f"/api/runs/{notal['id']}").get_json()
    assert "wandb.ai" in detail["chart"]["wandb_url"]
    assert detail["run"]["seed"] == 42
    assert "progress-reward" in detail["run"]["name"]
    assert "TAL" not in detail["run"]["name"]
    assert detail["model_io"]["actor_dim"] == 1097
    assert detail["timeline"][0]["slug"] == "entropy-ab-austin-A"
    assert detail["timeline"][-1]["slug"] == "opponents-notal-v6-B"
    chart = http.get(detail["chart"]["src"])
    assert chart.status_code == 200
    assert chart.mimetype == "image/png"
    assert chart.data[:8] == b"\x89PNG\r\n\x1a\n"

    ev = http.get("/api/evals").get_json()["evals"]
    laps = [row["flying_lap_s"] for row in ev]
    assert laps == sorted(laps)
    assert laps[0] == pytest.approx(55.8)

    videos = http.get("/api/videos").get_json()["videos"]
    assert {row["kind"] for row in videos} >= {"flying lap", "crash clip"}

    policies = http.get("/api/policies").get_json()["checkpoints"]
    assert any(row["step"] == 256_000_000 and row["is_best"] for row in policies)
    assert all("disk_path" not in row for row in policies)


def test_fastopp_search_and_note_does_not_train():
    http = client()
    found = http.get("/api/search?q=fastopp").get_json()
    assert any("fastopp" in row["slug"] for row in found["runs"])
    rules = [str(rule) for rule in http.application.url_map.iter_rules()]
    assert not any("train" in rule or rule.endswith("/start") for rule in rules)
    created = http.post("/api/notes", json={"body": "watch the line"})
    assert created.status_code == 201
    listed = http.get("/api/notes").get_json()["notes"]
    assert listed[0]["body"] == "watch the line"


def test_video_plays_and_stage_needs_tigris(tmp_path):
    clip = tmp_path / "lap.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=1",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
        capture_output=True,
    )
    http = client()
    video = http.get("/api/videos").get_json()["videos"][0]
    with connect() as conn:
        conn.execute(
            "UPDATE videos SET disk_path = %s WHERE id = %s",
            (str(clip), video["id"]),
        )
    streamed = http.get(video["stream"])
    assert streamed.status_code == 200
    assert streamed.mimetype == "video/mp4"
    assert len(streamed.data) > 100

    ckpt = http.get("/api/policies").get_json()["checkpoints"][0]
    missing = http.post(f"/api/checkpoints/{ckpt['id']}/stage")
    assert missing.status_code == 404

    weight = tmp_path / "policy.pt"
    weight.write_bytes(b"not a real weight")
    with connect() as conn:
        conn.execute(
            "UPDATE checkpoints SET disk_path = %s WHERE id = %s",
            (str(weight), ckpt["id"]),
        )
    staged = http.post(f"/api/checkpoints/{ckpt['id']}/stage")
    assert staged.status_code == 503
    assert b"dashboard/.env" in staged.data


def test_web_does_not_open_the_database():
    text = []
    for path in WEB.rglob("*"):
        if "node_modules" in path.parts or path.suffix not in {".ts", ".tsx", ".js", ".mjs"}:
            continue
        text.append(path.read_text())
    blob = "\n".join(text)
    assert "psycopg" not in blob
    assert "DATABASE_URL" not in blob
    assert "TIGRIS" not in blob

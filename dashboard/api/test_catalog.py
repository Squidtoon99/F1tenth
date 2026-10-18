"""Catalog from the store TSV plus later arms. Display names stay public."""

from __future__ import annotations

from pathlib import Path

from catalog import attempts, public_text, videos_for
from live import LIVE_IDS, apply

WEB = Path(__file__).resolve().parents[1] / "web"


def test_tsv_and_later_arms_are_listed():
    rows = attempts()
    slugs = {row["slug"] for row in rows}
    assert "entropy-ab-austin-C" in slugs
    assert "opponents-v6-B" in slugs
    assert "spatialcaps-v6-B" in slugs
    assert "actionchunk-notal-v6-B" in slugs
    assert "tinylidarnet-v6-B" in slugs
    for slug in LIVE_IDS:
        assert slug in slugs
    assert any(row["result"] == "fail" for row in rows)
    assert any(row["result"] == "pass" for row in rows)


def test_public_names_drop_the_old_acronym():
    blob = " ".join(f"{row['name']} {row['one_change']}" for row in attempts())
    assert "TAL" not in blob
    assert "no-TAL" not in blob
    assert "notal" not in blob.lower()
    assert " no-" not in f" {blob.lower()}"
    names = {row["slug"]: row["name"] for row in attempts()}
    assert "progress-reward" in names["opponents-notal-v6-B"]
    assert "progress-reward" in names["actionchunk-notal-v6-B"]
    assert names["opponents-notal-v6-B"].startswith("Opponents")
    assert "no-" not in names["opponents-notal-v6-B"]
    assert public_text("TAL off") == "fixed-speed reward off"


def test_videos_are_real_files_or_absent():
    for row in attempts():
        video = row["video"]
        if video is None:
            continue
        assert video["disk_path"]
        from pathlib import Path
        assert Path(video["disk_path"]).is_file()
    missing = videos_for("entropy-ab-austin-A")
    assert missing == []


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


def test_live_overlay_marks_training():
    row = next(item for item in attempts() if item["slug"] == "spatialcaps-v2-B")
    updated = apply(row, {"state": "running", "transitions": 12_000_000})
    assert updated["result"] == "running"
    assert updated["transitions"] == 12_000_000
    assert updated["wandb_run_id"] == "spatialcaps-v2-B"

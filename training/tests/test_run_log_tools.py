"""Tests for run.log contamination sidecars."""

from __future__ import annotations

import json

from run_log_tools import (
    excluded_transition_ranges,
    filter_log_text,
    filter_parsed_rows,
    load_log_contamination,
)


def test_filter_log_text_drops_foreign_startup(tmp_path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    log = run_dir / "run.log"
    log.write_text(
        "[2026-07-29 15:56:21] standalone_trainer INFO: ticks=50 transitions=51200\n"
        "[2026-07-29 16:00:02] standalone_trainer INFO: Run id: run-a\n"
        "[2026-07-29 16:00:02] standalone_trainer INFO: total_transitions=300000000\n"
        "[2026-07-29 16:00:04] standalone_trainer INFO: ticks=100 transitions=102400\n",
        encoding="utf-8",
    )
    (run_dir / "log_contamination.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "kind": "foreign_trainer_startup",
                        "started_at": "2026-07-29T16:00:02-07:00",
                        "ended_at": "2026-07-29T16:00:03-07:00",
                        "line_markers": ["total_transitions=300000000"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    filtered = filter_log_text(log.read_text(encoding="utf-8"), log)
    assert "total_transitions=300000000" not in filtered
    assert "transitions=51200" in filtered
    assert "transitions=102400" in filtered


def test_filter_parsed_rows_honors_transition_ranges(tmp_path):
    run_dir = tmp_path / "run-b"
    run_dir.mkdir()
    log = run_dir / "run.log"
    log.write_text("placeholder\n", encoding="utf-8")
    (run_dir / "log_contamination.json").write_text(
        json.dumps({"exclude_transition_ranges": [[100, 200]]}),
        encoding="utf-8",
    )
    rows = [
        {"t": 50, "speed": 1.0},
        {"t": 150, "speed": 2.0},
        {"t": 400_000_000, "speed": 5.0},
    ]
    kept = filter_parsed_rows(rows, log, trans_key="t")
    assert [r["t"] for r in kept] == [50, 400_000_000]
    assert excluded_transition_ranges(log) == [(100, 200)]
    assert load_log_contamination(log)["exclude_transition_ranges"] == [[100, 200]]

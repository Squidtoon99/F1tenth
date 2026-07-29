"""Helpers for run.log contamination sidecars used by gate/verdict tooling."""

from __future__ import annotations

import json
import re
from pathlib import Path

LOG_CONTAMINATION_NAME = "log_contamination.json"


def log_contamination_path(run_dir: Path) -> Path:
    return run_dir / LOG_CONTAMINATION_NAME


def load_log_contamination(run_log: Path) -> dict:
    path = log_contamination_path(run_log.parent)
    if not path.is_file():
        return {"events": [], "exclude_transition_ranges": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("events", [])
    data.setdefault("exclude_transition_ranges", [])
    return data


def excluded_transition_ranges(run_log: Path) -> list[tuple[int, int]]:
    data = load_log_contamination(run_log)
    ranges: list[tuple[int, int]] = []
    for item in data.get("exclude_transition_ranges", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            ranges.append((int(item[0]), int(item[1])))
    for event in data.get("events", []):
        tr = event.get("transition_range")
        if (
            isinstance(tr, (list, tuple))
            and len(tr) == 2
            and tr[0] is not None
            and tr[1] is not None
        ):
            ranges.append((int(tr[0]), int(tr[1])))
    return ranges


def transition_is_excluded(transitions: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= transitions <= hi for lo, hi in ranges)


def filter_parsed_rows(rows: list[dict], run_log: Path, *, trans_key: str) -> list[dict]:
    ranges = excluded_transition_ranges(run_log)
    if not ranges:
        return rows
    return [
        row for row in rows
        if not transition_is_excluded(int(row[trans_key]), ranges)
    ]


def filter_log_text(text: str, run_log: Path) -> str:
    data = load_log_contamination(run_log)
    events = data.get("events", [])
    if not events:
        return text
    kept: list[str] = []
    skip_until_ts = None
    for line in text.splitlines():
        m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
        ts = m.group(1) if m else None
        if skip_until_ts is not None:
            if ts is not None and ts > skip_until_ts:
                skip_until_ts = None
            else:
                continue
        drop = False
        for event in events:
            if event.get("kind") != "foreign_trainer_startup":
                continue
            start = str(event.get("started_at", ""))[:19].replace("T", " ")
            end = str(event.get("ended_at", ""))[:19].replace("T", " ")
            if ts is not None and start and end and start <= ts <= end:
                drop = True
                break
            for marker in event.get("line_markers", []):
                if marker and marker in line:
                    drop = True
                    skip_until_ts = end or None
                    break
        if not drop:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")

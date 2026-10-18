"""Poll wandb and local trainer dirs for runs that are still training.

Read-only. Does not start, stop, or signal any trainer.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from netrc import netrc
from pathlib import Path

LIVE_IDS = (
    "spatialcaps-v2-B",
    "simv2-progress-s42",
    "gigaflow-v2",
    "sophy-multi-v2-B",
)

LOCAL_DIRS = {
    "spatialcaps-v2-B": Path(
        "/home/ubuntu/projects/F1tenth-spatialcaps-v2-B/outputs/runs/"
        "spatialcaps-v2-B"
    ),
    "sophy-multi-v2-B": Path(
        "/home/ubuntu/projects/F1tenth-baseline-v5/outputs/runs/"
        "sophy-multi-v2-B"
    ),
    "simv2-progress-s42": Path(
        "/home/ubuntu/projects/F1tenth-simv2-progress-s42/outputs/runs/"
        "simv2-progress-s42"
    ),
    "gigaflow-v2": Path(
        "/home/ubuntu/projects/F1tenth-gigaflow-v2/outputs/runs/gigaflow-v2"
    ),
}

ENTITY = "squidtoon99-ut-dallas"
PROJECT = "f1tenth-genesis"
GRAPHQL = "https://api.wandb.ai/graphql"
TTL_S = 12.0

_CACHE = {"at": 0.0, "rows": {}}
_TRANS_RE = re.compile(r"transitions=(\d+)")
_LIFE_RE = re.compile(r"episode_lifespan=([0-9.]+)")
_CRASH_RE = re.compile(r"scn/solo_flying:.*?crash=([0-9.]+)")
_COMP_RE = re.compile(r"scn/solo_flying:.*?lap_completion=([0-9.]+)")


def _api_key() -> str | None:
    env = os.environ.get("WANDB_API_KEY")
    if env:
        return env
    path = Path.home() / ".netrc"
    if not path.is_file():
        return None
    try:
        parsed = netrc(str(path))
        auth = parsed.authenticators("api.wandb.ai")
    except (OSError, ValueError):
        return None
    if not auth:
        return None
    return auth[2]


def _graphql(query: str, variables: dict) -> dict:
    key = _api_key()
    if not key:
        return {}
    payload = json.dumps({"query": query, "variables": variables}).encode()
    request = urllib.request.Request(
        GRAPHQL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def _wandb_runs() -> dict[str, dict]:
    query = """
    query($entity: String!, $name: String!) {
      project(name: $name, entityName: $entity) {
        runs(first: 80, order: "-createdAt") {
          edges { node { displayName name state summaryMetrics } }
        }
      }
    }
    """
    body = _graphql(query, {"entity": ENTITY, "name": PROJECT})
    edges = (
        ((body.get("data") or {}).get("project") or {})
        .get("runs", {})
        .get("edges") or []
    )
    found = {}
    for edge in edges:
        node = edge.get("node") or {}
        wid = node.get("displayName") or node.get("name")
        if wid not in LIVE_IDS:
            continue
        summary = node.get("summaryMetrics") or {}
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {}
        found[wid] = {
            "state": (node.get("state") or "").lower(),
            "transitions": _summary_num(summary, (
                "transitions", "global_step", "trainer/transitions",
            )),
            "completion": _summary_num(summary, (
                "eval/flying_full_lap", "scn/solo_flying/lap_completion",
            )),
            "crash": _summary_num(summary, (
                "eval/flying_crash", "scn/solo_flying/crash",
            )),
            "source": "wandb",
        }
    return found


def _summary_num(summary: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in summary and summary[key] is not None:
            try:
                return float(summary[key])
            except (TypeError, ValueError):
                continue
    return None


def _local_run(slug: str) -> dict | None:
    root = LOCAL_DIRS.get(slug)
    if root is None or not root.is_dir():
        return None
    status = root / "watch_status.json"
    if status.is_file():
        try:
            data = json.loads(status.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        if data:
            alive = bool(data.get("trainer_alive")) and not data.get("finished")
            return {
                "state": "running" if alive else "finished",
                "transitions": data.get("transitions"),
                "completion": data.get("completion"),
                "crash": data.get("crash"),
                "lifespan": data.get("lifespan"),
                "source": "local",
            }
    log = root / "run.log"
    if not log.is_file():
        return None
    try:
        text = log.read_text(errors="replace")[-8000:]
    except OSError:
        return None
    trans = _last_num(_TRANS_RE, text)
    if trans is None:
        return None
    return {
        "state": "running",
        "transitions": trans,
        "completion": _last_num(_COMP_RE, text),
        "crash": _last_num(_CRASH_RE, text),
        "lifespan": _last_num(_LIFE_RE, text),
        "source": "local",
    }


def _last_num(pattern: re.Pattern, text: str):
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def overlay(force: bool = False) -> dict[str, dict]:
    now = time.time()
    if not force and _CACHE["rows"] and now - _CACHE["at"] < TTL_S:
        return _CACHE["rows"]
    rows = {wid: {"state": "unknown", "source": None} for wid in LIVE_IDS}
    for slug, local in ((wid, _local_run(wid)) for wid in LIVE_IDS):
        if local:
            rows[slug] = local
    try:
        remote = _wandb_runs()
    except Exception:
        remote = {}
    for slug, remote_row in remote.items():
        current = rows.get(slug) or {}
        merged = {**remote_row}
        if current.get("transitions") and not merged.get("transitions"):
            merged["transitions"] = current["transitions"]
        if current.get("source") == "local" and current.get("state") == "running":
            merged["state"] = "running"
            merged["source"] = "local+wandb"
        rows[slug] = merged
    _CACHE["at"] = now
    _CACHE["rows"] = rows
    return rows


def apply(attempt: dict, live: dict | None) -> dict:
    if not live:
        return attempt
    out = dict(attempt)
    state = (live.get("state") or "").lower()
    if state in {"running", "running_wandb"}:
        out["verdict"] = "running"
        out["result"] = "running"
        out["stop_reason"] = None
    elif state in {"finished", "crashed", "failed"} and out.get("verdict") == "running":
        out["verdict"] = "stopped" if state == "finished" else "fail"
        out["result"] = "fail" if state != "finished" else "stopped"
        out["stop_reason"] = state
    if live.get("transitions") is not None:
        out["transitions"] = int(live["transitions"])
    if live.get("completion") is not None and out.get("completion") is None:
        out["completion"] = live["completion"]
    if live.get("crash") is not None and out.get("crash") is None:
        out["crash"] = live["crash"]
    out["live"] = {
        "state": state or None,
        "source": live.get("source"),
        "transitions": live.get("transitions"),
        "completion": live.get("completion"),
        "crash": live.get("crash"),
        "lifespan": live.get("lifespan"),
    }
    return out

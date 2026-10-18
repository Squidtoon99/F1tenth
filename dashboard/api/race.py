"""Garage race catalog and the shared gigaflow replay session."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "gigaflow" / "src", ROOT / "libs" / "f1tenth_policy"):
    entry = str(extra)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from gigaflow_f1tenth.config import load_config  # noqa: E402
from gigaflow_f1tenth.viewer.drivers import describe_checkpoint  # noqa: E402
from gigaflow_f1tenth.viewer.http_session import (  # noqa: E402
    HttpRaceSession,
    default_device,
)
from gigaflow_f1tenth.viewer.server import CHECKPOINT_GLOBS  # noqa: E402

_session: HttpRaceSession | None = None


def config_path() -> Path:
    raw = os.environ.get("GIGAFLOW_CONFIG")
    if raw:
        return Path(raw)
    return ROOT / "gigaflow" / "configs" / "default.yaml"


def cache_dir() -> Path:
    raw = os.environ.get("GIGAFLOW_TRACK_CACHE")
    if raw:
        return Path(raw)
    return ROOT / "gigaflow" / "cache"


def viewer_dist() -> Path:
    return ROOT / "gigaflow" / "viewer" / "dist"


def checkpoint_roots() -> list[Path]:
    raw = os.environ.get("GIGAFLOW_CHECKPOINT_ROOTS", "")
    roots = [Path(item) for item in raw.split(":") if item]
    roots.extend(
        [
            ROOT / "gigaflow" / "outputs",
            ROOT / "training" / "outputs",
        ]
    )
    return [path for path in roots if path.is_dir()]


def session() -> HttpRaceSession:
    global _session
    if _session is None:
        _session = HttpRaceSession(config_path(), cache_dir(), default_device())
    return _session


def _disk_models(cfg) -> list[dict]:
    found = []
    seen: set[str] = set()
    for root in checkpoint_roots():
        paths: list[Path] = []
        for pattern in CHECKPOINT_GLOBS:
            paths.extend(path for path in root.rglob(pattern) if path.is_file())
        for path in sorted(set(paths)):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            described = describe_checkpoint(path, cfg)
            if described is None:
                continue
            seen.add(resolved)
            relative = path.parent.relative_to(root)
            run = root.name if relative == Path(".") else str(relative)
            found.append({**described, "run": run, "step": None, "id": None})
    return found


def _garage_models(conn, cfg) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.id, c.step, c.disk_path, e.name AS run_name
        FROM checkpoints c
        JOIN runs r ON r.id = c.run_id
        JOIN experiments e ON e.id = r.experiment_id
        ORDER BY e.created_at, c.step
        """
    ).fetchall()
    models = []
    for row in rows:
        disk = row["disk_path"] or ""
        path = Path(disk) if disk else None
        item = {
            "id": str(row["id"]),
            "run": row["run_name"],
            "step": int(row["step"]),
            "name": path.name if path else f"policy_{row['step']}.pt",
            "path": None,
            "kind": None,
            "available": False,
        }
        if path is not None and path.is_file():
            described = describe_checkpoint(path, cfg)
            if described is not None:
                item.update(described)
                item["available"] = True
        models.append(item)
    return models


def race_catalog(conn) -> dict:
    cfg = load_config(config_path())
    garage = _garage_models(conn, cfg) if conn is not None else []
    disk = _disk_models(cfg)
    garage_paths = {row["path"] for row in garage if row.get("path")}
    extras = [row for row in disk if row["path"] not in garage_paths]
    for row in extras:
        row["available"] = True
    return {
        "models": garage + extras,
        "viewer": (viewer_dist() / "index.html").is_file(),
        "cache": cache_dir().is_dir(),
    }


def race_tracks() -> dict:
    cache = cache_dir()
    if not (cache / "atlas.npz").is_file():
        return {"tracks": [], "cache": False}
    from gigaflow_f1tenth.tracks import load_atlas

    view = load_atlas(str(cache)).view()
    return {"tracks": [str(track) for track in view.track_ids], "cache": True}

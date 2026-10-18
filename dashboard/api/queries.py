"""Read models for the garage pages. Disk paths stay on the server."""

from __future__ import annotations

import uuid

from catalog import (
    all_evals,
    ancestor_path,
    attempts,
    evals_for,
    find_attempt,
    videos_for,
)
from live import LIVE_IDS, apply, overlay
from seed_data import BUDGETS, EPISODE_MARKS

KEEP = "keep"
PASS = "pass"


def _num(value):
    if value is None:
        return None
    return float(value)


def _live_map():
    try:
        return overlay()
    except Exception:
        return {}


def _catalog(live=None):
    rows = attempts()
    live = live if live is not None else _live_map()
    return [apply(row, live.get(row["slug"])) for row in rows]


def _pub_attempt(row):
    return {
        "id": row["run_id"],
        "experiment_id": row["id"],
        "name": row["name"],
        "slug": row["slug"],
        "verdict": row["result"],
        "one_change": row["one_change"],
        "wandb_run_id": row["wandb_run_id"],
        "wandb_url": row["wandb_url"],
        "seed": row["seed"],
        "plant": row["plant"],
        "transitions": row["transitions"],
        "stop_reason": row["stop_reason"],
        "started_at": None,
        "ended_at": None,
        "budget": row.get("budget") or BUDGETS.get(row["wandb_run_id"]),
        "result": row["result"],
        "live": row.get("live"),
    }


def _pub_run(row):
    return {
        "id": row["id"],
        "experiment_id": row["experiment_id"],
        "name": row["name"],
        "slug": row["slug"],
        "verdict": row["verdict"],
        "one_change": row["one_change"],
        "wandb_run_id": row["wandb_run_id"],
        "wandb_url": row["wandb_url"],
        "seed": row["seed"],
        "plant": row["plant"],
        "transitions": row["transitions"],
        "stop_reason": row["stop_reason"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "budget": BUDGETS.get(row["wandb_run_id"]),
    }


def _runs(conn):
    return conn.execute(
        """
        SELECT r.*, e.slug, e.name, e.verdict, e.one_change, e.kind, e.created_at
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        ORDER BY e.created_at
        """
    ).fetchall()


def _best_laps(conn):
    return conn.execute(
        """
        SELECT e.id, e.slug, e.name, e.verdict, e.created_at, e.parent_id,
               e.one_change, r.id AS run_id, r.wandb_run_id, r.transitions,
               MIN(v.flying_lap_s) AS lap,
               (ARRAY_AGG(v.completion ORDER BY v.flying_lap_s))[1] AS completion,
               (ARRAY_AGG(v.crash ORDER BY v.flying_lap_s))[1] AS crash,
               (ARRAY_AGG(v.overtakes ORDER BY v.flying_lap_s))[1] AS overtakes,
               (ARRAY_AGG(c.id ORDER BY v.flying_lap_s))[1] AS checkpoint_id,
               (ARRAY_AGG(c.s3_key ORDER BY v.flying_lap_s))[1] AS s3_key
        FROM experiments e
        JOIN runs r ON r.experiment_id = e.id
        LEFT JOIN checkpoints c ON c.run_id = r.id
        LEFT JOIN evals v ON v.checkpoint_id = c.id
        GROUP BY e.id, r.id
        ORDER BY e.created_at
        """
    ).fetchall()


def _improvement(metric, current, previous):
    if current is None or previous is None:
        return None
    if metric in ("flying_lap_s", "crash"):
        return previous - current
    return current - previous


def overview(conn):
    rows = _catalog()
    scored = [row for row in rows if row.get("lap") is not None]
    keeps = [row for row in scored if row["result"] == PASS]
    best = min(keeps, key=lambda row: row["lap"]) if keeps else None
    previous = None
    if best is not None:
        earlier = [row for row in keeps if row["slug"] != best["slug"]]
        if earlier:
            previous = min(
                earlier,
                key=lambda row: abs((row["lap"] or 0) - best["lap"]),
            )
            slower = [row for row in earlier if (row["lap"] or 0) > best["lap"]]
            if slower:
                previous = min(slower, key=lambda row: row["lap"] or 0)

    def stat(label, field, metric, unit):
        current = _num(best[field]) if best else None
        prior = _num(previous[field]) if previous else None
        return {
            "label": label,
            "value": current,
            "unit": unit,
            "delta": _improvement(metric, current, prior),
            "versus": previous["name"] if previous else None,
        }

    laps = [
        {
            "label": row["name"],
            "lap": _num(row["lap"]),
            "verdict": row["result"],
        }
        for row in scored
    ]
    marked = None
    for row in rows:
        marks = EPISODE_MARKS.get(row["wandb_run_id"])
        if marks and (marked is None or len(marks) > len(marked["marks"])):
            marked = {"run_name": row["name"], "marks": marks}

    gauge = None
    live_running = [row for row in rows if row["result"] == "running"]
    focus = live_running[0] if live_running else best
    if focus is not None:
        budget = focus.get("budget") or BUDGETS.get(focus["wandb_run_id"])
        if focus.get("transitions") is not None and budget:
            gauge = {
                "used": focus["transitions"],
                "budget": budget,
                "label": focus["name"],
            }

    latest = next((row for row in reversed(rows) if row.get("crash") is not None), None)
    db_best = None
    try:
        db_best = _best_laps(conn)
    except Exception:
        db_best = []
    ckpt = None
    if best is not None:
        for row in db_best:
            if row["slug"] == best["slug"]:
                ckpt = row
                break

    best_card = None
    if best is not None:
        best_card = {
            "name": best["name"],
            "lap": _num(best["lap"]),
            "completion": _num(best["completion"]),
            "checkpoint_id": ckpt["checkpoint_id"] if ckpt else None,
            "staged": bool(ckpt and ckpt["s3_key"]),
        }

    return {
        "stats": [
            stat("Best flying lap", "lap", "flying_lap_s", "s"),
            stat("Completion", "completion", "completion", ""),
            stat("Crashes", "crash", "crash", ""),
            stat("Overtakes", "overtakes", "overtakes", ""),
        ],
        "laps": laps,
        "episode": marked,
        "gauge": gauge,
        "scenarios": [{"name": "Austin", "fraction": 1.0}],
        "crash": {
            "rate": _num(latest["crash"]) if latest else None,
            "label": latest["name"] if latest else None,
        },
        "best": best_card,
        "live": [row["slug"] for row in live_running],
    }


def tree(conn):
    del conn
    return {
        "nodes": [
            {
                "id": row["id"],
                "slug": row["slug"],
                "name": row["name"],
                "parent_id": row["parent_id"],
                "parent_slug": row.get("parent_slug"),
                "one_change": row["one_change"],
                "verdict": row["result"],
                "lap": _num(row.get("lap")),
                "completion": _num(row.get("completion")),
                "run_id": row["run_id"],
                "wandb_run_id": row["wandb_run_id"],
                "live": row.get("live"),
                "video": row.get("video"),
            }
            for row in _catalog()
        ],
        "live": list(LIVE_IDS),
    }


def runs(conn):
    del conn
    return {"runs": [_pub_attempt(row) for row in _catalog()]}


def _evals_for_run(conn, run_id):
    return conn.execute(
        """
        SELECT v.*, c.step, c.is_best, c.is_final, c.run_id
        FROM evals v
        JOIN checkpoints c ON c.id = v.checkpoint_id
        WHERE c.run_id = %s
        ORDER BY v.flying_lap_s ASC NULLS LAST, c.step
        """,
        (run_id,),
    ).fetchall()


def _videos_for_run(conn, run_id):
    return conn.execute(
        """
        SELECT v.id, v.kind, v.duration_s, v.width, v.height, v.s3_key,
               e.name AS run_name, c.step
        FROM videos v
        LEFT JOIN checkpoints c ON c.id = v.checkpoint_id
        LEFT JOIN runs r ON r.id = c.run_id
        LEFT JOIN experiments e ON e.id = r.experiment_id
        WHERE c.run_id = %s
        ORDER BY v.kind
        """,
        (run_id,),
    ).fetchall()


def run_detail(conn, run_id):
    attempt = find_attempt(str(run_id))
    if attempt is None:
        return None
    live = _live_map()
    attempt = apply(attempt, live.get(attempt["slug"]))
    catalog_evals = evals_for(attempt)
    db_evals = []
    db_videos = []
    try:
        db_evals = _evals_for_run(conn, attempt["run_id"])
        db_videos = [
            {**video, "stream": f"/api/videos/{video['id']}/stream"}
            for video in _videos_for_run(conn, attempt["run_id"])
        ]
    except Exception:
        db_evals = []
        db_videos = []
    videos = attempt.get("videos") or videos_for(attempt["slug"])
    if db_videos:
        seen = {str(video["id"]) for video in videos}
        for video in db_videos:
            if str(video["id"]) not in seen:
                videos.append(video)
    series = []
    if attempt.get("lap") is not None:
        series.append({"step": attempt.get("transitions") or 0, "lap": attempt["lap"]})
    marks = EPISODE_MARKS.get(attempt["wandb_run_id"]) or []
    return {
        "run": _pub_attempt(attempt),
        "experiment": {
            "id": attempt["id"],
            "slug": attempt["slug"],
            "name": attempt["name"],
            "verdict": attempt["result"],
            "one_change": attempt["one_change"],
            "kind": attempt.get("kind"),
            "parent_id": attempt.get("parent_id"),
        },
        "evals": db_evals or catalog_evals,
        "videos": videos,
        "model_io": attempt["model_io"],
        "timeline": ancestor_path(attempt["slug"]),
        "series": {
            "laps": series,
            "episode": marks,
        },
        "chart": {
            "title": attempt["name"],
            "src": f"/api/runs/{attempt['run_id']}/chart.png",
            "wandb_url": attempt["wandb_url"],
        },
    }


def evals(conn):
    del conn
    return {"evals": all_evals()}


def videos(conn):
    rows = []
    seen = set()
    for attempt in _catalog():
        for video in attempt.get("videos") or []:
            if str(video["id"]) in seen:
                continue
            seen.add(str(video["id"]))
            rows.append({
                **video,
                "run_id": attempt["run_id"],
                "run_name": attempt["name"],
                "step": attempt.get("transitions"),
            })
    try:
        db_rows = conn.execute(
            """
            SELECT v.id, v.kind, v.duration_s, v.width, v.height, v.s3_key,
                   e.name AS run_name, c.step, r.id AS run_id
            FROM videos v
            LEFT JOIN checkpoints c ON c.id = v.checkpoint_id
            LEFT JOIN runs r ON r.id = c.run_id
            LEFT JOIN experiments e ON e.id = r.experiment_id
            ORDER BY e.created_at NULLS LAST, v.kind
            """
        ).fetchall()
    except Exception:
        db_rows = []
    for row in db_rows:
        if str(row["id"]) in seen:
            continue
        row["stream"] = f"/api/videos/{row['id']}/stream"
        rows.append(row)
    return {"videos": rows}


def policies(conn):
    rows = conn.execute(
        """
        SELECT c.id, c.step, c.sha256, c.bytes, c.is_best, c.is_final, c.s3_key,
               e.name AS run_name, r.id AS run_id, e.slug
        FROM checkpoints c
        JOIN runs r ON r.id = c.run_id
        JOIN experiments e ON e.id = r.experiment_id
        ORDER BY e.created_at, c.step
        """
    ).fetchall()
    for row in rows:
        row["staged"] = bool(row["s3_key"])
        row["key"] = row["s3_key"] or (
            f"dashboard/checkpoints/{row['slug']}/policy_{row['step']}.pt"
        )
        del row["s3_key"]
    return {"checkpoints": rows}


def files(conn):
    ckpts = policies(conn)["checkpoints"]
    vids = videos(conn)["videos"]
    items = []
    for row in ckpts:
        items.append({
            "kind": "checkpoint",
            "name": f"policy_{row['step']}.pt",
            "run_name": row["run_name"],
            "bytes": row["bytes"],
            "key": row["key"],
            "staged": row["staged"],
        })
    for row in vids:
        items.append({
            "kind": row["kind"],
            "name": (row["s3_key"] or "").rsplit("/", 1)[-1],
            "run_name": row["run_name"],
            "bytes": None,
            "key": row["s3_key"],
            "staged": bool(row["s3_key"]),
        })
    return {"files": items}


def search(conn, q: str):
    needle = q.strip().lower()
    hits = []
    for row in _catalog():
        blob = " ".join([
            row["name"], row["slug"], row["wandb_run_id"],
            row.get("one_change") or "", str(row.get("seed") or ""),
        ]).lower()
        if needle in blob:
            hits.append({
                "id": row["run_id"],
                "name": row["name"],
                "slug": row["slug"],
                "wandb_run_id": row["wandb_run_id"],
                "seed": row.get("seed"),
            })
        if len(hits) >= 8:
            break
    ckpts = []
    try:
        like = f"%{q.strip()}%"
        ckpts = conn.execute(
            """
            SELECT c.id, c.step, e.name AS run_name, r.id AS run_id
            FROM checkpoints c
            JOIN runs r ON r.id = c.run_id
            JOIN experiments e ON e.id = r.experiment_id
            WHERE e.name ILIKE %s OR CAST(c.step AS text) ILIKE %s
               OR e.slug ILIKE %s
            ORDER BY c.step
            LIMIT 8
            """,
            (like, like, like),
        ).fetchall()
    except Exception:
        ckpts = []
    return {"runs": hits, "checkpoints": ckpts}


def wandb_gallery(conn):
    del conn
    return {
        "runs": [
            {
                "id": row["run_id"],
                "name": row["name"],
                "wandb_url": row["wandb_url"],
                "src": f"/api/runs/{row['run_id']}/chart.png",
            }
            for row in _catalog()
            if row["wandb_url"]
        ]
    }


def export_payload(conn):
    return {
        "experiments": conn.execute(
            """
            SELECT id, slug, name, parent_id, kind, one_change, verdict, created_at
            FROM experiments ORDER BY created_at
            """
        ).fetchall(),
        "runs": [
            {key: row[key] for key in row.keys() if key != "disk_path"}
            for row in _runs(conn)
        ],
        "evals": evals(conn)["evals"],
    }


def notes(conn):
    return {
        "notes": conn.execute(
            "SELECT id, body, created_at FROM notes ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
    }


def add_note(conn, body: str):
    row = conn.execute(
        """
        INSERT INTO notes (id, body) VALUES (%s, %s)
        RETURNING id, body, created_at
        """,
        (uuid.uuid4(), body),
    ).fetchone()
    return row


def checkpoint_row(conn, ckpt_id):
    return conn.execute(
        """
        SELECT c.*, e.slug, e.name
        FROM checkpoints c
        JOIN runs r ON r.id = c.run_id
        JOIN experiments e ON e.id = r.experiment_id
        WHERE c.id = %s
        """,
        (ckpt_id,),
    ).fetchone()


def set_checkpoint_key(conn, ckpt_id, key: str) -> None:
    conn.execute(
        "UPDATE checkpoints SET s3_key = %s WHERE id = %s",
        (key, ckpt_id),
    )


def catalog_video(video_id):
    token = str(video_id)
    for attempt in attempts():
        for video in attempt.get("videos") or []:
            if str(video["id"]) == token:
                return video
    return None


def video_row(conn, video_id):
    found = catalog_video(video_id)
    if found:
        return found
    return conn.execute(
        "SELECT * FROM videos WHERE id = %s",
        (video_id,),
    ).fetchone()


def chart_inputs(conn, run_id):
    attempt = find_attempt(str(run_id))
    if attempt is not None:
        ev = []
        if attempt.get("lap") is not None:
            ev.append({
                "flying_lap_s": attempt["lap"],
                "step": attempt.get("transitions"),
            })
        return (
            {
                "id": attempt["run_id"],
                "wandb_run_id": attempt["wandb_run_id"],
                "name": attempt["name"],
            },
            ev,
        )
    row = conn.execute(
        """
        SELECT r.id, r.wandb_run_id, e.name
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        WHERE r.id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    ev = conn.execute(
        """
        SELECT v.flying_lap_s, c.step
        FROM evals v
        JOIN checkpoints c ON c.id = v.checkpoint_id
        WHERE c.run_id = %s AND v.flying_lap_s IS NOT NULL
        """,
        (run_id,),
    ).fetchall()
    return row, ev

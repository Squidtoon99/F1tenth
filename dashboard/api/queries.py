"""Read models for the garage pages. Disk paths stay on the server."""

from __future__ import annotations

import uuid

from seed_data import BUDGETS, EPISODE_MARKS

KEEP = "keep"


def _num(value):
    if value is None:
        return None
    return float(value)


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
    rows = _best_laps(conn)
    scored = [row for row in rows if row["lap"] is not None]
    keeps = [row for row in scored if row["verdict"] == KEEP]
    best = min(keeps, key=lambda row: row["lap"]) if keeps else None
    previous = None
    if best is not None:
        earlier = [
            row for row in keeps
            if row["created_at"] < best["created_at"]
        ]
        if earlier:
            previous = max(earlier, key=lambda row: row["created_at"])

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
            "verdict": row["verdict"],
        }
        for row in scored
    ]
    marked = None
    for row in rows:
        marks = EPISODE_MARKS.get(row["wandb_run_id"])
        if marks and (marked is None or len(marks) > len(marked["marks"])):
            marked = {"run_name": row["name"], "marks": marks}

    gauge = None
    if best is not None:
        budget = BUDGETS.get(best["wandb_run_id"])
        if best["transitions"] is not None and budget:
            gauge = {
                "used": best["transitions"],
                "budget": budget,
                "label": best["name"],
            }

    latest = conn.execute(
        """
        SELECT v.crash, e.name
        FROM evals v
        JOIN checkpoints c ON c.id = v.checkpoint_id
        JOIN runs r ON r.id = c.run_id
        JOIN experiments e ON e.id = r.experiment_id
        WHERE v.crash IS NOT NULL
        ORDER BY e.created_at DESC, c.step DESC
        LIMIT 1
        """
    ).fetchone()

    best_card = None
    if best is not None:
        best_card = {
            "name": best["name"],
            "lap": _num(best["lap"]),
            "completion": _num(best["completion"]),
            "checkpoint_id": best["checkpoint_id"],
            "staged": bool(best["s3_key"]),
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
    }


def tree(conn):
    rows = _best_laps(conn)
    return {
        "nodes": [
            {
                "id": row["id"],
                "slug": row["slug"],
                "name": row["name"],
                "parent_id": row["parent_id"],
                "one_change": row["one_change"],
                "verdict": row["verdict"],
                "lap": _num(row["lap"]),
                "run_id": row["run_id"],
            }
            for row in rows
        ]
    }


def runs(conn):
    return {"runs": [_pub_run(row) for row in _runs(conn)]}


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
    row = conn.execute(
        """
        SELECT r.*, e.slug, e.name, e.verdict, e.one_change, e.kind,
               e.created_at, e.parent_id
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        WHERE r.id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    evals = _evals_for_run(conn, run_id)
    videos = [
        {**video, "stream": f"/api/videos/{video['id']}/stream"}
        for video in _videos_for_run(conn, run_id)
    ]
    return {
        "run": _pub_run(row),
        "experiment": {
            "id": row["experiment_id"],
            "slug": row["slug"],
            "name": row["name"],
            "verdict": row["verdict"],
            "one_change": row["one_change"],
            "kind": row["kind"],
            "parent_id": row["parent_id"],
        },
        "evals": evals,
        "videos": videos,
        "chart": {
            "title": row["name"],
            "src": f"/api/runs/{row['id']}/chart.png",
            "wandb_url": row["wandb_url"],
        },
    }


def evals(conn):
    rows = conn.execute(
        """
        SELECT v.*, c.step, c.is_best, c.is_final, e.name AS run_name,
               e.slug, r.id AS run_id
        FROM evals v
        JOIN checkpoints c ON c.id = v.checkpoint_id
        JOIN runs r ON r.id = c.run_id
        JOIN experiments e ON e.id = r.experiment_id
        ORDER BY v.flying_lap_s ASC NULLS LAST, c.step
        """
    ).fetchall()
    return {"evals": rows}


def videos(conn):
    rows = conn.execute(
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
    for row in rows:
        row["stream"] = f"/api/videos/{row['id']}/stream"
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
    like = f"%{q.strip()}%"
    run_rows = conn.execute(
        """
        SELECT r.id, e.name, e.slug, r.wandb_run_id, r.seed
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        WHERE e.name ILIKE %s OR e.slug ILIKE %s OR r.wandb_run_id ILIKE %s
           OR CAST(r.seed AS text) ILIKE %s
        ORDER BY e.created_at
        LIMIT 8
        """,
        (like, like, like, like),
    ).fetchall()
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
    return {"runs": run_rows, "checkpoints": ckpts}


def wandb_gallery(conn):
    rows = _runs(conn)
    return {
        "runs": [
            {
                "id": row["id"],
                "name": row["name"],
                "wandb_url": row["wandb_url"],
                "src": f"/api/runs/{row['id']}/chart.png",
            }
            for row in rows
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


def video_row(conn, video_id):
    return conn.execute(
        "SELECT * FROM videos WHERE id = %s",
        (video_id,),
    ).fetchone()


def chart_inputs(conn, run_id):
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

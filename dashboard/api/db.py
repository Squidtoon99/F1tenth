"""Postgres connection, schema, and the one-time seed."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from seed_data import NS_NAME, experiments

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).with_name("schema.sql")
NS = uuid.UUID("a3f1c2d4-5b6e-4789-8abc-def012345678")


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing. Put it in dashboard/.env")
    return psycopg.connect(url, row_factory=dict_row)


def _eid(slug: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:experiment:{slug}")


def _rid(wandb: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:run:{wandb}")


def _cid(wandb: str, step: int) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:ckpt:{wandb}:{step}")


def _vid(key: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:video:{key}")


def _evid(ckpt: uuid.UUID, scenario: str) -> uuid.UUID:
    return uuid.uuid5(NS, f"{NS_NAME}:eval:{ckpt}:{scenario}")


def _ts(value):
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def apply_schema(conn) -> None:
    sql = SCHEMA.read_text()
    for statement in sql.split(";"):
        chunk = statement.strip()
        if chunk:
            conn.execute(chunk)


def drop_all(conn) -> None:
    conn.execute(
        "DROP TABLE IF EXISTS notes, videos, evals, checkpoints, runs, "
        "experiments CASCADE"
    )


def _ordered(rows):
    by_slug = {exp["slug"]: exp for exp in rows}
    ordered = []
    seen = set()

    def visit(exp):
        if exp["slug"] in seen:
            return
        if exp["parent"]:
            visit(by_slug[exp["parent"]])
        seen.add(exp["slug"])
        ordered.append(exp)

    for exp in rows:
        visit(exp)
    return ordered


def seed(conn) -> None:
    count = conn.execute("SELECT count(*) AS n FROM experiments").fetchone()["n"]
    if count:
        return
    rows = _ordered(experiments())
    for exp in rows:
        conn.execute(
            """
            INSERT INTO experiments
                (id, slug, name, parent_id, kind, one_change, verdict, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                _eid(exp["slug"]),
                exp["slug"],
                exp["name"],
                _eid(exp["parent"]) if exp["parent"] else None,
                exp["kind"],
                exp["one_change"],
                exp["verdict"],
                _ts(exp["created_at"]),
            ),
        )
        run = exp["run"]
        run_id = _rid(run["wandb_run_id"])
        conn.execute(
            """
            INSERT INTO runs (
                id, experiment_id, wandb_run_id, wandb_url, seed, plant,
                transitions, stop_reason, started_at, ended_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                _eid(exp["slug"]),
                run["wandb_run_id"],
                run["wandb_url"],
                run["seed"],
                run["plant"],
                run["transitions"],
                run["stop_reason"],
                _ts(run["started_at"]),
                _ts(run["ended_at"]),
            ),
        )
        for ckpt in exp["checkpoints"]:
            ckpt_id = _cid(run["wandb_run_id"], ckpt["step"])
            conn.execute(
                """
                INSERT INTO checkpoints (
                    id, run_id, step, s3_key, sha256, bytes, is_best, is_final,
                    disk_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    ckpt_id,
                    run_id,
                    ckpt["step"],
                    ckpt["s3_key"],
                    ckpt["sha256"],
                    ckpt["bytes"],
                    ckpt["is_best"],
                    ckpt["is_final"],
                    ckpt["disk_path"],
                ),
            )
            for ev in ckpt["evals"]:
                ev_id = _evid(ckpt_id, ev["scenario"])
                conn.execute(
                    """
                    INSERT INTO evals (
                        id, checkpoint_id, scenario, flying_lap_s, completion,
                        crash, overtakes, steer_reversal, spawn_mps
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ev_id,
                        ckpt_id,
                        ev["scenario"],
                        ev["flying_lap_s"],
                        ev["completion"],
                        ev["crash"],
                        ev["overtakes"],
                        ev["steer_reversal"],
                        ev["spawn_mps"],
                    ),
                )
                for video in ckpt["videos"]:
                    conn.execute(
                        """
                        INSERT INTO videos (
                            id, checkpoint_id, eval_id, kind, s3_key,
                            duration_s, width, height, disk_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            _vid(video["s3_key"]),
                            ckpt_id,
                            ev_id,
                            video["kind"],
                            video["s3_key"],
                            video["duration_s"],
                            video["width"],
                            video["height"],
                            video["disk_path"],
                        ),
                    )


def ensure_db() -> None:
    with connect() as conn:
        apply_schema(conn)
        seed(conn)


def reset_db() -> None:
    with connect() as conn:
        drop_all(conn)
        apply_schema(conn)
        seed(conn)

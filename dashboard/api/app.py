"""Garage API. Postgres and Tigris are reached only from this process."""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request, send_file
from flask.json.provider import DefaultJSONProvider

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from charts import render_chart, series_for  # noqa: E402
from db import connect, ensure_db  # noqa: E402
from queries import (  # noqa: E402
    add_note,
    chart_inputs,
    checkpoint_row,
    evals,
    export_payload,
    files,
    notes,
    overview,
    policies,
    run_detail,
    runs,
    search,
    set_checkpoint_key,
    tree,
    video_row,
    videos,
    wandb_gallery,
)
from storage import configured, open_object, upload  # noqa: E402

_ready = False


class Provider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


def _init():
    global _ready
    if not _ready:
        ensure_db()
        _ready = True


def create_app() -> Flask:
    app = Flask(__name__)
    app.json_provider_class = Provider

    @app.before_request
    def prepare():
        _init()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/overview")
    def overview_route():
        with connect() as conn:
            return overview(conn)

    @app.get("/api/tree")
    def tree_route():
        with connect() as conn:
            return tree(conn)

    @app.get("/api/runs")
    def runs_route():
        with connect() as conn:
            return runs(conn)

    @app.get("/api/runs/<uuid:run_id>")
    def run_route(run_id):
        with connect() as conn:
            payload = run_detail(conn, run_id)
        if payload is None:
            abort(404)
        return payload

    @app.get("/api/runs/<uuid:run_id>/chart.png")
    def chart_route(run_id):
        with connect() as conn:
            found = chart_inputs(conn, run_id)
        if found is None:
            abort(404)
        row, ev = found
        title, points = series_for(row["wandb_run_id"], ev)
        png = render_chart(f"{row['name']} — {title}", points, "s")
        return Response(png, mimetype="image/png")

    @app.get("/api/evals")
    def evals_route():
        with connect() as conn:
            return evals(conn)

    @app.get("/api/videos")
    def videos_route():
        with connect() as conn:
            return videos(conn)

    @app.get("/api/videos/<uuid:video_id>/stream")
    def stream_route(video_id):
        with connect() as conn:
            row = video_row(conn, video_id)
        if row is None:
            abort(404)
        return _stream_video(row)

    @app.get("/api/policies")
    def policies_route():
        with connect() as conn:
            return policies(conn)

    @app.get("/api/files")
    def files_route():
        with connect() as conn:
            return files(conn)

    @app.get("/api/wandb")
    def wandb_route():
        with connect() as conn:
            return wandb_gallery(conn)

    @app.get("/api/search")
    def search_route():
        q = (request.args.get("q") or "").strip()
        if not q:
            return {"runs": [], "checkpoints": []}
        with connect() as conn:
            return search(conn, q)

    @app.get("/api/export")
    def export_route():
        with connect() as conn:
            payload = export_payload(conn)
        response = jsonify(payload)
        response.headers["Content-Disposition"] = "attachment; filename=garage.json"
        return response

    @app.get("/api/notes")
    def notes_route():
        with connect() as conn:
            return notes(conn)

    @app.post("/api/notes")
    def add_note_route():
        data = request.get_json(silent=True) or {}
        body = (data.get("body") or "").strip()
        if not body:
            abort(400, "note is empty")
        if len(body) > 4000:
            abort(400, "note is too long")
        with connect() as conn:
            return add_note(conn, body), 201

    @app.post("/api/checkpoints/<uuid:ckpt_id>/stage")
    def stage_route(ckpt_id):
        with connect() as conn:
            row = checkpoint_row(conn, ckpt_id)
            if row is None:
                abort(404)
            if row["s3_key"]:
                return {"s3_key": row["s3_key"]}
            path = row["disk_path"] or ""
            if not path or not os.path.isfile(path):
                abort(404, "checkpoint is not on this disk")
            key = f"dashboard/checkpoints/{row['slug']}/policy_{row['step']}.pt"
            try:
                upload(path, key)
            except RuntimeError:
                abort(502, "Tigris request failed")
            set_checkpoint_key(conn, ckpt_id, key)
        return {
            "s3_key": key,
            "download": f"/api/checkpoints/{ckpt_id}/download",
        }

    @app.get("/api/checkpoints/<uuid:ckpt_id>/download")
    def download_route(ckpt_id):
        with connect() as conn:
            row = checkpoint_row(conn, ckpt_id)
        if row is None:
            abort(404)
        if not row["s3_key"]:
            abort(409, "file is only on disk; copy it into the bucket first")
        try:
            obj = open_object(row["s3_key"], request.headers.get("Range"))
        except RuntimeError:
            abort(502, "Tigris request failed")
        headers = {
            "Content-Disposition": f'attachment; filename="policy_{row["step"]}.pt"',
            "Content-Type": "application/octet-stream",
            "Accept-Ranges": "bytes",
        }
        if obj.get("ContentLength") is not None:
            headers["Content-Length"] = str(obj["ContentLength"])
        if obj.get("ContentRange"):
            headers["Content-Range"] = obj["ContentRange"]
        status = 206 if request.headers.get("Range") else 200
        return Response(
            obj["Body"].iter_chunks(64 * 1024),
            status=status,
            headers=headers,
        )

    return app


def _stream_video(row):
    path = row["disk_path"] or ""
    if path and os.path.isfile(path):
        return send_file(path, mimetype="video/mp4", conditional=True)
    if row["s3_key"] and configured():
        try:
            obj = open_object(row["s3_key"], request.headers.get("Range"))
        except RuntimeError:
            abort(404, "video is not in the bucket")
        headers = {
            "Content-Type": "video/mp4",
            "Accept-Ranges": "bytes",
        }
        if obj.get("ContentLength") is not None:
            headers["Content-Length"] = str(obj["ContentLength"])
        if obj.get("ContentRange"):
            headers["Content-Range"] = obj["ContentRange"]
        status = 206 if request.headers.get("Range") else 200
        return Response(
            obj["Body"].iter_chunks(64 * 1024),
            status=status,
            headers=headers,
        )
    abort(404, "video is not on this disk")


app = create_app()


def main():
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is missing. Put it in dashboard/.env")
    app.run(host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()

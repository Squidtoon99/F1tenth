#!/usr/bin/env python3
"""One-shot backfill of completed async CPU eval media into an active W&B run.

Attaches as a non-primary shared writer when the installed SDK supports it and
the target run can accept shared mode. Logs via custom axis ``eval/source_step``
(never ``step=<cadence>``), verifies filestream success before writing
``wandb_uploaded.json`` markers, then exits.

If the run already has history and was not started in shared mode, W&B returns
HTTP 409 (``cannot enable shared mode for run with existing history``). In that
case this tool reports a blocker rather than claiming a successful history sync.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gigaflow_f1tenth.async_cpu_eval import (  # noqa: E402
    EVAL_CONFIG_FILENAME,
    REPORT_FILENAME,
    collect_media_paths,
    read_status,
    read_uploaded,
    upload_completed_eval,
    write_uploaded,
)
from gigaflow_f1tenth.config import (  # noqa: E402
    config_from_dict,
    load_config,
    replace_wandb_config,
)
from gigaflow_f1tenth.wandb_log import (  # noqa: E402
    build_wandb_session,
    read_run_meta,
)

_EVAL_DIR_RE = re.compile(r"^eval_(\d{6})$")
_SHARED_BLOCK_RE = re.compile(
    r"cannot enable shared mode for run .* with existing history",
    re.IGNORECASE,
)


def _latest_metrics_step(run_dir: Path) -> int | None:
    newest: tuple[int, float] | None = None
    for path in run_dir.glob("metrics_*.json"):
        try:
            step = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        mtime = path.stat().st_mtime
        if newest is None or mtime > newest[1] or (
            mtime == newest[1] and step > newest[0]
        ):
            newest = (step, mtime)
    return None if newest is None else int(newest[0])


def _completed_eval_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in sorted(run_dir.glob("eval_*")):
        if not path.is_dir():
            continue
        match = _EVAL_DIR_RE.match(path.name)
        if match is None:
            continue
        source_step = int(match.group(1))
        status = read_status(path) or {}
        if status.get("state") != "completed":
            continue
        if not (path / REPORT_FILENAME).is_file():
            continue
        if read_uploaded(path) is not None:
            continue
        found.append((source_step, path))
    return found


def _load_experiment_config(run_dir: Path):
    for candidate in sorted(run_dir.glob("eval_*/" + EVAL_CONFIG_FILENAME)):
        try:
            return config_from_dict(json.loads(candidate.read_text(encoding="utf-8")))
        except Exception:
            continue
    smoke = ROOT / "configs" / "smoke.yaml"
    return load_config(smoke)


def _newest_wandb_run_dir(run_dir: Path, run_id: str) -> Path | None:
    wb = run_dir / "wandb"
    if not wb.is_dir():
        return None
    matches = sorted(
        wb.glob(f"run-*-{run_id}"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _shared_mode_blocked(run_dir: Path, run_id: str) -> str | None:
    latest = _newest_wandb_run_dir(run_dir, run_id)
    if latest is None:
        return None
    for name in ("debug-internal.log", "debug-core.log", "debug.log"):
        path = latest / "logs" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _SHARED_BLOCK_RE.search(text) or "409 Conflict" in text:
            return (
                "W&B refused shared-mode filestream "
                "(cannot enable shared mode for a run that already has history). "
                "History/metrics were not synced. Start future runs with "
                "mode=shared on the primary if concurrent writers are required; "
                "otherwise rely on the parent trainer's custom-axis upload path."
            )
    return None


def _history_synced(run_dir: Path, run_id: str) -> bool:
    latest = _newest_wandb_run_dir(run_dir, run_id)
    if latest is None:
        return False
    internal = latest / "logs" / "debug-internal.log"
    if not internal.is_file():
        return False
    text = internal.read_text(encoding="utf-8", errors="replace")
    if _SHARED_BLOCK_RE.search(text) or "409 Conflict" in text:
        return False
    return "history_lines" in text and "fatal error" not in text


def _attach_shared_session(
    *,
    run_dir: Path,
    entity: str | None,
    project: str,
    run_id: str,
):
    import wandb

    try:
        probe = wandb.Settings(mode="shared", x_primary=False)
        if str(probe.mode) != "shared":
            raise RuntimeError(f"unexpected mode={probe.mode!r}")
    except Exception as exc:
        raise RuntimeError(
            f"wandb shared mode unsupported by installed SDK: {exc}"
        ) from exc

    cfg = replace_wandb_config(
        _load_experiment_config(run_dir),
        enabled=True,
        mode="online",
        project=project,
        entity=entity,
        run_id=run_id,
        resume="allow",
    )
    session = build_wandb_session(cfg, run_dir=run_dir)
    settings = wandb.Settings(
        mode="shared",
        x_primary=False,
        x_update_finish_state=False,
        x_label="async-eval-backfill",
    )
    init_kwargs = {
        "id": run_id,
        "project": project,
        "resume": "allow",
        "reinit": True,
        "settings": settings,
        "dir": str(run_dir),
    }
    if entity:
        init_kwargs["entity"] = entity
    run = wandb.init(**init_kwargs)
    session._started = True
    session._run = run
    session._run_id = getattr(run, "id", None) or run_id
    session._logging_disabled = False
    session.define_eval_axis()
    return session


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--entity", type=str, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List completed unlogged eval dirs without uploading",
    )
    p.add_argument(
        "--post-check-s",
        type=float,
        default=20.0,
        help="Seconds to wait before verifying training metrics advanced",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run dir missing: {run_dir}", file=sys.stderr)
        return 2

    meta = read_run_meta(run_dir).get("wandb") or {}
    run_id = args.run_id or meta.get("run_id")
    project = args.project or meta.get("project")
    entity = args.entity if args.entity is not None else meta.get("entity")
    if not run_id or not project:
        print("need --run-id/--project or run_meta.json wandb fields", file=sys.stderr)
        return 2

    pending = _completed_eval_dirs(run_dir)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "run_id": run_id,
                "project": project,
                "entity": entity,
                "pending": [
                    {
                        "source_step": step,
                        "dir": str(path),
                        "media_count": len(collect_media_paths(path)),
                    }
                    for step, path in pending
                ],
                "latest_metrics_step": _latest_metrics_step(run_dir),
            },
            indent=2,
        )
    )
    if args.dry_run or not pending:
        return 0

    before_step = _latest_metrics_step(run_dir)
    try:
        session = _attach_shared_session(
            run_dir=run_dir,
            entity=entity,
            project=str(project),
            run_id=str(run_id),
        )
    except RuntimeError as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 3

    staged: list[dict] = []
    try:
        for source_step, path in pending:
            # Shared mode ignores wandb.log(step=...); custom axis carries cadence.
            marker = upload_completed_eval(
                session,
                source_step=source_step,
                out_dir=path,
                train_step=None,
                mark_backfill=True,
                persist_marker=False,
            )
            if marker is None:
                print(f"failed source_step={source_step}")
                continue
            staged.append({"path": str(path), **dict(marker)})
            print(
                f"staged eval/source_step={source_step} "
                f"media={marker['media_count']}"
            )
    finally:
        session.finish()

    block = _shared_mode_blocked(run_dir, str(run_id))
    synced = _history_synced(run_dir, str(run_id)) and block is None
    uploaded: list[dict] = []
    if synced:
        for row in staged:
            path = Path(row["path"])
            marker = {k: v for k, v in row.items() if k != "path"}
            write_uploaded(path, marker)
            uploaded.append(marker)
    else:
        reason = block or "filestream history sync was not confirmed"
        print(f"blocker: {reason}", file=sys.stderr)

    if args.post_check_s > 0:
        time.sleep(float(args.post_check_s))
    after_step = _latest_metrics_step(run_dir)
    result = {
        "uploaded": uploaded,
        "staged_but_unsynced": [] if synced else staged,
        "history_synced": synced,
        "shared_mode_blocker": block,
        "metrics_step_before": before_step,
        "metrics_step_after": after_step,
        "training_advanced_or_steady": (
            after_step is not None
            and before_step is not None
            and after_step >= before_step
        ),
        "wandb_url": (
            f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            if entity
            else f"https://wandb.ai/{project}/runs/{run_id}"
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if synced else 3


if __name__ == "__main__":
    raise SystemExit(main())

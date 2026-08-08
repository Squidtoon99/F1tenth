"""Isolated asynchronous CPU evaluation for cadenced mid-train checks.

Cadenced eval runs in a child process with CUDA hidden so the parent GPU
trainer never shares the device with evaluation allocations. The parent owns
the W&B run and only uploads completed local reports/media after reaping.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from gigaflow_f1tenth.artifacts import export_actor_artifact
from gigaflow_f1tenth.config import ExperimentConfig, config_from_dict, config_to_dict
from gigaflow_f1tenth.evaluation import (
    EvalMetrics,
    EvalReport,
    load_actor_from_checkpoint,
    resolve_eval_device,
    resolve_eval_num_worlds,
    run_evaluation,
)

STATUS_FILENAME = "eval_status.json"
REPORT_FILENAME = "eval_report.json"
UPLOADED_FILENAME = "wandb_uploaded.json"
ACTOR_SNAPSHOT = "actor_snapshot.pt"
EVAL_CONFIG_FILENAME = "eval_config.json"
WORKER_LOG_FILENAME = "eval_worker.log"
SKIP_REASON_INFLIGHT = "previous_eval_inflight"

_STATE_PENDING = "pending"
_STATE_RUNNING = "running"
_STATE_COMPLETED = "completed"
_STATE_FAILED = "failed"
_STATE_SKIPPED = "skipped"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def eval_output_dir(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / f"eval_{int(step):06d}"


def status_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / STATUS_FILENAME


def write_status(out_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = status_path(out)
    body = dict(payload)
    body["updated_at"] = _utc_now()
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_status(out_dir: str | Path) -> dict[str, Any] | None:
    path = status_path(out_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cpu_hidden_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that hides every CUDA device from the child process."""
    env = dict(os.environ if base is None else base)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Keep child from accidentally attaching to the parent W&B run.
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_RESUME", None)
    return env


def assert_no_cuda(context: str = "cpu eval") -> dict[str, Any]:
    """Fail if this process can see a CUDA device (must run after CUDA hide)."""
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    info = {
        "cuda_visible_devices": visible,
        "torch_cuda_available": available,
        "cuda_device_count": count,
    }
    if available or count > 0:
        raise RuntimeError(
            f"{context}: CUDA must be hidden "
            f"(CUDA_VISIBLE_DEVICES={visible!r}, "
            f"is_available={available}, device_count={count})"
        )
    if visible not in {"", "-1"}:
        # Empty string is the intentional hide; reject accidental GPU lists.
        raise RuntimeError(
            f"{context}: expected CUDA_VISIBLE_DEVICES='' or '-1', got {visible!r}"
        )
    return info


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_cpu_eval_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Freeze an eval-only config: CPU device + eval world count.

    Keeps ``worlds.num_worlds`` at the training scale so PPO layout validation
    (minibatch vs transitions) stays meaningful. Evaluation scale is applied
    only via ``evaluation.num_worlds`` inside suite overrides.
    """
    raw = config_to_dict(cfg)
    worlds = dict(raw["worlds"])
    worlds["device"] = "cpu"
    raw["worlds"] = worlds
    evaluation = dict(raw["evaluation"])
    evaluation["device"] = "cpu"
    evaluation["num_worlds"] = int(resolve_eval_num_worlds(cfg))
    raw["evaluation"] = evaluation
    # Child never owns W&B.
    wb = dict(raw.get("wandb", {}))
    wb["enabled"] = False
    wb["mode"] = "disabled"
    raw["wandb"] = wb
    return config_from_dict(raw)


def export_cpu_actor_snapshot(
    cfg: ExperimentConfig, actor: Any, path: str | Path
) -> Path:
    """Write an actor-only CPU snapshot (no CUDA tensors retained in the file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_actor_artifact(cfg, actor, str(path))
    # Prove artifact tensors are host-side.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["actor_state_dict"]
    for key, tensor in state.items():
        if hasattr(tensor, "is_cuda") and tensor.is_cuda:
            raise RuntimeError(f"actor snapshot retained CUDA tensor: {key}")
    return path


def load_reports_from_json(path: str | Path) -> list[EvalReport]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reports: list[EvalReport] = []
    for row in payload.get("reports", []):
        metrics = EvalMetrics(**row["metrics"])
        reports.append(
            EvalReport(
                suite=str(row["suite"]),
                seed=int(row["seed"]),
                metrics=metrics,
                extras=dict(row.get("extras") or {}),
            )
        )
    return reports


def collect_media_paths(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    if not out.is_dir():
        return []
    media: list[Path] = []
    for pattern in ("*.png", "*.gif", "*.mp4", "*.webm"):
        media.extend(out.glob(pattern))
    return sorted(media)


def uploaded_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / UPLOADED_FILENAME


def read_uploaded(out_dir: str | Path) -> dict[str, Any] | None:
    path = uploaded_path(out_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_uploaded(out_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = uploaded_path(out)
    body = dict(payload)
    body.setdefault("uploaded_at", _utc_now())
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_eval_global_step(
    *,
    train_step: int | None = None,
    session: Any | None = None,
    source_step: int | None = None,
) -> int | None:
    """Monotonic W&B global history step for a delayed eval upload.

    Never returns the eval cadence/source step. Prefer the caller's current
    train step and the session's last logged global step; ``None`` omits an
    explicit global step (SDK appends safely). ``source_step`` is accepted only
    so callers cannot accidentally use it as the global history step.
    """
    del source_step  # cadence belongs on eval/source_step, never global step=
    candidates: list[int] = []
    if train_step is not None:
        candidates.append(int(train_step))
    last = getattr(session, "last_step", None) if session is not None else None
    if last is not None:
        candidates.append(int(last))
    if not candidates:
        return None
    return max(candidates)


# Backward-compatible name used by earlier step-order drafts/tests.
resolve_eval_upload_step = resolve_eval_global_step


def upload_completed_eval(
    session: Any | None,
    *,
    source_step: int,
    out_dir: str | Path,
    status: Mapping[str, Any] | None = None,
    train_step: int | None = None,
    training_num_worlds: int | None = None,
    mark_backfill: bool = False,
    persist_marker: bool = True,
) -> dict[str, Any] | None:
    """Upload one completed eval dir using custom axis ``eval/source_step``.

    Global W&B ``step`` stays monotonic with training; cadence is only the
    custom eval axis value. Returns the upload marker payload, or ``None``.
    """
    out = Path(out_dir)
    prior = read_uploaded(out)
    if prior is not None:
        return dict(prior)
    report_path = out / REPORT_FILENAME
    if not report_path.is_file():
        warnings.warn(
            f"async CPU eval completed without {REPORT_FILENAME} at {out}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    status_payload = dict(status or read_status(out) or {})
    reports = load_reports_from_json(report_path)
    media = collect_media_paths(out)
    global_step = resolve_eval_global_step(
        train_step=train_step,
        session=session,
    )
    if session is not None and getattr(session, "active", False):
        worlds_fallback = (
            float(training_num_worlds)
            if training_num_worlds is not None
            else float(status_payload.get("training_num_worlds") or 0.0)
        )
        session.log_evaluation(
            reports,
            source_step=int(source_step),
            media_paths=media,
            global_step=global_step,
            extra_metrics={
                "eval/async/elapsed_s": float(status_payload.get("elapsed_s") or 0.0),
                "eval/async/num_worlds": float(status_payload.get("num_worlds") or 0.0),
                "eval/async/training_num_worlds": worlds_fallback,
                "eval/async/cuda_device_count": float(
                    status_payload.get("cuda_device_count") or 0.0
                ),
                "eval/async/media_count": float(len(media)),
                "eval/async/backfill": 1.0 if mark_backfill else 0.0,
                "eval/async/global_step": (
                    float(global_step) if global_step is not None else -1.0
                ),
            },
        )
    marker = {
        "source_step": int(source_step),
        "global_step": global_step,
        "upload_step": global_step,
        "media_count": len(media),
        "num_reports": len(reports),
        "backfill": bool(mark_backfill),
        "report_path": str(report_path),
    }
    if persist_marker:
        write_uploaded(out, marker)
    return marker


def run_cpu_eval_worker(
    *,
    config_path: str | Path,
    actor_path: str | Path,
    output_dir: str | Path,
    step: int,
) -> int:
    """Child-process entry: CPU-only evaluation with local JSON/media outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    status: dict[str, Any] = {
        "state": _STATE_RUNNING,
        "pid": os.getpid(),
        "step": int(step),
        "started_at": _utc_now(),
        "config_path": str(config_path),
        "actor_path": str(actor_path),
        "output_dir": str(out_dir),
        "device": "cpu",
    }
    write_status(out_dir, status)
    try:
        cuda_info = assert_no_cuda("cpu eval worker")
        status.update(cuda_info)
        write_status(out_dir, status)

        cfg = build_cpu_eval_config(
            config_from_dict(
                json.loads(Path(config_path).read_text(encoding="utf-8"))
            )
        )
        status["num_worlds"] = int(resolve_eval_num_worlds(cfg))
        status["training_num_worlds"] = int(cfg.worlds.num_worlds)
        status["training_num_worlds_note"] = (
            "evaluation.num_worlds is eval-only; "
            "worlds.num_worlds remains the training scale"
        )
        write_status(out_dir, status)

        actor = load_actor_from_checkpoint(cfg, actor_path, device="cpu")
        actor.eval()
        reports = run_evaluation(
            cfg,
            suites=cfg.evaluation.suite,
            actor=actor,
            device="cpu",
            output_dir=out_dir,
        )
        elapsed = time.perf_counter() - t0
        status.update(
            {
                "state": _STATE_COMPLETED,
                "finished_at": _utc_now(),
                "elapsed_s": elapsed,
                "exit_code": 0,
                "num_reports": len(reports),
                "report_path": str(out_dir / REPORT_FILENAME),
                "media_count": len(collect_media_paths(out_dir)),
                "error": None,
            }
        )
        write_status(out_dir, status)
        return 0
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        status.update(
            {
                "state": _STATE_FAILED,
                "finished_at": _utc_now(),
                "elapsed_s": elapsed,
                "exit_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_status(out_dir, status)
        print(status["traceback"], file=sys.stderr)
        return 1


@dataclass
class _InflightEval:
    step: int
    out_dir: Path
    proc: subprocess.Popen
    launched_at: float


class AsyncCpuEvalManager:
    """At-most-one in-flight CPU eval subprocess with skip/coalesce semantics."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        run_dir: str | Path | None,
        python_executable: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.python_executable = python_executable or sys.executable
        self._inflight: _InflightEval | None = None
        self.last_launch_result: str | None = None
        self.last_skip: dict[str, Any] | None = None
        self.completed_uploads: list[int] = []
        self.completed_upload_records: list[dict[str, Any]] = []

    @property
    def inflight(self) -> bool:
        return self._inflight is not None

    @property
    def inflight_pid(self) -> int | None:
        if self._inflight is None:
            return None
        return int(self._inflight.proc.pid)

    def enabled_for_cadence(self) -> bool:
        return resolve_eval_device(self.cfg) == "cpu"

    def poll_and_upload(
        self,
        session: Any | None,
        *,
        train_step: int | None = None,
    ) -> dict[str, Any] | None:
        """Reap a finished child (if any) and upload local artifacts via parent W&B."""
        handle = self._inflight
        if handle is None:
            return None
        rc = handle.proc.poll()
        if rc is None:
            return read_status(handle.out_dir)
        # Drain and reap to avoid zombies.
        try:
            handle.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
            handle.proc.communicate(timeout=5)
        except Exception:
            pass
        status = read_status(handle.out_dir) or {}
        final_rc = handle.proc.returncode
        if final_rc is None:
            final_rc = rc
        status["exit_code"] = int(final_rc)
        if status.get("state") not in {_STATE_COMPLETED, _STATE_FAILED}:
            status["state"] = (
                _STATE_COMPLETED if final_rc == 0 else _STATE_FAILED
            )
            if final_rc != 0 and not status.get("error"):
                status["error"] = f"worker_exit_{final_rc}"
            write_status(handle.out_dir, status)
        if final_rc == 0:
            self._upload_completed(
                session,
                handle.step,
                handle.out_dir,
                status,
                train_step=train_step,
            )
        elif not str(status.get("termination_reason") or "").startswith(
            "trainer_shutdown_"
        ):
            warnings.warn(
                f"async CPU eval failed step={handle.step} "
                f"pid={handle.proc.pid} rc={rc} "
                f"error={status.get('error')!r} "
                f"log={handle.out_dir / WORKER_LOG_FILENAME}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._inflight = None
        return status

    def try_launch(
        self,
        *,
        step: int,
        actor: Any,
        session: Any | None = None,
    ) -> str:
        """Launch CPU eval or skip/coalesce if one is already running."""
        self.poll_and_upload(session, train_step=int(step))
        if self.run_dir is None:
            self.last_launch_result = "no_run_dir"
            return self.last_launch_result
        out_dir = eval_output_dir(self.run_dir, step)
        if self._inflight is not None:
            skip = {
                "state": _STATE_SKIPPED,
                "reason": SKIP_REASON_INFLIGHT,
                "step": int(step),
                "inflight_step": int(self._inflight.step),
                "inflight_pid": int(self._inflight.proc.pid),
                "started_at": _utc_now(),
                "device": "cpu",
                "num_worlds": int(resolve_eval_num_worlds(self.cfg)),
                "training_num_worlds": int(self.cfg.worlds.num_worlds),
            }
            write_status(out_dir, skip)
            self.last_skip = skip
            self.last_launch_result = "skipped_inflight"
            return self.last_launch_result

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / EVAL_CONFIG_FILENAME
        actor_path = out_dir / ACTOR_SNAPSHOT
        cfg_path.write_text(
            json.dumps(config_to_dict(self.cfg), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        export_cpu_actor_snapshot(self.cfg, actor, actor_path)
        write_status(
            out_dir,
            {
                "state": _STATE_PENDING,
                "step": int(step),
                "started_at": _utc_now(),
                "device": "cpu",
                "num_worlds": int(resolve_eval_num_worlds(self.cfg)),
                "training_num_worlds": int(self.cfg.worlds.num_worlds),
                "actor_path": str(actor_path),
                "config_path": str(cfg_path),
            },
        )
        log_path = out_dir / WORKER_LOG_FILENAME
        cmd = [
            self.python_executable,
            "-m",
            "gigaflow_f1tenth.async_cpu_eval",
            "--config",
            str(cfg_path),
            "--actor",
            str(actor_path),
            "--output-dir",
            str(out_dir),
            "--step",
            str(int(step)),
        ]
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                env=cpu_hidden_env(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_fh.close()
            raise
        # Parent keeps the log handle only until Popen steals the fd; close our copy.
        log_fh.close()
        self._inflight = _InflightEval(
            step=int(step),
            out_dir=out_dir,
            proc=proc,
            launched_at=time.perf_counter(),
        )
        write_status(
            out_dir,
            {
                "state": _STATE_RUNNING,
                "pid": int(proc.pid),
                "step": int(step),
                "started_at": _utc_now(),
                "device": "cpu",
                "num_worlds": int(resolve_eval_num_worlds(self.cfg)),
                "training_num_worlds": int(self.cfg.worlds.num_worlds),
                "actor_path": str(actor_path),
                "config_path": str(cfg_path),
                "worker_log": str(log_path),
            },
        )
        self.last_launch_result = "launched"
        return self.last_launch_result

    def wait(
        self,
        session: Any | None = None,
        *,
        timeout_s: float | None = None,
        train_step: int | None = None,
    ) -> dict[str, Any] | None:
        """Block until the in-flight eval finishes (or timeout), then upload."""
        handle = self._inflight
        if handle is None:
            return None
        try:
            handle.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"async CPU eval still running after timeout "
                f"(pid={handle.proc.pid}, step={handle.step}); leaving child alive",
                RuntimeWarning,
                stacklevel=2,
            )
            return read_status(handle.out_dir)
        return self.poll_and_upload(session, train_step=train_step)

    def shutdown(
        self,
        session: Any | None = None,
        *,
        kill: bool = False,
        train_step: int | None = None,
        grace_s: float = 30.0,
        term_wait_s: float = 10.0,
        kill_wait_s: float = 5.0,
    ) -> None:
        del kill
        handle = self._inflight
        if handle is None:
            return
        sent_term = False
        if handle.proc.poll() is None:
            try:
                handle.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                status = read_status(handle.out_dir) or {}
                status.update(
                    {
                        "state": _STATE_FAILED,
                        "termination_reason": "trainer_shutdown_sigterm",
                        "partial_report": (handle.out_dir / REPORT_FILENAME).is_file(),
                    }
                )
                write_status(handle.out_dir, status)
                try:
                    os.killpg(handle.proc.pid, signal.SIGTERM)
                    sent_term = True
                except ProcessLookupError:
                    pass
        if sent_term:
            deadline = time.monotonic() + term_wait_s
            while (
                _process_group_exists(handle.proc.pid)
                and time.monotonic() < deadline
            ):
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            if _process_group_exists(handle.proc.pid):
                status = read_status(handle.out_dir) or {}
                status["termination_reason"] = "trainer_shutdown_sigkill"
                write_status(handle.out_dir, status)
                try:
                    os.killpg(handle.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    handle.proc.wait(timeout=kill_wait_s)
                except subprocess.TimeoutExpired:
                    warnings.warn(
                        f"async CPU eval process group did not reap "
                        f"(pid={handle.proc.pid}, step={handle.step})",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        elif handle.proc.poll() is None:
            try:
                handle.proc.wait(timeout=term_wait_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(handle.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                handle.proc.wait(timeout=kill_wait_s)
        try:
            self.poll_and_upload(session, train_step=train_step)
        finally:
            self._inflight = None

    def _upload_completed(
        self,
        session: Any | None,
        step: int,
        out_dir: Path,
        status: Mapping[str, Any],
        *,
        train_step: int | None = None,
    ) -> None:
        marker = upload_completed_eval(
            session,
            source_step=int(step),
            out_dir=out_dir,
            status=status,
            train_step=train_step,
            training_num_worlds=int(self.cfg.worlds.num_worlds),
            mark_backfill=False,
        )
        if marker is None:
            return
        self.completed_uploads.append(int(step))
        self.completed_upload_records.append(dict(marker))


def _parse_args(argv: Sequence[str] | None = None):
    import argparse

    p = argparse.ArgumentParser(description="CPU-only gigaflow eval worker")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--actor", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_cpu_eval_worker(
        config_path=args.config,
        actor_path=args.actor,
        output_dir=args.output_dir,
        step=args.step,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared per-run output directory layout for standalone training."""

from __future__ import annotations

from pathlib import Path

# Each training run: outputs/runs/<run-id>/{checkpoints/,run.log,config.json}
RUNS_ROOT = Path("outputs/runs")
ORCHESTRATOR_META_DIR = RUNS_ROOT / "_orchestrator"


def default_run_dir(run_id: str, *, root: Path | None = None) -> Path:
    base = root if root is not None else Path.cwd()
    return base / RUNS_ROOT / run_id


def checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def run_log_path(run_dir: Path) -> Path:
    return run_dir / "run.log"


def config_snapshot_path(run_dir: Path) -> Path:
    return run_dir / "config.json"


def run_lock_path(run_dir: Path) -> Path:
    return run_dir / "run.lock"

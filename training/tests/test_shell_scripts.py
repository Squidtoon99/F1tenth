"""Launcher and watcher shell scripts must parse under bash -n."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _shell_scripts() -> list[Path]:
    roots = (_REPO / "tools", _REPO / "training" / "outputs" / "experiments")
    scripts: list[Path] = []
    for root in roots:
        if root.is_dir():
            scripts.extend(sorted(root.rglob("*.sh")))
    return scripts


def test_shell_scripts_pass_bash_n():
    scripts = _shell_scripts()
    assert scripts, "expected shell scripts under tools/ and training/outputs/experiments/"
    failures: list[str] = []
    for script in scripts:
        proc = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip() or "bash -n failed"
            failures.append(f"{script}: {detail}")
    assert not failures, "shell syntax errors:\n" + "\n".join(failures)

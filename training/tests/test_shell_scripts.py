"""Launcher and watcher shell scripts must parse under bash -n."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TRAINING = _REPO / "training"
_PCPLUS2B_LOG = _TRAINING / "outputs/runs/pcplus2b-a001/run.log"
_REGRESSED_LOG = (
    _TRAINING
    / "outputs/runs/progab-control-a001/run.log.attempt7-regressed-20260729"
)
_LIVE_CHAIN_LOG = _TRAINING / "outputs/runs/progab-a001-chain.log"


def _shell_scripts() -> list[Path]:
    roots = (_REPO / "tools", _REPO / "training" / "outputs" / "experiments")
    scripts: list[Path] = []
    for root in roots:
        if root.is_dir():
            scripts.extend(sorted(root.rglob("*.sh")))
    return scripts


def _chain_log_tail() -> str:
    if not _LIVE_CHAIN_LOG.is_file():
        return ""
    return _LIVE_CHAIN_LOG.read_text(errors="replace")[-4096:]


def _trainer_pids() -> set[str]:
    proc = subprocess.run(
        ["pgrep", "-f", r"[.]venv/bin/python.*standalone_trainer\.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


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


def test_progab_gates_sourceable_without_launching_trainer():
    before = _trainer_pids()
    chain_before = _chain_log_tail()

    proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source tools/progab_gates.sh && "
                "type check_interim_400m_gate check_validity_1b_gate latest_transitions"
            ),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    chain_proc = subprocess.run(
        [
            "bash",
            "-c",
            (
                "source tools/progab_chain.sh && "
                "type check_interim_400m_gate monitor_interim_gate run_arm"
            ),
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert chain_proc.returncode == 0, chain_proc.stderr or chain_proc.stdout

    time.sleep(1)
    after = _trainer_pids()
    assert after == before, f"sourcing progab scripts spawned trainers: {after - before}"

    chain_after = _chain_log_tail()
    assert chain_after == chain_before, "sourcing progab_chain.sh appended to chain log"


def test_progab_gates_against_reference_logs():
    assert _PCPLUS2B_LOG.is_file(), f"missing reference log {_PCPLUS2B_LOG}"
    assert _REGRESSED_LOG.is_file(), f"missing regressed log {_REGRESSED_LOG}"

    healthy_400m = subprocess.run(
        [
            "bash",
            "-c",
            f'source tools/progab_gates.sh; check_interim_400m_gate "{_PCPLUS2B_LOG}"',
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert healthy_400m.returncode == 0, healthy_400m.stdout + healthy_400m.stderr
    assert "INTERIM_400M" in healthy_400m.stdout
    assert "pass=True" in healthy_400m.stdout

    healthy_1b = subprocess.run(
        [
            "bash",
            "-c",
            f'source tools/progab_gates.sh; check_validity_1b_gate "{_PCPLUS2B_LOG}"',
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert healthy_1b.returncode == 0, healthy_1b.stdout + healthy_1b.stderr
    assert "VALIDITY_1B" in healthy_1b.stdout
    assert "pass=True" in healthy_1b.stdout

    regressed_400m = subprocess.run(
        [
            "bash",
            "-c",
            f'source tools/progab_gates.sh; check_interim_400m_gate "{_REGRESSED_LOG}"',
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert regressed_400m.returncode != 0, (
        "regressed attempt-7 log must fail 400M gate: " + regressed_400m.stdout
    )
    assert "INTERIM_400M" in regressed_400m.stdout
    assert "pass=False" in regressed_400m.stdout


def test_progab_gates_no_data_on_truncated_log(tmp_path):
    truncated = tmp_path / "truncated.log"
    truncated.write_text(
        "[2026-07-29 15:00:00] standalone_trainer INFO: ticks=100 transitions=1000000\n"
    )
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'source tools/progab_gates.sh; check_interim_400m_gate "{truncated}"',
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "INTERIM_400M NO_DATA" in proc.stdout

    proc_1b = subprocess.run(
        [
            "bash",
            "-c",
            f'source tools/progab_gates.sh; check_validity_1b_gate "{truncated}"',
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_1b.returncode != 0
    assert "VALIDITY_1B NO_DATA" in proc_1b.stdout

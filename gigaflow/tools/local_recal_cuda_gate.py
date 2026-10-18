#!/usr/bin/env python3
"""Local RTX 4080 gate for recalibrated safety policy."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "_runtime_recal_local_gate.yaml"
MIN_UPDATES = 200


def main() -> int:
    if not CONFIG.is_file():
        raise SystemExit(f"missing runtime config: {CONFIG}")
    run_dir = Path(tempfile.mkdtemp(prefix="recal_gate_"))
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        cmd = [
            sys.executable,
            "-m",
            "gigaflow_f1tenth.cli",
            "train",
            "--config",
            str(CONFIG),
            "--device",
            "cuda",
            "--num-updates",
            str(MIN_UPDATES + 20),
            "--run-dir",
            str(run_dir),
            "--checkpoint-interval",
            "100",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"training failed rc={proc.returncode}")

        metrics_files = sorted(run_dir.glob("metrics_*.json"))
        if len(metrics_files) < MIN_UPDATES // 10:
            raise SystemExit(
                f"expected at least {MIN_UPDATES // 10} metric files, got {len(metrics_files)}"
            )
        max_update = int(metrics_files[-1].stem.split("_")[-1])
        if max_update < MIN_UPDATES:
            raise SystemExit(f"expected update >= {MIN_UPDATES}, got {max_update}")

        ok = 0
        for path in metrics_files:
            progress = json.loads(path.read_text())["progress"]
            checks = {
                "finite": all(
                    math.isfinite(float(progress[key]))
                    for key in (
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "grad_norm",
                        "candidate_kl",
                        "full_rollout_kl",
                    )
                ),
                "parity": float(progress["pre_update_approx_kl"]) == 0.0,
                "no_rollback_stop": float(progress["rollback_stop_requested"]) == 0.0,
            }
            if all(checks.values()):
                ok += 1
        if ok < len(metrics_files) * 0.95:
            raise SystemExit(f"metric gate failed: {ok}/{len(metrics_files)} rows ok")

        ckpt = sorted(run_dir.glob("ckpt_*.pt"))
        if not ckpt:
            raise SystemExit("missing checkpoint")
        resume_cmd = [
            sys.executable,
            "-m",
            "gigaflow_f1tenth.cli",
            "train",
            "--config",
            str(CONFIG),
            "--device",
            "cuda",
            "--num-updates",
            "1",
            "--run-dir",
            str(run_dir / "resume"),
            "--resume-from",
            str(ckpt[-1]),
        ]
        resume = subprocess.run(resume_cmd, cwd=str(ROOT), env=env, check=False)
        if resume.returncode != 0:
            raise SystemExit(f"resume failed rc={resume.returncode}")

        print(
            json.dumps(
                {
                    "gate": "LOCAL_RECAL_CUDA_OK",
                    "updates": max_update,
                    "metrics_files": len(metrics_files),
                    "rollback_stop_rows": sum(
                        1
                        for path in metrics_files
                        if float(
                            json.loads(path.read_text())["progress"][
                                "rollback_stop_requested"
                            ]
                        )
                        > 0.0
                    ),
                },
                indent=2,
            )
        )
        return 0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

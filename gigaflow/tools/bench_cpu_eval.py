#!/usr/bin/env python3
"""Benchmark CPU evaluation wall-time scaling vs world count.

Runs the real CPU eval worker under CUDA_VISIBLE_DEVICES="" for each world
count, with a freshly exported actor snapshot. Prints JSON summarizing elapsed
time and whether a full production suite at 512 worlds is feasible relative to
the cadenced eval interval.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gigaflow_f1tenth.artifacts import export_actor_artifact  # noqa: E402
from gigaflow_f1tenth.async_cpu_eval import (  # noqa: E402
    REPORT_FILENAME,
    STATUS_FILENAME,
    cpu_hidden_env,
    read_status,
)
from gigaflow_f1tenth.config import config_to_dict, load_config  # noqa: E402
from gigaflow_f1tenth.model import build_actor  # noqa: E402


def _run_one(
    *,
    cfg_path: Path,
    actor_path: Path,
    out_dir: Path,
    python: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench_worker.log"
    cmd = [
        python,
        "-m",
        "gigaflow_f1tenth.async_cpu_eval",
        "--config",
        str(cfg_path),
        "--actor",
        str(actor_path),
        "--output-dir",
        str(out_dir),
        "--step",
        "0",
    ]
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.run(
            cmd,
            env=cpu_hidden_env(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    status = read_status(out_dir) or {}
    return {
        "exit_code": int(proc.returncode),
        "elapsed_s": elapsed,
        "status_elapsed_s": status.get("elapsed_s"),
        "cuda_device_count": status.get("cuda_device_count"),
        "torch_cuda_available": status.get("torch_cuda_available"),
        "num_worlds": status.get("num_worlds"),
        "state": status.get("state"),
        "error": status.get("error"),
        "report_exists": (out_dir / REPORT_FILENAME).is_file(),
        "status_path": str(out_dir / STATUS_FILENAME),
        "log_path": str(log_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "production_h100.yaml",
    )
    p.add_argument(
        "--worlds",
        type=str,
        default="1,2,4,8,16,23,32",
        help="Comma-separated evaluation world counts to measure",
    )
    p.add_argument(
        "--soak-steps",
        type=int,
        default=40,
        help="Horizon cap for scaling probe (use 2000 for full-suite estimate)",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated eval seeds (default single seed for scaling)",
    )
    p.add_argument(
        "--suites",
        type=str,
        default="solo",
        help="Comma-separated suites (default solo for scaling)",
    )
    p.add_argument(
        "--viz-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--eval-interval-updates",
        type=int,
        default=100,
        help="Cadence used to judge feasibility vs training",
    )
    p.add_argument(
        "--update-wall-s",
        type=float,
        default=20.0,
        help="Assumed GPU train update wall time for feasibility window",
    )
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    base = load_config(args.config)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    world_counts = [int(x) for x in args.worlds.split(",") if x.strip()]

    rows = []
    with tempfile.TemporaryDirectory(prefix="bench_cpu_eval_") as tmp:
        tmp_path = Path(tmp)
        actor = build_actor(base)
        actor_path = tmp_path / "actor_snapshot.pt"
        export_actor_artifact(base, actor, str(actor_path))

        for n_worlds in world_counts:
            cfg = replace(
                base,
                worlds=replace(base.worlds, num_worlds=512, device="cuda"),
                evaluation=replace(
                    base.evaluation,
                    device="cpu",
                    num_worlds=int(n_worlds),
                    soak_steps=int(args.soak_steps),
                    seeds=seeds,
                    suite=suites,
                    viz_enabled=bool(args.viz_enabled),
                ),
                wandb=replace(base.wandb, enabled=False, mode="disabled"),
            )
            cfg_path = tmp_path / f"cfg_w{n_worlds}.json"
            cfg_path.write_text(
                json.dumps(config_to_dict(cfg), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            out_dir = tmp_path / f"eval_w{n_worlds}"
            print(f"[bench] worlds={n_worlds} starting", flush=True)
            row = _run_one(
                cfg_path=cfg_path,
                actor_path=actor_path,
                out_dir=out_dir,
                python=sys.executable,
            )
            row["requested_num_worlds"] = int(n_worlds)
            row["soak_steps"] = int(args.soak_steps)
            row["suites"] = list(suites)
            row["seeds"] = list(seeds)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    # Linear fit elapsed ~ a + b * worlds for feasibility projection.
    xs = [float(r["requested_num_worlds"]) for r in rows if r["exit_code"] == 0]
    ys = [float(r["elapsed_s"]) for r in rows if r["exit_code"] == 0]
    slope = intercept = None
    if len(xs) >= 2:
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs) or 1.0
        slope = num / den
        intercept = y_mean - slope * x_mean

    cadence_window_s = float(args.eval_interval_updates) * float(args.update_wall_s)
    # Full-suite multiplier from this probe (suites*seeds*soak relative to prod).
    prod_suites = len(base.evaluation.suite)
    prod_seeds = len(base.evaluation.seeds)
    prod_soak = int(base.evaluation.soak_steps)
    probe_suites = max(len(suites), 1)
    probe_seeds = max(len(seeds), 1)
    probe_soak = max(int(args.soak_steps), 1)
    suite_scale = (prod_suites * prod_seeds * prod_soak) / (
        probe_suites * probe_seeds * probe_soak
    )

    def project(n: int) -> float | None:
        if slope is None or intercept is None:
            return None
        return max(0.0, (intercept + slope * float(n)) * suite_scale)

    proj_512 = project(512)
    # Prefer full training world count when the projected full suite fits the
    # cadence window; otherwise pick the smallest representative that fits.
    candidate_worlds = [1, 2, 4, 8, 16, 23, 32, 64]
    if proj_512 is not None and proj_512 <= cadence_window_s:
        chosen = 512
    else:
        chosen = None
        for n in candidate_worlds:
            est = project(n)
            if est is not None and est <= 0.8 * cadence_window_s:
                chosen = n
                break

    summary = {
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "cuda_visible_devices_parent": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_num_worlds_note": (
            "Training remains worlds.num_worlds (512 for production). "
            "evaluation.num_worlds is eval-only."
        ),
        "probe": {
            "soak_steps": int(args.soak_steps),
            "suites": list(suites),
            "seeds": list(seeds),
            "viz_enabled": bool(args.viz_enabled),
            "suite_scale_to_production": suite_scale,
        },
        "cadence_window_s": cadence_window_s,
        "fit": {"intercept_s": intercept, "slope_s_per_world": slope},
        "projected_full_suite_s": {
            "w23": project(23),
            "w32": project(32),
            "w64": project(64),
            "w512": proj_512,
        },
        "recommended_eval_num_worlds": chosen,
        "rows": rows,
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(r["exit_code"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark sensor_racer recurrent inference (CPU/CUDA/compile)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "racing_rl", "f1tenth_rl_agent"))

from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402
from f1tenth_rl_agent.policy_model import (  # noqa: E402
    load_sensor_actor,
    load_sensor_obs_norm,
)
from f1tenth_rl_agent.sensor_inference_runtime import SensorInferenceRuntime  # noqa: E402
from f1tenth_rl_agent.sensor_preprocessing import (  # noqa: E402
    pack_actor_observation,
    pack_lidar_from_scan,
)


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }


def _synthetic_obs(rng: np.random.Generator) -> np.ndarray:
    lidar = np.full(si.LIDAR_DIM, 2.5, dtype=np.float32)
    imu = np.array([0.1, 0.0, si.GRAVITY_MS2, 0.0, 0.0, 0.05], dtype=np.float32)
    steer_hist = np.zeros(4, dtype=np.float32)
    return pack_actor_observation(
        lidar,
        imu,
        speed_mps=3.0,
        vesc_current_a=1.0,
        throttle_current=0.2,
        throttle_predecessor=0.1,
        executed_steer_history=steer_hist,
    )


def _bench_backend(
    checkpoint: str,
    device_str: str,
    *,
    iters: int,
    soak_iters: int,
    use_compile: bool,
    use_pinned_h2d: bool | None,
) -> dict:
    device = torch.device(device_str)
    actor = load_sensor_actor(checkpoint, "actor", device)
    norm = load_sensor_obs_norm(checkpoint, device, si.OBS_NORM_EPS, si.OBS_NORM_CLIP)
    runtime = SensorInferenceRuntime(
        actor,
        norm,
        device,
        warmup_iters=10,
        use_compile=use_compile,
        use_pinned_h2d=use_pinned_h2d,
    )

    rng = np.random.default_rng(0)
    obs = _synthetic_obs(rng)
    np.copyto(runtime.host_obs_buffer, obs)

    preprocess_ms = []
    h2d_ms = []
    infer_ms = []
    d2h_ms = []
    total_ms = []
    deadline_misses = 0
    period_ms = 1000.0 / si.CONTROL_HZ

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device)
    else:
        mem_before = 0

    for i in range(iters):
        tick_t0 = time.perf_counter()
        pack_t0 = time.perf_counter()
        lidar = pack_lidar_from_scan(
            float(si.LIDAR_ANGLE_MIN),
            float(si.LIDAR_ANGLE_INCREMENT),
            [2.5] * si.LIDAR_DIM,
            out=np.full(si.LIDAR_DIM, si.LIDAR_RANGE_MAX, dtype=np.float32),
        )
        pack_actor_observation(
            lidar,
            np.array([0.1, 0.0, si.GRAVITY_MS2, 0.0, 0.0, 0.05], dtype=np.float32),
            3.0,
            0.2,
            0.2,
            0.1,
            np.zeros(4, dtype=np.float32),
            out=runtime.host_obs_buffer,
        )
        preprocess_ms.append((time.perf_counter() - pack_t0) * 1000.0)
        action, timings = runtime.infer_host_obs()
        tick_ms = (time.perf_counter() - tick_t0) * 1000.0
        h2d_ms.append(timings.h2d_ms)
        infer_ms.append(timings.infer_ms)
        d2h_ms.append(timings.d2h_ms)
        total_ms.append(tick_ms)
        if tick_ms > 1.25 * period_ms:
            deadline_misses += 1
        _ = action
        if i % 50 == 0:
            runtime.reset_hidden()

    soak_deadline_misses = 0
    for _ in range(soak_iters):
        tick_t0 = time.perf_counter()
        _, _ = runtime.infer_host_obs()
        tick_ms = (time.perf_counter() - tick_t0) * 1000.0
        if tick_ms > 1.25 * period_ms:
            soak_deadline_misses += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        mem_after = torch.cuda.memory_allocated(device)
        mem_peak = torch.cuda.max_memory_allocated(device)
    else:
        mem_after = 0
        mem_peak = 0

    return {
        "device": device_str,
        "use_compile": use_compile,
        "use_pinned_h2d": use_pinned_h2d,
        "iters": iters,
        "soak_iters": soak_iters,
        "deadline_misses": deadline_misses,
        "soak_deadline_misses": soak_deadline_misses,
        "preprocess": _percentiles(preprocess_ms),
        "h2d": _percentiles(h2d_ms),
        "infer": _percentiles(infer_ms),
        "d2h": _percentiles(d2h_ms),
        "tick_total": _percentiles(total_ms),
        "cuda_memory_allocated_bytes": int(mem_after),
        "cuda_memory_peak_bytes": int(mem_peak),
        "cuda_memory_delta_bytes": int(mem_after - mem_before),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "SENSOR_POLICY_CKPT",
            "/home/ubuntu/projects/F1tenth/training/outputs/runs/"
            "274164fb/checkpoints/policy_834560000.pt",
        ),
    )
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--soak-iters", type=int, default=12000)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    results = {
        "checkpoint": args.checkpoint,
        "control_hz": si.CONTROL_HZ,
        "note": "Local RTX benchmark; not Orin certification.",
        "backends": [],
    }

    backends: list[tuple[str, bool, bool | None]] = [("cpu", False, None)]
    if torch.cuda.is_available():
        backends.append(("cuda", False, True))
        backends.append(("cuda", False, False))
        backends.append(("cuda", True, True))

    for device_str, use_compile, pinned in backends:
        label = device_str
        if use_compile:
            label += "+compile"
        if pinned is False:
            label += "+no_pin"
        print(f"benchmarking {label}...", flush=True)
        try:
            row = _bench_backend(
                args.checkpoint,
                device_str,
                iters=args.iters,
                soak_iters=args.soak_iters,
                use_compile=use_compile,
                use_pinned_h2d=pinned,
            )
            row["label"] = label
            results["backends"].append(row)
            print(json.dumps(row, indent=2))
        except Exception as exc:  # noqa: BLE001
            results["backends"].append(
                {
                    "label": label,
                    "error": str(exc),
                }
            )
            print(f"failed {label}: {exc}", file=sys.stderr)

    text = json.dumps(results, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

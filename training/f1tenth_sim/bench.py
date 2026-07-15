"""JSON benchmark for the Warp vehicle kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from .params import VehicleParams
from .sim_warp import WarpVehicleSim


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(args, num_envs, device):
    params = VehicleParams.from_config({"tire_friction": 0.9})
    sim = WarpVehicleSim(params, num_envs, device=device)
    position = torch.zeros(num_envs, 3, device=device)
    quaternion = torch.zeros(num_envs, 4, device=device)
    quaternion[:, 0] = 1.0
    speed = torch.zeros(num_envs, device=device)
    actions = torch.tensor([0.5, 0.1], device=device).repeat(num_envs, 1)
    samples = []
    hashes = []
    for _ in range(args.repeats):
        sim.reset(None, position, quaternion, speed)
        for _ in range(args.warmup):
            sim.step(actions)
        _sync(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            sim.step(actions)
        _sync(device)
        elapsed = time.perf_counter() - start
        samples.append(num_envs * args.steps / elapsed)
        digest = hashlib.sha256(
            sim.read_state()["base_pos"].cpu().numpy().tobytes()
        ).hexdigest()
        hashes.append(digest)
    median = statistics.median(samples)
    return {
        "num_envs": num_envs,
        "median_transitions_per_second": median,
        "p95_transitions_per_second": float(np.percentile(samples, 5.0)),
        "median_ns_per_env_step": 1.0e9 / median,
        "samples_transitions_per_second": samples,
        "deterministic": len(set(hashes)) == 1,
        "output_sha256": hashes[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 12288, 32768, 65536],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    report = {
        "benchmark": "warp_f1tenth_vehicle",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": wp.__version__,
        "results": [_run(args, count, device) for count in args.envs],
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

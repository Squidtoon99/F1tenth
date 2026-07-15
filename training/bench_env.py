"""JSON-emitting benchmark for the complete Warp environment tick."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from standalone_trainer import select_device


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _confidence_interval(samples):
    generator = np.random.default_rng(0)
    values = np.asarray(samples, dtype=np.float64)
    means = np.empty(2000, dtype=np.float64)
    for index in range(means.size):
        means[index] = generator.choice(values, size=values.size).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _hash_outputs(env):
    digest = hashlib.sha256()
    digest.update(env.obs_buf.detach().cpu().numpy().tobytes())
    digest.update(env.reward_buf.detach().cpu().numpy().tobytes())
    digest.update(env.reset_buf.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _benchmark_count(args, cfg, device, num_envs):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    construction_start = time.perf_counter()
    env = F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    _synchronize(device)
    construction_seconds = time.perf_counter() - construction_start
    actions = torch.zeros(num_envs, 2, device=device)
    actions[:, 0] = 0.5
    actions[:, 1] = 0.1
    samples = []
    hashes = []
    for _ in range(args.repeats):
        env.reset(seed=args.seed)
        for _ in range(args.warmup):
            env.step(actions, n_steps=env.control_interval)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions, n_steps=env.control_interval)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        samples.append(num_envs * args.steps / elapsed)
        hashes.append(_hash_outputs(env))

    result = {
        "num_envs": num_envs,
        "construction_seconds": construction_seconds,
        "launches_per_tick": env.step_launch_count,
        "median_transitions_per_second": statistics.median(samples),
        "p95_transitions_per_second": float(np.percentile(samples, 5.0)),
        "bootstrap_mean_95ci": _confidence_interval(samples),
        "median_ns_per_env_step": 1.0e9 / statistics.median(samples),
        "samples_transitions_per_second": samples,
        "deterministic": len(set(hashes)) == 1,
        "output_sha256": hashes[0],
    }
    if device.type == "cuda":
        result["peak_torch_bytes"] = torch.cuda.max_memory_allocated(device)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 12288, 32768, 65536],
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--opponent", choices=["none", "scripted"], default="none")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1.0e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = (
        None if args.opponent == "none" else args.opponent
    )
    report = {
        "benchmark": "warp_f1tenth_environment",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "torch_cuda": torch.version.cuda,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "opponent": args.opponent,
        "results": [
            _benchmark_count(args, cfg, device, count) for count in args.envs
        ],
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

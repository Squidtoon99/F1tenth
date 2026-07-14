"""Reproducible TorchSim throughput benchmark and physics-parity guard."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_env.env import F1tenthEnv  # noqa: E402
from standalone_trainer import select_device  # noqa: E402


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_info() -> dict[str, float]:
    fields = "temperature.gpu,clocks.sm,utilization.gpu,memory.used"
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        vals = [float(value.strip()) for value in text.split(",")]
        return dict(zip(("temp_c", "sm_clock_mhz", "util_pct", "memory_mib"), vals))
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def _bootstrap_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    medians = np.median(rng.choice(data, (10_000, len(data))), axis=1)
    low, high = np.percentile(medians, (2.5, 97.5))
    return float(low), float(high)


def _build_env(args: argparse.Namespace, *, deterministic: bool) -> F1tenthEnv:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = args.track
    cfg["env"]["opponent_strategy"] = None if deterministic else args.opponent
    cfg["env"]["episode_length"] = 1e9
    cfg["env"]["term_not_moving_time_s"] = 1e9
    cfg["env"]["term_oob_max_consecutive"] = 10**9
    cfg["env"]["term_heading_error_rad"] = 100.0
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0 if deterministic else 1.0
    cfg["env"]["reset_speed_max_mps"] = 0.0 if deterministic else 4.0
    cfg["env"]["domain_randomization"] = {
        **cfg["env"]["domain_randomization"],
        "enabled": not deterministic,
    }
    opponent = cfg["env"]["opponent_strategy"]
    cfg["obs"]["enable_opponent_obs"] = opponent is not None
    cfg["obs"]["num_obs"] = 384 + (
        int(cfg["obs"]["opponent_obs_dim"]) if opponent is not None else 0
    )
    env_cfg = {
        "launch_strategy": "fixed" if deterministic else "uniform_jittered",
        "launch_strategy_data": {"num_cars": args.num_envs},
        **cfg["env"],
    }
    return F1tenthEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(args: argparse.Namespace, device: torch.device) -> dict:
    env = _build_env(args, deterministic=False)
    control_interval = int(env.env_cfg["control_interval"])
    actions = torch.zeros(
        args.num_envs, 2, dtype=rt.tc_float, device=device
    )
    actions[:, 0] = 0.5
    samples = []
    try:
        for _ in range(args.warmup):
            env.step(actions, n_steps=control_interval)
        _sync(device)
        for rep in range(args.reps):
            gpu_before = _gpu_info()
            start = time.perf_counter()
            for _ in range(args.steps):
                env.step(actions, n_steps=control_interval)
            _sync(device)
            elapsed = time.perf_counter() - start
            samples.append(
                {
                    "rep": rep,
                    "elapsed_s": elapsed,
                    "ticks_per_s": args.steps / elapsed,
                    "transitions_per_s": args.steps * args.num_envs / elapsed,
                    "gpu_before": gpu_before,
                    "gpu_after": _gpu_info(),
                }
            )
    finally:
        env.close()
    rates = [sample["transitions_per_s"] for sample in samples]
    low, high = _bootstrap_ci(rates, args.seed)
    result = {
        "mode": "bench",
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "reps": args.reps,
        "opponent": args.opponent,
        "median_transitions_per_s": statistics.median(rates),
        "ci95_transitions_per_s": [low, high],
        "relative_noise_pct": (
            100.0 * (high - low) / max(statistics.median(rates), 1e-9)
        ),
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "samples": samples,
    }
    return result


def _trajectory(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    env = _build_env(args, deterministic=True)
    control_interval = int(env.env_cfg["control_interval"])
    frames: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("X", "Y", "yaw", "vx", "vy", "r", "omega", "slip", "load")
    }
    try:
        _seed(args.seed)
        env.reset()
        for step in range(args.ref_steps):
            phase = step / max(args.ref_steps - 1, 1)
            action = torch.empty(
                args.num_envs, 2, dtype=rt.tc_float, device=device
            )
            action[:, 0] = 0.35 + 0.1 * np.sin(phase * 4.0 * np.pi)
            action[:, 1] = 0.12 * np.sin(phase * 2.0 * np.pi)
            env.step(action, n_steps=control_interval)
            sim = env.backend.sim
            wheel = sim.read_wheel_state()
            for key in ("X", "Y", "yaw", "vx", "vy", "r", "omega"):
                frames[key].append(sim.s[key].detach().cpu())
            frames["slip"].append(wheel["tyre_slip"].detach().cpu())
            frames["load"].append(wheel["tyre_load"].detach().cpu())
    finally:
        env.close()
    return {key: torch.stack(values) for key, values in frames.items()}


def save_reference(args: argparse.Namespace, device: torch.device) -> dict:
    trajectory = _trajectory(args, device)
    Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectory, args.reference)
    return {"mode": "save-ref", "reference": args.reference, "keys": list(trajectory)}


def check_reference(args: argparse.Namespace, device: torch.device) -> dict:
    expected = torch.load(args.reference, map_location="cpu", weights_only=True)
    actual = _trajectory(args, device)
    errors = {
        key: float((actual[key] - expected[key]).abs().max()) for key in expected
    }
    finite = all(bool(torch.isfinite(value).all()) for value in actual.values())
    passed = finite and max(errors.values(), default=0.0) <= args.tolerance
    return {
        "mode": "check-ref",
        "reference": args.reference,
        "tolerance": args.tolerance,
        "finite": finite,
        "max_abs_error": max(errors.values(), default=0.0),
        "errors": errors,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "save-ref", "check-ref"), default="bench")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--ref-steps", type=int, default=100)
    parser.add_argument("--reference", default="outputs/perf/physics_reference.pt")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--track", default="Austin")
    parser.add_argument("--opponent", choices=("none", "scripted", "mixed"), default="mixed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.opponent == "none":
        args.opponent = None
    device = select_device(args.device)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    _seed(args.seed)
    if args.mode == "bench":
        result = benchmark(args, device)
    elif args.mode == "save-ref":
        result = save_reference(args, device)
    else:
        result = check_reference(args, device)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

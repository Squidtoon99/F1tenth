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
from standalone_trainer import build_env_cfg, select_device

# Trainer default; used as the sensor throughput-gate env count.
CANONICAL_NUM_ENVS = 512
GATE_TARGET_RATIO = 0.70
GATE_FLOOR_RATIO = 0.50
DUAL_VIEW_FLOOR_RATIO = 0.50


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


def _hash_outputs(env, with_sensors):
    digest = hashlib.sha256()
    digest.update(env.obs_buf.detach().cpu().numpy().tobytes())
    digest.update(env.reward_buf.detach().cpu().numpy().tobytes())
    digest.update(env.reset_buf.detach().cpu().numpy().tobytes())
    if with_sensors:
        digest.update(env.actor_obs_buf.detach().cpu().numpy().tobytes())
        if env._should_render_opponent_actor():
            digest.update(env.opponent_actor_obs_buf.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _apply_sensor_knobs(cfg, args):
    sensor = cfg["sensor"]
    if args.num_beams is not None:
        sensor["num_beams"] = int(args.num_beams)
    if args.beam_decimation is not None:
        sensor["beam_decimation"] = int(args.beam_decimation)
    if args.max_march_steps is not None:
        sensor["max_march_steps"] = int(args.max_march_steps)
    if int(sensor.get("beam_decimation", 1)) != 1:
        raise ValueError("bench_env requires beam_decimation=1 for actor layout")
    if int(sensor.get("num_beams", 1081)) != 1081:
        raise ValueError("bench_env requires num_beams=1081 for actor layout")
    return sensor


def _seed_policy_opponent(env, device):
    """Load a real sensor-dim snapshot so dual-view policy forward is exercised."""
    if env._policy_opponent is None:
        return False
    dim = int(env.num_actor_obs)
    env.refresh_opponent_policy(
        env._policy_opponent.actor.state_dict(),
        torch.zeros(dim, device=device, dtype=torch.float32),
        torch.ones(dim, device=device, dtype=torch.float32),
    )
    # Warm CUDA graphs / inductor kernels outside the timed window.
    with torch.inference_mode():
        for _ in range(3):
            env._policy_opponent.act_observation(env.opponent_actor_obs_buf)
    _synchronize(device)
    return True


def _benchmark_count(args, cfg, device, num_envs, with_sensors, *, seed_policy=False):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    construction_start = time.perf_counter()
    env = F1tenthEnv(
        num_envs=num_envs,
        env_cfg=build_env_cfg(cfg),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    policy_seeded = False
    if seed_policy:
        policy_seeded = _seed_policy_opponent(env, device)
    _synchronize(device)
    construction_seconds = time.perf_counter() - construction_start
    actions = torch.zeros(num_envs, 2, device=device)
    actions[:, 0] = 0.5
    actions[:, 1] = 0.1
    samples = []
    hashes = []
    for _ in range(args.repeats):
        env.reset(seed=args.seed, with_sensors=with_sensors)
        for _ in range(args.warmup):
            env.step(actions, n_steps=env.control_interval, with_sensors=with_sensors)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions, n_steps=env.control_interval, with_sensors=with_sensors)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        samples.append(num_envs * args.steps / elapsed)
        hashes.append(_hash_outputs(env, with_sensors))

    result = {
        "num_envs": num_envs,
        "with_sensors": with_sensors,
        "num_lidar_beams": int(env.num_lidar_beams),
        "max_march_steps": int(env._sensor_params.max_march_steps),
        "construction_seconds": construction_seconds,
        "launches_per_tick": env.step_launch_count,
        "median_transitions_per_second": statistics.median(samples),
        "p95_transitions_per_second": float(np.percentile(samples, 5.0)),
        "bootstrap_mean_95ci": _confidence_interval(samples),
        "median_ns_per_env_step": 1.0e9 / statistics.median(samples),
        "samples_transitions_per_second": samples,
        "deterministic": len(set(hashes)) == 1,
        "output_sha256": hashes[0],
        "policy_snapshot_seeded": policy_seeded,
    }
    if device.type == "cuda":
        result["peak_torch_bytes"] = torch.cuda.max_memory_allocated(device)
    env.close()
    return result


def _gate_status(ratio):
    if ratio is None:
        return None
    if ratio >= GATE_TARGET_RATIO:
        return "pass_70"
    if ratio >= GATE_FLOOR_RATIO:
        return "pass_50_floor"
    return "fail"


def _compare_count(args, cfg, device, num_envs):
    off = _benchmark_count(args, cfg, device, num_envs, with_sensors=False)
    on = _benchmark_count(args, cfg, device, num_envs, with_sensors=True)
    off_tps = float(off["median_transitions_per_second"])
    on_tps = float(on["median_transitions_per_second"])
    ratio = on_tps / off_tps if off_tps > 0.0 else None
    return {
        "num_envs": num_envs,
        "sensor_off": off,
        "sensor_on": on,
        "sensor_off_median_tps": off_tps,
        "sensor_on_median_tps": on_tps,
        "sensor_on_over_off_ratio": ratio,
        "gate_target_ratio": GATE_TARGET_RATIO,
        "gate_floor_ratio": GATE_FLOOR_RATIO,
        "gate_status": _gate_status(ratio),
    }


def _compare_dual_view(args, cfg, device, num_envs):
    """Ego-sensor (1v0) vs full-parity policy self-play dual-view throughput.

    Reports the dual/ego ratio for later verification gates; does not fail the
    process on ratio alone.
    """
    ego_cfg = copy.deepcopy(cfg)
    ego_cfg["env"]["opponent_strategy"] = None
    dual_cfg = copy.deepcopy(cfg)
    dual_cfg["env"]["opponent_strategy"] = "policy"
    ego = _benchmark_count(args, ego_cfg, device, num_envs, with_sensors=True)
    dual = _benchmark_count(
        args, dual_cfg, device, num_envs, with_sensors=True, seed_policy=True
    )
    ego_tps = float(ego["median_transitions_per_second"])
    dual_tps = float(dual["median_transitions_per_second"])
    ratio = dual_tps / ego_tps if ego_tps > 0.0 else None
    status = None
    if ratio is not None:
        status = "pass_50_floor" if ratio >= DUAL_VIEW_FLOOR_RATIO else "fail"
    return {
        "num_envs": num_envs,
        "ego_sensor": ego,
        "dual_view_policy": dual,
        "ego_sensor_median_tps": ego_tps,
        "dual_view_median_tps": dual_tps,
        "dual_over_ego_ratio": ratio,
        "dual_view_floor_ratio": DUAL_VIEW_FLOOR_RATIO,
        "dual_view_status": status,
        "launches_ego": ego["launches_per_tick"],
        "launches_dual": dual["launches_per_tick"],
        "policy_snapshot_seeded": dual.get("policy_snapshot_seeded", False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Env counts to benchmark. Default: canonical "
            f"{CANONICAL_NUM_ENVS} for --compare-sensors, else the legacy "
            "scaling sweep."
        ),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--opponent",
        choices=["none", "scripted", "policy"],
        default="none",
        help="Opponent mode for single-path benchmarks (policy enables dual-view sensors).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare-sensors",
        action="store_true",
        help="Measure with_sensors off vs on and report the TPS ratio gate.",
    )
    parser.add_argument(
        "--compare-dual-view",
        action="store_true",
        help=(
            "Measure ego-sensor (1v0) vs full-parity policy self-play dual-view "
            "throughput (reports ratio; hard gates deferred to verification)."
        ),
    )
    parser.add_argument(
        "--with-sensors",
        action="store_true",
        help="Benchmark only the with_sensors=True path (ignored with --compare-sensors).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Override sensor.num_beams (native UST-10LX count before decimation).",
    )
    parser.add_argument(
        "--beam-decimation",
        type=int,
        default=None,
        help="Override sensor.beam_decimation (must be 1; actor layout rejects others).",
    )
    parser.add_argument(
        "--max-march-steps",
        type=int,
        default=None,
        help="Override sensor.max_march_steps sphere-trace iteration cap.",
    )
    args = parser.parse_args()

    if args.envs is None:
        if args.compare_sensors or args.compare_dual_view:
            args.envs = [CANONICAL_NUM_ENVS]
        else:
            args.envs = [256, 1024, 4096, 12288, 32768, 65536]

    device = select_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
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
    sensor = _apply_sensor_knobs(cfg, args)
    report = {
        "benchmark": "warp_f1tenth_environment",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "torch_cuda": torch.version.cuda,
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "opponent": args.opponent,
        "seed": args.seed,
        "compare_sensors": bool(args.compare_sensors),
        "compare_dual_view": bool(args.compare_dual_view),
        "canonical_num_envs": CANONICAL_NUM_ENVS,
        "sensor_knobs": {
            "num_beams": sensor["num_beams"],
            "beam_decimation": sensor["beam_decimation"],
            "max_march_steps": sensor["max_march_steps"],
        },
    }
    if args.compare_dual_view:
        report["results"] = [
            _compare_dual_view(args, cfg, device, count) for count in args.envs
        ]
    elif args.compare_sensors:
        report["results"] = [
            _compare_count(args, cfg, device, count) for count in args.envs
        ]
    else:
        report["results"] = [
            _benchmark_count(
                args, cfg, device, count, with_sensors=bool(args.with_sensors)
            )
            for count in args.envs
        ]
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

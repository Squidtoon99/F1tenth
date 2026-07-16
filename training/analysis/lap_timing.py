"""Solo lap-timing benchmark for trained policy checkpoints.

Two modes:

* single checkpoint (worker) -- roll out one ``policy_*.pt`` solo (1v0) across
  many parallel envs under domain randomization, detect finish-line crossings via
  the env's ``lap_count_buf`` and emit per-lap times as JSON.

    python lap_timing.py --checkpoint .../policy_123.pt --out /tmp/x.json

* orchestrator (``--run-dir``) -- discover every checkpoint in a run, launch the
  single-checkpoint workers in a process pool, aggregate into a CSV of lap-time
  distributions, print a top-10 table, and plot lap time vs training progress.

    python lap_timing.py --run-dir outputs/runs/<id> --workers 8

Lap timing is line-to-line: a lap is counted only between two consecutive
finish-line crossings within a single uninterrupted episode (a reset/crash
discards the in-progress lap). Domain randomization is left ON so the reported
distribution reflects pace across randomized race conditions; the same seed is
used for every checkpoint so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

CKPT_RE = re.compile(r"policy_(\d+)\.pt$")
CRASH_TERM_KEYS = ("out_of_bounds", "not_moving", "invalid_state", "collision")
BENCHMARK_VERSION = 3
LAP_TIMING_SPAWN_POLICY = "centerline_tangent"


class NoZeroCrashReferenceError(Exception):
    """No checkpoint qualified with zero crashes and enough clean laps."""


class NoQualifiedReferenceError(Exception):
    """No checkpoint qualified with enough clean laps for seeding."""


REFERENCE_MODE_ZERO_CRASH = "zero-crash"
REFERENCE_MODE_MIN_CRASH = "min-crash"
REFERENCE_MODES = (REFERENCE_MODE_ZERO_CRASH, REFERENCE_MODE_MIN_CRASH)


def _read_config_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "config" in data:
        return data["config"]
    return data


def load_lap_timing_config(
    checkpoint: Path,
    *,
    config: str | Path | None = None,
) -> tuple[dict, str]:
    from standalone_trainer import DEFAULT_CONFIG

    if config is not None:
        path = Path(config).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Lap timing config not found: {path}")
        return _read_config_json(path), str(path)
    cfg_path = checkpoint.parent.parent / "config.json"
    if cfg_path.exists():
        return _read_config_json(cfg_path), str(cfg_path.resolve())
    return copy.deepcopy(DEFAULT_CONFIG), "DEFAULT_CONFIG"


def lap_timing_config_sha(cfg: dict, track: str) -> str:
    payload = json.dumps(
        {"track": track, "env": cfg.get("env", {}), "obs": cfg.get("obs", {})},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _transitions_from_name(path: Path) -> int:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def classify_step_terminations(
    done: list[bool] | tuple[bool, ...],
    termination_extras: dict,
) -> tuple[int, int, dict[str, int], int]:
    """Classify episode ends on this control step.

    Returns (crashes, timeouts, crash_breakdown, dropped_in_progress_laps).
    A crash is any done env whose termination is not solely ``time_out``.
    """
    term = termination_extras.get("termination", {})
    term_np = {
        key: value.detach().cpu().numpy().astype(bool)
        for key, value in term.items()
    }
    crashes = 0
    timeouts = 0
    breakdown = {key: 0 for key in CRASH_TERM_KEYS}
    dropped = 0
    for e, is_done in enumerate(done):
        if not is_done:
            continue
        timed_out = bool(term_np.get("time_out", [False] * len(done))[e])
        crashed = False
        for key in CRASH_TERM_KEYS:
            if key in term_np and term_np[key][e]:
                breakdown[key] += 1
                crashed = True
        if timed_out and not crashed:
            timeouts += 1
        elif crashed:
            crashes += 1
    return crashes, timeouts, breakdown, dropped


def apply_domain_randomization_setting(cfg: dict, *, enabled: bool) -> None:
    cfg["env"]["domain_randomization"] = {
        **cfg["env"].get("domain_randomization", {}),
        "enabled": enabled,
    }


def apply_lap_timing_spawn_policy(env_cfg: dict) -> None:
    """Spawn on the centerline, aligned with the local tangent (lap timing only)."""
    env_cfg["reset_lateral_offset_m"] = 0.0
    env_cfg["reset_yaw_jitter_rad"] = 0.0


def benchmark_config_fingerprint(
    *,
    track: str,
    num_envs: int,
    steps: int,
    seed: int,
    device: str,
    precision: str,
    domain_randomization_enabled: bool = True,
    config_source: str | None = None,
    config_sha: str | None = None,
) -> dict:
    fp = {
        "version": BENCHMARK_VERSION,
        "track": track,
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "device": device,
        "precision": precision,
        "domain_randomization_enabled": domain_randomization_enabled,
        "spawn_policy": LAP_TIMING_SPAWN_POLICY,
    }
    if config_source is not None:
        fp["config_source"] = config_source
    if config_sha is not None:
        fp["config_sha"] = config_sha
    return fp


def is_valid_cached_result(data: dict, fingerprint: dict) -> bool:
    if data.get("benchmark_config") != fingerprint:
        return False
    if data.get("n_laps") is None:
        return False
    if data.get("mean") is None and int(data.get("n_laps", 0)) > 0:
        return False
    return True


def default_equivalent_tolerance_s(run_dir: Path | None = None) -> float:
    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())["config"]
            sim_dt = float(cfg["env"].get("sim_dt", 0.005))
            control_interval = int(cfg["env"].get("control_interval", 10))
            return sim_dt * control_interval
    return 0.05


def _qualified_rows(rows: list[dict], min_laps: int) -> list[dict]:
    return [
        r for r in rows
        if int(r.get("n_laps", 0)) >= min_laps and r.get("mean") is not None
    ]


def _seed_sort_key(row: dict) -> tuple:
    return (
        float(row["mean"]),
        int(row.get("crashes", 0)),
        float(row["min"]) if row.get("min") is not None else float("inf"),
        int(row.get("transitions", 0)),
    )


def _min_crash_reference_key(row: dict) -> tuple:
    return (
        float(row["mean"]),
        float(row["min"]) if row.get("min") is not None else float("inf"),
        int(row.get("transitions", 0)),
    )


def select_tournament_seeds(
    rows: list[dict],
    *,
    min_laps: int = 20,
    equivalent_tolerance_s: float,
    reference_mode: str = REFERENCE_MODE_ZERO_CRASH,
) -> dict:
    """Select seeds at or faster than the reference mean (+ tolerance)."""
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            f"Unknown reference_mode {reference_mode!r}; "
            f"expected one of {REFERENCE_MODES}"
        )

    qualified = _qualified_rows(rows, min_laps)
    if not qualified:
        raise NoQualifiedReferenceError(
            f"No checkpoint has >= {min_laps} clean laps with a mean lap time"
        )

    if reference_mode == REFERENCE_MODE_ZERO_CRASH:
        pool = [r for r in qualified if int(r.get("crashes", 0)) == 0]
        if not pool:
            raise NoZeroCrashReferenceError(
                f"No checkpoint has >= {min_laps} clean laps and zero crashes"
            )
        min_crash_count = 0
        reference = min(pool, key=lambda r: float(r["mean"]))
    else:
        min_crash_count = min(int(r.get("crashes", 0)) for r in qualified)
        pool = [
            r for r in qualified if int(r.get("crashes", 0)) == min_crash_count
        ]
        reference = min(pool, key=_min_crash_reference_key)

    cutoff = float(reference["mean"]) + equivalent_tolerance_s
    selected = [r for r in qualified if float(r["mean"]) <= cutoff]
    selected.sort(key=_seed_sort_key)
    for i, row in enumerate(selected, 1):
        row = dict(row)
        row["seed_rank"] = i
        row["seed_metric"] = "mean"
        selected[i - 1] = row
    return {
        "reference_mode": reference_mode,
        "min_crash_count": min_crash_count,
        "reference_checkpoint": reference["checkpoint"],
        "reference_transitions": int(reference.get("transitions", -1)),
        "reference_mean_s": float(reference["mean"]),
        "reference_min_s": (
            float(reference["min"]) if reference.get("min") is not None else None
        ),
        "reference_n_laps": int(reference.get("n_laps", 0)),
        "reference_crashes": int(reference.get("crashes", 0)),
        "equivalent_tolerance_s": equivalent_tolerance_s,
        "cutoff_mean_s": cutoff,
        "min_laps": min_laps,
        "total_benchmarked": len(rows),
        "qualified_count": len(qualified),
        "zero_crash_qualified_count": sum(
            1 for r in qualified if int(r.get("crashes", 0)) == 0
        ),
        "selected_count": len(selected),
        "selected": selected,
    }


def load_results_from_json_dir(json_dir: Path, fingerprint: dict) -> list[dict]:
    paths = sorted(json_dir.glob("policy_*.json"), key=lambda p: _transitions_from_name(p))
    if not paths:
        raise FileNotFoundError(f"No per-checkpoint JSON in {json_dir}")
    results: list[dict] = []
    for path in paths:
        data = json.loads(path.read_text())
        if not is_valid_cached_result(data, fingerprint):
            raise ValueError(
                f"Cached result does not match benchmark config: {path.name}"
            )
        results.append(data)
    return results


def write_tournament_seed_outputs(
    seed_info: dict,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_csv_path = out_dir / "tournament_seed.csv"
    seed_summary_path = out_dir / "tournament_seed_summary.json"
    _write_tournament_seed_csv(seed_info, seed_csv_path)
    seed_summary_path.write_text(json.dumps(seed_info, indent=2))
    return seed_csv_path, seed_summary_path


# --------------------------------------------------------------------------- #
# Single-checkpoint worker
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(
    checkpoint: str,
    track: str,
    num_envs: int,
    steps: int,
    seed: int,
    device_str: str,
    precision: str,
    benchmark_config: dict | None = None,
    disable_domain_randomization: bool = False,
    config: str | Path | None = None,
) -> dict:
    import numpy as np
    import torch

    from evaluation import deterministic_rollout
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt
    from standalone_trainer import (
        ObsNormalizer,
        build_models,
        episode_length_for_track,
        select_device,
    )

    ckpt_path = Path(checkpoint).resolve()
    cfg, config_source = load_lap_timing_config(ckpt_path, config=config)

    cfg["env"]["track"] = track
    cfg["env"]["opponent_strategy"] = None
    if disable_domain_randomization:
        apply_domain_randomization_setting(cfg, enabled=False)
    cfg["env"]["episode_length"] = episode_length_for_track(
        track=track,
        workspace_dir=str(Path(__file__).resolve().parent),
        ref_lap_speed_mps=float(cfg["env"].get("expected_lap_speed_mps", 3.5)),
        lap_multiplier=float(cfg["env"].get("episode_lap_multiplier", 3.0)),
    )

    device = select_device(device_str)
    rt.configure(
        float_dtype=torch.float64 if precision == "64" else torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )

    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    apply_lap_timing_spawn_policy(env_cfg)
    control_interval = int(cfg["env"]["control_interval"])
    clip_actions = float(cfg["env"]["clip_actions"])

    env = F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    control_dt = float(env.control_dt)

    models, _ = build_models(cfg, device)
    normalizer = ObsNormalizer(
        obs_dim=cfg["obs"]["num_obs"],
        device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    models.actor.eval()

    prev_lap = np.zeros(num_envs, dtype=np.int64)
    last_cross_step = np.full(num_envs, -1, dtype=np.int64)
    have_full_start = np.zeros(num_envs, dtype=bool)
    lap_times: list[float] = []
    crashes = 0
    timeouts = 0
    dropped_laps = 0
    crash_breakdown = {key: 0 for key in CRASH_TERM_KEYS}

    def on_step(step, rollout_env, _sb, _reward, done, extras):
        nonlocal prev_lap, crashes, timeouts, dropped_laps
        lap = extras["metrics"]["lap_count"].detach().cpu().numpy().astype(np.int64)
        dn = done.detach().cpu().numpy().astype(bool)
        step_crashes, step_timeouts, step_breakdown, _ = classify_step_terminations(
            dn, extras,
        )
        crashes += step_crashes
        timeouts += step_timeouts
        for key, count in step_breakdown.items():
            crash_breakdown[key] += count

        crossed = lap > prev_lap
        for e in range(num_envs):
            if crossed[e] and not dn[e]:
                if have_full_start[e]:
                    lap_times.append((step - last_cross_step[e]) * control_dt)
                last_cross_step[e] = step
                have_full_start[e] = True
            if dn[e]:
                if have_full_start[e]:
                    dropped_laps += 1
                have_full_start[e] = False
                last_cross_step[e] = -1
        prev_lap = lap

    deterministic_rollout(
        env,
        models.actor,
        normalizer.normalize,
        num_steps=steps,
        control_interval=control_interval,
        clip_actions=clip_actions,
        seed=seed,
        callback=on_step,
    )
    env.close()

    laps = np.asarray(lap_times, dtype=np.float64)
    result = {
        "checkpoint": ckpt_path.name,
        "transitions": _transitions_from_name(ckpt_path),
        "track": track,
        "config_source": config_source,
        "config_sha": lap_timing_config_sha(cfg, track),
        "num_envs": num_envs,
        "steps": steps,
        "n_laps": int(laps.size),
        "min": float(laps.min()) if laps.size else None,
        "max": float(laps.max()) if laps.size else None,
        "mean": float(laps.mean()) if laps.size else None,
        "median": float(np.median(laps)) if laps.size else None,
        "std": float(laps.std()) if laps.size else None,
        "p05": float(np.percentile(laps, 5)) if laps.size else None,
        "crashes": crashes,
        "timeouts": timeouts,
        "dropped_laps": dropped_laps,
        "crash_breakdown": crash_breakdown,
        "laps": [round(x, 4) for x in lap_times],
    }
    if benchmark_config is not None:
        result["benchmark_config"] = benchmark_config
    return result


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _warm_warp_kernel_cache(
    *,
    track: str,
    device_str: str,
    precision: str,
) -> None:
    """Populate the Warp JIT cache in-process before spawning worker subprocesses."""
    import copy

    import torch

    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt
    from standalone_trainer import DEFAULT_CONFIG, select_device

    device = select_device(device_str)
    rt.configure(
        float_dtype=torch.float64 if precision == "64" else torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = track
    cfg["env"]["opponent_strategy"] = None
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": 1},
        **cfg["env"],
    }
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        env.reset(seed=0)
        action = torch.zeros(1, 2, device=device, dtype=torch.float32)
        env.step(action, n_steps=env.control_interval)
    finally:
        env.close()


def _run_pool(
    ckpts: list[Path],
    args: argparse.Namespace,
    json_dir: Path,
    fingerprint: dict,
) -> tuple[list[dict], list[str]]:
    pending = list(ckpts)
    running: list[tuple[subprocess.Popen, Path, Path]] = []
    results: list[dict] = []
    failed: list[str] = []
    done = 0
    total = len(ckpts)

    def launch(ckpt: Path) -> tuple[subprocess.Popen, Path, Path]:
        jpath = json_dir / f"{ckpt.stem}.json"
        env = dict(os.environ)
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--checkpoint", str(ckpt), "--out", str(jpath),
            "--track", args.track, "--num-envs", str(args.num_envs),
            "--steps", str(args.steps), "--seed", str(args.seed),
            "--device", args.device, "--precision", args.precision,
        ]
        if args.disable_domain_randomization:
            cmd.append("--disable-domain-randomization")
        if args.config:
            cmd += ["--config", str(args.config)]
        p = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return p, ckpt, jpath

    while pending or running:
        while pending and len(running) < args.workers:
            ckpt = pending.pop(0)
            jpath = json_dir / f"{ckpt.stem}.json"
            if jpath.exists():
                try:
                    cached = json.loads(jpath.read_text())
                except json.JSONDecodeError:
                    cached = None
                if cached and is_valid_cached_result(cached, fingerprint):
                    results.append(cached)
                    done += 1
                    r = cached
                    print(
                        f"[{done}/{total}] {ckpt.name}: cached "
                        f"n={r['n_laps']} mean={r['mean']} crashes={r.get('crashes', 0)}"
                    )
                    continue
            running.append(launch(ckpt))
        still: list[tuple[subprocess.Popen, Path, Path]] = []
        for p, ckpt, jpath in running:
            rc = p.poll()
            if rc is None:
                still.append((p, ckpt, jpath))
                continue
            done += 1
            if rc == 0 and jpath.exists():
                results.append(json.loads(jpath.read_text()))
                r = results[-1]
                print(
                    f"[{done}/{total}] {ckpt.name}: "
                    f"n={r['n_laps']} min={r['min']} mean={r['mean']} "
                    f"crashes={r.get('crashes', 0)}"
                )
            else:
                err = p.stderr.read().decode()[-500:] if p.stderr else ""
                print(f"[{done}/{total}] {ckpt.name}: FAILED rc={rc}\n{err}")
                failed.append(ckpt.name)
        running = still
        if running:
            time.sleep(0.5)

    return results, failed


def run_orchestrator(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("policy_*.pt"), key=_transitions_from_name)
    if args.limit:
        ckpts = ckpts[:: max(1, len(ckpts) // args.limit)][: args.limit]
    if not ckpts:
        sys.exit(f"No checkpoints found in {ckpt_dir}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "per_checkpoint"
    json_dir.mkdir(exist_ok=True)

    dr_enabled = not args.disable_domain_randomization
    config_source = None
    config_sha = None
    if args.config:
        config_source = str(Path(args.config).resolve())
        config_sha = lap_timing_config_sha(
            _read_config_json(Path(args.config)), args.track,
        )
    fingerprint = benchmark_config_fingerprint(
        track=args.track,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        domain_randomization_enabled=dr_enabled,
        config_source=config_source,
        config_sha=config_sha,
    )
    dr_label = "DR on" if dr_enabled else "nominal (DR off)"
    tolerance = (
        args.equivalent_tolerance_s
        if args.equivalent_tolerance_s is not None
        else default_equivalent_tolerance_s(run_dir)
    )

    print(
        f"Timing {len(ckpts)} checkpoints with {args.workers} workers "
        f"({args.num_envs} envs x {args.steps} steps each, {dr_label})..."
    )
    if args.workers > 1:
        print("Warming Warp kernel cache before worker pool...")
        _warm_warp_kernel_cache(
            track=args.track,
            device_str=args.device,
            precision=args.precision,
        )
    t0 = time.time()
    results, failed = _run_pool(ckpts, args, json_dir, fingerprint)

    if failed and args.workers > 1:
        retry_workers = max(1, args.workers // 2)
        print(
            f"\nRetrying {len(failed)} failed checkpoints "
            f"with {retry_workers} workers..."
        )
        retry_ckpts = [ckpt_dir / name for name in failed]
        args_retry = argparse.Namespace(**vars(args))
        args_retry.workers = retry_workers
        retry_results, still_failed = _run_pool(
            retry_ckpts, args_retry, json_dir, fingerprint,
        )
        ok_names = {r["checkpoint"] for r in retry_results}
        results = [r for r in results if r["checkpoint"] not in failed] + retry_results
        failed = [n for n in failed if n not in ok_names] + still_failed

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["transitions"])
    _write_csv(results, out_dir / "car_lap_times.csv")
    _print_table(results)
    _plot(results, out_dir / "lap_times_by_checkpoint.png")

    try:
        seed_info = select_tournament_seeds(
            results,
            min_laps=args.min_laps,
            equivalent_tolerance_s=tolerance,
            reference_mode=args.reference_mode,
        )
        seed_csv_path, seed_summary_path = write_tournament_seed_outputs(
            seed_info, out_dir,
        )
        _print_seed_selection(seed_info)
        print(f"\nSeed CSV:     {seed_csv_path}")
        print(f"Seed summary: {seed_summary_path}")
    except (NoZeroCrashReferenceError, NoQualifiedReferenceError) as exc:
        print(f"\nTournament seed selection skipped: {exc}")

    if failed:
        fail_path = out_dir / "failed_checkpoints.txt"
        fail_path.write_text("\n".join(sorted(failed)) + "\n")
        print(f"\nFailed ({len(failed)}): {fail_path}")

    print(f"\nElapsed: {elapsed / 60:.1f} min")
    print(f"Coverage: {len(results)}/{len(ckpts)} checkpoints")
    print(f"CSV:  {out_dir / 'car_lap_times.csv'}")
    print(f"Plot: {out_dir / 'lap_times_by_checkpoint.png'}")


def _write_csv(results: list[dict], path: Path) -> None:
    cols = [
        "checkpoint", "transitions", "track", "n_laps",
        "min", "max", "mean", "median", "std", "p05",
        "crashes", "timeouts", "dropped_laps",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c) for c in cols])


def _write_tournament_seed_csv(seed_info: dict, path: Path) -> None:
    cols = [
        "seed_rank", "seed_metric", "checkpoint", "transitions", "n_laps",
        "min", "mean", "crashes", "reference_mode", "reference_checkpoint",
        "reference_mean_s", "min_crash_count", "equivalent_tolerance_s",
        "cutoff_mean_s",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        meta = {
            "seed_metric": "mean",
            "reference_mode": seed_info["reference_mode"],
            "reference_checkpoint": seed_info["reference_checkpoint"],
            "reference_mean_s": seed_info["reference_mean_s"],
            "min_crash_count": seed_info["min_crash_count"],
            "equivalent_tolerance_s": seed_info["equivalent_tolerance_s"],
            "cutoff_mean_s": seed_info["cutoff_mean_s"],
        }
        for row in seed_info["selected"]:
            w.writerow({**meta, **row})


def _print_seed_selection(seed_info: dict) -> None:
    print(
        f"\n=== Tournament seeds "
        f"(mode={seed_info['reference_mode']} "
        f"ref={seed_info['reference_checkpoint']} "
        f"mean={seed_info['reference_mean_s']:.2f}s "
        f"cutoff={seed_info['cutoff_mean_s']:.2f}s, "
        f"selected={seed_info['selected_count']}) ==="
    )
    print(
        f"{'rank':>4} {'checkpoint':>22} {'trans':>10} "
        f"{'laps':>5} {'min':>7} {'mean':>7} {'crashes':>7}"
    )
    for i, r in enumerate(seed_info["selected"], 1):
        print(
            f"{i:>4} {r['checkpoint']:>22} {r['transitions']:>10} "
            f"{r['n_laps']:>5} {r['min']:>7.2f} {r['mean']:>7.2f} "
            f"{r.get('crashes', 0):>7}"
        )


def _print_table(results: list[dict]) -> None:
    ranked = [r for r in results if r["n_laps"] > 0 and r["mean"] is not None]
    ranked.sort(key=lambda r: r["mean"])
    print("\n=== Top 10 checkpoints by mean lap time (fastest first) ===")
    print(
        f"{'rank':>4} {'checkpoint':>22} {'transitions':>13} "
        f"{'laps':>5} {'min':>7} {'mean':>7} {'max':>7} "
        f"{'std':>6} {'crashes':>7}"
    )
    for i, r in enumerate(ranked[:10], 1):
        print(
            f"{i:>4} {r['checkpoint']:>22} {r['transitions']:>13} "
            f"{r['n_laps']:>5} {r['min']:>7.2f} {r['mean']:>7.2f} "
            f"{r['max']:>7.2f} {r['std']:>6.2f} "
            f"{r.get('crashes', 0):>7}"
        )
    if not ranked:
        print("  (no checkpoint completed a clean lap)")


def _plot(results: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable ({exc}); skipping plot")
        return
    pts = [r for r in results if r["n_laps"] > 0 and r["mean"] is not None]
    if not pts:
        print("No clean laps to plot")
        return
    x = [r["transitions"] / 1e6 for r in pts]
    mean = [r["mean"] for r in pts]
    lo = [r["min"] for r in pts]
    hi = [r["max"] for r in pts]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, lo, hi, alpha=0.2, color="tab:blue", label="min-max range")
    ax.plot(x, mean, "-o", ms=3, color="tab:blue", label="mean lap time")
    ax.set_xlabel("Training progress (millions of transitions)")
    ax.set_ylabel("Lap time (s)")
    ax.set_title("Solo lap time vs training progress")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solo lap-timing benchmark")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="Worker mode: write this checkpoint's JSON result here.")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Orchestrator mode: time every checkpoint in this run.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Orchestrator output dir (CSV + plot + per-ckpt JSON).")
    p.add_argument("--track", type=str, default="Austin")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Evenly subsample to at most N checkpoints (0 = all).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    p.add_argument("--min-laps", type=int, default=20,
                   help="Min clean laps to qualify for tournament seeding.")
    p.add_argument(
        "--equivalent-tolerance-s", type=float, default=None,
        help="Mean lap times within this many seconds of the reference are "
        "equivalent (default: one control tick from run config).",
    )
    p.add_argument(
        "--disable-domain-randomization", "--nominal",
        action="store_true",
        dest="disable_domain_randomization",
        help="Disable env domain randomization (nominal/tournament physics).",
    )
    p.add_argument(
        "--reference-mode",
        choices=REFERENCE_MODES,
        default=REFERENCE_MODE_ZERO_CRASH,
        help="Reference checkpoint rule for tournament seeding.",
    )
    p.add_argument(
        "--select-only",
        action="store_true",
        help="Reuse per-checkpoint JSON in --out-dir and write seed outputs only.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Canonical config.json for all checkpoints (cross-run comparison).",
    )
    return p.parse_args()


def run_select_only(args: argparse.Namespace) -> None:
    if not args.out_dir:
        sys.exit("--select-only requires --out-dir")
    out_dir = Path(args.out_dir).resolve()
    json_dir = out_dir / "per_checkpoint"
    if not json_dir.is_dir():
        sys.exit(f"Per-checkpoint JSON dir not found: {json_dir}")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else out_dir.parent
    dr_enabled = not args.disable_domain_randomization
    config_source = None
    config_sha = None
    if args.config:
        config_source = str(Path(args.config).resolve())
        config_sha = lap_timing_config_sha(
            _read_config_json(Path(args.config)), args.track,
        )
    fingerprint = benchmark_config_fingerprint(
        track=args.track,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        domain_randomization_enabled=dr_enabled,
        config_source=config_source,
        config_sha=config_sha,
    )
    tolerance = (
        args.equivalent_tolerance_s
        if args.equivalent_tolerance_s is not None
        else default_equivalent_tolerance_s(run_dir)
    )

    results = load_results_from_json_dir(json_dir, fingerprint)
    results.sort(key=lambda r: r["transitions"])
    seed_info = select_tournament_seeds(
        results,
        min_laps=args.min_laps,
        equivalent_tolerance_s=tolerance,
        reference_mode=args.reference_mode,
    )
    seed_csv_path, seed_summary_path = write_tournament_seed_outputs(
        seed_info, out_dir,
    )
    _print_seed_selection(seed_info)
    print(f"\nLoaded {len(results)} cached results from {json_dir}")
    print(f"Seed CSV:     {seed_csv_path}")
    print(f"Seed summary: {seed_summary_path}")


def main() -> None:
    args = parse_args()
    if args.select_only:
        run_select_only(args)
        return
    if args.run_dir:
        if args.out_dir is None:
            args.out_dir = str(Path(args.run_dir).resolve() / "lap_timing")
        run_orchestrator(args)
        return
    if not args.checkpoint or not args.out:
        sys.exit("Worker mode needs --checkpoint and --out (or use --run-dir).")
    import torch
    torch.set_num_threads(1)
    config_source = None
    config_sha = None
    if args.config:
        config_source = str(Path(args.config).resolve())
        config_sha = lap_timing_config_sha(
            _read_config_json(Path(args.config)), args.track,
        )
    fingerprint = benchmark_config_fingerprint(
        track=args.track,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        domain_randomization_enabled=not args.disable_domain_randomization,
        config_source=config_source,
        config_sha=config_sha,
    )
    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        track=args.track,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        device_str=args.device,
        precision=args.precision,
        benchmark_config=fingerprint,
        disable_domain_randomization=args.disable_domain_randomization,
        config=args.config,
    )
    Path(args.out).write_text(json.dumps(result))
    print(
        f"{result['checkpoint']}: n_laps={result['n_laps']} "
        f"min={result['min']} mean={result['mean']} crashes={result['crashes']}"
    )


if __name__ == "__main__":
    main()

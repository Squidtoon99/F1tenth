#!/usr/bin/env python3
"""Deterministic full-lap eval for PPO sensor-policy checkpoints.

A lap is net along-track progress of at least one track length from the
episode's own spawn, not a start/finish line crossing. Each scenario uses a
fixed set of evenly spaced centerline spawns (standing or flying per scenario).

    python eval_full_lap.py --checkpoint .../policy_320000000.pt
    python eval_full_lap.py --run-dir outputs/runs/<id> --out-dir /tmp/full-lap
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

CKPT_RE = re.compile(r"policy_(\d+)\.pt$")
CRASH_TERM_KEYS = ("out_of_bounds", "not_moving", "invalid_state", "collision")
DEFAULT_CONFIG_REL = Path("configs") / "courtyard_2_e2e_ppo_multi_scenario.json"
DEFAULT_N_SPAWNS = 64
DEFAULT_NUM_ENVS = 8
DEFAULT_MEMORY_FRACTION = 0.15
EVAL_VERSION = 2
PROGRESS_EPS_M = 1e-4
ACTION_CHANNEL_NAMES = ("throttle", "steer")
DEFAULT_REVERSAL_DEADBAND = 0.0
FOLLOW_OPPONENT_AHEAD_M = 3.0
FOLLOW_OPPONENT_SPEED_MPS = 2.0
SCRIPTED_OVERTAKE_NAME = "scripted_overtake"
DISTURBANCE_STEER_GAINS = (1.0, 1.21)
DISTURBANCE_ACTION_LATENCY_STEPS = 2
OFF_HEADING_YAW_JITTER_RAD = 0.4
HELD_OUT_OPPONENT_SPEED_MPS = 6.0
OFF_HEADING_NAME = "off_heading"
HELD_OUT_OVERTAKE_NAME = "scripted_overtake_6mps"


def _transitions_from_name(path: Path) -> int:
    match = CKPT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def wrapped_delta_s(current, previous, length):
    """Shortest-arc along-track delta in (-L/2, L/2]."""
    length = np.asarray(length, dtype=np.float64)
    delta = np.asarray(current, dtype=np.float64) - np.asarray(
        previous, dtype=np.float64
    )
    half = 0.5 * length
    delta = np.where(delta > half, delta - length, delta)
    delta = np.where(delta < -half, delta + length, delta)
    if np.ndim(delta) == 0:
        return float(delta)
    return delta


def wrapped_delta_torch(current, previous, length):
    delta = current - previous
    half = 0.5 * length
    delta = torch.where(delta > half, delta - length, delta)
    return torch.where(delta < -half, delta + length, delta)


def count_progress_overtakes(
    prev_ego_s,
    ego_s,
    prev_opp_s,
    opp_s,
    both_on_track,
    length,
):
    """1 where ego along-track s crosses opponent s from behind.

    A pass is the wrapped gap ``opp_s - ego_s`` going from ``> 0`` (opponent
    ahead) to ``<= 0`` (ego even or ahead) in one step, while both cars are on
    track. Relative progress larger than ``0.1 L`` is ignored as a wrap artifact.
    """
    max_rel = 0.1 * length
    if torch.is_tensor(ego_s):
        prev_gap = wrapped_delta_torch(prev_opp_s, prev_ego_s, length)
        gap = wrapped_delta_torch(opp_s, ego_s, length)
        delta = prev_gap - gap
        passed = (
            both_on_track
            & (prev_gap > 0)
            & (gap <= 0)
            & (delta > 0)
            & (delta <= max_rel)
        )
        return passed.to(dtype=torch.int32)
    prev_gap = wrapped_delta_s(prev_opp_s, prev_ego_s, length)
    gap = wrapped_delta_s(opp_s, ego_s, length)
    delta = prev_gap - gap
    passed = (
        np.asarray(both_on_track, dtype=bool)
        & (prev_gap > 0)
        & (gap <= 0)
        & (delta > 0)
        & (delta <= max_rel)
    )
    return passed.astype(np.int32)


def even_spawn_s(length: float, n: int) -> np.ndarray:
    if int(n) <= 0:
        raise ValueError(f"n_spawns must be positive, got {n!r}")
    length = float(length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"track length must be positive and finite, got {length}")
    return np.arange(int(n), dtype=np.float64) * length / float(n)


def pose_at_s(track, s: float) -> tuple[np.ndarray, float]:
    """Centerline pose (xy, yaw) at along-track distance ``s`` (wraps)."""
    length = float(track.length)
    s = float(s) % length
    cumulative = np.asarray(track.cumulative_length, dtype=np.float64)
    idx = int(np.searchsorted(cumulative, s, side="right") - 1)
    idx = max(0, min(idx, int(cumulative.shape[0]) - 1))
    seg_len = float(track.segment_length[idx])
    alpha = 0.0 if seg_len <= 1e-9 else (s - float(cumulative[idx])) / seg_len
    alpha = min(max(alpha, 0.0), 1.0)
    point = np.asarray(track.point, dtype=np.float64)
    nxt = (idx + 1) % point.shape[0]
    xy = point[idx] + alpha * (point[nxt] - point[idx])
    tangent = np.asarray(track.tangent[idx], dtype=np.float64)
    yaw = math.atan2(float(tangent[1]), float(tangent[0]))
    return xy.astype(np.float32), yaw


def spawn_speed_mps(env_cfg: dict) -> float:
    stationary = float(env_cfg.get("reset_stationary_probability", 0.0))
    if stationary >= 1.0:
        return 0.0
    return float(env_cfg.get("expected_lap_speed_mps", 3.5))


def classify_terminations(done, extras) -> tuple[np.ndarray, np.ndarray]:
    """Return (crash_mask, timeout_mask) for the current step."""
    if torch.is_tensor(done):
        done_np = done.detach().cpu().numpy().astype(bool)
    else:
        done_np = np.asarray(done, dtype=bool)
    term = extras.get("termination", {})
    crash = np.zeros(done_np.shape[0], dtype=bool)
    for key in CRASH_TERM_KEYS:
        values = term.get(key)
        if values is None:
            continue
        crash |= np.asarray(values.detach().cpu().numpy(), dtype=bool)
    timeout = np.zeros(done_np.shape[0], dtype=bool)
    timed = term.get("time_out")
    if timed is not None:
        timeout = np.asarray(timed.detach().cpu().numpy(), dtype=bool) & ~crash
    crash &= done_np
    timeout &= done_np
    return crash, timeout


def action_delta_metrics(actions, *, deadband=DEFAULT_REVERSAL_DEADBAND):
    """Per-channel reversal rate and mean |Δaction| for one episode.

    Reversal rate is the fraction of consecutive first-difference pairs whose
    product is strictly below ``-deadband`` (a sign flip above the deadband).
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(
            f"actions must have shape (T, C), got {tuple(actions.shape)}"
        )
    steps, channels = actions.shape
    deadband = float(deadband)
    names = ACTION_CHANNEL_NAMES[:channels]
    if steps < 2:
        out = {f"mean_abs_delta_{name}": 0.0 for name in names}
        out.update({f"reversal_rate_{name}": 0.0 for name in names})
        out["n_action_delta"] = 0
        out["n_action_triplet"] = 0
        return out
    delta = np.diff(actions, axis=0)
    mean_abs = np.mean(np.abs(delta), axis=0)
    n_triplet = max(steps - 2, 0)
    if n_triplet == 0:
        rates = np.zeros(channels, dtype=np.float64)
    else:
        prod = delta[1:] * delta[:-1]
        rates = np.mean(prod < -deadband, axis=0)
    out = {f"mean_abs_delta_{name}": float(mean_abs[i]) for i, name in enumerate(names)}
    out.update(
        {f"reversal_rate_{name}": float(rates[i]) for i, name in enumerate(names)}
    )
    out["n_action_delta"] = int(steps - 1)
    out["n_action_triplet"] = int(n_triplet)
    return out


def _combine_action_delta_metrics(rows):
    names = ACTION_CHANNEL_NAMES
    n_delta = 0
    n_triplet = 0
    abs_sum = {name: 0.0 for name in names}
    flip_sum = {name: 0.0 for name in names}
    for row in rows:
        n_d = int(row.get("n_action_delta", 0))
        n_t = int(row.get("n_action_triplet", 0))
        n_delta += n_d
        n_triplet += n_t
        for name in names:
            abs_sum[name] += float(row.get(f"mean_abs_delta_{name}", 0.0)) * n_d
            flip_sum[name] += float(row.get(f"reversal_rate_{name}", 0.0)) * n_t
    out = {
        "n_action_delta": n_delta,
        "n_action_triplet": n_triplet,
    }
    for name in names:
        out[f"mean_abs_delta_{name}"] = (
            abs_sum[name] / n_delta if n_delta else 0.0
        )
        out[f"reversal_rate_{name}"] = (
            flip_sum[name] / n_triplet if n_triplet else 0.0
        )
    return out


def summarize_episodes(
    episodes: list[dict],
    *,
    track: str,
    track_length_m: float,
    spawn_speed_mps_value: float,
) -> dict:
    n = len(episodes)
    if n == 0:
        raise ValueError("summarize_episodes requires at least one episode")
    full_laps = [ep for ep in episodes if ep["full_lap"]]
    lap_times = [
        float(ep["lap_time_s"])
        for ep in full_laps
        if ep.get("lap_time_s") is not None
    ]
    progress = np.asarray([ep["progress_m"] for ep in episodes], dtype=np.float64)
    overtakes = np.asarray(
        [ep.get("overtake_count", 0) for ep in episodes], dtype=np.float64
    )
    n_crash = sum(1 for ep in episodes if ep["crash"])
    n_timeout = sum(1 for ep in episodes if ep["timeout"])
    breakdown = {key: 0 for key in CRASH_TERM_KEYS}
    for ep in episodes:
        reason = ep.get("crash_reason")
        if reason in breakdown:
            breakdown[reason] += 1
    return {
        "n_episodes": n,
        "n_full_laps": len(full_laps),
        "full_lap_completion_rate": len(full_laps) / n,
        "crash_rate": n_crash / n,
        "timeout_rate": n_timeout / n,
        "mean_progress_m": float(progress.mean()),
        "mean_progress_frac": float(progress.mean() / float(track_length_m)),
        "mean_lap_time_s": float(np.mean(lap_times)) if lap_times else None,
        "best_lap_time_s": float(np.min(lap_times)) if lap_times else None,
        "mean_overtake_count": float(overtakes.mean()),
        "track": track,
        "track_length_m": float(track_length_m),
        "spawn_speed_mps": float(spawn_speed_mps_value),
        "crash_breakdown": breakdown,
        "episodes": episodes,
        **_combine_action_delta_metrics(episodes),
    }


def _actor_obs(obs, *, with_sensors: bool) -> torch.Tensor:
    if with_sensors:
        if not isinstance(obs, dict) or "actor" not in obs:
            raise TypeError("with_sensors=True requires obs['actor']")
        return obs["actor"].to(torch.float32)
    if isinstance(obs, dict):
        raise TypeError("with_sensors=False expects a flat observation tensor")
    return obs.to(torch.float32)


def _pad_batch(values: np.ndarray, n: int) -> np.ndarray:
    if values.shape[0] == n:
        return values
    pad = np.repeat(values[-1:], n - values.shape[0], axis=0)
    return np.concatenate([values, pad], axis=0)


def _load_eval_config(path: Path) -> tuple[dict, str]:
    from standalone_trainer import build_config, load_config_patch, parse_args

    text = path.read_text()
    raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("config"), dict):
        return copy.deepcopy(raw["config"]), str(path.resolve())
    patch, _meta = load_config_patch(str(path))
    args, explicit = parse_args(["--config", str(path)])
    return build_config(args, patch=patch, explicit=explicit), str(path.resolve())


def resolve_eval_config(
    checkpoint: Path | None,
    config: str | Path | None,
) -> tuple[dict, str]:
    if config is not None:
        return _load_eval_config(Path(config).resolve())
    if checkpoint is not None:
        snapshot = checkpoint.parent.parent / "config.json"
        if snapshot.is_file():
            return _load_eval_config(snapshot)
    shipped = _TRAINING_DIR / DEFAULT_CONFIG_REL
    if shipped.is_file():
        return _load_eval_config(shipped)
    raise FileNotFoundError("No eval config: pass --config or a run snapshot")


def cap_cuda_memory(fraction: float, device: torch.device) -> None:
    if device.type != "cuda" or fraction <= 0.0 or fraction >= 1.0:
        return
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=index)


def _prepare_eval_cfg(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["opponent_strategy"] = None
    cfg["env"]["domain_randomization"] = {
        **cfg["env"].get("domain_randomization", {}),
        "enabled": False,
    }
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    return cfg


def _prepare_scripted_overtake_cfg(cfg: dict) -> dict:
    cfg = _prepare_eval_cfg(cfg)
    cfg["env"]["opponent_strategy"] = "scripted"
    return cfg


def _flying_eval_source(resolved: list[dict]) -> dict:
    return next(
        (spec for spec in resolved if spec["name"] == "solo_flying"),
        resolved[0],
    )


def extra_scripted_overtake_spec(resolved: list[dict]) -> dict | None:
    """Follow-video scripted opponents on the flying (or first) track."""
    if not resolved:
        return None
    if any(spec["name"] == SCRIPTED_OVERTAKE_NAME for spec in resolved):
        return None
    source = _flying_eval_source(resolved)
    return {
        "name": SCRIPTED_OVERTAKE_NAME,
        "cfg": _prepare_scripted_overtake_cfg(source["cfg"]),
        "opponent_ahead_m": FOLLOW_OPPONENT_AHEAD_M,
        "opponent_speed_mps": FOLLOW_OPPONENT_SPEED_MPS,
    }


def extra_disturbance_specs(resolved: list[dict]) -> list[dict]:
    """Flying eval with steer-gain and extra action-latency knobs; other DR off."""
    if not resolved:
        return []
    existing = {spec["name"] for spec in resolved}
    source = _flying_eval_source(resolved)
    jobs = []
    for gain in DISTURBANCE_STEER_GAINS:
        name = f"disturbance_steer_gain_{str(gain).replace('.', '_')}"
        if name in existing:
            continue
        cfg = _prepare_eval_cfg(source["cfg"])
        cfg["env"]["steer_gain"] = float(gain)
        warp = dict(cfg["env"].get("warp_sim") or {})
        warp["steer_gain"] = float(gain)
        cfg["env"]["warp_sim"] = warp
        jobs.append(
            {
                "name": name,
                "cfg": cfg,
                "opponent_ahead_m": None,
                "opponent_speed_mps": None,
            }
        )
        existing.add(name)
    latency_name = "disturbance_action_latency_2"
    if latency_name not in existing:
        cfg = _prepare_eval_cfg(source["cfg"])
        steps = int(DISTURBANCE_ACTION_LATENCY_STEPS)
        cfg["env"]["simulate_action_latency"] = steps
        dr = dict(cfg["env"].get("domain_randomization") or {})
        dr["enabled"] = False
        dr["action_latency_steps_range"] = [steps, steps]
        cfg["env"]["domain_randomization"] = dr
        jobs.append(
            {
                "name": latency_name,
                "cfg": cfg,
                "opponent_ahead_m": None,
                "opponent_speed_mps": None,
            }
        )
    return jobs


def extra_generalization_specs(resolved: list[dict]) -> list[dict]:
    """Off-heading flying start and a held-out scripted opponent speed."""
    if not resolved:
        return []
    existing = {spec["name"] for spec in resolved}
    source = _flying_eval_source(resolved)
    jobs = []
    if OFF_HEADING_NAME not in existing:
        cfg = _prepare_eval_cfg(source["cfg"])
        cfg["env"]["reset_spawn_yaw_jitter_rad"] = float(OFF_HEADING_YAW_JITTER_RAD)
        jobs.append(
            {
                "name": OFF_HEADING_NAME,
                "cfg": cfg,
                "opponent_ahead_m": None,
                "opponent_speed_mps": None,
                "yaw_jitter_rad": float(OFF_HEADING_YAW_JITTER_RAD),
            }
        )
    if HELD_OUT_OVERTAKE_NAME not in existing:
        jobs.append(
            {
                "name": HELD_OUT_OVERTAKE_NAME,
                "cfg": _prepare_scripted_overtake_cfg(source["cfg"]),
                "opponent_ahead_m": FOLLOW_OPPONENT_AHEAD_M,
                "opponent_speed_mps": float(HELD_OUT_OPPONENT_SPEED_MPS),
            }
        )
    return jobs


def evaluate_policy_on_env(
    env,
    actor,
    normalize,
    *,
    n_spawns: int,
    seed: int,
    with_sensors: bool,
    clip_actions: float,
    spawn_speed: float,
    reversal_deadband: float = DEFAULT_REVERSAL_DEADBAND,
    opponent_ahead_m: float | None = None,
    opponent_speed_mps: float | None = None,
    yaw_jitter_rad: float | None = None,
) -> dict:
    """Roll the mean policy from evenly spaced centerline spawns (one episode each)."""
    from evaluation import actor_is_recurrent
    from f1tenth_env import runtime as rt
    from f1tenth_policy.layout import executed_plant_action

    track = str(env.env_cfg.get("track"))
    length = float(env.track_length)
    spawn_s = even_spawn_s(length, n_spawns)
    poses = []
    yaws = []
    for s_m in spawn_s:
        xy, yaw = pose_at_s(env._track_host, float(s_m))
        poses.append(xy)
        yaws.append(yaw)
    poses = np.stack(poses, axis=0)
    yaws = np.asarray(yaws, dtype=np.float32)
    if yaw_jitter_rad is not None and float(yaw_jitter_rad) != 0.0:
        rng = np.random.default_rng(int(seed))
        yaws = yaws + rng.uniform(
            -float(yaw_jitter_rad),
            float(yaw_jitter_rad),
            size=yaws.shape,
        ).astype(np.float32)
    speeds = np.full(n_spawns, float(spawn_speed), dtype=np.float32)
    opp_poses = None
    opp_yaws = None
    if opponent_ahead_m is not None:
        ahead = []
        ahead_yaw = []
        for s_m in spawn_s:
            xy, yaw = pose_at_s(env._track_host, float(s_m) + float(opponent_ahead_m))
            ahead.append(xy)
            ahead_yaw.append(yaw)
        opp_poses = np.stack(ahead, axis=0)
        opp_yaws = np.asarray(ahead_yaw, dtype=np.float32)
    opp_speed = (
        None
        if opponent_speed_mps is None
        else np.full(n_spawns, float(opponent_speed_mps), dtype=np.float32)
    )

    recurrent = actor_is_recurrent(actor)
    num_envs = int(env.num_envs)
    control_dt = float(env.control_dt)
    max_steps = int(env.max_episode_steps)
    max_ds = 0.1 * length
    episodes: list[dict] = []

    for start in range(0, n_spawns, num_envs):
        sl = slice(start, min(start + num_envs, n_spawns))
        k = sl.stop - sl.start
        batch_pose = _pad_batch(poses[sl], num_envs)
        batch_yaw = _pad_batch(yaws[sl], num_envs)
        batch_speed = _pad_batch(speeds[sl], num_envs)
        reset_kwargs = {
            "seed": int(seed + start),
            "with_sensors": with_sensors,
        }
        if opp_poses is not None:
            reset_kwargs["opponent_pose"] = _pad_batch(opp_poses[sl], num_envs)
            reset_kwargs["opponent_yaw"] = _pad_batch(opp_yaws[sl], num_envs)
            if opp_speed is not None:
                reset_kwargs["opponent_speed"] = _pad_batch(opp_speed[sl], num_envs)
        raw_obs, extras = env.reset_to(
            batch_pose,
            batch_yaw,
            batch_speed,
            **reset_kwargs,
        )
        obs = _actor_obs(raw_obs, with_sensors=with_sensors)
        hidden = None
        if recurrent:
            hidden = actor.initial_hidden(
                num_envs, device=obs.device, dtype=torch.float32
            )
        prev_s = extras["metrics"]["s"].detach().to(torch.float32).clone()
        opp_s_key = extras.get("metrics", {}).get("opponent_s")
        has_opponent = bool(getattr(env, "has_opponent", False))
        if has_opponent and opp_s_key is not None:
            prev_opp_s = opp_s_key.detach().to(torch.float32).clone()
        else:
            prev_opp_s = None
        progress = torch.zeros(num_envs, device=obs.device, dtype=torch.float32)
        overtake_count = torch.zeros(
            num_envs, device=obs.device, dtype=torch.int32
        )
        completed = torch.zeros(num_envs, device=obs.device, dtype=torch.bool)
        lap_time = torch.full(
            (num_envs,), float("nan"), device=obs.device, dtype=torch.float32
        )
        alive = torch.zeros(num_envs, dtype=torch.bool, device=obs.device)
        alive[:k] = True
        crash = np.zeros(num_envs, dtype=bool)
        timeout = np.zeros(num_envs, dtype=bool)
        crash_reason = [None] * num_envs
        steps_alive = np.zeros(num_envs, dtype=np.int32)
        action_traj = [[] for _ in range(k)]

        with torch.no_grad():
            for step in range(max_steps):
                if not bool(alive.any()):
                    break
                model_obs = normalize(obs)
                if recurrent:
                    actions, _, hidden = actor.step(
                        model_obs,
                        hidden,
                        reset_mask=None,
                        deterministic=True,
                        with_logprob=False,
                    )
                else:
                    actions, _ = actor(
                        model_obs, deterministic=True, with_logprob=False
                    )
                actions = actions.clamp(-clip_actions, clip_actions)
                plant_actions = executed_plant_action(actions)
                alive_np = alive.detach().cpu().numpy()
                act_np = plant_actions.detach().cpu().numpy()
                for i in range(k):
                    if alive_np[i]:
                        action_traj[i].append(np.asarray(act_np[i], dtype=np.float64))
                raw_obs, _reward, done, extras = env.step(
                    plant_actions.to(rt.tc_float),
                    n_steps=env.control_interval,
                    with_sensors=with_sensors,
                )
                obs = _actor_obs(raw_obs, with_sensors=with_sensors)
                done_b = done.to(device=obs.device, dtype=torch.bool)
                s_now = extras["metrics"]["s"].detach().to(torch.float32)
                term_s = env._env.tensor["terminal_s"].detach().to(torch.float32)
                s_used = torch.where(done_b, term_s, s_now)
                ds = wrapped_delta_torch(s_used, prev_s, length)
                ds = ds.clamp(-max_ds, max_ds)
                progress = torch.where(alive, progress + ds, progress)
                if prev_opp_s is not None:
                    opp_now = extras["metrics"]["opponent_s"].detach().to(
                        torch.float32
                    )
                    passed = count_progress_overtakes(
                        prev_s,
                        s_used,
                        prev_opp_s,
                        opp_now,
                        alive,
                        length,
                    )
                    overtake_count = overtake_count + passed
                    prev_opp_s = torch.where(alive, opp_now, prev_opp_s)
                newly = (
                    alive
                    & ~completed
                    & (progress >= (length - PROGRESS_EPS_M))
                )
                t_now = float(step + 1) * control_dt
                lap_time = torch.where(
                    newly, torch.full_like(lap_time, t_now), lap_time
                )
                completed = completed | newly
                crash_step, timeout_step = classify_terminations(done_b.cpu(), extras)
                newly_done = alive.cpu().numpy() & done_b.cpu().numpy()
                for i in range(k):
                    if not newly_done[i]:
                        continue
                    crash[i] = bool(crash_step[i])
                    timeout[i] = bool(timeout_step[i])
                    steps_alive[i] = step + 1
                    if crash[i]:
                        term = extras.get("termination", {})
                        for key in CRASH_TERM_KEYS:
                            values = term.get(key)
                            if values is None:
                                continue
                            if bool(values[i].detach().cpu().item()):
                                crash_reason[i] = key
                                break
                alive = alive & ~done_b
                if recurrent and hidden is not None:
                    hidden[done_b] = 0
                prev_s = torch.where(alive, s_used, prev_s)

        still = alive.detach().cpu().numpy()
        for i in range(k):
            if still[i] and steps_alive[i] == 0:
                steps_alive[i] = max_steps
                timeout[i] = True

        progress_np = progress.detach().cpu().numpy()
        lap_np = lap_time.detach().cpu().numpy()
        completed_np = completed.detach().cpu().numpy()
        overtake_np = overtake_count.detach().cpu().numpy()
        for i in range(k):
            lap = float(lap_np[i]) if completed_np[i] and math.isfinite(lap_np[i]) else None
            traj = action_traj[i]
            if traj:
                act_stats = action_delta_metrics(
                    np.stack(traj, axis=0), deadband=reversal_deadband
                )
            else:
                act_stats = action_delta_metrics(
                    np.zeros((0, 2), dtype=np.float64), deadband=reversal_deadband
                )
            episodes.append(
                {
                    "spawn_s": float(spawn_s[start + i]),
                    "progress_m": float(progress_np[i]),
                    "full_lap": bool(completed_np[i]),
                    "lap_time_s": lap,
                    "overtake_count": int(overtake_np[i]),
                    "crash": bool(crash[i]),
                    "timeout": bool(timeout[i]),
                    "crash_reason": crash_reason[i],
                    "steps": int(steps_alive[i]),
                    **act_stats,
                }
            )

    return summarize_episodes(
        episodes,
        track=track,
        track_length_m=length,
        spawn_speed_mps_value=spawn_speed,
    )


def _open_env(cfg: dict, num_envs: int):
    from f1tenth_env import F1tenthEnv
    from standalone_trainer import build_env_cfg

    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": num_envs},
    )
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _identity_normalize(obs):
    return obs


def load_actor_bundle(checkpoint: Path, device: torch.device, cfg: dict):
    """Load a sensor-policy actor + obs normalizer from checkpoint metadata.

    Uses the artifact's own ``actor_architecture`` rather than the installed
    package's current format constants, so older run directories still eval.
    """
    from f1tenth_policy import (
        ObsNormalizer,
        actor_from_architecture,
        normalize_actor_architecture,
    )

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = payload.get("actor_architecture")
    if not isinstance(architecture, dict):
        raise ValueError(
            f"{checkpoint.name}: missing actor_architecture metadata"
        )
    architecture = normalize_actor_architecture(architecture)
    actor_obs_dim = int(payload.get("actor_obs_dim", cfg["obs"]["num_actor_obs"]))
    expected_obs = int(cfg["obs"]["num_actor_obs"])
    if actor_obs_dim != expected_obs:
        raise ValueError(
            f"{checkpoint.name}: actor_obs_dim={actor_obs_dim} "
            f"!= config {expected_obs}"
        )
    if "actor" not in payload:
        raise ValueError(f"{checkpoint.name}: missing actor weights")
    if "obs_norm" not in payload:
        raise ValueError(
            f"{checkpoint.name}: sensor policy artifact is missing obs_norm"
        )
    actor = actor_from_architecture(architecture).to(
        device=device, dtype=torch.float32
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    normalizer = ObsNormalizer(
        obs_dim=actor_obs_dim,
        device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    normalizer.load_state_dict(payload["obs_norm"])
    return actor, normalizer, architecture, payload


def evaluate_checkpoint(
    checkpoint: str | Path | None,
    *,
    config: str | Path | None = None,
    n_spawns: int = DEFAULT_N_SPAWNS,
    num_envs: int = DEFAULT_NUM_ENVS,
    seed: int = 0,
    device_str: str = "auto",
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    with_sensors: bool = True,
    actor=None,
    normalize=None,
    reversal_deadband: float | None = None,
) -> dict:
    from f1tenth_env import runtime as rt
    from standalone_trainer import resolve_training_scenarios, select_device

    ckpt_path = Path(checkpoint).resolve() if checkpoint is not None else None
    cfg, config_source = resolve_eval_config(ckpt_path, config)
    cfg = _prepare_eval_cfg(cfg)
    device = select_device(device_str)
    cap_cuda_memory(memory_fraction, device)
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )

    if actor is None:
        if ckpt_path is None or not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        actor, normalizer, _arch, _payload = load_actor_bundle(
            ckpt_path, device, cfg
        )
        normalize = normalizer.normalize
    elif normalize is None:
        normalize = _identity_normalize

    workspace = str(_TRAINING_DIR)
    resolved = resolve_training_scenarios(cfg, max(int(num_envs), 1), workspace)
    if not resolved:
        resolved = [
            {
                "name": str(cfg["env"].get("track", "default")),
                "weight": 1.0,
                "num_envs": int(num_envs),
                "cfg": cfg,
            }
        ]

    clip_actions = float(cfg["env"]["clip_actions"])
    if reversal_deadband is None:
        reversal_deadband = float(
            cfg.get("ppo", {}).get("reversal_deadband", DEFAULT_REVERSAL_DEADBAND)
        )
    jobs = [
        {
            "name": spec["name"],
            "cfg": _prepare_eval_cfg(spec["cfg"]),
            "opponent_ahead_m": None,
            "opponent_speed_mps": None,
        }
        for spec in resolved
    ]
    extra = extra_scripted_overtake_spec(resolved)
    if extra is not None:
        jobs.append(extra)
    jobs.extend(extra_disturbance_specs(resolved + jobs))
    jobs.extend(extra_generalization_specs(resolved + jobs))

    scenarios = {}
    for job in jobs:
        scn_cfg = job["cfg"]
        speed = spawn_speed_mps(scn_cfg["env"])
        env = _open_env(scn_cfg, int(num_envs))
        try:
            scenarios[job["name"]] = evaluate_policy_on_env(
                env,
                actor,
                normalize,
                n_spawns=int(n_spawns),
                seed=int(seed),
                with_sensors=with_sensors,
                clip_actions=clip_actions,
                spawn_speed=speed,
                reversal_deadband=float(reversal_deadband),
                opponent_ahead_m=job.get("opponent_ahead_m"),
                opponent_speed_mps=job.get("opponent_speed_mps"),
                yaw_jitter_rad=job.get("yaw_jitter_rad"),
            )
        finally:
            env.close()

    out = {
        "eval_version": EVAL_VERSION,
        "checkpoint": None if ckpt_path is None else ckpt_path.name,
        "transitions": None if ckpt_path is None else _transitions_from_name(ckpt_path),
        "config_source": config_source,
        "n_spawns": int(n_spawns),
        "num_envs": int(num_envs),
        "seed": int(seed),
        "deterministic": True,
        "domain_randomization": False,
        "memory_fraction": float(memory_fraction),
        "with_sensors": bool(with_sensors),
        "scenarios": scenarios,
    }
    return out


def default_json_path(checkpoint: Path, out_dir: Path | None) -> Path:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{checkpoint.stem}.full_lap.json"
    return checkpoint.with_suffix(".full_lap.json")


def wandb_metrics(result: dict) -> dict:
    payload = {}
    for name, row in result.get("scenarios", {}).items():
        prefix = f"eval/{name}"
        payload[f"{prefix}/full_lap_completion_rate"] = row["full_lap_completion_rate"]
        payload[f"{prefix}/crash_rate"] = row["crash_rate"]
        payload[f"{prefix}/timeout_rate"] = row["timeout_rate"]
        payload[f"{prefix}/mean_progress_m"] = row["mean_progress_m"]
        payload[f"{prefix}/mean_progress_frac"] = row["mean_progress_frac"]
        payload[f"{prefix}/n_episodes"] = row["n_episodes"]
        payload[f"{prefix}/mean_overtake_count"] = row.get(
            "mean_overtake_count", 0.0
        )
        if row.get("mean_lap_time_s") is not None:
            payload[f"{prefix}/mean_lap_time_s"] = row["mean_lap_time_s"]
        if row.get("best_lap_time_s") is not None:
            payload[f"{prefix}/best_lap_time_s"] = row["best_lap_time_s"]
        for key in (
            "reversal_rate_throttle",
            "reversal_rate_steer",
            "mean_abs_delta_throttle",
            "mean_abs_delta_steer",
        ):
            if key in row:
                payload[f"{prefix}/{key}"] = row[key]
    return payload


def log_wandb(result: dict, *, run_id: str, project: str | None = None) -> None:
    import wandb

    init_kwargs = {
        "project": project or os.getenv("WANDB_PROJECT", "f1tenth-genesis"),
        "id": run_id,
        "resume": "allow",
        "mode": os.getenv("WANDB_MODE", "online"),
    }
    if entity := os.getenv("WANDB_ENTITY"):
        init_kwargs["entity"] = entity
    run = wandb.init(**init_kwargs)
    step = result.get("transitions")
    payload = wandb_metrics(result)
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=int(step))
    run.finish()


def _print_result(result: dict) -> None:
    ckpt = result.get("checkpoint") or "(actor)"
    print(f"{ckpt}  transitions={result.get('transitions')}")
    for name, row in result["scenarios"].items():
        mean_t = row["mean_lap_time_s"]
        best_t = row["best_lap_time_s"]
        mean_s = "n/a" if mean_t is None else f"{mean_t:.3f}s"
        best_s = "n/a" if best_t is None else f"{best_t:.3f}s"
        print(
            f"  {name}: full_lap={row['full_lap_completion_rate']:.3f} "
            f"crash={row['crash_rate']:.3f} timeout={row['timeout_rate']:.3f} "
            f"progress={row['mean_progress_m']:.1f}m "
            f"({row['mean_progress_frac']:.2f} L) "
            f"mean_lap={mean_s} best_lap={best_s} "
            f"overtakes={row.get('mean_overtake_count', 0.0):.3f} "
            f"n={row['n_episodes']}"
        )
        if "reversal_rate_throttle" in row:
            print(
                f"    reversal throttle={row['reversal_rate_throttle']:.4f} "
                f"steer={row.get('reversal_rate_steer', 0.0):.4f} "
                f"|Δa| throttle={row.get('mean_abs_delta_throttle', 0.0):.4f} "
                f"steer={row.get('mean_abs_delta_steer', 0.0):.4f}"
            )


def discover_checkpoints(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("policy_*.pt"), key=_transitions_from_name)
    if not ckpts:
        raise FileNotFoundError(f"No policy_*.pt checkpoints in {ckpt_dir}")
    return ckpts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic full-lap PPO eval")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--n-spawns", type=int, default=DEFAULT_N_SPAWNS)
    p.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--memory-fraction",
        type=float,
        default=DEFAULT_MEMORY_FRACTION,
        help="torch.cuda.set_per_process_memory_fraction (CUDA only)",
    )
    p.add_argument("--wandb-run-id", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Run-dir mode: reuse JSON that already exists",
    )
    p.add_argument(
        "--reversal-deadband",
        type=float,
        default=None,
        help="sign-flip deadband for eval reversal rate (default: config/0)",
    )
    return p.parse_args(argv)


def _eval_one(args: argparse.Namespace, checkpoint: Path, out_path: Path) -> dict:
    result = evaluate_checkpoint(
        checkpoint,
        config=args.config,
        n_spawns=args.n_spawns,
        num_envs=args.num_envs,
        seed=args.seed,
        device_str=args.device,
        memory_fraction=args.memory_fraction,
        reversal_deadband=args.reversal_deadband,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    _print_result(result)
    print(f"  wrote {out_path}")
    if args.wandb_run_id:
        log_wandb(
            result,
            run_id=args.wandb_run_id,
            project=args.wandb_project,
        )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        ckpts = discover_checkpoints(run_dir)
        out_dir = Path(args.out_dir).resolve() if args.out_dir else None
        results = []
        t0 = time.time()
        for ckpt in ckpts:
            out_path = default_json_path(ckpt, out_dir)
            if args.skip_existing and out_path.is_file():
                cached = json.loads(out_path.read_text())
                if cached.get("eval_version") == EVAL_VERSION:
                    results.append(cached)
                    print(f"{ckpt.name}: cached {out_path}")
                    continue
            results.append(_eval_one(args, ckpt, out_path))
        summary_dir = out_dir if out_dir is not None else run_dir
        summary_path = summary_dir / "full_lap_eval.json"
        summary_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(f"\nSummary: {summary_path}")
        print(f"Elapsed: {(time.time() - t0) / 60.0:.1f} min")
        return
    if not args.checkpoint:
        sys.exit("Need --checkpoint or --run-dir")
    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    out_path = Path(args.out).resolve() if args.out else default_json_path(ckpt, out_dir)
    _eval_one(args, ckpt, out_path)


if __name__ == "__main__":
    main()

"""1v1 head-to-head tournament for trained policy checkpoints.

Match format (two swapped-seat legs per pairing):
  * each leg is a head-to-head shotgun start -- both cars side by side at the
    start line; leg 1 places model A on the positive lateral half, leg 2 swaps;
  * first car to complete N laps (default 10) wins the leg;
  * a crash (leaving the track, or a hard collision) is NOT terminal: the car is
    respawned facing the track direction, on the side of the track it went off,
    held stationary for 2 s, then resumes racing;
  * match winner = more leg wins; 1-1 splits break on aggregate laps, progress,
    penalties, then seed rank (see ``decide_match_winner``).

Both cars are driven by their own checkpoint through the *identical* ego
observation + inference pipeline (``build_symmetric_agent_obs`` +
``dual_policy_step``); ``env.step()`` / ``PolicyOpponent`` are never used during a
race. Bracket: triple elimination (3 losses knocks a model out). Seeding uses
solo lap timing CSV; collisions caused are tallied per model as a standings
tie-break.

Modes:
  worker:   --model-a A.pt --model-b B.pt --out result.json
  bracket:  --run-dir <run> [--seed-csv ...] [--checkpoints-dir ...]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_env import runtime as rt  # noqa: E402
from f1tenth_env.eval_viz import RolloutVisualizer, yaw_from_quat_wxyz  # noqa: E402
from f1tenth_env.geom import quat_to_xyz  # noqa: E402
from f1tenth_env.kernel import (  # noqa: E402
    contact_stage_kernel,
    observation_stage_kernel,
    physics_stage_kernel,
    reset_to_kernel,
)
from f1tenth_env.observations import build_observation, obs_opponent  # noqa: E402
from f1tenth_env.terminations import collision_mask, init_termination_params  # noqa: E402
from f1tenth_env.utils import build_step_state, build_track_cache  # noqa: E402
from f1tenth_env.utils import compute_oob_from_boundary_state  # noqa: E402
from standalone_trainer import (  # noqa: E402
    DEFAULT_CONFIG,
    ObsNormalizer,
    build_models,
    select_device,
)

OOB_CONSECUTIVE = 2
COLLISION_TERM_SPEED = 2.0
CKPT_RE = re.compile(r"policy_(\d+)\.pt$")
DEFAULT_ELIM_LOSSES = 3
SEED_METRIC_MEAN = "mean"
SEED_METRIC_MIN = "min"
_RACE_GEOM_CACHE = "race"


# --------------------------------------------------------------------------- #
# Warp env adapters (head-to-head race engine)
# --------------------------------------------------------------------------- #
def _warp_stream(env: F1tenthEnv):
    if env.device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(env.device))
    return None


def _warp_yaw(base_quat: torch.Tensor) -> torch.Tensor:
    return quat_to_xyz(base_quat, rpy=True, degrees=False)[:, 2]


def _warp_track_geom(env: F1tenthEnv) -> dict:
    cache = env.track_state.setdefault("track_geom_cache", {})
    geom = cache.get(_RACE_GEOM_CACHE)
    if geom is None:
        geom = build_track_cache(env.track_state["centerline"], env.device)
        cache[_RACE_GEOM_CACHE] = geom
    return geom


def _warp_centerline_frame(env: F1tenthEnv, idx: torch.Tensor):
    geom = _warp_track_geom(env)
    tangent = geom["seg"][idx]
    tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)
    p_curr = geom["C"][idx]
    return tangent, normal, p_curr


def _warp_closest_centerline_indices(env: F1tenthEnv, pos_xy: torch.Tensor):
    ss = build_step_state(
        base_pos=torch.cat(
            [pos_xy, torch.zeros(pos_xy.shape[0], 1, device=env.device)],
            dim=1,
        ),
        track_state=env.track_state,
        device=env.device,
        cache_id=_RACE_GEOM_CACHE,
    )
    return ss["frenet"]["best_idx"]


def _warp_build_step_state(env: F1tenthEnv, which: str) -> dict:
    st = env.read_state()
    if which == "ego":
        base_pos = st["base_pos"]
        which_ws = "ego"
    else:
        base_pos = st["opp_base_pos"]
        which_ws = "opp"
    ss = build_step_state(
        base_pos=base_pos,
        track_state=env.track_state,
        device=env.device,
        cache_id=_RACE_GEOM_CACHE,
    )
    ws = env.read_wheel_state(which_ws)
    ss["tyre_slip"] = ws["tyre_slip"]
    ss["tyre_load"] = ws["tyre_load"]
    if which == "opp":
        ss["opp_vel_world"] = st["opp_vel_world"]
    return ss


def _warp_agent_dict(
    pos: torch.Tensor,
    quat: torch.Tensor,
    vel_world: torch.Tensor,
    step_state: dict,
) -> dict[str, torch.Tensor]:
    return {
        "pos_xy": pos[:, :2],
        "yaw": _warp_yaw(quat),
        "vel_xy": vel_world[:, :2],
        "s": step_state["frenet"]["s"],
        "ey": step_state["boundary"]["ey"],
        "L": step_state["frenet"]["L"],
    }


def _warp_apply_opponent_range_mask(
    block: torch.Tensor,
    s_self: torch.Tensor,
    s_other: torch.Tensor,
    track_len: torch.Tensor,
    obs_cfg: dict,
) -> torch.Tensor:
    gap = s_other - s_self
    half = 0.5 * track_len
    gap = torch.where(gap > half, gap - track_len, gap)
    gap = torch.where(gap < -half, gap + track_len, gap)
    ahead = float(obs_cfg.get("opp_obs_ahead_m", 40.0))
    behind = float(obs_cfg.get("opp_obs_behind_m", 20.0))
    visible = (gap <= ahead) & (gap >= -behind)
    return torch.where(visible.unsqueeze(-1), block, torch.zeros_like(block))


def _warp_collision_state(env: F1tenthEnv) -> dict[str, torch.Tensor]:
    st = env.read_state()
    car_length = float(env.env_cfg.get("car_length", 0.568))
    car_width = float(env.env_cfg.get("car_width", 0.296))
    overlap = collision_mask(
        st["base_pos"][:, :2],
        st["opp_base_pos"][:, :2],
        _warp_yaw(st["base_quat"]),
        _warp_yaw(st["opp_base_quat"]),
        car_length,
        car_width,
    )
    return {
        "overlap": overlap,
        "closing_speed": env._env.tensor["contact_closing_speed"],
    }


# --------------------------------------------------------------------------- #
# Pure helpers (bracket / seeding / decisions)
# --------------------------------------------------------------------------- #
def opponent_race_obs_dim(obs_cfg: dict) -> int:
    """Observation width for symmetric 1v1 racing from a base config."""
    base = int(obs_cfg["num_obs"])
    if obs_cfg.get("enable_opponent_obs", False):
        return base
    return base + int(obs_cfg.get("opponent_obs_dim", 6))


def _transitions_from_name(name: str) -> int:
    m = CKPT_RE.search(name)
    return int(m.group(1)) if m else -1


def _float_or_none(val) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _int_or_zero(val) -> int:
    if val is None or val == "":
        return 0
    return int(val)


def seed_sort_key(row: dict, metric: str = SEED_METRIC_MIN) -> tuple:
    min_lap = _float_or_none(row.get("min") or row.get("min_s"))
    crashes = _int_or_zero(row.get("crashes"))
    mean_lap = _float_or_none(row.get("mean") or row.get("mean_s"))
    transitions = _int_or_zero(row.get("transitions"))
    if min_lap is None:
        min_lap = float("inf")
    if mean_lap is None:
        mean_lap = float("inf")
    if metric == SEED_METRIC_MEAN:
        return (mean_lap, crashes, min_lap, transitions)
    return (min_lap, crashes, mean_lap, transitions)


def rows_have_explicit_seed_rank(rows: list[dict]) -> bool:
    if not rows:
        return False
    ranked = [
        r for r in rows
        if str(r.get("seed_rank", "")).strip() not in ("", "None")
    ]
    return len(ranked) == len(rows)


def infer_seed_metric(rows: list[dict]) -> str:
    """Infer seed ordering metric when CSV has no explicit seed_rank.

    * ``seed_metric`` column from lap_timing selector -> use that value.
    * ``reference_mode`` present (tournament_seed.csv) -> mean.
    * Legacy lap_bench schema (``min_s``/``mean_s``, no lap_timing markers) -> min.
    * Otherwise default to mean (lap_timing ``car_lap_times.csv`` schema).
    """
    for row in rows:
        metric = str(row.get("seed_metric", "")).strip().lower()
        if metric in (SEED_METRIC_MEAN, SEED_METRIC_MIN):
            return metric
    if rows and str(rows[0].get("reference_mode", "")).strip():
        return SEED_METRIC_MEAN
    if rows and ("min_s" in rows[0] or "mean_s" in rows[0]):
        if "reference_mode" not in rows[0] and "seed_metric" not in rows[0]:
            return SEED_METRIC_MIN
    return SEED_METRIC_MEAN


def row_is_seedable(row: dict, metric: str) -> bool:
    n_laps = _int_or_zero(row.get("n_laps"))
    if n_laps <= 0:
        return False
    if metric == SEED_METRIC_MEAN:
        return _float_or_none(row.get("mean") or row.get("mean_s")) is not None
    return _float_or_none(row.get("min") or row.get("min_s")) is not None


def load_seed_rows(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return []
    out = []
    for row in rows:
        ckpt = row.get("checkpoint") or row.get("model")
        if not ckpt:
            continue
        row = dict(row)
        row["checkpoint"] = ckpt.strip()
        out.append(row)
    return out


def seed_candidates_from_rows(
    rows: list[dict], candidates: int, ckpt_dir: Path,
) -> list[str]:
    preserve_rank = rows_have_explicit_seed_rank(rows)
    metric = infer_seed_metric(rows)
    usable = []
    for row in rows:
        if not row_is_seedable(row, metric):
            continue
        name = row["checkpoint"]
        if not (ckpt_dir / name).exists():
            raise FileNotFoundError(
                f"Seed checkpoint missing: {ckpt_dir / name} "
                f"(from {row.get('checkpoint', name)})"
            )
        usable.append(row)
    if preserve_rank:
        usable.sort(key=lambda r: int(r["seed_rank"]))
    else:
        usable.sort(key=lambda r: seed_sort_key(r, metric))
    names = [r["checkpoint"] for r in usable[:candidates]]
    if not names:
        raise ValueError(f"No seeded checkpoints with clean laps in {ckpt_dir}")
    basenames = [Path(n).name for n in names]
    if len(basenames) != len(set(basenames)):
        dupes = sorted({n for n in basenames if basenames.count(n) > 1})
        raise ValueError(f"Duplicate checkpoint basenames in seed list: {dupes}")
    return names


def validate_checkpoint_dir(ckpt_dir: Path, names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        base = Path(name).name
        if base in seen:
            raise ValueError(f"Duplicate checkpoint basename: {base}")
        seen.add(base)
        path = ckpt_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")


def match_id(round_no: int, ordinal: int, model_a: str, model_b: str,
             rematch: int = 0) -> str:
    a = Path(model_a).stem
    b = Path(model_b).stem
    base = f"r{round_no:03d}_m{ordinal:03d}__{a}__vs__{b}"
    return base if rematch == 0 else f"{base}__rematch{rematch}"


def decide_winner(
    *,
    laps_a: int,
    laps_b: int,
    prog_a: float,
    prog_b: float,
    crashes_a: int,
    crashes_b: int,
    collisions_a: int,
    collisions_b: int,
    target_laps: int,
    capped: bool,
    seed_rank_a: int,
    seed_rank_b: int,
) -> tuple[str, str]:
    """Return (winner 'a'|'b', finish_reason). Lower seed_rank is better."""
    if laps_a >= target_laps and laps_b < target_laps:
        return "a", "laps"
    if laps_b >= target_laps and laps_a < target_laps:
        return "b", "laps"
    if laps_a >= target_laps and laps_b >= target_laps:
        if prog_a > prog_b:
            return "a", "laps_both_progress"
        if prog_b > prog_a:
            return "b", "laps_both_progress"
    if laps_a != laps_b:
        return ("a", "tie_break_laps") if laps_a > laps_b else ("b", "tie_break_laps")
    if prog_a != prog_b:
        return ("a", "tie_break_progress") if prog_a > prog_b else (
            "b", "tie_break_progress"
        )
    pen_a = crashes_a + collisions_a
    pen_b = crashes_b + collisions_b
    if pen_a != pen_b:
        return ("a", "tie_break_crashes") if pen_a < pen_b else (
            "b", "tie_break_crashes"
        )
    if seed_rank_a != seed_rank_b:
        return ("a", "tie_break_seed") if seed_rank_a < seed_rank_b else (
            "b", "tie_break_seed"
        )
    if capped:
        return "a", "time_cap"
    return "a", "tie_default"


def decide_match_winner(
    leg1: dict,
    leg2: dict,
    *,
    seed_rank_a: int,
    seed_rank_b: int,
) -> tuple[str, str]:
    """Aggregate two swapped-seat legs into one match winner.

    Each leg uses ``decide_winner``. The match winner is whoever wins more legs.
    On a 1-1 split, break ties by aggregate total laps (A vs B across both
    legs), then aggregate progress, then fewer total crashes+collisions, then
    better (lower) seed_rank.
    """
    wins_a = sum(1 for leg in (leg1, leg2) if leg["winner"] == "a")
    wins_b = 2 - wins_a
    if wins_a > wins_b:
        return "a", "legs_won"
    if wins_b > wins_a:
        return "b", "legs_won"

    laps_a = leg1["laps_a"] + leg2["laps_a"]
    laps_b = leg1["laps_b"] + leg2["laps_b"]
    if laps_a != laps_b:
        return ("a", "tie_break_agg_laps") if laps_a > laps_b else (
            "b", "tie_break_agg_laps"
        )
    prog_a = leg1["progress_a"] + leg2["progress_a"]
    prog_b = leg1["progress_b"] + leg2["progress_b"]
    if prog_a != prog_b:
        return ("a", "tie_break_agg_progress") if prog_a > prog_b else (
            "b", "tie_break_agg_progress"
        )
    pen_a = (
        leg1["crashes_a"] + leg1["collisions_a"]
        + leg2["crashes_a"] + leg2["collisions_a"]
    )
    pen_b = (
        leg1["crashes_b"] + leg1["collisions_b"]
        + leg2["crashes_b"] + leg2["collisions_b"]
    )
    if pen_a != pen_b:
        return ("a", "tie_break_agg_penalties") if pen_a < pen_b else (
            "b", "tie_break_agg_penalties"
        )
    if seed_rank_a != seed_rank_b:
        return ("a", "tie_break_seed") if seed_rank_a < seed_rank_b else (
            "b", "tie_break_seed"
        )
    return "a", "tie_default"


def _leg_result_from_race(
    out: dict,
    *,
    leg: int,
    sim_side_positive_a: bool,
    target_laps: int,
    seed_rank_a: int,
    seed_rank_b: int,
) -> dict:
    sl, ol = int(out["sim_laps"][0]), int(out["opp_laps"][0])
    sp, op = float(out["sim_prog"][0]), float(out["opp_prog"][0])
    winner, finish_reason = decide_winner(
        laps_a=sl, laps_b=ol, prog_a=sp, prog_b=op,
        crashes_a=int(out["sim_crashes"][0]),
        crashes_b=int(out["opp_crashes"][0]),
        collisions_a=int(out["collide_sim"][0]),
        collisions_b=int(out["collide_opp"][0]),
        target_laps=target_laps, capped=bool(out["capped"]),
        seed_rank_a=seed_rank_a, seed_rank_b=seed_rank_b,
    )
    return {
        "leg": leg,
        "sim_side_positive_a": sim_side_positive_a,
        "winner": winner,
        "finish_reason": finish_reason,
        "laps_a": sl,
        "laps_b": ol,
        "progress_a": sp,
        "progress_b": op,
        "crashes_a": int(out["sim_crashes"][0]),
        "crashes_b": int(out["opp_crashes"][0]),
        "collisions_a": int(out["collide_sim"][0]),
        "collisions_b": int(out["collide_opp"][0]),
        "steps": out["steps"],
        "race_time_s": out["race_time_s"],
        "capped": bool(out["capped"]),
    }


def select_bye(
    players: list[str],
    losses: dict[str, int],
    seed_rank: dict[str, int],
    bye_counts: dict[str, int],
) -> str:
    """Pick a bye fairly: fewest prior byes, then higher loss bucket, then seed."""
    return min(
        players,
        key=lambda m: (bye_counts.get(m, 0), losses[m], seed_rank[m]),
    )


def pair_avoid_rematch(
    players: list[str],
    losses: dict[str, int],
    seed_rank: dict[str, int],
    last_opponent: dict[str, str | None],
) -> list[tuple[str, str]]:
    """Pair active players, preferring same loss tier but crossing tiers if needed."""
    if len(players) < 2:
        return []
    ordered = sorted(players, key=lambda m: (losses[m], seed_rank[m]))
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []

    tiers: dict[int, list[str]] = {}
    for m in ordered:
        tiers.setdefault(losses[m], []).append(m)
    for loss in sorted(tiers):
        tier = sorted(tiers[loss], key=lambda m: seed_rank[m])
        i = 0
        while i < len(tier):
            if tier[i] in used:
                i += 1
                continue
            partner = None
            for j in range(i + 1, len(tier)):
                if tier[j] in used:
                    continue
                if last_opponent.get(tier[i]) != tier[j]:
                    partner = tier[j]
                    break
            if partner is None:
                for j in range(i + 1, len(tier)):
                    if tier[j] not in used:
                        partner = tier[j]
                        break
            if partner is None:
                i += 1
                continue
            pairs.append((tier[i], partner))
            used.add(tier[i])
            used.add(partner)
            i += 1

    leftover = [m for m in ordered if m not in used]
    i = 0
    while i + 1 < len(leftover):
        a, b = leftover[i], leftover[i + 1]
        if last_opponent.get(a) == b and i + 2 < len(leftover):
            leftover[i + 1], leftover[i + 2] = leftover[i + 2], leftover[i + 1]
            b = leftover[i + 1]
        pairs.append((a, b))
        used.add(a)
        used.add(b)
        i += 2
    return pairs


def bracket_config_fingerprint(
    *,
    track: str,
    laps: int,
    freeze_s: float,
    seed: int,
    candidates: int,
    elim_losses: int,
    seeds: list[str],
    seed_csv: str | None,
    checkpoints_dir: str,
) -> dict:
    return {
        "track": track,
        "laps": laps,
        "freeze_s": freeze_s,
        "seed": seed,
        "candidates": candidates,
        "elim_losses": elim_losses,
        "seeds": seeds,
        "seed_csv": str(seed_csv) if seed_csv else None,
        "checkpoints_dir": str(checkpoints_dir),
    }


def validate_resume_config(saved: dict, current: dict) -> None:
    for key in (
        "track", "laps", "freeze_s", "seed", "candidates", "elim_losses",
        "checkpoints_dir",
    ):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"Bracket resume config mismatch on {key!r}: "
                f"log={saved.get(key)!r} current={current.get(key)!r}"
            )
    if saved.get("seeds") != current.get("seeds"):
        raise ValueError("Bracket resume seed list does not match current config")


def pin_actions(actions: torch.Tensor, hold: torch.Tensor) -> torch.Tensor:
    frozen = hold > 0
    if not bool(frozen.any()):
        return actions
    return torch.where(frozen.unsqueeze(1), torch.zeros_like(actions), actions)


def hard_collision_fault(
    hard: torch.Tensor,
    sim_prog: torch.Tensor,
    opp_prog: torch.Tensor,
    sim_active: torch.Tensor,
    opp_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sim_at_fault, opp_at_fault) for hard collisions."""
    both = hard & sim_active & opp_active
    behind_sim = both & (sim_prog <= opp_prog)
    behind_opp = both & (sim_prog > opp_prog)
    sim_only = hard & sim_active & ~opp_active
    opp_only = hard & opp_active & ~sim_active
    sim_fault = behind_sim | sim_only
    opp_fault = behind_opp | opp_only
    return sim_fault, opp_fault


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 4)))


# --------------------------------------------------------------------------- #
# Config / policy loading
# --------------------------------------------------------------------------- #
def build_race_config(cfg: dict, track: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["track"] = track
    cfg["env"]["domain_randomization"] = {
        **cfg["env"].get("domain_randomization", {}),
        "enabled": False,
    }
    cfg["env"]["simulate_action_latency"] = False
    cfg["env"]["opponent_strategy"] = "policy"
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_oob_max_consecutive"] = 10**9
    cfg["env"]["term_not_moving_time_s"] = 1e9
    cfg["env"]["term_heading_error_rad"] = 1e9
    cfg["env"]["episode_length"] = 1e9
    race_dim = opponent_race_obs_dim(cfg["obs"])
    cfg["obs"]["enable_opponent_obs"] = True
    cfg["obs"]["num_obs"] = race_dim
    return cfg


def validate_policy_bundle(cfg: dict, ckpt: Path, device: torch.device) -> str | None:
    expected = int(cfg["obs"]["num_obs"])
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    ckpt_dim = payload.get("obs_dim")
    if ckpt_dim is not None and int(ckpt_dim) != expected:
        return (
            f"{ckpt.name}: checkpoint obs_dim={ckpt_dim} "
            f"!= race obs_dim={expected}"
        )
    if "obs_norm" in payload:
        mean = payload["obs_norm"].get("mean")
        if mean is not None and int(mean.shape[0]) != expected:
            return (
                f"{ckpt.name}: normalizer dim={mean.shape[0]} "
                f"!= race obs_dim={expected}"
            )
    actor_sd = payload.get("actor", {})
    for key, tensor in actor_sd.items():
        if key.endswith("weight") and tensor.ndim == 2:
            if int(tensor.shape[1]) != expected:
                return (
                    f"{ckpt.name}: actor input dim={tensor.shape[1]} "
                    f"!= race obs_dim={expected}"
                )
            break
    return None


def load_policy_bundle(cfg: dict, ckpt: Path, device: torch.device):
    err = validate_policy_bundle(cfg, ckpt, device)
    if err:
        raise ValueError(err)
    models, _ = build_models(cfg, device)
    normalizer = ObsNormalizer(
        obs_dim=cfg["obs"]["num_obs"], device=device,
        eps=float(cfg["obs"].get("norm_eps", 1e-8)),
        clip=float(cfg["obs"].get("norm_clip", 10.0)),
    )
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    models.actor.load_state_dict(payload["actor"])
    if "obs_norm" in payload:
        normalizer.load_state_dict(payload["obs_norm"])
    models.actor.eval()
    return models.actor, normalizer


def make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    env = F1tenthEnv(
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"], show_viewer=False, enable_recording=False,
    )
    env._race_opp_last_actions = torch.zeros(
        num_envs, 2, device=env.device, dtype=torch.float32,
    )
    env.term_params = init_termination_params(cfg["env"], env.control_dt)
    return env


def load_base_config(model_a: Path) -> dict:
    cfg_path = model_a.parent.parent / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())["config"]
    return copy.deepcopy(DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# Symmetric dual-policy race engine
# --------------------------------------------------------------------------- #
def _fill_fixed_pose(buffer, values, num_envs: int, cols: int) -> None:
    tensor = torch.as_tensor(values, device=buffer.device, dtype=torch.float32)
    shape = (num_envs, cols) if cols > 1 else (num_envs,)
    tensor = tensor.reshape(-1, cols) if cols > 1 else tensor.reshape(-1)
    if tensor.shape[0] == 1:
        tensor = tensor.expand(*shape).contiguous()
    buffer.copy_(tensor)


def _place_race_cars(
    env: F1tenthEnv,
    *,
    ego_xy: torch.Tensor,
    ego_yaw: torch.Tensor,
    ego_speed: torch.Tensor,
    opp_xy: torch.Tensor,
    opp_yaw: torch.Tensor,
    opp_speed: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Place ego/opponent poses through reset_to_kernel (keeps Warp state consistent)."""
    n = env.num_envs
    if mask is None:
        env._reset_mask.fill_(True)
    else:
        env._reset_mask.copy_(mask.to(env.device))
    _fill_fixed_pose(env._fixed_ego_pose, ego_xy, n, 2)
    _fill_fixed_pose(env._fixed_ego_yaw, ego_yaw, n, 1)
    _fill_fixed_pose(env._fixed_ego_speed, ego_speed, n, 1)
    _fill_fixed_pose(env._fixed_opp_pose, opp_xy, n, 2)
    _fill_fixed_pose(env._fixed_opp_yaw, opp_yaw, n, 1)
    _fill_fixed_pose(env._fixed_opp_speed, opp_speed, n, 1)
    wp.launch(
        reset_to_kernel,
        dim=n,
        inputs=[
            env._reset_mask_wp,
            wp.from_torch(env._fixed_ego_pose, dtype=wp.vec2f),
            wp.from_torch(env._fixed_ego_yaw),
            wp.from_torch(env._fixed_ego_speed),
            wp.from_torch(env._fixed_opp_pose, dtype=wp.vec2f),
            wp.from_torch(env._fixed_opp_yaw),
            wp.from_torch(env._fixed_opp_speed),
            env._ego.buffers,
            env._opponent.buffers,
            env._env.buffers,
            env._track.data,
            env._sim_params,
            env._obs_params,
            env._reset_params,
            env._opponent_params,
            env._raw_obs_wp[0],
            env._raw_obs_wp[1],
            env._obs_wp[0],
            env._obs_wp[1],
            env._opponent_obs_wp,
        ],
        device=env.wp_device,
        stream=_warp_stream(env),
    )
    wp.synchronize()
    env.obs_buf = env._obs[env._active_obs]


def _read_car_poses(env: F1tenthEnv):
    st = env.read_state()
    ego_xy = st["base_pos"][:, :2]
    opp_xy = st["opp_base_pos"][:, :2]
    ego_yaw = _warp_yaw(st["base_quat"])
    opp_yaw = _warp_yaw(st["opp_base_quat"])
    ego_speed = torch.linalg.norm(st["base_vel_world"][:, :2], dim=-1)
    opp_speed = torch.linalg.norm(st["opp_vel_world"][:, :2], dim=-1)
    return ego_xy, ego_yaw, ego_speed, opp_xy, opp_yaw, opp_speed


def _clear_env_termination_state(env: F1tenthEnv, mask: torch.Tensor) -> None:
    tensors = env._env.tensor
    for key in (
        "oob_streak", "stopped_streak", "done", "term_timeout", "term_oob",
        "term_stopped", "term_invalid", "term_collision", "done_flags",
    ):
        if tensors[key].dtype == torch.bool:
            tensors[key][mask] = False
        else:
            tensors[key][mask] = 0


def _pin_cars(
    env: F1tenthEnv,
    *,
    ego_xy: torch.Tensor,
    ego_yaw: torch.Tensor,
    opp_xy: torch.Tensor,
    opp_yaw: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    zero = torch.zeros(env.num_envs, device=env.device, dtype=rt.tc_float)
    _place_race_cars(
        env,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        ego_speed=zero,
        opp_xy=opp_xy,
        opp_yaw=opp_yaw,
        opp_speed=zero,
        mask=mask,
    )


def apply_shotgun_start(env: F1tenthEnv, sim_side_positive: torch.Tensor) -> None:
    n = env.num_envs
    idx = torch.zeros(n, dtype=torch.long, device=env.device)
    tangent, normal, p_curr = _warp_centerline_frame(env, idx)
    yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    w_l = env.track_state["w_tr_left_torch"][idx].to(rt.tc_float)
    w_r = env.track_state["w_tr_right_torch"][idx].to(rt.tc_float)
    half = torch.minimum(w_l, w_r) * 0.4
    sim_lat = torch.where(sim_side_positive, half, -half)
    sim_xy = p_curr + normal * sim_lat.unsqueeze(1)
    opp_xy = p_curr + normal * (-sim_lat).unsqueeze(1)
    zero = torch.zeros(n, device=env.device, dtype=rt.tc_float)
    _place_race_cars(
        env,
        ego_xy=sim_xy,
        ego_yaw=yaw,
        ego_speed=zero,
        opp_xy=opp_xy,
        opp_yaw=yaw,
        opp_speed=zero,
    )


def build_symmetric_agent_obs(env: F1tenthEnv, agent: str) -> torch.Tensor:
    if agent == "sim":
        return env.obs_buf

    st = env.read_state()
    opp_read = env._read_vehicle(env._opponent.tensor)
    opp_ss = _warp_build_step_state(env, "opp")
    ego_ss = _warp_build_step_state(env, "ego")
    opp_block = obs_opponent(
        _warp_agent_dict(
            st["opp_base_pos"], st["opp_base_quat"], st["opp_vel_world"], opp_ss,
        ),
        _warp_agent_dict(
            st["base_pos"], st["base_quat"], st["base_vel_world"], ego_ss,
        ),
        env.obs_cfg,
    )
    opp_block = _warp_apply_opponent_range_mask(
        opp_block,
        opp_ss["frenet"]["s"],
        ego_ss["frenet"]["s"],
        opp_ss["frenet"]["L"],
        env.obs_cfg,
    )
    return build_observation(
        num_obs=env.obs_cfg["num_obs"],
        num_envs=env.num_envs,
        base_lin_vel=opp_read["base_lin_vel"],
        base_ang_vel=opp_read["base_ang_vel"],
        base_lin_acc=opp_read["base_lin_acc"],
        last_actions=env._race_opp_last_actions,
        base_pos=st["opp_base_pos"],
        base_quat=st["opp_base_quat"],
        obs_cfg=env.obs_cfg,
        step_state=opp_ss,
        device=env.device,
        opponent_block=opp_block,
    )


def dual_policy_step(env, sim_actions, opp_actions, n_steps, clip_actions) -> None:
    env.actions = torch.clip(sim_actions, -clip_actions, clip_actions).to(
        device=env.device, dtype=torch.float32,
    ).contiguous()
    opp_actions = torch.clip(opp_actions, -clip_actions, clip_actions).to(
        device=env.device, dtype=torch.float32,
    ).contiguous()
    inactive = 1 - env._active_obs
    wp.launch(
        physics_stage_kernel,
        dim=2 * env.num_envs,
        inputs=[
            wp.from_torch(env.actions, dtype=wp.vec2f),
            wp.from_torch(opp_actions, dtype=wp.vec2f),
            env._ego.buffers,
            env._opponent.buffers,
            env._env.physics_buffers,
            env._track.data,
            env._sim_params,
            env._reset_params,
            env._opponent_params,
            env.num_envs,
            env.control_interval,
        ],
        device=env.wp_device,
        stream=_warp_stream(env),
    )
    wp.launch(
        contact_stage_kernel,
        dim=env.num_envs,
        inputs=[
            env._ego.buffers,
            env._opponent.buffers,
            env._env.physics_buffers,
            env._opponent_params,
        ],
        device=env.wp_device,
        stream=_warp_stream(env),
    )
    wp.launch(
        observation_stage_kernel,
        dim=env.num_envs,
        inputs=[
            env._ego.buffers,
            env._opponent.buffers,
            env._env.buffers,
            env._track.data,
            env._obs_params,
            env._reset_params,
            env._raw_obs_wp[env._active_obs],
            env._raw_obs_wp[inactive],
            env._obs_wp[inactive],
            env._opponent_obs_wp,
        ],
        device=env.wp_device,
        stream=_warp_stream(env),
    )
    wp.synchronize()
    env._active_obs = inactive
    env.obs_buf = env._obs[inactive]
    env.last_actions.copy_(env.actions)
    env._race_opp_last_actions.copy_(opp_actions)


def _unwrap_ds(prev_s, s, length):
    ds = s - prev_s
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)
    return ds


def _progress_ds(prev_s, s, length, suppress):
    ds = _unwrap_ds(prev_s, s, length)
    glitch = ds.abs() > (0.5 * length)
    ds = torch.where(suppress | glitch, torch.zeros_like(ds), ds)
    return ds.clamp_min(0.0), s.detach().clone()


def _crashed_side_respawn(env, which, mask):
    st = env.read_state()
    pos_xy = (st["base_pos"] if which == "ego" else st["opp_base_pos"])[:, :2]
    idx_full = _warp_closest_centerline_indices(env, pos_xy)
    idx = idx_full[mask]
    tangent, normal, p_curr = _warp_centerline_frame(env, idx)
    yaw_part = torch.atan2(tangent[:, 1], tangent[:, 0])
    e = ((pos_xy[mask] - p_curr) * normal).sum(-1)
    w_l = env.track_state["w_tr_left_torch"][idx].to(rt.tc_float)
    w_r = env.track_state["w_tr_right_torch"][idx].to(rt.tc_float)
    side_w = torch.where(e >= 0, w_l, w_r)
    lat = torch.sign(e) * (0.5 * side_w)
    rxy = p_curr + normal * lat.unsqueeze(1)
    ego_xy, ego_yaw, ego_speed, opp_xy, opp_yaw, opp_speed = _read_car_poses(env)
    if which == "ego":
        ego_xy = ego_xy.clone()
        ego_yaw = ego_yaw.clone()
        ego_xy[mask] = rxy.to(rt.tc_float)
        ego_yaw[mask] = yaw_part.to(rt.tc_float)
        ego_speed = torch.where(mask, torch.zeros_like(ego_speed), ego_speed)
    else:
        opp_xy = opp_xy.clone()
        opp_yaw = opp_yaw.clone()
        opp_xy[mask] = rxy.to(rt.tc_float)
        opp_yaw[mask] = yaw_part.to(rt.tc_float)
        opp_speed = torch.where(mask, torch.zeros_like(opp_speed), opp_speed)
    _place_race_cars(
        env,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        ego_speed=ego_speed,
        opp_xy=opp_xy,
        opp_yaw=opp_yaw,
        opp_speed=opp_speed,
        mask=mask,
    )
    if which == "ego":
        t = env._ego.tensor
        return t["x"].clone(), t["y"].clone(), t["yaw"].clone()
    t = env._opponent.tensor
    return t["x"].clone(), t["y"].clone(), t["yaw"].clone()


def lateral_error_at(env: F1tenthEnv, which: str) -> torch.Tensor:
    st = env.read_state()
    pos_xy = (st["base_pos"] if which == "ego" else st["opp_base_pos"])[:, :2]
    idx = _warp_closest_centerline_indices(env, pos_xy)
    _, normal, p_curr = _warp_centerline_frame(env, idx)
    return ((pos_xy - p_curr) * normal).sum(-1)


def tangent_yaw_at(env: F1tenthEnv, which: str) -> torch.Tensor:
    st = env.read_state()
    pos_xy = (st["base_pos"] if which == "ego" else st["opp_base_pos"])[:, :2]
    idx = _warp_closest_centerline_indices(env, pos_xy)
    tangent, _, _ = _warp_centerline_frame(env, idx)
    return torch.atan2(tangent[:, 1], tangent[:, 0])


def force_car_oob(env: F1tenthEnv, which: str, side_positive: bool) -> float:
    """Teleport the car outside the track on the requested side; return lateral sign."""
    st = env.read_state()
    pos_xy = (st["base_pos"] if which == "ego" else st["opp_base_pos"])[:, :2]
    idx = _warp_closest_centerline_indices(env, pos_xy)
    tangent, normal, p_curr = _warp_centerline_frame(env, idx)
    yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
    w_l = env.track_state["w_tr_left_torch"][idx].to(rt.tc_float)
    w_r = env.track_state["w_tr_right_torch"][idx].to(rt.tc_float)
    margin = float(env.term_params.get("term_oob_margin_m", 0.0))
    if side_positive:
        beyond = w_l + margin + 1.0
    else:
        beyond = -(w_r + margin + 1.0)
    rxy = p_curr + normal * beyond.unsqueeze(1)
    ego_xy, ego_yaw, ego_speed, opp_xy, opp_yaw, opp_speed = _read_car_poses(env)
    if which == "ego":
        ego_xy = rxy.to(rt.tc_float)
        ego_yaw = yaw.to(rt.tc_float)
        ego_speed = torch.zeros_like(ego_speed)
    else:
        opp_xy = rxy.to(rt.tc_float)
        opp_yaw = yaw.to(rt.tc_float)
        opp_speed = torch.zeros_like(opp_speed)
    _place_race_cars(
        env,
        ego_xy=ego_xy,
        ego_yaw=ego_yaw,
        ego_speed=ego_speed,
        opp_xy=opp_xy,
        opp_yaw=opp_yaw,
        opp_speed=opp_speed,
    )
    return float(torch.sign(lateral_error_at(env, which)[0]).item())


def freeze_steps_for(env: F1tenthEnv, freeze_s: float) -> int:
    return max(1, int(round(float(freeze_s) / float(env.control_dt))))


def _apply_hold_pins(
    env: F1tenthEnv,
    active: torch.Tensor,
    sim_hold: torch.Tensor,
    opp_hold: torch.Tensor,
    fx_s: torch.Tensor,
    fy_s: torch.Tensor,
    fyaw_s: torch.Tensor,
    fx_o: torch.Tensor,
    fy_o: torch.Tensor,
    fyaw_o: torch.Tensor,
) -> None:
    holding = active & ((sim_hold > 0) | (opp_hold > 0))
    if not bool(holding.any()):
        return
    _pin_cars(
        env,
        ego_xy=torch.stack((fx_s, fy_s), dim=1),
        ego_yaw=fyaw_s,
        opp_xy=torch.stack((fx_o, fy_o), dim=1),
        opp_yaw=fyaw_o,
        mask=holding,
    )


def advance_hold_race_step(
    env: F1tenthEnv,
    *,
    sim_actions: torch.Tensor,
    opp_actions: torch.Tensor,
    control_interval: int,
    clip_actions: float,
    freeze_steps: int,
    sim_hold: torch.Tensor,
    opp_hold: torch.Tensor,
    fx_s: torch.Tensor,
    fy_s: torch.Tensor,
    fyaw_s: torch.Tensor,
    fx_o: torch.Tensor,
    fy_o: torch.Tensor,
    fyaw_o: torch.Tensor,
    sim_streak: torch.Tensor,
    opp_streak: torch.Tensor,
    sim_prog: torch.Tensor,
    opp_prog: torch.Tensor,
    sim_crashes: torch.Tensor,
    opp_crashes: torch.Tensor,
    collide_sim: torch.Tensor,
    collide_opp: torch.Tensor,
    prev_sim_s: torch.Tensor,
    prev_opp_s: torch.Tensor,
    active: torch.Tensor,
) -> dict:
    """One production race-loop iteration for hold/crash accounting (no policies)."""
    _apply_hold_pins(
        env, active, sim_hold, opp_hold, fx_s, fy_s, fyaw_s, fx_o, fy_o, fyaw_o,
    )

    sim_act = pin_actions(sim_actions.to(rt.tc_float), sim_hold)
    opp_act = pin_actions(opp_actions.to(rt.tc_float), opp_hold)
    holding = active & ((sim_hold > 0) | (opp_hold > 0))
    if not bool(holding.any()):
        dual_policy_step(env, sim_act, opp_act, control_interval, clip_actions)

    suppress = (sim_hold > 0) | (opp_hold > 0)
    sim_ss = _warp_build_step_state(env, "ego")
    opp_ss = _warp_build_step_state(env, "opp")
    length = sim_ss["frenet"]["L"]
    ds_s, prev_sim_s = _progress_ds(
        prev_sim_s, sim_ss["frenet"]["s"], length, suppress)
    ds_o, prev_opp_s = _progress_ds(
        prev_opp_s, opp_ss["frenet"]["s"], length, suppress)
    sim_prog = sim_prog + ds_s
    opp_prog = opp_prog + ds_o

    sim_active = active & (sim_hold == 0)
    opp_active = active & (opp_hold == 0)

    sim_oob, _ = compute_oob_from_boundary_state(
        sim_ss["boundary"],
        margin_m=float(env.term_params["term_oob_margin_m"]))
    opp_oob, _ = compute_oob_from_boundary_state(
        opp_ss["boundary"],
        margin_m=float(env.term_params["term_oob_margin_m"]))
    sim_streak = torch.where(
        sim_oob & sim_active, sim_streak + 1,
        torch.where(sim_active, torch.zeros_like(sim_streak), sim_streak),
    )
    opp_streak = torch.where(
        opp_oob & opp_active, opp_streak + 1,
        torch.where(opp_active, torch.zeros_like(opp_streak), opp_streak),
    )

    coll = _warp_collision_state(env)
    hard = coll["overlap"] & (coll["closing_speed"] > COLLISION_TERM_SPEED)
    sim_fault, opp_fault = hard_collision_fault(
        hard, sim_prog, opp_prog, sim_active, opp_active,
    )
    collide_sim = collide_sim + (sim_fault & active).to(torch.int32)
    collide_opp = collide_opp + (opp_fault & active).to(torch.int32)

    sim_crash = sim_active & ((sim_streak >= OOB_CONSECUTIVE) | sim_fault)
    opp_crash = opp_active & ((opp_streak >= OOB_CONSECUTIVE) | opp_fault)
    sim_crashes = sim_crashes + sim_crash.to(torch.int32)
    opp_crashes = opp_crashes + opp_crash.to(torch.int32)

    if bool(sim_crash.any()):
        fx_s2, fy_s2, fyaw_s2 = _crashed_side_respawn(env, "ego", sim_crash)
        _clear_env_termination_state(env, sim_crash)
        prev_sim_s = torch.where(
            sim_crash, _warp_build_step_state(env, "ego")["frenet"]["s"], prev_sim_s)
        sim_hold = torch.where(
            sim_crash, torch.full_like(sim_hold, freeze_steps), sim_hold)
        fx_s = torch.where(sim_crash, fx_s2, fx_s)
        fy_s = torch.where(sim_crash, fy_s2, fy_s)
        fyaw_s = torch.where(sim_crash, fyaw_s2, fyaw_s)
        sim_streak = torch.where(sim_crash, torch.zeros_like(sim_streak), sim_streak)
    if bool(opp_crash.any()):
        fx_o2, fy_o2, fyaw_o2 = _crashed_side_respawn(env, "opp", opp_crash)
        _clear_env_termination_state(env, opp_crash)
        prev_opp_s = torch.where(
            opp_crash, _warp_build_step_state(env, "opp")["frenet"]["s"], prev_opp_s)
        opp_hold = torch.where(
            opp_crash, torch.full_like(opp_hold, freeze_steps), opp_hold)
        fx_o = torch.where(opp_crash, fx_o2, fx_o)
        fy_o = torch.where(opp_crash, fy_o2, fy_o)
        fyaw_o = torch.where(opp_crash, fyaw_o2, fyaw_o)
        opp_streak = torch.where(opp_crash, torch.zeros_like(opp_streak), opp_streak)

    sim_hm = active & (sim_hold > 0)
    opp_hm = active & (opp_hold > 0)
    _apply_hold_pins(
        env, active, sim_hold, opp_hold, fx_s, fy_s, fyaw_s, fx_o, fy_o, fyaw_o,
    )
    if bool(sim_hm.any()):
        sim_hold = torch.where(sim_hm, sim_hold - 1, sim_hold)
    if bool(opp_hm.any()):
        opp_hold = torch.where(opp_hm, opp_hold - 1, opp_hold)

    return {
        "sim_hold": sim_hold, "opp_hold": opp_hold,
        "fx_s": fx_s, "fy_s": fy_s, "fyaw_s": fyaw_s,
        "fx_o": fx_o, "fy_o": fy_o, "fyaw_o": fyaw_o,
        "sim_streak": sim_streak, "opp_streak": opp_streak,
        "sim_prog": sim_prog, "opp_prog": opp_prog,
        "sim_crashes": sim_crashes, "opp_crashes": opp_crashes,
        "collide_sim": collide_sim, "collide_opp": collide_opp,
        "prev_sim_s": prev_sim_s, "prev_opp_s": prev_opp_s,
        "active": active,
    }


def race(env, sim_policy, opp_policy, *, target_laps, max_steps, sim_side_positive,
         control_interval, clip_actions, freeze_steps, seed, record=None,
         overlay_prefix=""):
    n = env.num_envs
    device = env.device
    sim_actor, sim_norm = sim_policy
    opp_actor, opp_norm = opp_policy
    control_dt = float(env.control_dt)
    py_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            torch.manual_seed(seed)
            env.reset()
            apply_shotgun_start(env, sim_side_positive)

            sim_laps = torch.zeros(n, dtype=torch.int32, device=device)
            opp_laps = torch.zeros(n, dtype=torch.int32, device=device)
            sim_prog = torch.zeros(n, dtype=rt.tc_float, device=device)
            opp_prog = torch.zeros(n, dtype=rt.tc_float, device=device)
            sim_crashes = torch.zeros(n, dtype=torch.int32, device=device)
            opp_crashes = torch.zeros(n, dtype=torch.int32, device=device)
            collide_sim = torch.zeros(n, dtype=torch.int32, device=device)
            collide_opp = torch.zeros(n, dtype=torch.int32, device=device)

            sim_ss = _warp_build_step_state(env, "ego")
            opp_ss = _warp_build_step_state(env, "opp")
            prev_sim_s = sim_ss["frenet"]["s"].detach().clone()
            prev_opp_s = opp_ss["frenet"]["s"].detach().clone()
            track_len = sim_ss["frenet"]["L"]

            sim_streak = torch.zeros(n, dtype=torch.int32, device=device)
            opp_streak = torch.zeros(n, dtype=torch.int32, device=device)
            sim_hold = torch.zeros(n, dtype=torch.int32, device=device)
            opp_hold = torch.zeros(n, dtype=torch.int32, device=device)
            fx_s = torch.zeros(n, dtype=rt.tc_float, device=device)
            fy_s = torch.zeros(n, dtype=rt.tc_float, device=device)
            fyaw_s = torch.zeros(n, dtype=rt.tc_float, device=device)
            fx_o = torch.zeros(n, dtype=rt.tc_float, device=device)
            fy_o = torch.zeros(n, dtype=rt.tc_float, device=device)
            fyaw_o = torch.zeros(n, dtype=rt.tc_float, device=device)
            active = torch.ones(n, dtype=torch.bool, device=device)
            step = 0

            with torch.no_grad():
                while step < max_steps and bool(active.any()):
                    _apply_hold_pins(
                        env, active, sim_hold, opp_hold,
                        fx_s, fy_s, fyaw_s, fx_o, fy_o, fyaw_o,
                    )

                    sim_obs = build_symmetric_agent_obs(env, "sim").to(torch.float32)
                    opp_obs = build_symmetric_agent_obs(env, "opp").to(torch.float32)
                    sim_act, _ = sim_actor(sim_norm.normalize(sim_obs),
                                           deterministic=True, with_logprob=False)
                    opp_act, _ = opp_actor(opp_norm.normalize(opp_obs),
                                           deterministic=True, with_logprob=False)
                    sim_act = pin_actions(sim_act.to(rt.tc_float), sim_hold)
                    opp_act = pin_actions(opp_act.to(rt.tc_float), opp_hold)
                    holding = active & ((sim_hold > 0) | (opp_hold > 0))
                    if not bool(holding.any()):
                        dual_policy_step(env, sim_act, opp_act, control_interval,
                                         clip_actions)
                    step += 1

                    suppress = (sim_hold > 0) | (opp_hold > 0)
                    if step <= 1:
                        suppress = torch.ones(n, dtype=torch.bool, device=device)

                    sim_ss = _warp_build_step_state(env, "ego")
                    opp_ss = _warp_build_step_state(env, "opp")
                    length = sim_ss["frenet"]["L"]
                    ds_s, prev_sim_s = _progress_ds(
                        prev_sim_s, sim_ss["frenet"]["s"], length, suppress)
                    ds_o, prev_opp_s = _progress_ds(
                        prev_opp_s, opp_ss["frenet"]["s"], length, suppress)
                    sim_prog = sim_prog + ds_s
                    opp_prog = opp_prog + ds_o
                    sim_laps = (sim_prog / length).floor().to(torch.int32)
                    opp_laps = (opp_prog / length).floor().to(torch.int32)

                    sim_active = active & (sim_hold == 0)
                    opp_active = active & (opp_hold == 0)

                    sim_oob, _ = compute_oob_from_boundary_state(
                        sim_ss["boundary"],
                        margin_m=float(env.term_params["term_oob_margin_m"]))
                    opp_oob, _ = compute_oob_from_boundary_state(
                        opp_ss["boundary"],
                        margin_m=float(env.term_params["term_oob_margin_m"]))
                    sim_streak = torch.where(
                        sim_oob & sim_active, sim_streak + 1,
                        torch.where(sim_active, torch.zeros_like(sim_streak), sim_streak),
                    )
                    opp_streak = torch.where(
                        opp_oob & opp_active, opp_streak + 1,
                        torch.where(opp_active, torch.zeros_like(opp_streak), opp_streak),
                    )

                    coll = _warp_collision_state(env)
                    hard = coll["overlap"] & (coll["closing_speed"] > COLLISION_TERM_SPEED)
                    sim_fault, opp_fault = hard_collision_fault(
                        hard, sim_prog, opp_prog, sim_active, opp_active,
                    )
                    collide_sim += (sim_fault & active).to(torch.int32)
                    collide_opp += (opp_fault & active).to(torch.int32)

                    sim_crash = sim_active & (
                        (sim_streak >= OOB_CONSECUTIVE) | sim_fault
                    )
                    opp_crash = opp_active & (
                        (opp_streak >= OOB_CONSECUTIVE) | opp_fault
                    )
                    sim_crashes += sim_crash.to(torch.int32)
                    opp_crashes += opp_crash.to(torch.int32)

                    if bool(sim_crash.any()):
                        fx_s2, fy_s2, fyaw_s2 = _crashed_side_respawn(
                            env, "ego", sim_crash)
                        _clear_env_termination_state(env, sim_crash)
                        prev_sim_s = torch.where(
                            sim_crash,
                            _warp_build_step_state(env, "ego")["frenet"]["s"],
                            prev_sim_s,
                        )
                        sim_hold = torch.where(
                            sim_crash, torch.full_like(sim_hold, freeze_steps), sim_hold)
                        fx_s = torch.where(sim_crash, fx_s2, fx_s)
                        fy_s = torch.where(sim_crash, fy_s2, fy_s)
                        fyaw_s = torch.where(sim_crash, fyaw_s2, fyaw_s)
                        sim_streak = torch.where(
                            sim_crash, torch.zeros_like(sim_streak), sim_streak)
                    if bool(opp_crash.any()):
                        fx_o2, fy_o2, fyaw_o2 = _crashed_side_respawn(
                            env, "opp", opp_crash)
                        _clear_env_termination_state(env, opp_crash)
                        prev_opp_s = torch.where(
                            opp_crash,
                            _warp_build_step_state(env, "opp")["frenet"]["s"],
                            prev_opp_s,
                        )
                        opp_hold = torch.where(
                            opp_crash, torch.full_like(opp_hold, freeze_steps), opp_hold)
                        fx_o = torch.where(opp_crash, fx_o2, fx_o)
                        fy_o = torch.where(opp_crash, fy_o2, fy_o)
                        fyaw_o = torch.where(opp_crash, fyaw_o2, fyaw_o)
                        opp_streak = torch.where(
                            opp_crash, torch.zeros_like(opp_streak), opp_streak)

                    sim_hm = active & (sim_hold > 0)
                    opp_hm = active & (opp_hold > 0)
                    _apply_hold_pins(
                        env, active, sim_hold, opp_hold,
                        fx_s, fy_s, fyaw_s, fx_o, fy_o, fyaw_o,
                    )
                    if bool(sim_hm.any()):
                        sim_hold = torch.where(sim_hm, sim_hold - 1, sim_hold)
                    if bool(opp_hm.any()):
                        opp_hold = torch.where(opp_hm, opp_hold - 1, opp_hold)

                    if record is not None:
                        _record_frame(
                            env, record, sim_laps, opp_laps, overlay_prefix, step,
                        )

                    finished = active & ((sim_laps >= target_laps) |
                                         (opp_laps >= target_laps))
                    active = active & ~finished
    finally:
        random.setstate(py_state)

    capped = step >= max_steps
    return {
        "sim_laps": sim_laps.cpu().numpy(), "opp_laps": opp_laps.cpu().numpy(),
        "sim_prog": (sim_prog / track_len).cpu().numpy(),
        "opp_prog": (opp_prog / track_len).cpu().numpy(),
        "sim_crashes": sim_crashes.cpu().numpy(),
        "opp_crashes": opp_crashes.cpu().numpy(),
        "collide_sim": collide_sim.cpu().numpy(),
        "collide_opp": collide_opp.cpu().numpy(),
        "steps": step, "capped": capped,
        "race_time_s": step * control_dt,
    }


def _record_frame(env, record, sim_laps, opp_laps, prefix, step):
    st = env.read_state()
    sim_xy = st["base_pos"][:, :2].cpu().numpy()
    sim_yaw = np.array([yaw_from_quat_wxyz(q.tolist()) for q in st["base_quat"]])
    speed = torch.linalg.norm(st["base_lin_vel"][:, :2], dim=-1).cpu().numpy()
    opp_xy = st["opp_base_pos"][:, :2].cpu().numpy()
    opp_yaw = np.array([yaw_from_quat_wxyz(q.tolist()) for q in st["opp_base_quat"]])
    extra = f"{prefix}  A={int(sim_laps[0])} B={int(opp_laps[0])} laps"
    record.render(ego_xy=sim_xy[:1], ego_yaw=sim_yaw[:1], speed=speed[:1],
                  opp_xy=opp_xy[:1], opp_yaw=opp_yaw[:1],
                  done=np.zeros(1, dtype=bool), extra_text=extra, step=step)


# --------------------------------------------------------------------------- #
# Match worker
# --------------------------------------------------------------------------- #
def race_match(
    model_a,
    model_b,
    track,
    laps,
    freeze_s,
    seed,
    device_str,
    precision,
    *,
    max_race_duration_s: float | None = None,
    max_steps: int | None = None,
    seed_rank_a: int = 0,
    seed_rank_b: int = 1,
    match_meta: dict | None = None,
    video_path=None,
) -> dict:
    a_path, b_path = Path(model_a).resolve(), Path(model_b).resolve()
    base = load_base_config(a_path)
    cfg = build_race_config(base, track)

    device = select_device(device_str)
    rt.configure(float_dtype=torch.float64 if precision == "64" else torch.float32,
                 int_dtype=torch.int32, dev=device, eps=1e-12)

    control_interval = int(cfg["env"]["control_interval"])
    clip = float(cfg["env"]["clip_actions"])
    env = make_env(cfg, 1)
    control_dt = float(env.control_dt)
    freeze_steps = max(1, int(round(freeze_s / control_dt)))
    if max_steps is None:
        if max_race_duration_s is not None:
            max_steps = max(1, int(max_race_duration_s / control_dt))
        else:
            max_steps = int(laps * 90.0 / control_dt)

    record = None
    try:
        bundle_a = load_policy_bundle(cfg, a_path, device)
        bundle_b = load_policy_bundle(cfg, b_path, device)
        side_pos = torch.ones(1, dtype=torch.bool, device=device)
        side_neg = torch.zeros(1, dtype=torch.bool, device=device)
        overlay = f"{a_path.stem} vs {b_path.stem}"

        if video_path is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            record = RolloutVisualizer(
                centerline=env.track_state["centerline"],
                w_tr_left=env.track_state["w_tr_left"],
                w_tr_right=env.track_state["w_tr_right"],
                car_length=float(cfg["env"].get("car_length", 0.568)),
                car_width=float(cfg["env"].get("car_width", 0.296)),
                num_show=1, mp4_path=str(video_path),
                fps=int(round(1.0 / control_dt)), has_opponent=True,
            )

        leg1_out = race(
            env, bundle_a, bundle_b, target_laps=laps, max_steps=max_steps,
            sim_side_positive=side_pos, control_interval=control_interval,
            clip_actions=clip, freeze_steps=freeze_steps, seed=seed,
            record=record, overlay_prefix=f"L1 {overlay}",
        )
        leg1 = _leg_result_from_race(
            leg1_out, leg=1, sim_side_positive_a=True, target_laps=laps,
            seed_rank_a=seed_rank_a, seed_rank_b=seed_rank_b,
        )

        leg2_out = race(
            env, bundle_a, bundle_b, target_laps=laps, max_steps=max_steps,
            sim_side_positive=side_neg, control_interval=control_interval,
            clip_actions=clip, freeze_steps=freeze_steps, seed=seed + 1,
            record=None, overlay_prefix=f"L2 {overlay}",
        )
        leg2 = _leg_result_from_race(
            leg2_out, leg=2, sim_side_positive_a=False, target_laps=laps,
            seed_rank_a=seed_rank_a, seed_rank_b=seed_rank_b,
        )
    finally:
        if record is not None:
            record.close()
        env.close()

    winner, finish_reason = decide_match_winner(
        leg1, leg2, seed_rank_a=seed_rank_a, seed_rank_b=seed_rank_b,
    )
    result = {
        "model_a": a_path.name,
        "model_b": b_path.name,
        "model_a_path": str(a_path),
        "model_b_path": str(b_path),
        "winner": winner,
        "finish_reason": finish_reason,
        "leg_wins_a": sum(1 for leg in (leg1, leg2) if leg["winner"] == "a"),
        "leg_wins_b": sum(1 for leg in (leg1, leg2) if leg["winner"] == "b"),
        "leg1": leg1,
        "leg2": leg2,
        "laps_a": leg1["laps_a"] + leg2["laps_a"],
        "laps_b": leg1["laps_b"] + leg2["laps_b"],
        "progress_a": leg1["progress_a"] + leg2["progress_a"],
        "progress_b": leg1["progress_b"] + leg2["progress_b"],
        "crashes_a": leg1["crashes_a"] + leg2["crashes_a"],
        "crashes_b": leg1["crashes_b"] + leg2["crashes_b"],
        "collisions_a": leg1["collisions_a"] + leg2["collisions_a"],
        "collisions_b": leg1["collisions_b"] + leg2["collisions_b"],
        "steps": leg1["steps"] + leg2["steps"],
        "race_time_s": leg1["race_time_s"] + leg2["race_time_s"],
        "capped": leg1["capped"] or leg2["capped"],
        "laps_target": laps,
        "seed_rank_a": seed_rank_a,
        "seed_rank_b": seed_rank_b,
    }
    if match_meta:
        result.update(match_meta)
    return result


# --------------------------------------------------------------------------- #
# Triple-elimination bracket orchestrator
# --------------------------------------------------------------------------- #
def _resolve_seed_csv(args, run_dir: Path) -> Path:
    if args.seed_csv:
        return Path(args.seed_csv).resolve()
    return run_dir / "lap_timing" / "car_lap_times.csv"


def _resolve_ckpt_dir(args, run_dir: Path) -> Path:
    if args.checkpoints_dir:
        return Path(args.checkpoints_dir).resolve()
    return run_dir / "checkpoints"


def _seed_candidates(args, run_dir: Path, ckpt_dir: Path) -> list[str]:
    csv_path = _resolve_seed_csv(args, run_dir)
    if not csv_path.exists():
        sys.exit(f"Seed CSV not found: {csv_path}")
    rows = load_seed_rows(csv_path)
    try:
        return seed_candidates_from_rows(rows, args.candidates, ckpt_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))


def _bracket_state_path(out_dir: Path) -> Path:
    return out_dir / "bracket_log.json"


def _load_or_init_bracket(args, run_dir: Path, ckpt_dir: Path, out_dir: Path):
    seeds = _seed_candidates(args, run_dir, ckpt_dir)
    fingerprint = bracket_config_fingerprint(
        track=args.track, laps=args.laps, freeze_s=args.freeze_s, seed=args.seed,
        candidates=args.candidates, elim_losses=args.elim_losses, seeds=seeds,
        seed_csv=str(_resolve_seed_csv(args, run_dir)),
        checkpoints_dir=str(ckpt_dir),
    )
    log_path = _bracket_state_path(out_dir)
    if log_path.exists():
        saved = json.loads(log_path.read_text())
        validate_resume_config(saved.get("config", {}), fingerprint)
        losses = {m: int(v) for m, v in saved["losses"].items()}
        collisions = {m: int(v) for m, v in saved["collisions"].items()}
        bye_counts = {m: int(v) for m, v in saved.get("bye_counts", {}).items()}
        last_opponent = {
            m: (v if v else None)
            for m, v in saved.get("last_opponent", {}).items()
        }
        match_log = list(saved.get("matches", []))
        completed_ids = {m["match_id"] for m in match_log if "match_id" in m}
        rnd = int(saved.get("round", 0))
        rematch_counts = {
            tuple(sorted((m["model_a"], m["model_b"]))): 0
            for m in match_log
        }
        for m in match_log:
            key = tuple(sorted((m["model_a"], m["model_b"])))
            tag = m.get("match_id", "")
            if "__rematch" in tag:
                try:
                    rematch_counts[key] = max(
                        rematch_counts.get(key, 0),
                        int(tag.rsplit("rematch", 1)[-1]),
                    )
                except ValueError:
                    pass
        seed_rank = {m: i for i, m in enumerate(seeds)}
        print(f"Resuming bracket from {log_path} (round {rnd}, "
              f"{len(match_log)} matches done)")
        return (
            seeds, seed_rank, losses, collisions, bye_counts, last_opponent,
            match_log, completed_ids, rnd, rematch_counts, fingerprint,
        )

    seed_rank = {m: i for i, m in enumerate(seeds)}
    losses = {m: 0 for m in seeds}
    collisions = {m: 0 for m in seeds}
    bye_counts = {m: 0 for m in seeds}
    last_opponent = {m: None for m in seeds}
    return (
        seeds, seed_rank, losses, collisions, bye_counts, last_opponent,
        [], set(), 0, {}, fingerprint,
    )


def _persist_bracket(
    out_dir: Path,
    *,
    fingerprint: dict,
    seeds: list[str],
    seed_rank: dict[str, int],
    losses: dict[str, int],
    collisions: dict[str, int],
    bye_counts: dict[str, int],
    last_opponent: dict[str, str | None],
    match_log: list[dict],
    rnd: int,
    champion: str | None,
) -> None:
    payload = {
        "config": fingerprint,
        "champion": champion,
        "round": rnd,
        "seeds": seeds,
        "seed_rank": {m: seed_rank[m] + 1 for m in seeds},
        "losses": losses,
        "collisions": collisions,
        "bye_counts": bye_counts,
        "last_opponent": last_opponent,
        "matches": match_log,
        "standings": [
            {"checkpoint": m, "seed": seed_rank[m] + 1,
             "losses": losses[m], "collisions": collisions[m],
             "byes": bye_counts.get(m, 0)}
            for m in seeds
        ],
    }
    atomic_write_json(_bracket_state_path(out_dir), payload)
    standings = sorted(seeds, key=lambda m: (losses[m], collisions[m], seed_rank[m]))
    with (out_dir / "standings.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "checkpoint", "seed", "losses", "collisions", "byes"])
        for i, m in enumerate(standings, 1):
            w.writerow([
                i, m, seed_rank[m] + 1, losses[m], collisions[m],
                bye_counts.get(m, 0),
            ])


def _dispatch_match(
    args,
    ckpt_dir: Path,
    a: str,
    b: str,
    *,
    match_id_str: str,
    round_no: int,
    ordinal: int,
    rematch: int,
    seed_rank_a: int,
    seed_rank_b: int,
    video: Path | None,
):
    out = Path(args.out_dir) / "matches" / f"{match_id_str}.json"
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--model-a", str(ckpt_dir / a),
        "--model-b", str(ckpt_dir / b),
        "--out", str(out),
        "--track", args.track,
        "--laps", str(args.laps),
        "--freeze-s", str(args.freeze_s),
        "--seed", str(args.seed),
        "--device", args.device,
        "--precision", args.precision,
        "--seed-rank-a", str(seed_rank_a),
        "--seed-rank-b", str(seed_rank_b),
        "--match-id", match_id_str,
        "--round", str(round_no),
        "--match-ordinal", str(ordinal),
        "--rematch-index", str(rematch),
    ]
    if args.max_race_duration_s is not None:
        cmd += ["--max-race-duration-s", str(args.max_race_duration_s)]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if video is not None:
        cmd += ["--video", str(video)]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    ), out, (a, b, match_id_str)


def _wait_matches(running, timeout_s: float | None):
    deadline = None if timeout_s is None else time.time() + timeout_s
    results = []
    while running:
        still = []
        for p, out, meta in running:
            if p.poll() is None:
                if deadline is not None and time.time() > deadline:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    err = p.stderr.read().decode() if p.stderr else ""
                    a, b, mid = meta
                    sys.exit(
                        f"MATCH TIMEOUT {mid} ({a} vs {b})\n{err[-2000:]}"
                    )
                still.append((p, out, meta))
                continue
            a, b, mid = meta
            if p.returncode == 0 and out.exists():
                results.append(json.loads(out.read_text()))
            else:
                err = p.stderr.read().decode() if p.stderr else ""
                sys.exit(
                    f"MATCH FAILED {mid} ({a} vs {b}) rc={p.returncode}\n{err}"
                )
        running = still
        if running:
            time.sleep(0.3)
    return results


def _print_dry_run(
    seeds: list[str],
    seed_rank: dict[str, int],
    elim_losses: int,
    max_rounds: int = 3,
) -> None:
    losses = {m: 0 for m in seeds}
    bye_counts = {m: 0 for m in seeds}
    last_opponent = {m: None for m in seeds}
    print(f"\n=== DRY RUN ({len(seeds)} seeds, elim at {elim_losses} losses) ===")
    for rnd in range(1, max_rounds + 1):
        active = [m for m in seeds if losses[m] < elim_losses]
        if len(active) <= 1:
            champ = active[0] if active else "none"
            print(f"\nChampion: {champ}")
            return
        players = sorted(active, key=lambda m: (losses[m], seed_rank[m]))
        bye = None
        if len(players) % 2 == 1:
            bye = select_bye(players, losses, seed_rank, bye_counts)
            players = [p for p in players if p != bye]
            bye_counts[bye] = bye_counts.get(bye, 0) + 1
        pairs = pair_avoid_rematch(players, losses, seed_rank, last_opponent)
        print(f"\nRound {rnd}: {len(pairs)} matches"
              f"{f', bye={bye}' if bye else ''}")
        for i, (a, b) in enumerate(pairs, 1):
            mid = match_id(rnd, i, a, b)
            print(
                f"  {mid}: {a} (#{seed_rank[a]+1}) vs {b} (#{seed_rank[b]+1})"
            )
    print("\n(Dry run shows initial pairing schedule only; no races simulated.)")


def run_bracket(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = _resolve_ckpt_dir(args, run_dir)
    if not ckpt_dir.is_dir():
        sys.exit(f"Checkpoint directory not found: {ckpt_dir}")
    if args.out_dir is None:
        args.out_dir = str(run_dir / "tournament")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    if args.dry_run:
        seeds = _seed_candidates(args, run_dir, ckpt_dir)
        seed_rank = {m: i for i, m in enumerate(seeds)}
        print(f"Seeded {len(seeds)} candidates:")
        for i, m in enumerate(seeds, 1):
            print(f"  #{i:>2} {m}")
        _print_dry_run(seeds, seed_rank, args.elim_losses)
        return

    (
        seeds, seed_rank, losses, collisions, bye_counts, last_opponent,
        match_log, completed_ids, rnd, rematch_counts, fingerprint,
    ) = _load_or_init_bracket(args, run_dir, ckpt_dir, out_dir)

    if not match_log:
        print(f"Seeded {len(seeds)} candidates (fastest solo lap first):")
        for i, m in enumerate(seeds, 1):
            print(f"  #{i:>2} {m}")

    rematch_counts = rematch_counts or {}
    champion = None
    while len([m for m in seeds if losses[m] < args.elim_losses]) > 1:
        rnd += 1
        active = sorted(
            (m for m in seeds if losses[m] < args.elim_losses),
            key=lambda m: (losses[m], seed_rank[m]),
        )
        bye = None
        players = list(active)
        if len(players) % 2 == 1:
            bye = select_bye(players, losses, seed_rank, bye_counts)
            players = [p for p in players if p != bye]
            bye_counts[bye] = bye_counts.get(bye, 0) + 1
        pairs = pair_avoid_rematch(players, losses, seed_rank, last_opponent)
        if not pairs and len(active) > 1:
            sys.exit(
                f"Bracket stuck in round {rnd}: no pairings for {active} "
                f"(bye={bye})"
            )
        final_round = len(pairs) == 1 and bye is None
        print(f"\n--- Round {rnd}: {len(pairs)} matches"
              f"{f', bye={bye}' if bye else ''} ---")

        queue = []
        for ordinal, (a, b) in enumerate(pairs, 1):
            key = tuple(sorted((a, b)))
            rematch = sum(
                1 for m in match_log
                if tuple(sorted((m["model_a"], m["model_b"]))) == key
            )
            mid = match_id(rnd, ordinal, a, b, rematch)
            if mid in completed_ids:
                continue
            queue.append((a, b, ordinal, rematch, mid))

        running = []
        while queue or running:
            while queue and len(running) < args.workers:
                a, b, ordinal, rematch, mid = queue.pop(0)
                vid = None
                if args.videos or final_round:
                    vid = video_dir / f"{mid}.mp4"
                running.append(_dispatch_match(
                    args, ckpt_dir, a, b,
                    match_id_str=mid,
                    round_no=rnd,
                    ordinal=ordinal,
                    rematch=rematch,
                    seed_rank_a=seed_rank[a],
                    seed_rank_b=seed_rank[b],
                    video=vid,
                ))
            batch = _wait_matches(running, args.match_timeout_s)
            running = []
            for r in batch:
                a, b = r["model_a"], r["model_b"]
                win = a if r["winner"] == "a" else b
                lose = b if win == a else a
                losses[lose] += 1
                collisions[a] += r["collisions_a"]
                collisions[b] += r["collisions_b"]
                last_opponent[a] = b
                last_opponent[b] = a
                key = tuple(sorted((a, b)))
                if r.get("rematch_index", 0) > 0:
                    rematch_counts[key] = max(
                        rematch_counts.get(key, 0), int(r["rematch_index"]),
                    )
                match_log.append(r)
                completed_ids.add(r["match_id"])
                print(
                    f"  {r['match_id']}: {a} legs {r['leg_wins_a']}-{r['leg_wins_b']} "
                    f"{b} agg A{r['laps_a']}-B{r['laps_b']} "
                    f"-> WIN {win} ({r['finish_reason']})"
                    f"  (loser {lose} now {losses[lose]}L"
                    f"{', capped' if r['capped'] else ''})"
                )

        _persist_bracket(
            out_dir,
            fingerprint=fingerprint,
            seeds=seeds,
            seed_rank=seed_rank,
            losses=losses,
            collisions=collisions,
            bye_counts=bye_counts,
            last_opponent=last_opponent,
            match_log=match_log,
            rnd=rnd,
            champion=None,
        )

    champion = next(m for m in seeds if losses[m] < args.elim_losses)
    _persist_bracket(
        out_dir,
        fingerprint=fingerprint,
        seeds=seeds,
        seed_rank=seed_rank,
        losses=losses,
        collisions=collisions,
        bye_counts=bye_counts,
        last_opponent=last_opponent,
        match_log=match_log,
        rnd=rnd,
        champion=champion,
    )
    standings = sorted(seeds, key=lambda m: (losses[m], collisions[m], seed_rank[m]))
    print("\n=========================================")
    print(f"  CHAMPION: {champion}")
    print("=========================================")
    print(f"{'rank':>4} {'checkpoint':>26} {'seed':>5} {'losses':>7} {'coll':>6}")
    for i, m in enumerate(standings[:10], 1):
        print(
            f"{i:>4} {m:>26} {seed_rank[m] + 1:>5} {losses[m]:>7} "
            f"{collisions[m]:>6}"
        )
    print(f"\nBracket log: {_bracket_state_path(out_dir)}")
    print(f"Standings:   {out_dir / 'standings.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1v1 triple-elimination tournament")
    p.add_argument("--model-a", type=str, default=None)
    p.add_argument("--model-b", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed-csv", type=str, default=None,
                   help="Lap-timing CSV for seeding (new or lap_bench schema).")
    p.add_argument("--checkpoints-dir", type=str, default=None,
                   help="Directory containing seeded policy_*.pt files.")
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--elim-losses", type=int, default=DEFAULT_ELIM_LOSSES)
    p.add_argument("--track", type=str, default="Austin")
    p.add_argument("--laps", type=int, default=10)
    p.add_argument("--freeze-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--match-timeout-s", type=float, default=None,
                   help="Per-match subprocess timeout (default: laps*180s).")
    p.add_argument("--max-race-duration-s", type=float, default=None,
                   help="Cap wall-clock race duration inside the worker.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Cap simulation steps inside the worker.")
    p.add_argument("--videos", action="store_true",
                   help="Render an MP4 for every match (default: final only).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print seeding and pairing only; do not load models.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--precision", type=str, default="32", choices=["32", "64"])
    p.add_argument("--seed-rank-a", type=int, default=0)
    p.add_argument("--seed-rank-b", type=int, default=1)
    p.add_argument("--match-id", type=str, default=None)
    p.add_argument("--round", type=int, default=0, dest="round_no")
    p.add_argument("--match-ordinal", type=int, default=0)
    p.add_argument("--rematch-index", type=int, default=0)
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.candidates < 2:
        sys.exit("--candidates must be >= 2")
    if args.laps < 1:
        sys.exit("--laps must be >= 1")
    if args.freeze_s < 0:
        sys.exit("--freeze-s must be >= 0")
    if args.workers < 1:
        sys.exit("--workers must be >= 1")
    if args.elim_losses < 1:
        sys.exit("--elim-losses must be >= 1")
    if args.match_timeout_s is None and args.run_dir:
        args.match_timeout_s = max(300.0, args.laps * 180.0)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.run_dir:
        run_bracket(args)
        return
    if not args.model_a or not args.model_b or not args.out:
        sys.exit("Worker needs --model-a, --model-b, --out (or use --run-dir).")
    torch.set_num_threads(1)
    match_meta = {}
    if args.match_id:
        match_meta = {
            "match_id": args.match_id,
            "round": args.round_no,
            "match_ordinal": args.match_ordinal,
            "rematch_index": args.rematch_index,
        }
    result = race_match(
        args.model_a, args.model_b, args.track, args.laps,
        args.freeze_s, args.seed, args.device, args.precision,
        max_race_duration_s=args.max_race_duration_s,
        max_steps=args.max_steps,
        seed_rank_a=args.seed_rank_a,
        seed_rank_b=args.seed_rank_b,
        match_meta=match_meta,
        video_path=args.video,
    )
    atomic_write_json(Path(args.out), result)
    print(
        f"{result['model_a']} legs {result['leg_wins_a']}-{result['leg_wins_b']} "
        f"{result['model_b']} agg A{result['laps_a']}-B{result['laps_b']} "
        f"-> {result['winner']} ({result['finish_reason']})"
    )


if __name__ == "__main__":
    main()

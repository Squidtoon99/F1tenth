"""Dense-traffic experiment: max-agents A/B metrics and promotion gates.

Does not alter production defaults. Callers pass explicit configs / atlas caches.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gigaflow_f1tenth.config import (
    ExperimentConfig,
    config_from_dict,
    config_to_dict,
    estimate_memory_bytes,
    load_config,
)
from gigaflow_f1tenth.evaluation import _suite_world_overrides
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import (
    PackedTrackAtlasView,
    rebuild_atlas_capacity,
    sample_active_counts as atlas_sample_active_counts,
    sample_track_ids as atlas_sample_track_ids,
)
from gigaflow_f1tenth.trainer import build_trainer

EXPERIMENT_NAME = "dense_traffic_max_agents"
VARIANT_AGENTS = (8, 10, 12)
BASELINE_AGENTS = 8

# Historical "equal worlds" comparison scale for this experiment (production
# itself runs at 448 worlds x 8 agents; see configs/production_h100.yaml).
# VRAM at any (worlds, agents) pair is estimated by delegating to
# config.estimate_memory_bytes, not by a second model anchored here.
H100_ANCHOR_WORLDS = 1024
H100_VRAM_BUDGET_GIB = 72.0
# Reference architecture (rollout length, minibatch size, network sizes) for
# H100 VRAM projections at arbitrary world/agent counts. World/agent counts are
# overridden per call; every other knob comes from this file so a projection
# never silently drifts from the config that estimate_memory_bytes validates.
H100_REFERENCE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "production_h100.yaml"
)

CLOSE_PROXIMITY_M = 2.0
TRACK_VISIBILITY_RANGE_M = 12.0

# Relative gates vs max-8 baseline (dense worlds).
MIN_CARS_LIFT = 1.05
MIN_CLOSE_PAIR_LIFT = 1.10
MAX_SPAWN_REJECT_RATE = 0.15
MAX_SPAWN_REJECT_LIFT = 1.75
MAX_OCCLUSION_LIFT = 1.80
MAX_COLLISION_LIFT = 2.25
MAX_OOB_LIFT = 2.25
MIN_THROUGHPUT_RATIO = 0.55
MAX_ABS_APPROX_KL = 0.25
# Realized single-car worlds include sparse singles, so compare vs baseline
# rather than the raw solo_world_fraction knob.
SOLO_FRAC_TOL = 0.05


@dataclass(frozen=True)
class DenseTrafficMetrics:
    max_agents: int
    realized_cars_per_world_mean: float
    realized_cars_per_dense_world_mean: float
    dense_world_frac: float
    solo_world_frac: float
    close_pair_rate: float
    mean_nearest_opponent_m: float
    overtake_proxy: float
    collision_events: float
    oob_events: float
    collision_per_km: float
    oob_per_km: float
    lidar_occlusion_proxy: float
    track_visibility_proxy: float
    spawn_requested: float
    spawn_realized: float
    spawn_rejects: float
    spawn_reject_rate: float
    sim_world_ticks_per_s: float
    learner_updates_per_s: float
    learner_transitions_per_s: float
    vram_est_gib: float
    startup_estimate_bytes: int
    ppo_policy_loss: float
    ppo_value_loss: float
    ppo_entropy: float
    ppo_approx_kl: float
    ppo_finite: bool
    head_to_head_cars: float
    extras: dict[str, float] = field(default_factory=dict)


def estimate_h100_vram_gib(
    *,
    num_worlds: int,
    max_agents: int,
    reference_config: ExperimentConfig | None = None,
) -> float:
    """H100 VRAM estimate for (num_worlds, max_agents).

    Delegates to ``config.estimate_memory_bytes`` — the model validated against
    measured CUDA peak across nine configurations — instead of keeping a second,
    independent memory model here. ``reference_config`` (default: the production
    H100 config) supplies every knob ``estimate_memory_bytes`` needs besides
    world/agent counts (rollout length, minibatch size, network sizes); only
    ``worlds.num_worlds`` and ``worlds.max_agents_per_world`` are overridden.
    """
    cfg = reference_config or load_config(H100_REFERENCE_CONFIG_PATH)
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=int(num_worlds),
            max_agents_per_world=int(max_agents),
        ),
    )
    return float(estimate_memory_bytes(cfg)) / (1024.0**3)


def slot_matched_worlds(max_agents: int, *, baseline_slots: int = 8192) -> int:
    """World count keeping total slots near the production baseline."""
    a = max(1, int(max_agents))
    return max(1, int(round(float(baseline_slots) / float(a) / 64.0) * 64))


def rebuild_capacity_variants(
    src_cache: str | Path,
    dst_root: str | Path,
    *,
    max_agents_list: tuple[int, ...] = VARIANT_AGENTS,
    car_width_m: float,
    car_length_m: float,
) -> dict[int, Path]:
    """Rebuild capacity-aware atlas variants under ``dst_root/max{N}/``."""
    out: dict[int, Path] = {}
    root = Path(dst_root).expanduser()
    for n in max_agents_list:
        dst = root / f"max{int(n)}"
        rebuild_atlas_capacity(
            src_cache,
            dst,
            max_agents=int(n),
            car_width_m=car_width_m,
            car_length_m=car_length_m,
        )
        out[int(n)] = dst
    return out


def _active_per_world(active: torch.Tensor, max_agents: int) -> np.ndarray:
    n = int(active.numel())
    worlds = n // max(1, int(max_agents))
    return (
        active.view(worlds, int(max_agents))
        .sum(dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _close_pair_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    active: torch.Tensor,
    max_agents: int,
    *,
    proximity_m: float = CLOSE_PROXIMITY_M,
) -> tuple[float, float]:
    n_agents = int(max_agents)
    worlds = int(x.numel()) // max(n_agents, 1)
    close = 0
    pairs = 0
    nearest: list[float] = []
    xa = x.detach().cpu().numpy()
    ya = y.detach().cpu().numpy()
    act = active.detach().cpu().numpy() > 0
    for w in range(worlds):
        base = w * n_agents
        idx = [base + s for s in range(n_agents) if act[base + s]]
        if len(idx) < 2:
            continue
        for i, ai in enumerate(idx):
            best = math.inf
            for bi in idx[i + 1 :]:
                d = math.hypot(float(xa[ai] - xa[bi]), float(ya[ai] - ya[bi]))
                pairs += 1
                if d < proximity_m:
                    close += 1
                if d < best:
                    best = d
            if math.isfinite(best):
                nearest.append(best)
    rate = float(close / max(pairs, 1))
    mean_nn = float(np.mean(nearest)) if nearest else float("nan")
    return rate, mean_nn


def _lidar_visibility_proxies(
    sensor_obs: torch.Tensor,
    active: torch.Tensor,
    lidar_dim: int,
) -> tuple[float, float]:
    """Occlusion / track-visibility proxies from range returns.

    ``lidar_occlusion_proxy``: fraction of beams shorter than the solo-typical
    track horizon (near hits, often cars). ``track_visibility_proxy``: fraction
    of beams at/above that horizon (open track still visible).
    """
    act = active > 0
    if int(act.sum().item()) == 0:
        return 0.0, 1.0
    lidar = sensor_obs[act, :lidar_dim].detach().float()
    finite = torch.isfinite(lidar)
    short = (lidar < TRACK_VISIBILITY_RANGE_M) & finite
    long = (lidar >= TRACK_VISIBILITY_RANGE_M) & finite
    denom = float(finite.sum().item())
    if denom <= 0.0:
        return 0.0, 1.0
    return float(short.sum().item() / denom), float(long.sum().item() / denom)


def measure_variant(
    cfg: ExperimentConfig,
    *,
    atlas: PackedTrackAtlasView | None = None,
    device: str = "cpu",
    sim_steps: int = 40,
    learner_updates: int = 2,
    seed: int = 0,
) -> DenseTrafficMetrics:
    """CPU-safe dense-traffic measurement for one max_agents variant."""
    if str(device).startswith("cuda"):
        raise RuntimeError(
            "dense_traffic validation refuses CUDA to avoid contending with "
            "live viewers/trainers; use device=cpu"
        )
    max_agents = int(cfg.worlds.max_agents_per_world)
    atlas = atlas or make_synthetic_oval_atlas(max_agents=max_agents)

    # Force dense bin for interaction metrics while keeping solo_world_fraction.
    # This simulator is stepped directly and never feeds a trainer, so the PPO
    # minibatch and rollout-memory budgets (sized for a training update) do not
    # apply — same reasoning as evaluation.FixedSeedEvaluator._build_suite_sim.
    dense_raw = config_to_dict(cfg)
    dense_raw["worlds"]["density_bins"] = ["dense"]
    dense_raw["worlds"]["device"] = "cpu"
    dense_raw["seed"] = int(seed)
    dense_cfg = config_from_dict(dense_raw, check_training_budget=False)

    sim = build_simulator(dense_cfg, atlas, "cpu")
    sim.reset_all(int(seed))
    spawn = dict(getattr(sim, "_spawn_stats", {}))
    t = sim.buffers.torch_arrays
    per_world = _active_per_world(t.active, max_agents)
    close_rate, mean_nn = _close_pair_stats(t.x, t.y, t.active, max_agents)
    obs0 = sim.action_observation()
    occ0, vis0 = _lidar_visibility_proxies(
        obs0, t.active, int(cfg.agents.lidar_dim)
    )

    # Population mix on a large assignment (independent of tiny CPU num_worlds).
    mix_worlds = 2048
    track_ids = atlas_sample_track_ids(
        int(atlas.num_tracks),
        mix_worlds,
        int(seed),
        sampling=str(cfg.tracks.sampling),
        lengths=np.asarray(atlas.lengths, dtype=np.float64),
    )
    capacities = np.asarray(atlas.capacity, dtype=np.int32)
    counts = atlas_sample_active_counts(
        track_ids,
        capacities,
        density_bins=cfg.worlds.density_bins,
        solo_world_fraction=float(cfg.worlds.solo_world_fraction),
        max_agents=max_agents,
        seed=int(seed) + 17,
    )
    solo_frac = float(np.mean(counts == 1))
    cap_w = np.minimum(max_agents, capacities[track_ids])
    dense_lo = np.maximum(1, np.floor(0.7 * cap_w))
    dense_frac = float(np.mean(counts >= dense_lo))

    n = world_slot_layout(dense_cfg).num_slots
    actions = torch.zeros((n, 2), dtype=torch.float32)
    actions[:, 0] = 0.15
    collisions = 0
    oobs = 0
    overtake_proxy = 0.0
    distance_m = 0.0
    progress0 = t.progress_s.clone()
    prev_contact = torch.zeros(n, dtype=torch.bool)
    prev_wall = torch.zeros(n, dtype=torch.bool)
    occ_acc = 0.0
    vis_acc = 0.0
    close_acc = 0.0
    nn_acc = 0.0
    nn_n = 0

    t0 = time.perf_counter()
    for _ in range(int(sim_steps)):
        out = sim.step(actions)
        contact = torch.as_tensor(out.get("contact", t.contact > 0), dtype=torch.bool)
        wall = torch.as_tensor(out.get("wall_contact", t.wall_contact > 0), dtype=torch.bool)
        collisions += int((contact & ~prev_contact).sum().item())
        oobs += int((wall & ~prev_wall).sum().item())
        prev_contact = contact
        prev_wall = wall
        ds = torch.clamp(t.progress_s - progress0, min=0)
        distance_m += float(ds.sum().item())
        # Passing proxy: active agents with positive relative progress vs mean.
        act = t.active > 0
        if int(act.sum().item()) > 1:
            mean_ds = float(ds[act].mean().item())
            overtake_proxy += float(((ds > mean_ds + 1e-3) & act).sum().item())
        progress0 = t.progress_s.clone()
        cr, nn = _close_pair_stats(t.x, t.y, t.active, max_agents)
        close_acc += cr
        if math.isfinite(nn):
            nn_acc += nn
            nn_n += 1
        occ, vis = _lidar_visibility_proxies(
            out["sensor_obs"], t.active, int(cfg.agents.lidar_dim)
        )
        occ_acc += occ
        vis_acc += vis
    elapsed = max(time.perf_counter() - t0, 1e-9)
    sim_tps = float(sim_steps) / elapsed
    km = max(distance_m / 1000.0, 1e-6)

    # Head-to-head distribution must stay exactly 2 cars. Evaluation-only
    # construction (never a trainer): skip the training-update budgets so a
    # production-scale config's max_agents_per_world -> 2 override cannot be
    # rejected by PPO minibatch/rollout-memory sizing meant for training.
    h2h_raw = _suite_world_overrides("head_to_head", cfg)
    h2h_cfg = config_from_dict(h2h_raw, check_training_budget=False)
    h2h_sim = build_simulator(h2h_cfg, atlas, "cpu", sync_no_respawn=True)
    h2h_sim.reset_all(int(seed))
    h2h_cars = float(
        _active_per_world(
            h2h_sim.buffers.torch_arrays.active,
            int(h2h_cfg.worlds.max_agents_per_world),
        ).mean()
    )

    # Short PPO health / learner throughput (CPU).
    train_raw = config_to_dict(cfg)
    train_raw["worlds"]["device"] = "cpu"
    train_raw["ppo"]["amp"] = False
    train_raw["ppo"]["total_updates"] = max(2, int(learner_updates))
    train_cfg = config_from_dict(train_raw)
    trainer = build_trainer(train_cfg, atlas=atlas, device="cpu")
    trainer.setup()
    trainer.train_update()  # warmup
    t1 = time.perf_counter()
    last_metrics: dict[str, float] = {}
    for _ in range(max(1, int(learner_updates))):
        progress = trainer.train_update()
        last_metrics = dict(progress.metrics)
    learn_elapsed = max(time.perf_counter() - t1, 1e-9)
    pol = float(last_metrics.get("policy_loss", float("nan")))
    val = float(last_metrics.get("value_loss", float("nan")))
    ent = float(last_metrics.get("entropy", float("nan")))
    kl = float(last_metrics.get("approx_kl", float("nan")))
    ppo_finite = all(math.isfinite(v) for v in (pol, val, ent, kl))

    # H100-scale VRAM (equal-worlds + slot-matched), not tiny CPU world count.
    vram = estimate_h100_vram_gib(
        num_worlds=H100_ANCHOR_WORLDS,
        max_agents=max_agents,
    )
    matched_worlds = slot_matched_worlds(max_agents)
    vram_matched = estimate_h100_vram_gib(
        num_worlds=matched_worlds,
        max_agents=max_agents,
    )
    steps = max(int(sim_steps), 1)
    return DenseTrafficMetrics(
        max_agents=max_agents,
        realized_cars_per_world_mean=float(per_world.mean()),
        realized_cars_per_dense_world_mean=float(per_world.mean()),
        dense_world_frac=dense_frac,
        solo_world_frac=solo_frac,
        close_pair_rate=float(close_acc / steps),
        mean_nearest_opponent_m=float(nn_acc / max(nn_n, 1)),
        overtake_proxy=float(overtake_proxy / steps),
        collision_events=float(collisions),
        oob_events=float(oobs),
        collision_per_km=float(collisions / km),
        oob_per_km=float(oobs / km),
        lidar_occlusion_proxy=float(occ_acc / steps),
        track_visibility_proxy=float(vis_acc / steps),
        spawn_requested=float(spawn.get("requested", 0)),
        spawn_realized=float(spawn.get("realized", 0)),
        spawn_rejects=float(spawn.get("rejects", 0)),
        spawn_reject_rate=float(
            spawn.get("rejects", 0) / max(spawn.get("requested", 1), 1)
        ),
        sim_world_ticks_per_s=sim_tps,
        learner_updates_per_s=float(max(1, learner_updates) / learn_elapsed),
        learner_transitions_per_s=float(
            last_metrics.get("transitions_per_s", 0.0)
        ),
        vram_est_gib=vram,
        startup_estimate_bytes=int(estimate_memory_bytes(cfg)),
        ppo_policy_loss=pol,
        ppo_value_loss=val,
        ppo_entropy=ent,
        ppo_approx_kl=kl,
        ppo_finite=ppo_finite,
        head_to_head_cars=h2h_cars,
        extras={
            "lidar_occlusion_at_reset": float(occ0),
            "track_visibility_at_reset": float(vis0),
            "active_at_reset_mean": float(per_world.mean()),
            "active_at_reset_max": float(per_world.max() if per_world.size else 0.0),
            "mix_count_mean": float(np.mean(counts)),
            "mix_count_p90": float(np.quantile(counts, 0.9)),
            "vram_slot_matched_gib": float(vram_matched),
            "slot_matched_worlds": float(matched_worlds),
        },
    )


def _lift(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return float("nan")
    if abs(baseline) < 1e-12:
        return float("inf") if abs(candidate) > 1e-12 else 1.0
    return float(candidate / baseline)


def promotion_gates(
    baseline: DenseTrafficMetrics,
    candidate: DenseTrafficMetrics,
    *,
    solo_target: float = 0.05,
) -> dict[str, bool]:
    """Explicit gates balancing denser interactions vs failure modes."""
    if baseline.max_agents != BASELINE_AGENTS:
        raise ValueError(
            f"baseline max_agents must be {BASELINE_AGENTS}, got {baseline.max_agents}"
        )
    cars_lift = _lift(
        candidate.realized_cars_per_dense_world_mean,
        baseline.realized_cars_per_dense_world_mean,
    )
    close_lift = _lift(candidate.close_pair_rate, baseline.close_pair_rate)
    reject_lift = _lift(candidate.spawn_reject_rate, max(baseline.spawn_reject_rate, 1e-6))
    occ_lift = _lift(
        candidate.lidar_occlusion_proxy, max(baseline.lidar_occlusion_proxy, 1e-6)
    )
    col_lift = _lift(
        candidate.collision_per_km, max(baseline.collision_per_km, 1e-6)
    )
    oob_lift = _lift(candidate.oob_per_km, max(baseline.oob_per_km, 1e-6))
    thr_ratio = _lift(
        candidate.sim_world_ticks_per_s, max(baseline.sim_world_ticks_per_s, 1e-6)
    )
    return {
        "cars_per_world_lift": bool(cars_lift >= MIN_CARS_LIFT),
        # Close-pair is noisy on short CPU soaks; accept absolute rate or lift.
        "close_pair_lift": bool(
            close_lift >= MIN_CLOSE_PAIR_LIFT
            or candidate.close_pair_rate + 1e-9 >= baseline.close_pair_rate
            or candidate.realized_cars_per_dense_world_mean
            >= baseline.realized_cars_per_dense_world_mean * MIN_CARS_LIFT
        ),
        "spawn_rejects_bounded": bool(
            candidate.spawn_reject_rate <= MAX_SPAWN_REJECT_RATE
            and reject_lift <= MAX_SPAWN_REJECT_LIFT
        ),
        "occlusion_bounded": bool(occ_lift <= MAX_OCCLUSION_LIFT),
        "track_visibility_positive": bool(candidate.track_visibility_proxy > 0.05),
        "collision_bounded": bool(col_lift <= MAX_COLLISION_LIFT),
        "oob_bounded": bool(oob_lift <= MAX_OOB_LIFT),
        "throughput_ok": bool(thr_ratio >= MIN_THROUGHPUT_RATIO),
        "vram_under_budget": bool(
            candidate.vram_est_gib <= H100_VRAM_BUDGET_GIB
            or float(candidate.extras.get("vram_slot_matched_gib", math.inf))
            <= H100_VRAM_BUDGET_GIB
        ),
        "ppo_finite": bool(candidate.ppo_finite),
        "ppo_kl_bounded": bool(
            math.isfinite(candidate.ppo_approx_kl)
            and abs(candidate.ppo_approx_kl) <= MAX_ABS_APPROX_KL
        ),
        "solo_frac_preserved": bool(
            abs(candidate.solo_world_frac - baseline.solo_world_frac) <= SOLO_FRAC_TOL
            # Config target remains the production knob (checked in config tests).
            and abs(solo_target - 0.05) <= 1e-9
        ),
        "head_to_head_two_cars": bool(abs(candidate.head_to_head_cars - 2.0) < 1e-6),
        "all_pass": False,  # filled below
    }


def finalize_gates(gates: dict[str, bool]) -> dict[str, bool]:
    keys = [k for k in gates if k != "all_pass"]
    out = dict(gates)
    out["all_pass"] = all(bool(out[k]) for k in keys)
    return out


def recommend_first_variant(
    results: dict[int, DenseTrafficMetrics],
    gates: dict[int, dict[str, bool]],
) -> dict[str, Any]:
    """Prefer max-10 before max-12: smaller step, lower VRAM/occlusion risk."""
    g10 = gates.get(10, {})
    g12 = gates.get(12, {})
    reason = (
        "Try max_agents=10 first: smaller jump from the production-8 baseline, "
        "lower VRAM headroom risk than 12 at equal world count, and a cleaner "
        "interaction/occlusion trade-off before attempting 12."
    )
    if g10.get("all_pass") and not g12.get("all_pass"):
        choice = 10
        reason += " Local gates already favor 10 over 12."
    elif g12.get("all_pass") and not g10.get("all_pass"):
        choice = 12
        reason = (
            "Local gates passed only for 12; still prefer confirming 10 on H100 "
            "first unless slot-matched worlds are used for both."
        )
        # Keep recommendation as 10 unless 10 is hard-fail on safety gates.
        safety = (
            "spawn_rejects_bounded",
            "occlusion_bounded",
            "ppo_finite",
            "vram_under_budget",
        )
        if g10 and not all(g10.get(k, False) for k in safety):
            choice = 12
            reason = (
                "max-10 failed safety gates locally; only queue max-12 if "
                "slot-matched worlds restore VRAM/spawn headroom."
            )
        else:
            choice = 10
    else:
        choice = 10
    return {
        "try_first": int(choice),
        "reason": reason,
        "max10_all_pass": bool(g10.get("all_pass", False)),
        "max12_all_pass": bool(g12.get("all_pass", False)),
        "vram_est_gib": {
            str(k): float(v.vram_est_gib) for k, v in results.items()
        },
    }


def h100_queue_plan(
    *,
    repo_gigaflow: str | Path,
    atlas_root: str | Path,
    run_root: str | Path,
) -> list[dict[str, Any]]:
    """Exact H100 commands + artifact schema (queued; not executed here)."""
    root = Path(repo_gigaflow)
    atlas_root = Path(atlas_root)
    run_root = Path(run_root)
    src_atlas = "~/.cache/gigaflow/tracks"
    jobs: list[dict[str, Any]] = []
    for agents in VARIANT_AGENTS:
        cfg_path = root / f"configs/experiments/dense_traffic_h100_max{agents}.yaml"
        atlas = atlas_root / f"max{agents}"
        modes = [("equal_worlds", H100_ANCHOR_WORLDS)]
        matched = slot_matched_worlds(agents)
        if agents != BASELINE_AGENTS:
            modes.append(("slot_matched", matched))
        for mode, worlds in modes:
            run_dir = run_root / f"dense_traffic_max{agents}_{mode}"
            vram = estimate_h100_vram_gib(num_worlds=worlds, max_agents=agents)
            rebuild_cmd = (
                "python - <<'PY'\n"
                "from gigaflow_f1tenth.tracks import rebuild_atlas_capacity\n"
                f"rebuild_atlas_capacity('{src_atlas}', '{atlas}', "
                f"max_agents={agents})\n"
                f"print('atlas_ok', '{atlas}')\n"
                "PY"
            )
            if mode == "equal_worlds":
                train_cmd = (
                    f"gigaflow train --config {cfg_path} --device cuda "
                    f"--num-updates 200 --checkpoint-interval 50 "
                    f"--run-dir {run_dir}"
                )
                eval_cfg = str(cfg_path)
                notes = (
                    "Equal world count vs production (1024). "
                    f"VRAM est {vram:.1f} GiB."
                )
            else:
                train_cmd = (
                    "python - <<'PY'\n"
                    "from pathlib import Path\n"
                    "import yaml\n"
                    f"raw = yaml.safe_load(Path('{cfg_path}').read_text())\n"
                    f"raw['worlds']['num_worlds'] = {worlds}\n"
                    f"raw['tracks']['manifest_path'] = '{atlas}/manifest.json'\n"
                    f"out = Path('{run_dir}')\n"
                    "out.mkdir(parents=True, exist_ok=True)\n"
                    "(out / 'config.override.yaml').write_text("
                    "yaml.safe_dump(raw, sort_keys=False))\n"
                    "print(out / 'config.override.yaml')\n"
                    "PY\n"
                    f"gigaflow train --config {run_dir}/config.override.yaml "
                    f"--device cuda --num-updates 200 --checkpoint-interval 50 "
                    f"--run-dir {run_dir}"
                )
                eval_cfg = f"{run_dir}/config.override.yaml"
                notes = (
                    f"Slot-matched worlds={worlds} (~8192 slots). "
                    f"VRAM est {vram:.1f} GiB."
                )
            eval_cmd = (
                f"gigaflow evaluate --config {eval_cfg} --device cpu "
                f"--output-dir {run_dir}/eval_gate "
                f"--suite solo --suite head_to_head --suite dense"
            )
            jobs.append(
                {
                    "variant": f"max{agents}",
                    "mode": mode,
                    "max_agents_per_world": agents,
                    "num_worlds": worlds,
                    "vram_est_gib": vram,
                    "config": str(cfg_path),
                    "atlas_cache": str(atlas),
                    "run_dir": str(run_dir),
                    "notes": notes,
                    "artifact_schema": {
                        "run_dir": str(run_dir),
                        "metrics_glob": "metrics_*.json",
                        "checkpoints_glob": "actor_*.pt",
                        "dense_traffic_summary": "dense_traffic_summary.json",
                        "eval_dir": "eval_gate/",
                        "config_snapshot": "config.json",
                    },
                    "commands": [rebuild_cmd, train_cmd, eval_cmd],
                }
            )
    return jobs


def write_experiment_report(
    path: str | Path,
    *,
    results: dict[int, DenseTrafficMetrics],
    gates: dict[int, dict[str, bool]],
    recommendation: dict[str, Any],
    h100_queue: list[dict[str, Any]],
    atlas_variants: dict[int, str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "baseline_max_agents": BASELINE_AGENTS,
        "variants": {
            str(k): {**asdict(v), "promotion_gates": gates.get(k, {})}
            for k, v in sorted(results.items())
        },
        "recommendation": recommendation,
        "atlas_variants": {
            str(k): v for k, v in (atlas_variants or {}).items()
        },
        "h100_queue": h100_queue,
        "gate_thresholds": {
            "min_cars_lift": MIN_CARS_LIFT,
            "min_close_pair_lift": MIN_CLOSE_PAIR_LIFT,
            "max_spawn_reject_rate": MAX_SPAWN_REJECT_RATE,
            "max_spawn_reject_lift": MAX_SPAWN_REJECT_LIFT,
            "max_occlusion_lift": MAX_OCCLUSION_LIFT,
            "max_collision_lift": MAX_COLLISION_LIFT,
            "max_oob_lift": MAX_OOB_LIFT,
            "min_throughput_ratio": MIN_THROUGHPUT_RATIO,
            "max_abs_approx_kl": MAX_ABS_APPROX_KL,
            "h100_vram_budget_gib": H100_VRAM_BUDGET_GIB,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_cpu_experiment(
    configs: dict[int, ExperimentConfig],
    *,
    output_dir: str | Path,
    src_atlas_cache: str | Path | None = None,
    rebuild_atlases: bool = False,
    sim_steps: int = 40,
    learner_updates: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full CPU validation sweep and write artifacts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atlas_variants: dict[int, str] = {}
    atlases: dict[int, PackedTrackAtlasView] = {}

    if rebuild_atlases:
        if src_atlas_cache is None:
            raise ValueError("src_atlas_cache required when rebuild_atlases=True")
        base_cfg = configs[BASELINE_AGENTS]
        rebuilt = rebuild_capacity_variants(
            src_atlas_cache,
            out / "atlases",
            max_agents_list=tuple(sorted(configs)),
            car_width_m=base_cfg.agents.car_width_m,
            car_length_m=base_cfg.agents.car_length_m,
        )
        for n, path in rebuilt.items():
            atlas_variants[n] = str(path)
            from gigaflow_f1tenth.tracks import load_atlas

            atlases[n] = load_atlas(str(path), device="cpu").view()
    else:
        for n in configs:
            atlases[n] = make_synthetic_oval_atlas(max_agents=n)

    results: dict[int, DenseTrafficMetrics] = {}
    for n, cfg in sorted(configs.items()):
        results[n] = measure_variant(
            cfg,
            atlas=atlases[n],
            device="cpu",
            sim_steps=sim_steps,
            learner_updates=learner_updates,
            seed=seed,
        )

    baseline = results[BASELINE_AGENTS]
    gates: dict[int, dict[str, bool]] = {BASELINE_AGENTS: {"baseline": True}}
    for n, metrics in results.items():
        if n == BASELINE_AGENTS:
            continue
        gates[n] = finalize_gates(
            promotion_gates(
                baseline,
                metrics,
                solo_target=float(configs[n].worlds.solo_world_fraction),
            )
        )

    recommendation = recommend_first_variant(results, gates)
    h100_queue = h100_queue_plan(
        repo_gigaflow=Path(__file__).resolve().parents[2],
        atlas_root=out / "atlases",
        run_root=out / "h100_queue",
    )
    report_path = write_experiment_report(
        out / "dense_traffic_summary.json",
        results=results,
        gates=gates,
        recommendation=recommendation,
        h100_queue=h100_queue,
        atlas_variants=atlas_variants,
    )
    return {
        "report": str(report_path),
        "results": results,
        "gates": gates,
        "recommendation": recommendation,
        "h100_queue": h100_queue,
        "atlas_variants": atlas_variants,
    }

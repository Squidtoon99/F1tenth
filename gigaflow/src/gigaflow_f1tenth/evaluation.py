"""Evaluation suites, multi-car diagnostics, soak tooling, and promotion gates."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
import torch

from gigaflow_f1tenth.config import (
    ConfigError,
    ExperimentConfig,
    config_from_dict,
    config_to_dict,
)
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.model import build_actor
from gigaflow_f1tenth.rewards import (
    deployment_style,
    styles_to_condition_batch,
)
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import PackedTrackAtlasView, load_atlas

SOLO_COMPLETION_MIN = 0.85
SOLO_PROGRESS_MIN_MPS = 4.5
SOLO_LAP_MAX_S = 75.0
HEAD_TO_HEAD_COMPLETION_MIN = 0.85
DENSE_COMPLETION_MIN = 0.50
DENSE_LAP_MAX_S = 80.0
COLLISION_MAX_PER_KM = 3.0
OOB_MAX_PER_KM = 0.5
CATASTROPHIC_COMPLETION_MAX = 0.25
CATASTROPHIC_PROGRESS_MAX_MPS = 2.0
CATASTROPHIC_OOB_MIN_PER_KM = 3.0
LAP_REGRESSION_MAX = 1.20
SUITE_THROUGHPUT_WEIGHTS = {
    "solo": 0.4,
    "head_to_head": 0.3,
    "dense": 0.3,
}

# Centerline progress, as a fraction of the agent's own track length, that
# counts as a completed lap for both completion_rate and lap_time_s.
LAP_COMPLETION_FRACTION = 0.95


@dataclass(frozen=True)
class EvalMetrics:
    lap_time_s: float | None
    completion_rate: float
    progress_rate_mps: float
    collision_per_km: float
    oob_per_km: float
    clean_overtakes: float
    stall_rate: float
    return_mean: float


@dataclass
class EvalReport:
    suite: str
    seed: int
    metrics: EvalMetrics
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorGateDecision:
    passed: bool
    catastrophic: bool
    stop_requested: bool
    warning_active: bool
    best_safe: bool
    feasible: bool
    score: float
    gates: dict[str, bool]


@dataclass
class BehaviorGateState:
    consecutive_failures: int = 0
    consecutive_passes: int = 0
    warning_active: bool = False
    feasible: bool = False
    best_safe_lap_s: float | None = None
    best_safe_score: float = float("-inf")
    frontier: list[tuple[tuple[float, float, float, float], float, int]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "consecutive_failures": int(self.consecutive_failures),
            "consecutive_passes": int(self.consecutive_passes),
            "warning_active": bool(self.warning_active),
            "feasible": bool(self.feasible),
            "best_safe_lap_s": self.best_safe_lap_s,
            "best_safe_score": float(self.best_safe_score),
            "frontier": [
                {"vector": list(vector), "score": score, "step": step}
                for vector, score, step in self.frontier
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BehaviorGateState":
        state = cls()
        state.consecutive_failures = int(payload.get("consecutive_failures", 0))
        state.consecutive_passes = int(payload.get("consecutive_passes", 0))
        state.warning_active = bool(payload.get("warning_active", False))
        state.feasible = bool(payload.get("feasible", False))
        lap = payload.get("best_safe_lap_s")
        state.best_safe_lap_s = None if lap is None else float(lap)
        state.best_safe_score = float(payload.get("best_safe_score", float("-inf")))
        frontier = []
        for row in payload.get("frontier", ()):
            frontier.append(
                (
                    tuple(float(value) for value in row["vector"]),
                    float(row["score"]),
                    int(row["step"]),
                )
            )
        state.frontier = frontier
        return state

    def observe(
        self,
        reports: list[EvalReport],
        *,
        required: tuple[str, ...],
        step: int,
        feasibility_passes: int = 2,
    ) -> BehaviorGateDecision:
        passes_required = max(1, int(feasibility_passes))
        gates = promotion_gates(
            reports,
            required=required,
            best_safe_lap_s=self.best_safe_lap_s,
        )
        catastrophic = catastrophic_behavior(reports)
        passed = all(gates.values()) and not catastrophic
        best_safe = False
        score = behavioral_score(reports)
        if passed:
            self.consecutive_failures = 0
            self.consecutive_passes += 1
            if self.consecutive_passes >= passes_required:
                self.warning_active = False
            if not self.feasible and self.consecutive_passes >= passes_required:
                self.feasible = True
            vector = behavior_pareto_vector(reports)
            if not any(pareto_dominates(row[0], vector) for row in self.frontier):
                self.frontier = [
                    row
                    for row in self.frontier
                    if not pareto_dominates(vector, row[0])
                ]
                self.frontier.append((vector, score, int(step)))
                if score > self.best_safe_score:
                    self.best_safe_score = score
                    best_safe = True
            aggregates = aggregate_behavior(reports)
            safe_laps = [
                float(row["lap_time_s"])
                for row in aggregates.values()
                if row["lap_time_s"] is not None
            ]
            if safe_laps:
                fastest = min(safe_laps)
                self.best_safe_lap_s = (
                    fastest
                    if self.best_safe_lap_s is None
                    else min(self.best_safe_lap_s, fastest)
                )
        else:
            self.consecutive_failures += 1
            self.consecutive_passes = 0
            self.warning_active = True
        if self.feasible:
            stop_requested = self.consecutive_failures >= passes_required
        else:
            stop_requested = False
        return BehaviorGateDecision(
            passed=passed,
            catastrophic=catastrophic,
            stop_requested=stop_requested,
            warning_active=self.warning_active,
            best_safe=best_safe,
            feasible=self.feasible,
            score=score,
            gates=gates,
        )


def aggregate_behavior(reports: list[EvalReport]) -> dict[str, dict[str, float | None]]:
    aggregates: dict[str, dict[str, float | None]] = {}
    for suite in SUITE_THROUGHPUT_WEIGHTS:
        rows = [report.metrics for report in reports if report.suite == suite]
        if not rows:
            continue
        lap_times = [row.lap_time_s for row in rows if row.lap_time_s is not None]
        aggregates[suite] = {
            "completion_rate": sum(row.completion_rate for row in rows) / len(rows),
            "progress_rate_mps": sum(row.progress_rate_mps for row in rows) / len(rows),
            "collision_per_km": sum(row.collision_per_km for row in rows) / len(rows),
            "oob_per_km": sum(row.oob_per_km for row in rows) / len(rows),
            "lap_time_s": (
                sum(float(value) for value in lap_times) / len(lap_times)
                if lap_times
                else None
            ),
        }
    return aggregates


def behavioral_score(reports: list[EvalReport]) -> float:
    aggregates = aggregate_behavior(reports)
    score = 0.0
    for suite, weight in SUITE_THROUGHPUT_WEIGHTS.items():
        row = aggregates.get(suite)
        if row is None or row["lap_time_s"] is None:
            continue
        lap_time = float(row["lap_time_s"])
        if lap_time > 0.0:
            score += (
                weight
                * 60.0
                * float(row["completion_rate"])
                / lap_time
            )
    return float(score)


def catastrophic_behavior(reports: list[EvalReport]) -> bool:
    return any(
        report.suite == "solo"
        and (
            report.metrics.completion_rate < CATASTROPHIC_COMPLETION_MAX
            or report.metrics.progress_rate_mps < CATASTROPHIC_PROGRESS_MAX_MPS
            or report.metrics.oob_per_km > CATASTROPHIC_OOB_MIN_PER_KM
        )
        for report in reports
    )


def behavior_pareto_vector(reports: list[EvalReport]) -> tuple[float, float, float, float]:
    aggregates = aggregate_behavior(reports)
    completions = [
        float(row["completion_rate"]) for row in aggregates.values()
    ]
    laps = [
        float(row["lap_time_s"])
        for row in aggregates.values()
        if row["lap_time_s"] is not None
    ]
    oobs = [float(row["oob_per_km"]) for row in aggregates.values()]
    collisions = [
        float(row["collision_per_km"]) for row in aggregates.values()
    ]
    return (
        sum(completions) / max(len(completions), 1),
        min(laps) if laps else float("inf"),
        sum(oobs) / max(len(oobs), 1),
        sum(collisions) / max(len(collisions), 1),
    )


def pareto_dominates(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_objectives = (left[0], -left[1], -left[2], -left[3])
    right_objectives = (right[0], -right[1], -right[2], -right[3])
    return all(a >= b for a, b in zip(left_objectives, right_objectives)) and any(
        a > b for a, b in zip(left_objectives, right_objectives)
    )


def promotion_gates(
    reports: list[EvalReport],
    *,
    required: tuple[str, ...],
    best_safe_lap_s: float | None = None,
) -> dict[str, bool]:
    if not reports:
        return {"has_reports": False}
    finite = all(
        math.isfinite(report.metrics.return_mean)
        and math.isfinite(report.metrics.collision_per_km)
        and math.isfinite(report.metrics.oob_per_km)
        and math.isfinite(report.metrics.progress_rate_mps)
        for report in reports
    )
    suites = {report.suite for report in reports}
    solo = [report.metrics for report in reports if report.suite == "solo"]
    head_to_head = [
        report.metrics for report in reports if report.suite == "head_to_head"
    ]
    dense = [report.metrics for report in reports if report.suite == "dense"]
    lap_values = [
        float(report.metrics.lap_time_s)
        for report in reports
        if report.metrics.lap_time_s is not None
    ]
    relative_lap_ok = True
    if best_safe_lap_s is not None and lap_values:
        relative_lap_ok = min(lap_values) <= LAP_REGRESSION_MAX * best_safe_lap_s
    return {
        "has_reports": True,
        "finite_metrics": finite,
        "required_suites_covered": set(required).issubset(suites),
        "solo_completion": "solo" not in required or bool(solo)
        and all(row.completion_rate >= SOLO_COMPLETION_MIN for row in solo),
        "solo_progress": "solo" not in required or bool(solo)
        and all(row.progress_rate_mps >= SOLO_PROGRESS_MIN_MPS for row in solo),
        "solo_lap": "solo" not in required or bool(solo)
        and all(
            row.lap_time_s is not None and row.lap_time_s <= SOLO_LAP_MAX_S
            for row in solo
        ),
        "solo_oob": "solo" not in required or bool(solo)
        and all(row.oob_per_km <= OOB_MAX_PER_KM for row in solo),
        "head_to_head_completion": "head_to_head" not in required or bool(head_to_head)
        and all(
            row.completion_rate >= HEAD_TO_HEAD_COMPLETION_MIN
            for row in head_to_head
        ),
        "head_to_head_oob": "head_to_head" not in required or bool(head_to_head)
        and all(row.oob_per_km <= OOB_MAX_PER_KM for row in head_to_head),
        "dense_completion": "dense" not in required or bool(dense)
        and all(row.completion_rate >= DENSE_COMPLETION_MIN for row in dense),
        "dense_lap": "dense" not in required or bool(dense)
        and all(
            row.lap_time_s is not None and row.lap_time_s <= DENSE_LAP_MAX_S
            for row in dense
        ),
        "dense_collisions": "dense" not in required or bool(dense)
        and all(row.collision_per_km <= COLLISION_MAX_PER_KM for row in dense),
        "dense_oob": "dense" not in required or bool(dense)
        and all(row.oob_per_km <= OOB_MAX_PER_KM for row in dense),
        "relative_lap": relative_lap_ok,
    }


@runtime_checkable
class Evaluator(Protocol):
    def run_suite(self, suite: str, seed: int) -> EvalReport:
        ...

    def promotion_gates(self, reports: list[EvalReport]) -> dict[str, bool]:
        ...


def required_suites(cfg: ExperimentConfig) -> tuple[str, ...]:
    return cfg.evaluation.suite


def resolve_eval_num_worlds(cfg: ExperimentConfig) -> int:
    """World count for evaluation only (may differ from training worlds)."""
    n = cfg.evaluation.num_worlds
    if n is None:
        return int(cfg.worlds.num_worlds)
    return int(n)


def resolve_eval_atlas(cfg: ExperimentConfig) -> PackedTrackAtlasView:
    """Atlas for evaluation: the configured manifest, never a silent stand-in.

    A configured manifest that fails to load is an error — substituting
    synthetic geometry would score the policy on a track it never races.
    """
    if cfg.tracks.manifest_path:
        return load_atlas(str(Path(cfg.tracks.manifest_path).expanduser())).view()
    return make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)


def resolve_eval_device(
    cfg: ExperimentConfig, override: str | None = None
) -> str:
    """Device for evaluation; ``evaluation.device`` overrides ``worlds.device``."""
    if override is not None:
        return str(override)
    if cfg.evaluation.device is not None:
        return str(cfg.evaluation.device)
    return str(cfg.worlds.device)


def ablation_config(
    cfg: ExperimentConfig,
    *,
    actor_variant: str | None = None,
    frame_stack: int | None = None,
    adaptive_filter_enabled: bool | None = None,
    reward_randomization: bool | None = None,
    condition_to_actor: bool | None = None,
    centralized_critic: bool | None = None,
) -> ExperimentConfig:
    """Return a config copy with ablation seams overridden."""
    raw = config_to_dict(cfg)
    ab = dict(raw.get("ablations", {}))
    if actor_variant is not None:
        ab["actor_variant"] = actor_variant
        raw["agents"]["actor_variant"] = actor_variant
    if frame_stack is not None:
        ab["frame_stack"] = frame_stack
        raw["agents"]["frame_stack"] = frame_stack
    if adaptive_filter_enabled is not None:
        ab["adaptive_filter_enabled"] = adaptive_filter_enabled
        raw["ppo"]["adaptive_filter_enabled"] = adaptive_filter_enabled
    if reward_randomization is not None:
        ab["reward_randomization"] = reward_randomization
        raw["reward_conditioning"]["randomize_styles"] = reward_randomization
    if condition_to_actor is not None:
        ab["condition_to_actor"] = condition_to_actor
        raw["reward_conditioning"]["expose_condition_to_actor"] = condition_to_actor
    if centralized_critic is not None:
        ab["centralized_critic"] = centralized_critic
    raw["ablations"] = ab
    return config_from_dict(raw)


def _suite_world_overrides(suite: str, cfg: ExperimentConfig) -> dict[str, Any]:
    raw = config_to_dict(cfg)
    worlds = dict(raw["worlds"])
    agents = dict(raw["agents"])
    evaluation = dict(raw["evaluation"])
    evaluation["sync_no_respawn"] = True
    agents["async_respawn"] = False
    # Eval world count is independent of training worlds.num_worlds.
    worlds["num_worlds"] = int(resolve_eval_num_worlds(cfg))
    # solo/head_to_head/surprise_braking pin the world to an exact learner
    # count (1, 2, 2) to measure that shape cleanly; a production config's
    # static_opponents_per_world (reserved out of max_agents_per_world) can
    # exceed those shrunk counts, leaving no room for a learner slot. These
    # suites predate static opponents and are defined as learners-only, so
    # drop static opponents there rather than fail the suite outright.
    if suite == "solo":
        worlds["max_agents_per_world"] = 1
        worlds["solo_world_fraction"] = 1.0
        worlds["density_bins"] = ["sparse"]
        worlds["static_opponents_per_world"] = 0
    elif suite == "head_to_head":
        worlds["max_agents_per_world"] = min(2, int(cfg.worlds.max_agents_per_world))
        worlds["solo_world_fraction"] = 0.0
        worlds["density_bins"] = ["pair"]
        worlds["static_opponents_per_world"] = 0
    elif suite == "dense":
        worlds["solo_world_fraction"] = 0.0
        worlds["density_bins"] = ["dense"]
    elif suite == "surprise_braking":
        worlds["max_agents_per_world"] = min(2, int(cfg.worlds.max_agents_per_world))
        raw["ablations"]["surprise_braking"] = True
        worlds["static_opponents_per_world"] = 0
    elif suite == "conservative_longform":
        raw["reward_conditioning"]["enabled"] = False
        raw["reward_conditioning"]["randomize_styles"] = False
        evaluation["conservative_deployment_style"] = "centered_high_collision"
        worlds["density_bins"] = ["dense"]
        worlds["solo_world_fraction"] = 0.0
    elif suite == "all_tracks":
        worlds["num_worlds"] = max(int(worlds["num_worlds"]), int(cfg.tracks.num_tracks))
    else:
        raise ConfigError(f"unknown evaluation suite: {suite!r}")
    raw["worlds"] = worlds
    raw["agents"] = agents
    raw["evaluation"] = evaluation
    return raw


def save_incident_window(
    states: list[np.ndarray],
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta or {},
        "states": [np.asarray(s, dtype=np.float32) for s in states],
    }
    torch.save(payload, path)
    return path


class FixedSeedEvaluator:
    """World-synchronous, no-respawn evaluation with fixed seeds."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        atlas: PackedTrackAtlasView | None = None,
        actor: Any | None = None,
        device: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.atlas = atlas if atlas is not None else resolve_eval_atlas(cfg)
        wanted = resolve_eval_device(cfg, device)
        if str(wanted).startswith("cuda") and not torch.cuda.is_available():
            wanted = "cpu"
        self.device = str(wanted)
        self.actor = actor
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_suite_sim(self, suite: str, seed: int):
        raw = _suite_world_overrides(suite, self.cfg)
        raw["seed"] = int(seed)
        # PPO minibatch sizing and the rollout memory estimate are budgets on a
        # training update that evaluation never runs, so a suite layout must not
        # be rejected by them. Anything else invalid fails the suite outright:
        # falling back to the base layout would score every suite on the same
        # worlds and silently erase the differences the suites exist to measure.
        try:
            suite_cfg = config_from_dict(raw, check_training_budget=False)
        except ConfigError as exc:
            raise ConfigError(
                f"evaluation suite {suite!r} world override is invalid: {exc}"
            ) from exc
        sim = build_simulator(
            suite_cfg, self.atlas, self.device, sync_no_respawn=True
        )
        sim.reset_all(seed)
        # Conservative deployment style for eval unless suite randomizes.
        style = deployment_style(suite_cfg.evaluation.conservative_deployment_style)
        sim.apply_styles([style for _ in range(sim.layout.num_slots)])
        return suite_cfg, sim

    def _policy_actions(self, sim, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = sim.layout.num_slots
        # Use sensors from reset/previous step; sim.step() rebuilds once afterward.
        obs = sim.action_observation().to(self.device)
        if self.actor is None:
            actions = torch.zeros(n, 2, device=self.device)
            actions[:, 0] = 0.2
            return actions, hidden
        cond = torch.as_tensor(
            styles_to_condition_batch(sim.styles, normalize=True),
            device=self.device,
            dtype=torch.float32,
        )
        active = (sim.buffers.torch_arrays.active > 0).to(device=self.device)
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        actions = torch.zeros(n, 2, device=self.device, dtype=torch.float32)
        next_hidden = hidden
        if idx.numel() == 0:
            return actions, next_hidden
        out = self.actor.forward(
            obs.index_select(0, idx),
            cond.index_select(0, idx),
            hidden.index_select(0, idx),
            reset_mask=None,
            deterministic=True,
        )
        actions.index_copy_(0, idx, out.actions.detach())
        next_hidden = hidden.clone()
        next_hidden.index_copy_(0, idx, out.hidden.detach())
        return actions, next_hidden

    def run_suite(self, suite: str, seed: int) -> EvalReport:
        cfg, sim = self._build_suite_sim(suite, seed)
        layout = world_slot_layout(cfg)
        n = layout.num_slots
        horizon = int(
            max(
                1,
                int(sim.buffers.torch_arrays.episode_horizon.max().item()),
            )
        )
        # Cap evaluation length for smoke / unit tests.
        horizon = min(horizon, max(20, int(cfg.evaluation.soak_steps)))
        if self.actor is not None:
            hidden = self.actor.initial_hidden(n, self.device)
        else:
            hidden = torch.zeros(n, cfg.agents.gru_hidden_dim, device=self.device)

        # Participants are frozen at reset: padded slots in a world below
        # capacity never race, so they must not dilute any denominator.
        participants = (sim.buffers.torch_arrays.active > 0).clone()
        num_participants = max(int(participants.sum().item()), 1)
        sim_participants = participants.to(device=sim.buffers.torch_device)
        lap_length = torch.as_tensor(
            [
                float(sim.geom.track_length[int(tid)])
                for tid in sim.buffers.torch_arrays.track_id.tolist()
            ],
            device=sim.buffers.torch_device,
        )
        # Control steps each participant took to reach its own lap threshold;
        # 0 means it never got there within the horizon.
        lap_steps = torch.zeros(
            n, dtype=torch.int64, device=sim.buffers.torch_device
        )

        returns = torch.zeros(n, device=self.device)
        collisions = 0
        oobs = 0
        stalls = 0
        progress0 = sim.buffers.torch_arrays.progress_s.clone()
        distance_m = 0.0
        incident_states: list[np.ndarray] = []
        window = int(cfg.evaluation.incident_window_steps)
        recent: list[np.ndarray] = []
        render_frames: list[np.ndarray] = []
        render_stride = max(
            1, math.ceil(horizon / int(cfg.evaluation.viz_max_frames))
        )
        world_slots = slice(0, layout.max_agents_per_world)
        viz_track_id = int(sim.buffers.torch_arrays.track_id[0].item())
        prev_contact = torch.zeros(n, dtype=torch.bool, device=sim.buffers.torch_device)
        prev_wall = torch.zeros(n, dtype=torch.bool, device=sim.buffers.torch_device)
        steps_run = 0

        for step in range(horizon):
            if (
                self.output_dir is not None
                and cfg.evaluation.viz_enabled
                and step % render_stride == 0
            ):
                t = sim.buffers.torch_arrays
                render_frames.append(
                    torch.stack(
                        (t.x, t.y, t.yaw, t.active.to(dtype=torch.float32)), dim=1
                    )[world_slots]
                    .detach()
                    .cpu()
                    .numpy()
                )
            actions, hidden = self._policy_actions(sim, hidden)
            # Optional surprise-braking: freeze slot 1 throttle.
            if bool(cfg.ablations.surprise_braking) and n >= 2:
                actions = actions.clone()
                actions[1:: layout.max_agents_per_world, 0] = -1.0
            before = sim.pack_state().detach().cpu().numpy()
            recent.append(before)
            if len(recent) > window:
                recent.pop(0)
            out = sim.step(actions)
            steps_run += 1
            returns = returns + out["rewards"].to(self.device)
            # Transition-time rising edges (avoids soft-wall / sticky-contact inflation).
            contact = out.get("contact")
            wall = out.get("wall_contact")
            if contact is None:
                contact = sim.buffers.torch_arrays.contact > 0
            if wall is None:
                wall = sim.buffers.torch_arrays.wall_contact > 0
            contact_b = torch.as_tensor(contact, dtype=torch.bool)
            wall_b = torch.as_tensor(wall, dtype=torch.bool)
            collisions += int((contact_b & ~prev_contact).sum().item())
            oobs += int((wall_b & ~prev_wall).sum().item())
            prev_contact = contact_b
            prev_wall = wall_b
            t = sim.buffers.torch_arrays
            stalls += int(((t.stalled_steps > 0) & (t.active > 0)).sum().item())
            # Approximate distance from progress channel if available.
            distance_m += float(
                torch.clamp(t.progress_s - progress0, min=0).sum().item()
            )
            progress0 = t.progress_s.clone()
            lap_steps[
                (lap_steps == 0)
                & sim_participants
                & (t.progress_s >= LAP_COMPLETION_FRACTION * lap_length)
            ] = steps_run
            done = out["done"].to(dtype=torch.bool)
            if bool(done.any()) and not incident_states:
                incident_states = list(recent)
            # Suites run without respawn, so a terminated participant stays
            # inactive; stop once none of them is racing.
            still_racing = (t.active > 0) & participants.to(device=t.active.device)
            if not bool(still_racing.any()):
                break

        elapsed_s = max(steps_run / cfg.agents.control_hz, 1e-6)
        km = max(distance_m / 1000.0, 1e-6)
        progress_s = sim.buffers.torch_arrays.progress_s.to(torch.float32)
        # Mean per-participant progress rate (not sum-over-world inflation).
        progress_rate = float(
            progress_s[sim_participants].sum().item()
            / elapsed_s
            / float(num_participants)
        )
        completed = (progress_s >= LAP_COMPLETION_FRACTION * lap_length)[
            sim_participants
        ]
        completion = float(completed.float().mean().item())
        # Mean over the participants that actually finished a lap; None when
        # nobody did, since averaging over an empty set would read as "fast".
        lap_steps_done = lap_steps[sim_participants & (lap_steps > 0)]
        lap_time_s = (
            float(lap_steps_done.to(torch.float32).mean().item())
            / cfg.agents.control_hz
            if lap_steps_done.numel() > 0
            else None
        )
        metrics = EvalMetrics(
            lap_time_s=lap_time_s,
            completion_rate=completion,
            progress_rate_mps=progress_rate,
            collision_per_km=float(collisions / km),
            oob_per_km=float(oobs / km),
            clean_overtakes=0.0,
            stall_rate=float(stalls / max(steps_run * num_participants, 1)),
            return_mean=float(
                returns[participants.to(device=returns.device)].mean().item()
            ),
        )
        extras = {
            "horizon_steps": float(horizon),
            "steps_run": float(steps_run),
            "distance_m": float(distance_m),
            "num_slots": float(n),
            "num_participants": float(num_participants),
            "collision_events": float(collisions),
            "oob_events": float(oobs),
            "num_lap_completers": float(lap_steps_done.numel()),
        }
        if (
            self.output_dir is not None
            and cfg.evaluation.viz_enabled
            and render_frames
        ):
            from gigaflow_f1tenth.visualization import FOLLOW_RADIUS_M, render_episode

            # Solo close-follow; multi-car suites keep whole-track framing.
            follow_radius = FOLLOW_RADIUS_M if suite == "solo" else None
            render_episode(
                self.atlas,
                viz_track_id,
                render_frames,
                self.output_dir / f"{suite}_seed{seed}",
                car_length=cfg.agents.car_length_m,
                car_width=cfg.agents.car_width_m,
                fps=cfg.evaluation.viz_fps,
                follow_radius=follow_radius,
            )
            if incident_states:
                save_incident_window(
                    incident_states,
                    self.output_dir / f"{suite}_seed{seed}_incident.pt",
                    meta={"suite": suite, "seed": seed},
                )
        return EvalReport(suite=suite, seed=int(seed), metrics=metrics, extras=extras)

    def promotion_gates(self, reports: list[EvalReport]) -> dict[str, bool]:
        return promotion_gates(
            reports,
            required=self.cfg.evaluation.suite,
        )


def build_evaluator(
    cfg: ExperimentConfig,
    *,
    atlas: PackedTrackAtlasView | None = None,
    actor: Any | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> FixedSeedEvaluator:
    return FixedSeedEvaluator(
        cfg, atlas=atlas, actor=actor, device=device, output_dir=output_dir
    )


def run_evaluation(
    cfg: ExperimentConfig,
    *,
    suites: tuple[str, ...] | None = None,
    atlas: PackedTrackAtlasView | None = None,
    actor: Any | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> list[EvalReport]:
    evaluator = build_evaluator(
        cfg, atlas=atlas, actor=actor, device=device, output_dir=output_dir
    )
    suites = suites or required_suites(cfg)
    reports: list[EvalReport] = []
    for suite in suites:
        for seed in cfg.evaluation.seeds:
            reports.append(evaluator.run_suite(suite, int(seed)))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "reports": [
                {
                    "suite": r.suite,
                    "seed": r.seed,
                    "metrics": r.metrics.__dict__,
                    "extras": r.extras,
                }
                for r in reports
            ],
            "promotion_gates": evaluator.promotion_gates(reports),
            "behavioral_score": behavioral_score(reports),
            "catastrophic": catastrophic_behavior(reports),
            "pareto_vector": behavior_pareto_vector(reports),
        }
        (out / "eval_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return reports


def run_soak(
    cfg: ExperimentConfig,
    *,
    steps: int | None = None,
    atlas: PackedTrackAtlasView | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
    mode: str = "random",
    actor=None,
) -> dict[str, Any]:
    """Random or policy-action soak: catch non-finite state / reset anomalies."""
    if mode not in {"random", "policy"}:
        raise ValueError(f"unsupported soak mode: {mode}")
    atlas = atlas or make_synthetic_oval_atlas(
        max_agents=cfg.worlds.max_agents_per_world
    )
    wanted = device or cfg.worlds.device
    if str(wanted).startswith("cuda") and not torch.cuda.is_available():
        wanted = "cpu"
    sim = build_simulator(cfg, atlas, str(wanted))
    n = world_slot_layout(cfg).num_slots
    steps = int(steps or cfg.evaluation.soak_steps)
    rng = np.random.default_rng(cfg.seed)
    policy = actor
    hidden = None
    pending_reset = torch.zeros(n, dtype=torch.bool, device=wanted)
    if mode == "policy":
        if policy is None:
            policy = build_actor(cfg).to(wanted)
            policy.eval()
        hidden = policy.initial_hidden(n, device=wanted)
    anomalies: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for step in range(steps):
        if mode == "random":
            actions = torch.as_tensor(
                rng.uniform(-1.0, 1.0, size=(n, 2)), dtype=torch.float32
            )
        else:
            assert policy is not None and hidden is not None
            obs = sim.rebuild_sensors().to(device=wanted)
            cond = torch.as_tensor(
                styles_to_condition_batch(sim.styles, normalize=True),
                device=wanted,
                dtype=torch.float32,
            )
            with torch.no_grad():
                out_a = policy.forward(
                    obs,
                    cond,
                    hidden,
                    reset_mask=pending_reset,
                    deterministic=False,
                )
            actions = out_a.actions.detach()
            hidden = out_a.hidden
        out = sim.step(actions)
        if mode == "policy":
            pending_reset = out["reset_mask"].to(device=wanted, dtype=torch.bool)
        state = sim.pack_state()
        if not torch.isfinite(state).all():
            anomalies.append({"step": step, "kind": "non_finite_state"})
            if output_dir is not None:
                save_incident_window(
                    [state.detach().cpu().numpy()],
                    Path(output_dir) / f"soak_nonfinite_{step}.pt",
                    meta={"step": step, "mode": mode},
                )
            break
        if int(out.get("broadphase_overflow", 0)) > 0:
            anomalies.append({"step": step, "kind": "broadphase_overflow"})
            break
        if not torch.isfinite(out["rewards"]).all():
            anomalies.append({"step": step, "kind": "non_finite_reward"})
            break
        if not torch.isfinite(out["sensor_obs"]).all():
            anomalies.append({"step": step, "kind": "non_finite_lidar"})
            break
    elapsed = time.perf_counter() - t0
    report = {
        "ok": len(anomalies) == 0,
        "mode": mode,
        "steps": steps,
        "elapsed_s": elapsed,
        "world_ticks_per_s": steps / max(elapsed, 1e-9),
        "anomalies": anomalies,
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "soak_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def load_actor_from_checkpoint(
    cfg: ExperimentConfig, checkpoint: str | Path, device: str = "cpu"
):
    """Load actor weights from a trainer checkpoint or deployable artifact."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(
        cfg,
        variant=cfg.ablations.actor_variant,
        frame_stack=cfg.ablations.frame_stack
        if cfg.ablations.actor_variant == "frame_stack"
        else 1,
    )
    if "actor_state_dict" in payload:
        actor.load_state_dict(payload["actor_state_dict"])
    elif "ppo" in payload and "actor" in payload["ppo"]:
        actor.load_state_dict(payload["ppo"]["actor"])
    else:
        raise KeyError("checkpoint missing actor weights")
    return actor.to(device)

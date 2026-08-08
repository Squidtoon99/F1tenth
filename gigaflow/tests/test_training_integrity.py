"""Multi-step training-integrity regressions (terminals, horizon, GAE, logging)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gigaflow_f1tenth.buffers import STATE_INDEX
from gigaflow_f1tenth.config import (
    ConfigError,
    config_from_dict,
    config_to_dict,
    load_config,
)
from gigaflow_f1tenth.evaluation import (
    BehaviorGateState,
    LAP_COMPLETION_FRACTION,
    SOLO_PROGRESS_MIN_MPS,
    EvalMetrics,
    EvalReport,
    _suite_world_overrides,
    behavioral_score,
    catastrophic_behavior,
    promotion_gates,
)
from gigaflow_f1tenth.evaluation import FixedSeedEvaluator
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.model import ActorOutput
from gigaflow_f1tenth.ppo import compute_gae
from gigaflow_f1tenth.rewards import deployment_style
from gigaflow_f1tenth.sim.geometry import (
    build_sim_geometry_from_atlas,
    make_synthetic_oval_atlas,
    make_two_track_atlas,
)
from gigaflow_f1tenth.sim.layout_local import (
    IMU_START,
    STEER_DELTA0,
    STEER_DELTA1,
    STEER_DELTA2,
    STEER_T,
    STEER_T1,
    STEER_T2,
    THROTTLE_CURRENT,
    THROTTLE_PRED,
    VESC_SPEED,
)
from gigaflow_f1tenth.wandb_log import summarize_rollout_aux

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _cfg(*, async_respawn: bool = True):
    cfg = load_config(SMOKE)
    worlds = replace(cfg.worlds, num_worlds=2, max_agents_per_world=2)
    agents = replace(cfg.agents, async_respawn=async_respawn)
    return replace(cfg, worlds=worlds, agents=agents)


def _place_slot(sim, geom, *, tid: int, local_seg: int, slot: int = 0) -> None:
    t = sim.buffers.torch_arrays
    a = int(geom.offsets[tid])
    glob = a + local_seg
    yaw = float(
        np.arctan2(geom.tangents_xy[glob, 1], geom.tangents_xy[glob, 0])
    )
    t.active[slot] = 1
    t.trainable[slot] = 1
    t.done[slot] = 0
    t.timeout[slot] = 0
    t.reset_mask[slot] = 0
    t.track_id[slot] = tid
    t.frenet_segment[slot] = local_seg
    t.x[slot] = float(geom.centerline_xy[glob, 0])
    t.y[slot] = float(geom.centerline_xy[glob, 1])
    t.yaw[slot] = yaw
    t.vx[slot] = 1.0
    t.vy[slot] = 0.0
    t.frenet_s[slot] = float(geom.cum_length[glob])
    t.prev_s[slot] = float(geom.cum_length[glob])
    t.frenet_ey[slot] = 0.0
    t.boundary_distance[slot] = 1.0
    t.episode_step[slot] = 0
    t.stalled_steps[slot] = 0
    t.wall_contact[slot] = 0
    t.contact[slot] = 0
    t.yaw_rate[slot] = 0.0


def test_async_full_oob_exposes_done_for_gae_and_respawns():
    """Fail-before: async respawn cleared done → GAE bootstrapped through death."""
    cfg = _cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    tid, local_seg = 1, 8
    a = int(geom.offsets[tid])
    glob = a + local_seg
    half_w = 0.5 * float(cfg.agents.car_width_m)
    nxy = geom.normals_xy[glob]
    oob = float(geom.widths_rl[glob, 1]) + half_w + 0.05

    _place_slot(sim, geom, tid=tid, local_seg=local_seg, slot=0)
    t.x[0] = float(geom.centerline_xy[glob, 0] + oob * nxy[0])
    t.y[0] = float(geom.centerline_xy[glob, 1] + oob * nxy[1])
    t.vx[0] = 1.0
    if sim._style_tensors is None:
        sim.apply_styles(sim.styles)

    # Multi-step: death then a post-respawn control tick.
    outs = []
    for _ in range(3):
        outs.append(sim.step(torch.zeros((n, 2))))

    death = outs[0]
    assert bool(death["done"][0].item())
    assert not bool(death["timeout"][0].item())
    assert bool(death["reset_mask"][0].item())
    assert bool(death["wall_contact"][0].item()) or float(
        death["reward_terms"]["boundary"][0]
    ) != 0.0
    # Live slot was respawned (active, done cleared) while transition kept flags.
    assert int(t.active[0].item()) == 1
    assert int(t.done[0].item()) == 0

    # GAE must cut on the death transition (done & ~timeout).
    rewards = torch.tensor([[1.0], [-5.0], [0.5]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    done = torch.tensor([[False], [True], [False]])
    timeout = torch.zeros_like(done)
    last = torch.tensor([0.5])
    adv_cut, _ = compute_gae(
        rewards, values, done, timeout, last, gamma=0.999, gae_lambda=0.95
    )
    done_bug = torch.zeros_like(done)  # pre-fix: respawn cleared done
    adv_bug, _ = compute_gae(
        rewards, values, done_bug, timeout, last, gamma=0.999, gae_lambda=0.95
    )
    assert adv_cut[1, 0].item() == pytest.approx(-5.0 - 0.5, abs=1e-5)
    assert abs(adv_bug[1, 0].item() - adv_cut[1, 0].item()) > 0.5


def test_horizon_timeout_returns_pre_respawn_next_state_for_bootstrap():
    """Fail-before: the bootstrap state was read after respawn had overwritten it."""
    cfg = _cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    _place_slot(sim, geom, tid=0, local_seg=4, slot=0)
    t.episode_horizon[0] = 2
    t.episode_step[0] = 0
    if sim._style_tensors is None:
        sim.apply_styles(sim.styles)

    out0 = sim.step(torch.zeros((n, 2)))
    assert not bool(out0["timeout"][0].item())
    out1 = sim.step(torch.zeros((n, 2)))
    assert bool(out1["done"][0].item())
    assert bool(out1["timeout"][0].item())
    assert bool(out1["reset_mask"][0].item())
    assert int(t.active[0].item()) == 1
    assert int(t.episode_step[0].item()) == 0  # respawned

    nxt = out1["next_compact_state"]
    live = sim.pack_state()
    # The truncated episode's own next state, not the fresh one.
    assert float(nxt[0, STATE_INDEX["active"]]) == 1.0
    assert float(nxt[0, STATE_INDEX["progress_s"]]) > 0.0
    assert float(live[0, STATE_INDEX["progress_s"]]) == 0.0
    # Only respawned rows may differ between the two snapshots.
    row_delta = (nxt - live).abs().sum(dim=-1)
    reset_rows = out1["reset_mask"]
    assert float(row_delta[0].item()) > 0.0
    assert float(row_delta[~reset_rows].max().item()) == 0.0

    # PPO GAE: a truncation bootstraps V(s') while a true terminal cuts.
    rewards = torch.zeros(3, 1)
    values = torch.ones(3, 1)
    next_values = torch.full((3, 1), 2.0)
    done = torch.tensor([[False], [True], [False]])
    timeout = torch.tensor([[False], [True], [False]])
    last = torch.tensor([2.0])
    adv_to, _ = compute_gae(
        rewards,
        values,
        done,
        timeout,
        last,
        gamma=0.9,
        gae_lambda=1.0,
        next_values=next_values,
    )
    adv_term, _ = compute_gae(
        rewards,
        values,
        done,
        torch.zeros_like(timeout),
        last,
        gamma=0.9,
        gae_lambda=1.0,
        next_values=next_values,
    )
    assert adv_to[1, 0].item() == pytest.approx(0.9 * 2.0 - 1.0, abs=1e-5)
    assert adv_term[1, 0].item() == pytest.approx(0.0 - 1.0, abs=1e-5)


def test_sync_no_respawn_keeps_terminal_reward_before_deactivating():
    """Fail-before: the terminal kernel deactivated the row and rewards zeroed it."""
    cfg = _cfg(async_respawn=False)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=True)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    style = deployment_style(cfg.evaluation.conservative_deployment_style)
    sim.apply_styles([style for _ in range(n)])
    tid, local_seg = 1, 8
    glob = int(geom.offsets[tid]) + local_seg
    nxy = geom.normals_xy[glob]
    oob = float(geom.widths_rl[glob, 1]) + 0.5 * float(cfg.agents.car_width_m) + 0.05
    _place_slot(sim, geom, tid=tid, local_seg=local_seg, slot=0)
    t.x[0] = float(geom.centerline_xy[glob, 0] + oob * nxy[0])
    t.y[0] = float(geom.centerline_xy[glob, 1] + oob * nxy[1])

    out = sim.step(torch.zeros((n, 2)))
    assert bool(out["done"][0].item())
    assert not bool(out["timeout"][0].item())
    assert float(out["reward_terms"]["boundary"][0].item()) == pytest.approx(
        -style.alpha_boundary, abs=1e-5
    )
    assert float(out["rewards"][0].item()) != 0.0
    # Retired only after its reward was recorded.
    assert int(t.active[0].item()) == 0
    assert int(t.trainable[0].item()) == 0


def test_async_respawn_clears_executed_command_history():
    """Fail-before: proprioception read the dead episode's command history."""
    cfg = _cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    _place_slot(sim, geom, tid=0, local_seg=4, slot=0)
    t.episode_horizon[0] = 2
    t.episode_step[0] = 0
    if sim._style_tensors is None:
        sim.apply_styles(sim.styles)

    actions = torch.zeros((n, 2))
    actions[0, 0] = 0.5
    actions[0, 1] = 0.5
    history = (
        t.executed_long_0,
        t.executed_long_1,
        t.executed_steer_0,
        t.executed_steer_1,
        t.executed_steer_2,
        t.executed_steer_3,
    )
    sim.step(actions)
    assert max(abs(float(arr[0].item())) for arr in history) > 0.0

    out = sim.step(actions)
    assert bool(out["reset_mask"][0].item())
    for arr in history:
        assert float(arr[0].item()) == 0.0
    obs = out["sensor_obs"][0]
    for channel in (
        THROTTLE_CURRENT,
        THROTTLE_PRED,
        STEER_T,
        STEER_T1,
        STEER_T2,
        STEER_DELTA0,
        STEER_DELTA1,
        STEER_DELTA2,
    ):
        assert float(obs[channel].item()) == 0.0


def test_catastrophic_wall_terminal_async_snapshot():
    cfg = _cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    tid, local_seg = 0, 6
    a = int(geom.offsets[tid])
    glob = a + local_seg
    half_w = 0.5 * float(cfg.agents.car_width_m)
    nxy = geom.normals_xy[glob]
    # Soft wall contact (footprint over boundary) at catastrophic speed.
    _place_slot(sim, geom, tid=tid, local_seg=local_seg, slot=0)
    lateral = float(geom.widths_rl[glob, 1]) - 0.5 * half_w
    t.x[0] = float(geom.centerline_xy[glob, 0] + lateral * nxy[0])
    t.y[0] = float(geom.centerline_xy[glob, 1] + lateral * nxy[1])
    t.vx[0] = 5.0  # > catastrophic_speed (4.0)
    if sim._style_tensors is None:
        sim.apply_styles(sim.styles)

    out = sim.step(torch.zeros((n, 2)))
    assert bool(out["done"][0].item())
    assert not bool(out["timeout"][0].item())
    assert bool(out["reset_mask"][0].item())
    assert bool(out["wall_contact"][0].item())


def test_rollout_reward_term_and_event_means_not_last_step():
    """Logging must use transition-time [T,S] means, not post-respawn snapshots."""
    valid = torch.ones(3, 2, dtype=torch.bool)
    # Step 0 soft wall; step 1 clear after "respawn"; step 2 clear.
    terms = {
        "boundary": torch.tensor(
            [[-2.5, 0.0], [0.0, 0.0], [0.0, 0.1]], dtype=torch.float32
        ),
        "progress": torch.tensor(
            [[0.1, 0.2], [0.3, 0.0], [0.4, 0.5]], dtype=torch.float32
        ),
    }
    wall = torch.tensor(
        [[True, False], [False, False], [False, False]], dtype=torch.bool
    )
    contact = torch.zeros_like(wall)
    contact[0, 1] = True

    class _EndSnap:
        active = torch.ones(2)
        contact = torch.zeros(2)  # post-respawn cleared
        wall_contact = torch.zeros(2)
        progress_s = torch.tensor([1.0, 2.0])

    reward_means, _, _, sim_stats = summarize_rollout_aux(
        rewards=terms["progress"],
        valid=valid,
        track_id=torch.tensor([0, 1]),
        active_end=_EndSnap.active,
        max_agents_per_world=2,
        reward_terms=terms,
        done=wall,  # reuse shape
        timeout=torch.zeros_like(wall),
        reset_mask=wall,
        contact=contact,
        wall_contact=wall,
        sim_arrays=_EndSnap(),
    )
    assert reward_means["boundary_mean"] == pytest.approx(
        terms["boundary"][valid].float().mean().item()
    )
    assert reward_means["boundary_mean"] < 0.0
    assert sim_stats["oob_frac"] == pytest.approx(1.0 / 6.0)
    assert sim_stats["oob_count"] == pytest.approx(1.0)
    assert sim_stats["collision_count"] == pytest.approx(1.0)
    # End snapshot alone would have reported 0 — transition path must win.
    assert sim_stats["oob_frac"] > 0.0


def test_solo_progress_floor_gate_detects_crawl_regression():
    cfg = load_config(SMOKE)
    ev = FixedSeedEvaluator(cfg, device="cpu")
    crawl = EvalReport(
        suite="solo",
        seed=0,
        metrics=EvalMetrics(
            lap_time_s=None,
            completion_rate=0.0,
            progress_rate_mps=0.03,
            collision_per_km=0.0,
            oob_per_km=10.0,
            clean_overtakes=0.0,
            stall_rate=0.0,
            return_mean=-1.0,
        ),
    )
    healthy = EvalReport(
        suite="solo",
        seed=0,
        metrics=EvalMetrics(
            lap_time_s=60.0,
            completion_rate=0.95,
            progress_rate_mps=5.0,
            collision_per_km=0.0,
            oob_per_km=0.1,
            clean_overtakes=0.0,
            stall_rate=0.0,
            return_mean=1.0,
        ),
    )
    gates_bad = ev.promotion_gates([crawl])
    gates_ok = ev.promotion_gates([healthy])
    assert gates_bad["solo_progress"] is False
    assert gates_ok["solo_progress"] is True
    assert gates_ok["solo_completion"] is True
    assert gates_ok["solo_lap"] is True
    assert SOLO_PROGRESS_MIN_MPS == pytest.approx(4.5)


def _behavior_reports(
    *,
    solo_completion,
    solo_lap,
    solo_progress,
    solo_oob,
    dense_collisions=2.0,
    dense_oob=0.2,
):
    common = {
        "clean_overtakes": 0.0,
        "stall_rate": 0.0,
        "return_mean": 1.0,
    }
    return [
        EvalReport(
            suite="solo",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=solo_lap,
                completion_rate=solo_completion,
                progress_rate_mps=solo_progress,
                collision_per_km=0.0,
                oob_per_km=solo_oob,
                **common,
            ),
        ),
        EvalReport(
            suite="head_to_head",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=68.0,
                completion_rate=0.90,
                progress_rate_mps=5.0,
                collision_per_km=1.0,
                oob_per_km=0.2,
                **common,
            ),
        ),
        EvalReport(
            suite="dense",
            seed=0,
            metrics=EvalMetrics(
                lap_time_s=72.0,
                completion_rate=0.60,
                progress_rate_mps=4.0,
                collision_per_km=dense_collisions,
                oob_per_km=dense_oob,
                **common,
            ),
        ),
    ]


def test_archived_behavior_examples_and_hysteresis():
    required = ("solo", "head_to_head", "dense")
    update_1100 = _behavior_reports(
        solo_completion=0.955,
        solo_lap=62.69,
        solo_progress=5.86,
        solo_oob=0.062,
    )
    update_2100 = _behavior_reports(
        solo_completion=0.990,
        solo_lap=95.19,
        solo_progress=3.976,
        solo_oob=0.1,
    )
    update_2700 = _behavior_reports(
        solo_completion=0.045,
        solo_lap=124.7,
        solo_progress=0.53,
        solo_oob=10.14,
    )
    reckless = _behavior_reports(
        solo_completion=0.99,
        solo_lap=55.0,
        solo_progress=6.5,
        solo_oob=0.943,
        dense_collisions=5.503,
        dense_oob=0.8,
    )

    assert all(promotion_gates(update_1100, required=required).values())
    assert not all(promotion_gates(update_2100, required=required).values())
    assert behavioral_score(update_1100) > behavioral_score(update_2100)
    assert catastrophic_behavior(update_2700) is True
    assert not all(promotion_gates(reckless, required=required).values())

    state = BehaviorGateState()
    first_pass = state.observe(update_1100, required=required, step=1100)
    assert first_pass.passed and first_pass.best_safe
    latch = state.observe(update_1100, required=required, step=1200)
    assert latch.passed and state.feasible
    first_failure = state.observe(update_2100, required=required, step=2100)
    assert first_failure.warning_active and not first_failure.stop_requested
    second_failure = state.observe(update_2100, required=required, step=2300)
    assert second_failure.stop_requested

    state = BehaviorGateState()
    state.observe(update_1100, required=required, step=1100)
    state.observe(update_2100, required=required, step=2100)
    one_pass = state.observe(update_1100, required=required, step=2300)
    assert one_pass.warning_active
    two_passes = state.observe(update_1100, required=required, step=2500)
    assert two_passes.warning_active is False
    feasible_state = BehaviorGateState()
    feasible_state.observe(update_1100, required=required, step=1100)
    feasible_state.observe(update_1100, required=required, step=1300)
    assert feasible_state.feasible
    first_catastrophic = feasible_state.observe(
        update_2700, required=required, step=2700
    )
    assert first_catastrophic.catastrophic and not first_catastrophic.stop_requested
    second_catastrophic = feasible_state.observe(
        update_2700, required=required, step=2900
    )
    assert second_catastrophic.stop_requested and second_catastrophic.catastrophic


def test_eval_metrics_ignore_padded_slots_when_capacity_grows():
    """Fail-before: denominators used layout.num_slots, so pads diluted metrics."""
    base = load_config(SMOKE)

    def _suite_cfg(max_agents: int):
        raw = config_to_dict(base)
        raw["worlds"]["max_agents_per_world"] = max_agents
        # One racer per world, so extra capacity is pure padding.
        raw["worlds"]["solo_world_fraction"] = 1.0
        return config_from_dict(raw)

    atlas = make_two_track_atlas(max_agents=4)
    narrow, wide = (
        FixedSeedEvaluator(
            _suite_cfg(max_agents), atlas=atlas, device="cpu"
        ).run_suite("all_tracks", 0)
        for max_agents in (2, 4)
    )
    assert narrow.extras["num_slots"] < wide.extras["num_slots"]
    assert narrow.extras["num_participants"] == wide.extras["num_participants"]
    assert wide.metrics.progress_rate_mps == pytest.approx(
        narrow.metrics.progress_rate_mps, rel=1e-6
    )
    assert wide.metrics.completion_rate == pytest.approx(
        narrow.metrics.completion_rate, abs=1e-9
    )
    assert wide.metrics.stall_rate == pytest.approx(
        narrow.metrics.stall_rate, rel=1e-6
    )
    assert wide.metrics.return_mean == pytest.approx(
        narrow.metrics.return_mean, rel=1e-5
    )


def _eval_suite_cfg():
    """Eval-scale config whose PPO minibatch only fits the base world layout."""
    raw = config_to_dict(load_config(SMOKE))
    raw["worlds"]["num_worlds"] = 32
    raw["worlds"]["max_agents_per_world"] = 8
    raw["worlds"]["density_bins"] = ["sparse", "medium", "dense"]
    raw["evaluation"]["num_worlds"] = 2
    raw["evaluation"]["soak_steps"] = 12
    raw["evaluation"]["viz_enabled"] = False
    raw["ppo"]["minibatch_size"] = 32 * 8 * raw["ppo"]["rollout_length"]
    return config_from_dict(raw)


def test_eval_suites_race_their_own_world_layout():
    """Fail-before: a suite override rejected by the PPO transition budget was
    swallowed, so solo/head_to_head/dense all raced the base 4-car layout."""
    cfg = _eval_suite_cfg()
    atlas = make_synthetic_oval_atlas(radius=20.0, max_agents=8)
    ev = FixedSeedEvaluator(cfg, atlas=atlas, device="cpu")

    agents_per_world = {"solo": 1, "head_to_head": 2, "dense": 8}
    populations = {}
    reports = {}
    for suite, agents in agents_per_world.items():
        suite_cfg, sim = ev._build_suite_sim(suite, 0)
        assert suite_cfg.worlds.max_agents_per_world == agents
        assert sim.layout.max_agents_per_world == agents
        active = (sim.buffers.torch_arrays.active > 0).reshape(-1, agents)
        populations[suite] = active.sum(dim=1).tolist()
        reports[suite] = ev.run_suite(suite, 0)

    assert populations["solo"] == [1, 1]
    assert populations["head_to_head"] == [2, 2]
    assert min(populations["dense"]) > 2
    assert reports["solo"].extras["num_participants"] == 2.0
    assert reports["head_to_head"].extras["num_participants"] == 4.0
    assert reports["dense"].extras["num_participants"] > 4.0

    # The original symptom: byte-identical numbers across differently sized fields.
    fingerprints = {
        suite: (
            report.metrics.return_mean,
            report.metrics.progress_rate_mps,
            report.extras["distance_m"],
        )
        for suite, report in reports.items()
    }
    assert len(set(fingerprints.values())) == len(fingerprints)


def test_unrepresentable_and_unknown_suites_fail_closed():
    cfg = _eval_suite_cfg()
    ev = FixedSeedEvaluator(
        cfg, atlas=make_synthetic_oval_atlas(radius=20.0, max_agents=8), device="cpu"
    )

    # Training-only budgets must not reject an eval layout...
    solo = _suite_world_overrides("solo", cfg)
    with pytest.raises(ConfigError, match="minibatch_size"):
        config_from_dict(solo)
    assert config_from_dict(solo, check_training_budget=False).worlds.num_worlds == 2

    # ...and a suite that cannot be built must raise rather than fall back.
    with pytest.raises(ConfigError, match="unknown evaluation suite"):
        ev.run_suite("head-to-head", 0)


def test_static_opponents_do_not_starve_fixed_learner_count_suites():
    """A production config's static_opponents_per_world must not break the
    solo/head_to_head/surprise_braking suites, which pin the world to an
    exact learner count (1, 2, 2) that predates static opponents."""
    raw = config_to_dict(_eval_suite_cfg())
    raw["worlds"]["static_opponents_per_world"] = 2
    cfg = config_from_dict(raw)

    for suite in ("solo", "head_to_head", "surprise_braking"):
        overridden = _suite_world_overrides(suite, cfg)
        assert overridden["worlds"]["static_opponents_per_world"] == 0
        suite_cfg = config_from_dict(overridden, check_training_budget=False)
        assert suite_cfg.worlds.static_opponents_per_world == 0

    # Dense keeps the full learner slot count, so static opponents still fit.
    dense = _suite_world_overrides("dense", cfg)
    assert dense["worlds"]["static_opponents_per_world"] == 2
    dense_cfg = config_from_dict(dense, check_training_budget=False)
    assert dense_cfg.worlds.static_opponents_per_world == 2


class CircleDriver:
    """Real closed-loop driver: hold a commanded speed on a circular track.

    Yaw rate is regulated to ``speed / radius`` (the rate that traces the
    centerline) with a lidar left/right correction that re-centers the car.
    """

    LEFT_BEAM = 900  # +90 deg on the 1081-beam, 270-deg scan
    RIGHT_BEAM = 180  # -90 deg
    YAW_RATE = IMU_START + 5

    def __init__(self, speed_mps: float, radius_m: float) -> None:
        self.speed = float(speed_mps)
        self.radius = float(radius_m)

    def initial_hidden(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, 1, device=device)

    def forward(self, obs, cond, hidden, reset_mask=None, deterministic=True):
        throttle = torch.clamp(2.0 * (self.speed - obs[:, VESC_SPEED]), -1.0, 1.0)
        centering = obs[:, self.LEFT_BEAM] - obs[:, self.RIGHT_BEAM]
        yaw_rate_target = self.speed / self.radius + 0.6 * centering
        steer = torch.clamp(
            3.0 * (yaw_rate_target - obs[:, self.YAW_RATE]), -1.0, 1.0
        )
        return ActorOutput(
            actions=torch.stack((throttle, steer), dim=1),
            log_prob=torch.zeros_like(throttle),
            entropy=torch.zeros_like(throttle),
            hidden=hidden,
        )


def _lap_time_cfg(*, max_agents: int, soak_steps: int):
    raw = config_to_dict(load_config(SMOKE))
    raw["worlds"]["num_worlds"] = 2
    raw["worlds"]["max_agents_per_world"] = max_agents
    raw["worlds"]["solo_world_fraction"] = 1.0
    raw["evaluation"]["soak_steps"] = soak_steps
    raw["evaluation"]["viz_enabled"] = False
    raw["ppo"]["minibatch_size"] = 2 * max_agents * raw["ppo"]["rollout_length"]
    return config_from_dict(raw)


def test_lap_time_tracks_commanded_speed_on_a_circular_track():
    """Fail-before: EvalMetrics.lap_time_s was hardcoded to None."""
    radius = 8.0
    atlas = make_synthetic_oval_atlas(radius=radius, max_agents=1)
    lap_distance = LAP_COMPLETION_FRACTION * float(atlas.lengths[0])
    cfg = _lap_time_cfg(max_agents=1, soak_steps=400)

    laps = {}
    for speed in (2.0, 3.0):
        report = FixedSeedEvaluator(
            cfg, atlas=atlas, actor=CircleDriver(speed, radius), device="cpu"
        ).run_suite("solo", 0)
        assert report.metrics.completion_rate == 1.0
        assert report.extras["num_lap_completers"] == report.extras["num_participants"]
        laps[speed] = report.metrics.lap_time_s

    for speed, lap_time in laps.items():
        assert lap_time == pytest.approx(lap_distance / speed, rel=0.10)
    # Halving the commanded speed must double the lap time.
    assert laps[2.0] / laps[3.0] == pytest.approx(1.5, rel=0.06)


def test_lap_time_is_none_without_a_completed_lap_and_ignores_pads():
    radius = 8.0
    atlas = make_synthetic_oval_atlas(radius=radius, max_agents=4)
    driver = CircleDriver(2.0, radius)

    unfinished = FixedSeedEvaluator(
        _lap_time_cfg(max_agents=1, soak_steps=20),
        atlas=atlas,
        actor=driver,
        device="cpu",
    ).run_suite("solo", 0)
    assert unfinished.metrics.completion_rate == 0.0
    assert unfinished.extras["num_lap_completers"] == 0.0
    assert unfinished.metrics.lap_time_s is None

    # Padded slots never race, so widening capacity must not move the metric.
    # (all_tracks keeps the configured capacity; solo would force it back to 1.)
    narrow, wide = (
        FixedSeedEvaluator(
            _lap_time_cfg(max_agents=max_agents, soak_steps=400),
            atlas=atlas,
            actor=driver,
            device="cpu",
        ).run_suite("all_tracks", 0)
        for max_agents in (1, 4)
    )
    assert narrow.extras["num_slots"] < wide.extras["num_slots"]
    assert narrow.extras["num_lap_completers"] == wide.extras["num_lap_completers"]
    # Sensor noise is seeded per flat slot, so the two runs are close rather
    # than identical; counting the three pads would instead quarter the mean.
    assert wide.metrics.lap_time_s == pytest.approx(
        narrow.metrics.lap_time_s, rel=0.05
    )


def test_private_style_diversity_across_slots_and_respawns():
    """Verify style sampling diversity; fail only on a reproducible defect."""
    cfg = _cfg(async_respawn=True)
    atlas = make_two_track_atlas()
    geom = build_sim_geometry_from_atlas(atlas)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    for slot in range(n):
        _place_slot(sim, geom, tid=slot % 2, local_seg=3, slot=slot)
    sim.resample_styles_for_mask(torch.ones(n, dtype=torch.bool), cfg.seed + 3)
    styles0 = sim.condition_tensor().detach().clone()
    # Distinct slots should not all collapse to one private style.
    pairwise = (styles0[0:1] - styles0).abs().sum(dim=-1)
    assert float(pairwise.max().item()) > 1e-3

    # Force respawn on slot 0 and confirm style can change with episode_id.
    t.reset_mask[:] = 0
    t.reset_mask[0] = 1
    t.episode_id[0] = int(t.episode_id[0].item()) + 1
    before = sim.condition_tensor()[0].detach().clone()
    sim.resample_styles_for_mask(t.reset_mask > 0, cfg.seed + 7)
    after = sim.condition_tensor()[0].detach().clone()
    # With a 256-row pool, a single resample usually changes; allow rare collision
    # by also checking another slot sample under a different episode id.
    changed = bool((before - after).abs().sum().item() > 1e-6)
    t.episode_id[1] = int(t.episode_id[1].item()) + 5
    t.reset_mask[:] = 0
    t.reset_mask[1] = 1
    b1 = sim.condition_tensor()[1].detach().clone()
    sim.resample_styles_for_mask(t.reset_mask > 0, cfg.seed + 7)
    a1 = sim.condition_tensor()[1].detach().clone()
    changed1 = bool((b1 - a1).abs().sum().item() > 1e-6)
    assert changed or changed1, "style scatter must vary across episode_id mixes"

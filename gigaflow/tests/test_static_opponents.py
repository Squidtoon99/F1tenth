"""Training-side gates for R2: light static opponents (immobile agent slots).

Placement/pinning geometry itself is exercised by the viewer's
``test_viewer_replay.py`` (same shared ``sim.spawn`` implementation); these
tests cover the training-only integration points: capacity reservation,
async-respawn safety, PPO transition exclusion, and the measurable LiDAR/
contact effect on learning agents.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gigaflow_f1tenth.config import ConfigError, config_from_dict, config_to_dict, load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import build_sim_geometry_from_atlas, make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import LIDAR_DIM
from gigaflow_f1tenth.sim.spawn import assign_world_tracks, sample_active_counts
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"


def _cfg(*, num_worlds: int = 2, max_agents_per_world: int = 3, static: int = 2, **overrides):
    raw = config_to_dict(load_config(SMOKE))
    raw["worlds"]["num_worlds"] = num_worlds
    raw["worlds"]["max_agents_per_world"] = max_agents_per_world
    raw["worlds"]["static_opponents_per_world"] = static
    raw["worlds"]["density_bins"] = ["dense"]
    raw["worlds"]["solo_world_fraction"] = 0.0
    for section, values in overrides.items():
        raw[section] = {**raw.get(section, {}), **values}
    return config_from_dict(raw)


def _static_mask(t) -> torch.Tensor:
    return (t.active > 0) & (t.trainable == 0)


def _learner_mask(t) -> torch.Tensor:
    return (t.active > 0) & (t.trainable > 0)


def test_config_rejects_invalid_static_opponent_counts():
    raw = config_to_dict(load_config(SMOKE))
    raw["worlds"]["max_agents_per_world"] = 4
    raw["worlds"]["static_opponents_per_world"] = -1
    with pytest.raises(ConfigError, match="non-negative"):
        config_from_dict(raw)

    raw["worlds"]["static_opponents_per_world"] = 4
    with pytest.raises(ConfigError, match="leave room for at least one"):
        config_from_dict(raw)


def test_static_opponents_count_toward_world_capacity_not_beyond_it():
    """Reserving static slots must not grow max_agents_per_world / num_slots."""
    cfg = _cfg()
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    layout = world_slot_layout(cfg)
    n_agents = cfg.worlds.max_agents_per_world
    assert layout.max_agents_per_world == n_agents
    assert layout.num_slots == cfg.worlds.num_worlds * n_agents

    t = sim.buffers.torch_arrays
    stride = n_agents
    for w in range(cfg.worlds.num_worlds):
        world_active = int(t.active[w * stride : (w + 1) * stride].sum().item())
        assert world_active <= stride
        world_static = int(_static_mask(t)[w * stride : (w + 1) * stride].sum().item())
        assert world_static == cfg.worlds.static_opponents_per_world


def test_reserved_capacity_shrinks_sampled_learner_counts():
    cfg = _cfg(max_agents_per_world=4, static=2)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    geom = build_sim_geometry_from_atlas(atlas)
    track_ids = assign_world_tracks(cfg, geom, cfg.seed)
    counts = sample_active_counts(cfg, geom, track_ids, cfg.seed + 17)
    # Effective learner ceiling is max_agents_per_world - static_opponents_per_world.
    effective_max = cfg.worlds.max_agents_per_world - cfg.worlds.static_opponents_per_world
    assert int(counts.max()) <= effective_max


def test_static_opponent_placement_is_deterministic_from_seed():
    cfg = _cfg()
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim_a = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    sim_b = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)

    ta, tb = sim_a.buffers.torch_arrays, sim_b.buffers.torch_arrays
    mask_a, mask_b = _static_mask(ta), _static_mask(tb)
    assert torch.equal(mask_a, mask_b)
    assert bool(mask_a.any())
    for field in ("x", "y", "yaw", "frenet_s", "frenet_ey", "boundary_distance"):
        va = getattr(ta, field)[mask_a]
        vb = getattr(tb, field)[mask_b]
        assert torch.allclose(va, vb, atol=1e-6)

    # A different seed's world/track assignment shifts placement.
    raw = config_to_dict(cfg)
    raw["seed"] = cfg.seed + 1
    cfg_c = config_from_dict(raw)
    sim_c = build_simulator(cfg_c, atlas, "cpu", sync_no_respawn=False)
    tc = sim_c.buffers.torch_arrays
    mask_c = _static_mask(tc)
    assert bool(mask_c.any())
    same_positions = torch.allclose(ta.x[mask_a], tc.x[mask_c], atol=1e-6) and torch.allclose(
        ta.y[mask_a], tc.y[mask_c], atol=1e-6
    )
    assert not same_positions


def test_static_opponent_passability_preserved():
    cfg = _cfg()
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    mask = _static_mask(t)
    assert bool(mask.any())
    # A corridor wide enough for a car to pass survives placement (never
    # jammed flush against the boundary).
    assert torch.all(t.boundary_distance[mask] > 0.14)


def test_static_opponent_zero_motion_under_contact_and_never_respawns():
    cfg = _cfg(num_worlds=1, max_agents_per_world=3, static=2)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    n = t.x.shape[0]
    static_idx = int(_static_mask(t).nonzero(as_tuple=False)[0].item())
    learner_idx = int(_learner_mask(t).nonzero(as_tuple=False)[0].item())

    ox, oy, oyaw = float(t.x[static_idx]), float(t.y[static_idx]), float(t.yaw[static_idx])
    # Drive the learner straight into the static opponent at speed.
    t.x[learner_idx] = ox - 0.3 * math.cos(oyaw)
    t.y[learner_idx] = oy - 0.3 * math.sin(oyaw)
    t.yaw[learner_idx] = oyaw
    t.prev_x[learner_idx] = t.x[learner_idx]
    t.prev_y[learner_idx] = t.y[learner_idx]
    t.prev_yaw[learner_idx] = oyaw
    t.vx[learner_idx] = 3.0
    t.vy[learner_idx] = 0.0
    t.track_id[learner_idx] = t.track_id[static_idx]
    t.frenet_segment[learner_idx] = t.frenet_segment[static_idx]

    actions = torch.zeros((n, 2))
    actions[learner_idx, 0] = 1.0
    saw_contact = False
    for _ in range(15):
        out = sim.step(actions)
        saw_contact = saw_contact or bool(out["contact"][learner_idx].item())
        assert float(t.x[static_idx].item()) == pytest.approx(ox, abs=1e-9)
        assert float(t.y[static_idx].item()) == pytest.approx(oy, abs=1e-9)
        assert float(t.yaw[static_idx].item()) == pytest.approx(oyaw, abs=1e-9)
        assert float(t.vx[static_idx].item()) == 0.0
        assert float(t.vy[static_idx].item()) == 0.0
        # Never mistaken for a terminated/respawned learner row.
        assert int(t.trainable[static_idx].item()) == 0
        assert int(t.active[static_idx].item()) == 1
        assert float(t.applied_effort[static_idx].item()) == -1.0
    assert saw_contact, "learner never made contact with the static opponent"


def test_static_opponent_changes_lidar_and_contact_for_learners():
    base_cfg = _cfg(num_worlds=1, max_agents_per_world=3, static=0)
    obstacle_cfg = _cfg(num_worlds=1, max_agents_per_world=3, static=2)
    atlas = make_synthetic_oval_atlas(max_agents=3)

    # Fix the probe pose from the obstacle sim's own deterministic placement,
    # then aim an identically-posed learner at that exact spot in both sims
    # so only the obstacle's presence differs between the two measurements.
    probe_sim = build_simulator(obstacle_cfg, atlas, "cpu", sync_no_respawn=False)
    probe_t = probe_sim.buffers.torch_arrays
    static_idx = int(_static_mask(probe_t).nonzero(as_tuple=False)[0].item())
    ox, oy, oyaw = (
        float(probe_t.x[static_idx]),
        float(probe_t.y[static_idx]),
        float(probe_t.yaw[static_idx]),
    )

    def _place_learner_and_measure(cfg):
        sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
        t = sim.buffers.torch_arrays
        learner_idx = int(_learner_mask(t).nonzero(as_tuple=False)[0].item())
        t.x[learner_idx] = ox - 2.0 * math.cos(oyaw)
        t.y[learner_idx] = oy - 2.0 * math.sin(oyaw)
        t.yaw[learner_idx] = oyaw
        t.vx[learner_idx] = 0.0
        t.vy[learner_idx] = 0.0
        for field in (
            "lidar_range_noise_std",
            "lidar_dropout_prob",
            "lidar_far_dropout_prob",
            "lidar_angle_bias",
            "lidar_extrinsic_x",
            "lidar_extrinsic_y",
            "lidar_extrinsic_yaw",
        ):
            getattr(t, field)[learner_idx] = 0.0
        t.lidar_sector_width[learner_idx] = 0
        sim.rebuild_sensors()
        forward_beam = float(t.sensor_obs[learner_idx, LIDAR_DIM // 2].item())

        t.x[learner_idx] = ox - 0.4 * cfg.agents.car_length_m * math.cos(oyaw)
        t.y[learner_idx] = oy - 0.4 * cfg.agents.car_length_m * math.sin(oyaw)
        t.prev_x[learner_idx] = t.x[learner_idx]
        t.prev_y[learner_idx] = t.y[learner_idx]
        t.vx[learner_idx] = 1.0
        actions = torch.zeros((t.x.shape[0], 2))
        out = sim.step(actions)
        return forward_beam, bool(out["contact"][learner_idx].item())

    beam_without, contact_without = _place_learner_and_measure(base_cfg)
    beam_with, contact_with = _place_learner_and_measure(obstacle_cfg)

    # With a static opponent in the way, the forward beam is much shorter and
    # driving into that spot registers contact; without it, neither happens.
    assert beam_with < 2.0
    assert beam_without > beam_with + 1.0
    assert contact_with
    assert not contact_without


def test_static_opponents_contribute_no_training_transitions():
    cfg = _cfg(
        num_worlds=2,
        max_agents_per_world=4,
        static=2,
        ppo={"rollout_length": 8, "minibatch_size": 16, "num_epochs": 1, "amp": False},
    )
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device="cpu")
    trainer.setup()
    t = trainer.sim.buffers.torch_arrays
    static_mask = _static_mask(t).clone()
    assert bool(static_mask.any())

    batch = trainer.collect_rollout()
    # No static slot ever contributes a valid PPO transition, at any step.
    assert not bool(batch.valid[:, static_mask].any())
    # Static slots still took the deterministic full-brake action every step.
    assert torch.all(batch.actions[:, static_mask, 0] == -1.0)
    assert torch.all(batch.actions[:, static_mask, 1] == 0.0)

    stats = trainer.ppo.update(batch, trainer.reconstruct_prepared(batch))
    assert math.isfinite(stats.policy_loss)
    assert math.isfinite(stats.value_loss)


def test_static_opponents_excluded_from_actor_forward_pass():
    """Static slots must not receive a policy forward pass at all."""
    cfg = _cfg(num_worlds=1, max_agents_per_world=3, static=2)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device="cpu")
    trainer.setup()

    calls: list[torch.Tensor] = []
    real_forward = trainer.actor.forward

    def _spy_forward(obs, cond, hidden, reset_mask=None, deterministic=False):
        calls.append(obs)
        return real_forward(
            obs, cond, hidden, reset_mask=reset_mask, deterministic=deterministic
        )

    trainer.actor.forward = _spy_forward
    trainer.collect_rollout()
    assert calls, "actor.forward was never invoked"
    total_rows = sum(int(c.shape[0]) for c in calls)
    max_possible = cfg.ppo.rollout_length * world_slot_layout(cfg).num_slots
    assert total_rows < max_possible


def test_static_opponent_style_is_never_resampled_after_placement():
    """A static slot's condition must be stable: it never re-enters a live episode."""
    cfg = _cfg(num_worlds=1, max_agents_per_world=3, static=2)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu", sync_no_respawn=False)
    t = sim.buffers.torch_arrays
    static_mask = _static_mask(t)
    before = sim.condition_tensor()[static_mask].clone()
    n = t.x.shape[0]
    actions = torch.zeros((n, 2))
    for _ in range(10):
        sim.step(actions)
    after = sim.condition_tensor()[static_mask].clone()
    assert torch.allclose(before, after)
    assert torch.all(t.reset_mask[static_mask] == 0)
    assert torch.all(t.episode_step[static_mask] == 0)

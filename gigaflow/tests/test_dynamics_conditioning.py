"""Behavioral gates for the conditioned dynamics scales (steer/accel/vmax)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from gigaflow_f1tenth.buffers import STATE_INDEX, pack_compact_agent_state
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator
from gigaflow_f1tenth.rewards import CONDITION_FIELD_NAMES, sample_private_styles
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import MAX_STEER_RAD, STEERING_DELTA_MAX_RAD
from gigaflow_f1tenth.sim.stages import physics_control_kernel
from gigaflow_f1tenth.sim.state import SimulatorBuffers
from gigaflow_f1tenth.sim.vehicle import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"

# Widest sampled range in the production configs: X(1.5).
WIDE_SCALES = (1.0 / 1.5, 1.0, 1.5)


def _drive_slots(
    field: str,
    values,
    action,
    steps: int,
    *,
    steering_action_mode: str = "delta",
):
    """Run the real physics kernel with one slot per value of ``field``.

    Every slot shares one buffer and one action, so the only difference between
    them is the field under test.
    """
    cfg = load_config(SMOKE)
    params = VehicleParams(steering_action_mode=steering_action_mode)
    buffers = SimulatorBuffers(cfg, "cpu")
    buffers.seed_defaults(params)
    sim_params = params.to_warp(
        sim_dt=cfg.agents.sim_dt, control_dt=1.0 / cfg.agents.control_hz
    )
    t = buffers.torch_arrays
    n = int(t.x.shape[0])
    assert len(values) <= n
    t.active[:] = 1
    for slot, value in enumerate(values):
        getattr(t, field)[slot] = float(value)
    actions = torch.zeros(n, 2, dtype=torch.float32)
    for slot in range(len(values)):
        actions[slot] = torch.as_tensor(action, dtype=torch.float32)
    wp_actions = wp.from_torch(actions, dtype=wp.vec2f)
    speeds: list[list[float]] = []
    steer: list[list[float]] = []
    for _ in range(steps):
        wp.launch(
            physics_control_kernel,
            dim=n,
            inputs=[
                buffers.vehicle_buffers(),
                buffers.active,
                wp_actions,
                buffers.prev_x,
                buffers.prev_y,
                buffers.prev_yaw,
                sim_params,
                int(cfg.agents.control_interval),
            ],
            device=buffers.wp_device,
        )
        speeds.append([float(t.vx[i]) for i in range(len(values))])
        steer.append([float(t.steer[i]) for i in range(len(values))])
    return speeds, steer


def test_steer_scale_gives_symmetric_authority_in_both_modes():
    """Fail-before: steer_scale was an additive trim, so full lock was one-sided."""
    for mode in ("delta", "position"):
        _, right = _drive_slots(
            "steer_scale", WIDE_SCALES, (0.0, 1.0), 12, steering_action_mode=mode
        )
        _, left = _drive_slots(
            "steer_scale", WIDE_SCALES, (0.0, -1.0), 12, steering_action_mode=mode
        )
        for slot, scale in enumerate(WIDE_SCALES):
            assert right[-1][slot] == pytest.approx(-left[-1][slot], abs=1e-6)
            assert abs(right[-1][slot]) <= MAX_STEER_RAD + 1e-6
            if mode == "delta":
                # Held lock saturates the mechanical limit whatever the gain.
                assert abs(right[-1][slot]) == pytest.approx(
                    MAX_STEER_RAD, rel=1e-3
                )
            else:
                assert abs(right[-1][slot]) == pytest.approx(
                    min(scale, 1.0) * MAX_STEER_RAD, rel=1e-3
                )


def test_steer_scale_is_a_gain_on_the_delta_command():
    """Fail-before: the production delta path ignored steer_scale entirely."""
    _, steer = _drive_slots("steer_scale", WIDE_SCALES, (0.0, 1.0), 1)
    first = steer[0]
    assert first[0] < first[1] < first[2]
    for slot, scale in enumerate(WIDE_SCALES):
        assert first[slot] == pytest.approx(
            scale * STEERING_DELTA_MAX_RAD, rel=1e-5
        )

    # A zero steering command must stay straight for every scale (no trim).
    _, centered = _drive_slots("steer_scale", WIDE_SCALES, (0.0, 0.0), 6)
    assert max(abs(v) for v in centered[-1]) == 0.0


def test_accel_scale_sets_the_effort_ramp_not_the_top_speed():
    """Fail-before: accel_scale was conditioned on but read by no kernel."""
    speeds, _ = _drive_slots("accel_scale", WIDE_SCALES, (1.0, 0.0), 40)
    early = speeds[1]
    assert early[0] < early[1] < early[2]
    assert early[2] > 1.3 * early[0]
    # Same sustained effort, so the slots converge once the ramp is done.
    late = speeds[-1]
    assert late[2] == pytest.approx(late[0], rel=0.02)


def test_vmax_scale_sets_the_attainable_speed():
    """Fail-before: vmax_scale was conditioned on but read by no kernel."""
    speeds, _ = _drive_slots("vmax_scale", WIDE_SCALES, (1.0, 0.0), 60)
    late = speeds[-1]
    assert late[0] < late[1] < late[2]
    assert late[1] > 1.1 * late[0]
    assert late[2] > 1.02 * late[1]


def test_dynamics_scales_reach_buffers_and_privileged_state():
    cfg = load_config(SMOKE)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cpu")
    n = sim.layout.num_slots
    styles = sample_private_styles(cfg, n, np.random.default_rng(0))
    sim.apply_styles(styles)
    t = sim.buffers.torch_arrays
    for name in ("steer_scale", "accel_scale", "vmax_scale"):
        expected = torch.as_tensor(
            [float(getattr(s, name)) for s in styles], dtype=torch.float32
        )
        assert torch.allclose(getattr(t, name), expected, atol=1e-6)
        assert name in CONDITION_FIELD_NAMES

    packed = pack_compact_agent_state(t)
    for name in ("accel_scale", "vmax_scale"):
        assert torch.allclose(
            packed[:, STATE_INDEX[name]], getattr(t, name), atol=1e-6
        )
    t.accel_scale.zero_()
    t.vmax_scale.zero_()
    sim.restore_state(packed)
    for name in ("accel_scale", "vmax_scale"):
        expected = torch.as_tensor(
            [float(getattr(s, name)) for s in styles], dtype=torch.float32
        )
        assert torch.allclose(getattr(t, name), expected, atol=1e-6)

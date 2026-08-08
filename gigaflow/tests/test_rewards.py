"""Focused tests for private conditioning and racing reward formulas."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from gigaflow_f1tenth import rewards as rewards_mod
from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.rewards import (
    ALPHA_PASSING_RANGE,
    CONDITION_DIM,
    CONDITION_FIELD_NAMES,
    compute_racing_rewards,
    condition_schema_fields,
    deployment_style,
    normalize_condition_vector,
    progress_delta,
    reward_collision,
    reward_lane_center,
    reward_passing_n,
    sample_private_styles,
    sample_x,
    styles_to_condition_batch,
)
from gigaflow_f1tenth.sim import reward_kernels

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"

DELETED_TERMS = ("lane_align", "reverse", "velocity", "timestep")


def test_condition_schema_dim():
    assert CONDITION_DIM == 10
    assert CONDITION_FIELD_NAMES == (
        "alpha_collision",
        "alpha_boundary",
        "alpha_l_center",
        "alpha_center_bias",
        "alpha_passing",
        "drive_scale",
        "steer_scale",
        "accel_scale",
        "vmax_scale",
        "mass_kg",
    )
    groups = [f["group"] for f in condition_schema_fields()]
    assert groups == ["reward"] * 5 + ["dynamics_estimable"] * 5


def test_deleted_reward_terms_leave_no_trace():
    """No lane-align / reverse / velocity / timestep contribution may survive."""
    for term in DELETED_TERMS:
        assert not hasattr(rewards_mod, f"reward_{term}")
        assert not any(term in name for name in CONDITION_FIELD_NAMES)
    assert not hasattr(rewards_mod, "reward_lane_align")
    assert not hasattr(rewards_mod, "reward_velocity_term")
    assert not hasattr(rewards_mod, "reward_timestep_term")
    # speed_forward was the C8 frame bug; the reverse term was its only consumer.
    assert "speed_forward" not in inspect.getsource(rewards_mod)
    assert "speed_forward" not in inspect.getsource(reward_kernels)

    cfg = load_config(SMOKE)
    for field in ("passing_k", "enable_velocity_term", "enable_active_timestep_term"):
        assert not hasattr(cfg.reward_conditioning, field)

    styles = sample_private_styles(cfg, 3, np.random.default_rng(0))
    terms = _terms(styles)
    assert set(terms.as_dict()) == {
        "total",
        "progress",
        "collision",
        "boundary",
        "lane_center",
        "passing",
    }


def _terms(styles, **kwargs):
    n = len(styles)
    base = dict(
        progress_ds=torch.zeros(n),
        speed_mps=torch.zeros(n),
        theta_f=torch.zeros(n),
        x_f_norm=torch.zeros(n),
        collision=torch.zeros(n),
        boundary=torch.zeros(n),
        styles=styles,
        dt=0.1,
    )
    base.update(kwargs)
    return compute_racing_rewards(**base)


def test_sample_x_balanced_and_in_range():
    rng = np.random.default_rng(0)
    a = 1.25
    samples = sample_x(rng, a, size=20_000)
    assert samples.min() >= 1.0 / a - 1e-6
    assert samples.max() <= a + 1e-6
    frac_below = float(np.mean(samples < 1.0))
    assert 0.45 <= frac_below <= 0.55


def test_private_style_sampling_deterministic():
    cfg = load_config(SMOKE)
    a = sample_private_styles(cfg, 8, np.random.default_rng(0))
    b = sample_private_styles(cfg, 8, np.random.default_rng(0))
    assert np.allclose(styles_to_condition_batch(a, normalize=False),
                       styles_to_condition_batch(b, normalize=False))
    vec = styles_to_condition_batch(a, normalize=True)
    assert vec.shape == (8, CONDITION_DIM)
    assert np.isfinite(vec).all()


def test_alpha_passing_sampled_over_full_range():
    cfg = load_config(SMOKE)
    styles = sample_private_styles(cfg, 20_000, np.random.default_rng(0))
    vals = np.asarray([s.alpha_passing for s in styles])
    lo, hi = ALPHA_PASSING_RANGE
    assert vals.min() >= lo
    assert vals.max() <= hi
    # U(0, 6): mean at the previous global constant, both tails reached.
    assert float(vals.mean()) == pytest.approx(3.0, abs=0.05)
    assert vals.min() < 0.05 and vals.max() > 5.95


def test_deployment_style_centered_high_collision():
    style = deployment_style("centered_high_collision")
    assert style.alpha_collision == 3.0
    assert style.alpha_center_bias == 0.0
    assert style.drive_scale == 1.0
    # Conservative deployment keeps the previous global passing aggression.
    assert style.alpha_passing == 3.0
    norm = normalize_condition_vector(style.raw_vector())
    assert norm.shape == (CONDITION_DIM,)


def test_progress_uncapped_signed_and_wrap():
    s = torch.tensor([2.0, 1.0])
    prev = torch.tensor([98.0, 5.0])
    length = torch.tensor([100.0, 100.0])
    ds = progress_delta(s, prev, length)
    # wrap forward across start/finish
    assert ds[0].item() == pytest.approx(4.0, abs=1e-5)
    # backward motion remains signed (uncapped by saturation shaping)
    assert ds[1].item() == pytest.approx(-4.0, abs=1e-5)
    reset = torch.tensor([True, False])
    ds_r = progress_delta(s, prev, length, reset_mask=reset)
    assert ds_r[0].item() == 0.0
    assert ds_r[1].item() == pytest.approx(-4.0, abs=1e-5)


def test_collision_speed_dependent():
    collision = torch.tensor([1.0, 0.0])
    speed = torch.tensor([10.0, 10.0])
    alpha = torch.tensor([1.0, 1.0])
    r = reward_collision(collision, speed, alpha)
    assert r[0].item() == pytest.approx(-(1.0 + 0.1 * 10.0))
    assert r[1].item() == 0.0


def test_passing_n_matches_1v1_and_averages():
    ego_ds = torch.tensor([0.5])
    opp_ds = torch.tensor([[0.1, 0.3]])
    ego_s = torch.tensor([5.0])
    opp_s = torch.tensor([[10.0, 12.0]])
    length = torch.tensor([100.0])
    active = torch.tensor([[True, True]])
    r, win = reward_passing_n(
        ego_ds, opp_ds, ego_s, opp_s, length, active, None, None, alpha_passing=3.0
    )
    mean_delta = 0.5 * ((0.5 - 0.1) + (0.5 - 0.3))
    assert r[0].item() == pytest.approx(3.0 * mean_delta, abs=1e-5)
    assert bool(win.all())

    # 1v1 reduces to alpha_passing * (ego_ds - opp_ds)
    r1, _ = reward_passing_n(
        ego_ds,
        torch.tensor([[0.1]]),
        ego_s,
        torch.tensor([[10.0]]),
        length,
        torch.tensor([[True]]),
        None,
        None,
        alpha_passing=3.0,
    )
    assert r1[0].item() == pytest.approx(3.0 * (0.5 - 0.1), abs=1e-5)


def test_passing_density_invariance_and_reset_gate():
    ego_ds = torch.tensor([0.4, 0.4])
    # env0: one gated opp; env1: three identical gated opps — mean must match
    opp_ds = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.1, 0.1]])
    ego_s = torch.tensor([0.0, 0.0])
    opp_s = torch.tensor([[5.0, 80.0, 90.0], [5.0, 6.0, 7.0]])
    length = torch.tensor([100.0, 100.0])
    active = torch.tensor([[True, False, False], [True, True, True]])
    r, prev = reward_passing_n(
        ego_ds, opp_ds, ego_s, opp_s, length, active, None, None, alpha_passing=3.0
    )
    assert r[0].item() == pytest.approx(r[1].item(), abs=1e-5)

    # after reset, previous gate must not keep inactive continuity
    reset = torch.tensor([True, False])
    r2, _ = reward_passing_n(
        ego_ds,
        opp_ds,
        ego_s,
        torch.tensor([[50.0, 80.0, 90.0], [50.0, 60.0, 70.0]]),
        length,
        active,
        prev,
        reset,
        alpha_passing=3.0,
    )
    assert r2[0].item() == pytest.approx(0.0, abs=1e-6)


def test_passing_gate_latches_past_the_window_until_reset():
    """Fail-before: the mirror returned in_window, so the gate unlatched early."""
    ego_ds = torch.tensor([0.4])
    opp_ds = torch.tensor([[0.1]])
    ego_s = torch.tensor([0.0])
    length = torch.tensor([200.0])
    active = torch.tensor([[True]])
    inside = torch.tensor([[10.0]])
    outside = torch.tensor([[90.0]])

    _, gate = reward_passing_n(
        ego_ds, opp_ds, ego_s, inside, length, active, None, None, alpha_passing=3.0
    )
    assert bool(gate[0, 0])

    reward, latched = reward_passing_n(
        ego_ds, opp_ds, ego_s, outside, length, active, gate, None, alpha_passing=3.0
    )
    assert bool(latched[0, 0])
    assert reward[0].item() == pytest.approx(3.0 * (0.4 - 0.1), abs=1e-5)

    _, after_reset = reward_passing_n(
        ego_ds,
        opp_ds,
        ego_s,
        outside,
        length,
        active,
        latched,
        torch.tensor([True]),
        alpha_passing=3.0,
    )
    assert not bool(after_reset[0, 0])


def test_alpha_passing_zero_disables_and_scales_linearly():
    ego_ds = torch.tensor([0.4])
    opp_ds = torch.tensor([[0.1]])
    ego_s = torch.tensor([0.0])
    opp_s = torch.tensor([[10.0]])
    length = torch.tensor([100.0])
    active = torch.tensor([[True]])

    def _r(alpha):
        r, _ = reward_passing_n(
            ego_ds,
            opp_ds,
            ego_s,
            opp_s,
            length,
            active,
            None,
            None,
            alpha_passing=torch.tensor([float(alpha)]),
        )
        return float(r[0])

    # A time-trial driver at the range floor ignores the term entirely.
    assert _r(0.0) == 0.0
    base = _r(1.0)
    assert base == pytest.approx(0.4 - 0.1, abs=1e-6)
    for alpha in (2.0, 3.0, 6.0):
        assert _r(alpha) == pytest.approx(alpha * base, abs=1e-6)


def test_alpha_passing_is_per_agent():
    ego_ds = torch.tensor([0.4, 0.4])
    opp_ds = torch.tensor([[0.1], [0.1]])
    ego_s = torch.tensor([0.0, 0.0])
    opp_s = torch.tensor([[10.0], [10.0]])
    length = torch.tensor([100.0, 100.0])
    active = torch.tensor([[True], [True]])
    r, _ = reward_passing_n(
        ego_ds,
        opp_ds,
        ego_s,
        opp_s,
        length,
        active,
        None,
        None,
        alpha_passing=torch.tensor([0.0, 6.0]),
    )
    assert float(r[0]) == 0.0
    assert float(r[1]) == pytest.approx(6.0 * 0.3, abs=1e-6)


def test_lane_center_hand_calculated_cases():
    """R = -α Δt 1_{cos θ_f > 0.5} |x_f_norm - bias|, clamped at |err| = 2."""
    dt = 0.1
    alpha = 5.0e-3
    # cos θ_f gate: 0 rad -> cos 1.0 (on); π/3 -> cos 0.5, not > 0.5 (off);
    # 2π/3 -> cos -0.5 (off). Errors: 0.4, 0.4, 0.4.
    x_f_norm = torch.tensor([0.4, 0.4, 0.4])
    theta_f = torch.tensor([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    out = reward_lane_center(
        x_f_norm, theta_f, torch.full((3,), alpha), torch.zeros(3), dt=dt
    )
    # float32 arithmetic: rel tolerance, not abs=1e-12 (below float32 resolution
    # at this magnitude).
    assert float(out[0]) == pytest.approx(-alpha * dt * 0.4, rel=1e-6)
    assert float(out[1]) == 0.0
    assert float(out[2]) == 0.0

    # Bias shifts the zero: |0.4 - 0.5| = 0.1 and |(-0.3) - 0.5| = 0.8.
    out2 = reward_lane_center(
        torch.tensor([0.4, -0.3]),
        torch.zeros(2),
        torch.full((2,), alpha),
        torch.full((2,), 0.5),
        dt=dt,
    )
    assert float(out2[0]) == pytest.approx(-alpha * dt * 0.1, rel=1e-6)
    assert float(out2[1]) == pytest.approx(-alpha * dt * 0.8, rel=1e-6)

    # Exactly at the centre the term vanishes; α = 0 disables it.
    zero = reward_lane_center(
        torch.zeros(1), torch.zeros(1), torch.full((1,), alpha), torch.zeros(1), dt=dt
    )
    assert float(zero[0]) == 0.0


def test_lane_center_is_bounded_linear_with_no_exponential():
    dt = 0.1
    alpha = 7.5e-3
    theta_f = torch.zeros(4)
    alpha_v = torch.full((4,), alpha)
    bias = torch.zeros(4)
    # Linear in the error below the cap: doubling the offset doubles the penalty.
    out = reward_lane_center(
        torch.tensor([0.25, 0.5, 1.0, 2.0]), theta_f, alpha_v, bias, dt=dt
    )
    for i, err in enumerate((0.25, 0.5, 1.0, 2.0)):
        # float32 arithmetic: rel tolerance, not abs=1e-12 (below float32
        # resolution at this magnitude).
        assert float(out[i]) == pytest.approx(-alpha * dt * err, rel=1e-6)

    # Beyond the cap the term saturates instead of growing (no explosion mode).
    spikes = reward_lane_center(
        torch.tensor([2.0, 1.0e3, 1.0e10, -1.0e10]),
        theta_f,
        alpha_v,
        bias,
        dt=dt,
    )
    assert torch.isfinite(spikes).all()
    capped = -alpha * dt * 2.0
    assert torch.allclose(spikes, torch.full((4,), capped), atol=1e-12)


def test_compute_racing_rewards_only_retained_terms():
    cfg = load_config(SMOKE)
    styles = sample_private_styles(cfg, 2, np.random.default_rng(1))
    terms = _terms(
        styles,
        progress_ds=torch.tensor([0.3, -0.1]),
        speed_mps=torch.tensor([2.0, 2.0]),
        theta_f=torch.tensor([0.0, 0.2]),
        x_f_norm=torch.tensor([0.1, -0.2]),
    )
    assert "comfort" not in terms.as_dict()
    for name in DELETED_TERMS:
        assert name not in terms.as_dict()
    assert terms.progress[0].item() == pytest.approx(0.3)
    assert terms.total.shape == (2,)
    # Total is exactly the five surviving contributions.
    expected = (
        terms.progress + terms.collision + terms.boundary
        + terms.lane_center + terms.passing
    )
    assert torch.allclose(terms.total, expected, atol=1e-7)


def test_torch_warp_reward_parity():
    """The Warp kernel and the Torch reference must agree term by term."""
    wp.init()
    n, n_others = 6, 2
    rng = np.random.default_rng(0)
    dt = 0.05

    yaw = np.array([0.0, 0.3, 1.2, 2.5, -0.4, -1.6], dtype=np.float32)
    tangent_yaw = np.array([0.0, 0.1, 0.0, 0.0, -0.2, 0.4], dtype=np.float32)
    vx = np.array([3.0, 5.0, -1.0, 0.0, 2.0, 4.0], dtype=np.float32)
    vy = np.array([0.0, 0.5, 0.2, 0.0, -0.3, 1.0], dtype=np.float32)
    frenet_ey = np.array([0.0, 0.4, -0.9, 3.0, -6.0, 0.2], dtype=np.float32)
    half_width = np.array([1.5, 1.5, 2.0, 1.0, 1.0, 2.5], dtype=np.float32)
    progress = np.array([0.4, -0.2, 0.0, 0.9, 0.1, -0.5], dtype=np.float32)
    contact = np.array([0, 1, 0, 1, 0, 0], dtype=np.uint8)
    wall = np.array([0, 0, 1, 0, 1, 0], dtype=np.uint8)
    frenet_s = np.array([0.0, 12.0, 30.0, 41.0, 55.0, 70.0], dtype=np.float32)
    track_length = np.full(n, 120.0, dtype=np.float32)
    reset_mask = np.zeros(n, dtype=np.uint8)
    opp_s = np.array(
        [[10.0, 100.0], [50.0, 13.0], [31.0, 32.0], [42.0, 43.0],
         [56.0, 57.0], [71.0, 72.0]],
        dtype=np.float32,
    )
    opp_ds = rng.uniform(-0.5, 0.5, size=(n, n_others)).astype(np.float32)
    opp_act = np.array(
        [[1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1]], dtype=np.uint8
    )
    prev_gate = np.zeros((n, n_others), dtype=np.uint8)
    alphas = {
        "alpha_collision": np.array(
            [0.0, 3.0, 1.5, 2.0, 0.5, 2.9], dtype=np.float32
        ),
        "alpha_boundary": np.array([0.0, 1.0, 3.0, 0.2, 2.0, 1.1], dtype=np.float32),
        "alpha_l_center": np.array(
            [2.5e-4, 7.5e-3, 3.8e-3, 1.0e-3, 5.0e-3, 6.0e-3], dtype=np.float32
        ),
        "alpha_center_bias": np.array(
            [0.0, 0.5, -0.5, 0.25, -0.1, 0.4], dtype=np.float32
        ),
        "alpha_passing": np.array([0.0, 6.0, 3.0, 1.0, 4.5, 2.0], dtype=np.float32),
    }

    device = "cpu"
    vehicles = reward_kernels.VehicleBuffers()
    vehicles.vx = wp.array(vx, dtype=wp.float32, device=device)
    vehicles.vy = wp.array(vy, dtype=wp.float32, device=device)
    vehicles.yaw = wp.array(yaw, dtype=wp.float32, device=device)
    out = {
        name: wp.zeros(n, dtype=wp.float32, device=device)
        for name in ("total", "progress", "collision", "boundary", "lane_center",
                     "passing")
    }
    out_gate = wp.zeros((n, n_others), dtype=wp.uint8, device=device)

    def _u8(a):
        return wp.array(a, dtype=wp.uint8, device=device)

    def _f32(a):
        return wp.array(a, dtype=wp.float32, device=device)

    wp.launch(
        reward_kernels.racing_reward_kernel,
        dim=n,
        inputs=[
            vehicles,
            _u8(np.ones(n, dtype=np.uint8)),
            _f32(progress),
            _u8(wall),
            _u8(contact),
            _f32(frenet_s),
            _f32(frenet_ey),
            _u8(reset_mask),
            _f32(track_length),
            _f32(half_width),
            _f32(tangent_yaw),
            _f32(alphas["alpha_collision"]),
            _f32(alphas["alpha_boundary"]),
            _f32(alphas["alpha_l_center"]),
            _f32(alphas["alpha_center_bias"]),
            _f32(alphas["alpha_passing"]),
            wp.array(opp_s, dtype=wp.float32, device=device),
            wp.array(opp_ds, dtype=wp.float32, device=device),
            wp.array(opp_act, dtype=wp.uint8, device=device),
            wp.array(prev_gate, dtype=wp.uint8, device=device),
            int(n_others),
            float(dt),
            out["total"],
            out["progress"],
            out["collision"],
            out["boundary"],
            out["lane_center"],
            out["passing"],
            out_gate,
        ],
        device=device,
    )
    wp.synchronize()

    style_map = {k: torch.from_numpy(v) for k, v in alphas.items()}
    theta_f = torch.from_numpy(yaw) - torch.from_numpy(tangent_yaw)
    theta_f = torch.atan2(torch.sin(theta_f), torch.cos(theta_f))
    passing, gate = reward_passing_n(
        torch.from_numpy(progress),
        torch.from_numpy(opp_ds),
        torch.from_numpy(frenet_s),
        torch.from_numpy(opp_s),
        torch.from_numpy(track_length),
        torch.from_numpy(opp_act).bool(),
        None,
        None,
        alpha_passing=style_map["alpha_passing"],
    )
    terms = compute_racing_rewards(
        progress_ds=torch.from_numpy(progress),
        speed_mps=torch.sqrt(
            torch.from_numpy(vx) ** 2 + torch.from_numpy(vy) ** 2
        ),
        theta_f=theta_f,
        x_f_norm=torch.from_numpy(frenet_ey) / torch.from_numpy(half_width),
        collision=torch.from_numpy(contact.astype(np.float32)),
        boundary=torch.from_numpy(wall.astype(np.float32)),
        styles=style_map,
        dt=dt,
        passing=passing,
    )

    assert np.array_equal(out_gate.numpy().astype(bool), gate.numpy())
    for name, tensor in terms.as_dict().items():
        assert np.allclose(out[name].numpy(), tensor.numpy(), atol=1e-6), name

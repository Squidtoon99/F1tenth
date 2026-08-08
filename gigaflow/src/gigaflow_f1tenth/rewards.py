"""Private reward/dynamics conditioning and paper-distributed racing rewards.

Racing keeps uncapped Frenet progress, collision, boundary, bounded-linear
lane-center shaping, and conditioned N-car GT Sophy-style passing; the urban
lane-align / reverse / velocity / timestep terms are deliberately absent.
Condition packing fills the fixed 10-D side channel declared in
ExperimentConfig.agents.condition_dim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gigaflow_f1tenth.config import ExperimentConfig

# Packed private-condition layout (must equal agents.condition_dim == 10):
# five reward coefficients followed by five estimable dynamics scales.
CONDITION_FIELD_NAMES: tuple[str, ...] = (
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
CONDITION_DIM = len(CONDITION_FIELD_NAMES)
assert CONDITION_DIM == 10

# Table A2 ranges (racing subset; comfort / goal / stop-line / lane-align omitted).
ALPHA_COLLISION_RANGE = (0.0, 3.0)
ALPHA_BOUNDARY_RANGE = (0.0, 3.0)
ALPHA_L_CENTER_RANGE = (2.5e-4, 7.5e-3)
ALPHA_CENTER_BIAS_RANGE = (-0.5, 0.5)
# Spans time-trial drivers at 0 through roughly twice the previous global 3.0.
ALPHA_PASSING_RANGE = (0.0, 6.0)
COLLISION_SPEED_COEF = 0.1
# Lateral error cap: far-OOB / projection spikes belong to boundary and collision.
LANE_CENTER_MAX_ERR = 2.0
REWARD_TOTAL_CLAMP = 100.0

# Estimable dynamics defaults (F1TENTH-ish); not imported from training/.
# Matches VehicleParams.mass, the physics default (sim/vehicle.py).
DEFAULT_MASS_KG = 3.74
MASS_KG_RANGE = (3.0, 4.0)

PASSING_GATE_AHEAD_M = 40.0
PASSING_GATE_BEHIND_M = 20.0
PROGRESS_MAX_STEP_FRAC = 0.05

DEPLOYMENT_STYLES: dict[str, dict[str, float]] = {
    "centered_high_collision": {
        "alpha_collision": 3.0,
        "alpha_boundary": 3.0,
        "alpha_l_center": 3.8e-3,
        "alpha_center_bias": 0.0,
        "alpha_passing": 3.0,
        "drive_scale": 1.0,
        "steer_scale": 1.0,
        "accel_scale": 1.0,
        "vmax_scale": 1.0,
        "mass_kg": DEFAULT_MASS_KG,
    },
}


@dataclass(frozen=True)
class PrivateStyle:
    """Per-episode private reward weights + estimable dynamics scales."""

    alpha_collision: float
    alpha_boundary: float
    alpha_l_center: float
    alpha_center_bias: float
    alpha_passing: float
    drive_scale: float
    steer_scale: float
    accel_scale: float
    vmax_scale: float
    mass_kg: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in CONDITION_FIELD_NAMES}

    def raw_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in CONDITION_FIELD_NAMES],
            dtype=np.float32,
        )


def condition_schema_fields() -> list[dict[str, Any]]:
    """Artifact / metadata description of the packed 10-D condition vector."""
    ranges = {
        "alpha_collision": ALPHA_COLLISION_RANGE,
        "alpha_boundary": ALPHA_BOUNDARY_RANGE,
        "alpha_l_center": ALPHA_L_CENTER_RANGE,
        "alpha_center_bias": ALPHA_CENTER_BIAS_RANGE,
        "alpha_passing": ALPHA_PASSING_RANGE,
        "drive_scale": "X(x_drive_a)",
        "steer_scale": "X(x_drive_a)",
        "accel_scale": "X(x_accel_a)",
        "vmax_scale": "X(x_accel_a)",
        "mass_kg": MASS_KG_RANGE,
    }
    groups = {
        "alpha_collision": "reward",
        "alpha_boundary": "reward",
        "alpha_l_center": "reward",
        "alpha_center_bias": "reward",
        "alpha_passing": "reward",
        "drive_scale": "dynamics_estimable",
        "steer_scale": "dynamics_estimable",
        "accel_scale": "dynamics_estimable",
        "vmax_scale": "dynamics_estimable",
        "mass_kg": "dynamics_estimable",
    }
    out = []
    for i, name in enumerate(CONDITION_FIELD_NAMES):
        out.append(
            {
                "index": i,
                "name": name,
                "group": groups[name],
                "range": ranges[name],
            }
        )
    return out


def sample_x(rng: np.random.Generator, a: float, size: int = 1) -> np.ndarray:
    """Gigaflow balanced multiplicative mix: X(a) = 0.5 U(a^-1,1) + 0.5 U(1,a)."""
    if a < 1.0:
        raise ValueError(f"X(a) requires a >= 1, got {a}")
    size = int(size)
    coin = rng.random(size)
    lo = rng.uniform(1.0 / a, 1.0, size=size)
    hi = rng.uniform(1.0, a, size=size)
    return np.where(coin < 0.5, lo, hi).astype(np.float32)


def _uniform(rng: np.random.Generator, lo: float, hi: float, size: int) -> np.ndarray:
    return rng.uniform(lo, hi, size=size).astype(np.float32)


def sample_private_styles(
    cfg: ExperimentConfig,
    num_agents: int,
    rng: np.random.Generator | None = None,
) -> list[PrivateStyle]:
    """Sample one private style per agent episode."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    n = int(num_agents)
    rc = cfg.reward_conditioning
    if not rc.enabled:
        style = deployment_style(cfg.evaluation.conservative_deployment_style)
        return [style for _ in range(n)]

    drive = sample_x(rng, rc.x_drive_a, n)
    steer = sample_x(rng, rc.x_drive_a, n)
    accel = sample_x(rng, rc.x_accel_a, n)
    vmax = sample_x(rng, rc.x_accel_a, n)
    mass = _uniform(rng, MASS_KG_RANGE[0], MASS_KG_RANGE[1], n)

    a_col = _uniform(rng, *ALPHA_COLLISION_RANGE, n)
    a_bnd = _uniform(rng, *ALPHA_BOUNDARY_RANGE, n)
    a_lc = _uniform(rng, *ALPHA_L_CENTER_RANGE, n)
    a_cb = _uniform(rng, *ALPHA_CENTER_BIAS_RANGE, n)
    a_pass = _uniform(rng, *ALPHA_PASSING_RANGE, n)

    styles = []
    for i in range(n):
        styles.append(
            PrivateStyle(
                alpha_collision=float(a_col[i]),
                alpha_boundary=float(a_bnd[i]),
                alpha_l_center=float(a_lc[i]),
                alpha_center_bias=float(a_cb[i]),
                alpha_passing=float(a_pass[i]),
                drive_scale=float(drive[i]),
                steer_scale=float(steer[i]),
                accel_scale=float(accel[i]),
                vmax_scale=float(vmax[i]),
                mass_kg=float(mass[i]),
            )
        )
    return styles


def deployment_style(name: str) -> PrivateStyle:
    if name not in DEPLOYMENT_STYLES:
        raise KeyError(f"unknown deployment style {name!r}")
    raw = DEPLOYMENT_STYLES[name]
    return PrivateStyle(**{k: float(raw[k]) for k in CONDITION_FIELD_NAMES})


def normalize_condition_vector(raw: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Map packed raw condition values into roughly [-1, 1] for the actor/critic."""
    is_torch = isinstance(raw, torch.Tensor)
    x = raw if is_torch else np.asarray(raw, dtype=np.float32)
    if x.shape[-1] != CONDITION_DIM:
        raise ValueError(f"condition dim {x.shape[-1]} != {CONDITION_DIM}")

    def _scale(val, lo, hi):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) if hi > lo else 1.0
        return (val - mid) / half

    out = []
    # Reward alphas
    out.append(_scale(x[..., 0], *ALPHA_COLLISION_RANGE))
    out.append(_scale(x[..., 1], *ALPHA_BOUNDARY_RANGE))
    out.append(_scale(x[..., 2], *ALPHA_L_CENTER_RANGE))
    out.append(_scale(x[..., 3], *ALPHA_CENTER_BIAS_RANGE))
    out.append(_scale(x[..., 4], *ALPHA_PASSING_RANGE))
    # Dynamics X(a) centered at 1; use log for multiplicative scales.
    for idx in (5, 6, 7, 8):
        if is_torch:
            out.append(torch.log(torch.clamp(x[..., idx], min=1e-6)))
        else:
            out.append(np.log(np.clip(x[..., idx], 1e-6, None)))
    out.append(_scale(x[..., 9], *MASS_KG_RANGE))

    if is_torch:
        return torch.stack(out, dim=-1)
    return np.stack(out, axis=-1).astype(np.float32)


def styles_to_condition_batch(
    styles: Sequence[PrivateStyle], *, normalize: bool = True
) -> np.ndarray:
    raw = np.stack([s.raw_vector() for s in styles], axis=0)
    return normalize_condition_vector(raw) if normalize else raw


def wrap_delta_s(ds: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    half = 0.5 * length
    ds = torch.where(ds > half, ds - length, ds)
    ds = torch.where(ds < -half, ds + length, ds)
    return ds


def progress_delta(
    s: torch.Tensor,
    prev_s: torch.Tensor,
    length: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
    max_step_frac: float = PROGRESS_MAX_STEP_FRAC,
) -> torch.Tensor:
    """Signed uncapped Frenet progress with wrap; zeros on reset rows."""
    ds = wrap_delta_s(s - prev_s, length)
    max_ds = float(max_step_frac) * length
    ds = torch.clamp(ds, min=-max_ds, max=max_ds)
    if reset_mask is not None:
        ds = torch.where(reset_mask, torch.zeros_like(ds), ds)
    return ds


def reward_collision(
    collision: torch.Tensor,
    speed_mps: torch.Tensor,
    alpha_collision: torch.Tensor,
) -> torch.Tensor:
    """R_collision = -(α_collision + 0.1 |v|) * 1_collision."""
    c = collision.to(dtype=speed_mps.dtype)
    return -(alpha_collision + COLLISION_SPEED_COEF * speed_mps.abs()) * c


def reward_boundary(
    boundary: torch.Tensor, alpha_boundary: torch.Tensor
) -> torch.Tensor:
    """R_off-road = -α_boundary * 1_boundary."""
    return -alpha_boundary * boundary.to(dtype=alpha_boundary.dtype)


def reward_lane_center(
    x_f_norm: torch.Tensor,
    theta_f: torch.Tensor,
    alpha_l_center: torch.Tensor,
    alpha_center_bias: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """R_center = -α_l_center Δt 1_{cos θ_f > 0.5} |x_f_norm - α_center_bias|.

    Width-normalized Frenet offset so α_center_bias is track-portable. Bounded
    linear only: environment variation, not a racing objective.
    """
    cos_t = torch.cos(theta_f)
    active = (cos_t > 0.5).to(dtype=x_f_norm.dtype)
    err = torch.clamp((x_f_norm - alpha_center_bias).abs(), max=LANE_CENTER_MAX_ERR)
    return -alpha_l_center * float(dt) * active * err


def reward_passing_n(
    ego_ds: torch.Tensor,
    opp_ds: torch.Tensor,
    ego_s: torch.Tensor,
    opp_s: torch.Tensor,
    length: torch.Tensor,
    opp_active: torch.Tensor,
    prev_in_window: torch.Tensor | None,
    reset_mask: torch.Tensor | None,
    alpha_passing: torch.Tensor | float,
    ahead_m: float = PASSING_GATE_AHEAD_M,
    behind_m: float = PASSING_GATE_BEHIND_M,
) -> tuple[torch.Tensor, torch.Tensor]:
    """N-car GT Sophy passing: α_passing * mean_gated(ego_ds - opp_ds).

    Shapes: ego_* [B], opp_* [B, N], opp_active [B, N], length [B] or scalar;
    ``alpha_passing`` is per-agent [B] (a scalar broadcasts). Returns
    (reward [B], gate [B, N]); the gate is latched for the episode, so feeding it
    back as ``prev_in_window`` keeps a pass credited until reset.
    """
    if opp_s.dim() == 1:
        opp_s = opp_s.unsqueeze(-1)
        opp_ds = opp_ds.unsqueeze(-1)
        opp_active = opp_active.unsqueeze(-1)
        if prev_in_window is not None and prev_in_window.dim() == 1:
            prev_in_window = prev_in_window.unsqueeze(-1)

    gap = opp_s - ego_s.unsqueeze(-1)
    if length.dim() == 0:
        length_b = length.expand(ego_s.shape[0])
    else:
        length_b = length
    half = 0.5 * length_b.unsqueeze(-1)
    gap = torch.where(gap > half, gap - length_b.unsqueeze(-1), gap)
    gap = torch.where(gap < -half, gap + length_b.unsqueeze(-1), gap)

    in_window = (gap <= float(ahead_m)) & (gap >= -float(behind_m)) & opp_active
    if prev_in_window is None:
        gate = in_window
    else:
        cont = prev_in_window & opp_active
        if reset_mask is not None:
            cont = cont & (~reset_mask.unsqueeze(-1))
        gate = in_window | cont

    delta = (ego_ds.unsqueeze(-1) - opp_ds) * gate.to(dtype=ego_ds.dtype)
    gated = gate.to(dtype=ego_ds.dtype)
    denom = gated.sum(dim=-1).clamp_min(1.0)
    mean_delta = delta.sum(dim=-1) / denom
    mean_delta = torch.where(gated.sum(dim=-1) > 0, mean_delta, torch.zeros_like(mean_delta))
    alpha = torch.as_tensor(alpha_passing, device=ego_ds.device, dtype=ego_ds.dtype)
    return alpha * mean_delta, gate


@dataclass
class RewardTerms:
    total: torch.Tensor
    progress: torch.Tensor
    collision: torch.Tensor
    boundary: torch.Tensor
    lane_center: torch.Tensor
    passing: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "progress": self.progress,
            "collision": self.collision,
            "boundary": self.boundary,
            "lane_center": self.lane_center,
            "passing": self.passing,
        }


def style_vector(
    styles: Sequence[PrivateStyle] | Mapping[str, torch.Tensor],
    name: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(styles, Mapping):
        return styles[name].to(device=device, dtype=dtype)
    return torch.tensor(
        [float(getattr(s, name)) for s in styles], device=device, dtype=dtype
    )


def style_vectors_from_list(
    styles: Sequence[PrivateStyle], device: torch.device | str, dtype=torch.float32
) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(
            [float(getattr(s, name)) for s in styles], device=device, dtype=dtype
        )
        for name in CONDITION_FIELD_NAMES
    }


def compute_rewards_from_sim_arrays(
    arrays: Any,
    *,
    styles: Sequence[PrivateStyle],
    track_length: torch.Tensor,
    half_width: torch.Tensor,
    tangent_yaw: torch.Tensor,
    dt: float,
    prev_passing_gate: torch.Tensor | None,
    max_agents_per_world: int,
) -> tuple[RewardTerms, torch.Tensor]:
    """Compose conditioned racing rewards from live simulator SoA views.

    ``track_length`` / ``half_width`` / ``tangent_yaw`` are per-slot [S].
    Returns (terms, next_passing_gate [S, N]).
    """
    device = arrays.x.device
    dtype = torch.float32
    s = int(arrays.x.shape[0])
    progress_ds = arrays.rewards.to(device=device, dtype=dtype).clone()
    speed = torch.sqrt(
        arrays.vx.to(dtype=dtype) ** 2 + arrays.vy.to(dtype=dtype) ** 2
    )
    yaw = arrays.yaw.to(dtype=dtype)
    theta_f = yaw - tangent_yaw.to(device=device, dtype=dtype)
    # wrap to [-pi, pi]
    theta_f = torch.atan2(torch.sin(theta_f), torch.cos(theta_f))
    hw = half_width.to(device=device, dtype=dtype).clamp_min(1.0e-3)
    x_f_norm = arrays.frenet_ey.to(dtype=dtype) / hw
    collision = arrays.contact.to(dtype=dtype)
    boundary = arrays.wall_contact.to(dtype=dtype)

    n_agents = int(max_agents_per_world)
    n_worlds = s // max(n_agents, 1)
    n_others = max(n_agents - 1, 0)
    ego_ds = progress_ds
    ego_s = arrays.frenet_s.to(dtype=dtype)
    length = track_length.to(device=device, dtype=dtype)
    active = arrays.active.to(dtype=torch.bool)
    reset = arrays.reset_mask.to(dtype=torch.bool)

    if n_others == 0:
        passing = torch.zeros(s, device=device, dtype=dtype)
        gate = torch.zeros(s, 0, device=device, dtype=torch.bool)
    else:
        # Vectorized same-world opponent gather (no per-slot host sync).
        s_w = ego_s.view(n_worlds, n_agents)
        ds_w = progress_ds.view(n_worlds, n_agents)
        act_w = active.view(n_worlds, n_agents)
        s_mat = s_w.unsqueeze(1).expand(n_worlds, n_agents, n_agents)
        ds_mat = ds_w.unsqueeze(1).expand(n_worlds, n_agents, n_agents)
        act_mat = act_w.unsqueeze(1).expand(n_worlds, n_agents, n_agents)
        off_diag = ~torch.eye(n_agents, dtype=torch.bool, device=device)
        opp_s = s_mat[:, off_diag].view(n_worlds, n_agents, n_others).reshape(s, n_others)
        opp_ds = ds_mat[:, off_diag].view(n_worlds, n_agents, n_others).reshape(
            s, n_others
        )
        opp_act = act_mat[:, off_diag].view(n_worlds, n_agents, n_others).reshape(
            s, n_others
        )
        prev = prev_passing_gate
        if prev is not None and prev.numel() == 0:
            prev = None
        passing, gate = reward_passing_n(
            ego_ds,
            opp_ds,
            ego_s,
            opp_s,
            length,
            opp_act,
            prev,
            reset,
            style_vector(styles, "alpha_passing", device, dtype),
        )
        # Warp mirror clears the gate row of an inactive ego slot.
        gate = gate & active.unsqueeze(-1)

    terms = compute_racing_rewards(
        progress_ds=progress_ds,
        speed_mps=speed,
        theta_f=theta_f,
        x_f_norm=x_f_norm,
        collision=collision,
        boundary=boundary,
        styles=styles,
        dt=dt,
        passing=passing,
    )
    return terms, gate


def compute_racing_rewards(
    *,
    progress_ds: torch.Tensor,
    speed_mps: torch.Tensor,
    theta_f: torch.Tensor,
    x_f_norm: torch.Tensor,
    collision: torch.Tensor,
    boundary: torch.Tensor,
    styles: Sequence[PrivateStyle] | Mapping[str, torch.Tensor],
    dt: float,
    passing: torch.Tensor | None = None,
) -> RewardTerms:
    """Compose racing rewards; progress is uncapped signed Δs."""
    device = progress_ds.device
    dtype = progress_ds.dtype
    b = progress_ds.shape[0]

    def _vec(name: str) -> torch.Tensor:
        return style_vector(styles, name, device, dtype)

    if isinstance(styles, Sequence) and len(styles) != b:
        raise ValueError(f"styles len {len(styles)} != batch {b}")

    progress = progress_ds
    col = reward_collision(collision, speed_mps, _vec("alpha_collision"))
    bnd = reward_boundary(boundary, _vec("alpha_boundary"))
    center = reward_lane_center(
        x_f_norm, theta_f, _vec("alpha_l_center"), _vec("alpha_center_bias"), dt
    )
    pas = passing if passing is not None else torch.zeros_like(progress)

    total = torch.clamp(
        progress + col + bnd + center + pas,
        min=-REWARD_TOTAL_CLAMP,
        max=REWARD_TOTAL_CLAMP,
    )
    return RewardTerms(
        total=total,
        progress=progress,
        collision=col,
        boundary=bnd,
        lane_center=center,
        passing=pas,
    )

from typing import Any

import torch

from . import runtime as rt
from .utils import compute_oob_from_boundary_state


def init_reward_state(
    reward_scales: dict[str, float],
    num_envs: int,
    device: torch.device,
) -> dict[str, Any]:
    episode_sums = {
        name: torch.zeros((num_envs,), dtype=rt.tc_float, device=device)
        for name in reward_scales.keys()
    }
    return {
        "reward_scales": reward_scales,
        "episode_sums": episode_sums,
        "last_reward_terms": {},
        "prev_s": None,
        "prev_step_counter": None,
        "last_progress_ds": torch.zeros((num_envs,), dtype=rt.tc_float, device=device),
        "prev_off_track": None,
    }


def ensure_progress_delta(
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    lap_count_buf: torch.Tensor,
) -> dict[str, Any]:
    if "progress_ds" in step_state:
        return step_state

    frenet_state = step_state["frenet"]
    s = frenet_state["s"].reshape(-1)
    length = frenet_state["L"]
    step_now = episode_steps_buf.to(dtype=rt.tc_float)
    batch = s.shape[0]

    prev_step_counter = reward_state["prev_step_counter"]
    prev_s = reward_state["prev_s"]
    if prev_step_counter is None or prev_step_counter.numel() != batch:
        prev_step_counter = step_now.detach().clone()
    if prev_s is None or prev_s.numel() != batch:
        prev_s = s.detach().clone()

    prev_step = prev_step_counter.reshape(-1)
    prev_s_flat = prev_s.reshape(-1)
    reset_mask = step_now < prev_step

    ds = s - prev_s_flat
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)

    max_step_frac = float(reward_cfg.get("progress_max_step_frac", 0.05))
    max_ds = max_step_frac * length
    ds = ds.clamp(min=-max_ds, max=max_ds)
    ds = torch.where(reset_mask, torch.zeros_like(ds), ds)

    lap_cross = (
        (~reset_mask) & (prev_s_flat > 0.9 * length) & (s < 0.1 * length) & (ds > 0.0)
    )
    lap_count_buf += lap_cross.to(dtype=lap_count_buf.dtype)
    # Per-step lap-completion events so the trainer can log an actual completion
    # count instead of the step-averaged lap_count level.
    reward_state["last_lap_cross"] = lap_cross.detach()

    reward_state["prev_s"] = s.detach().clone()
    reward_state["prev_step_counter"] = step_now.detach().clone()
    reward_state["last_progress_ds"] = ds.detach().clone()

    step_state["progress_ds"] = ds
    step_state["track_length"] = length
    return step_state


def sync_progress_state_for_resets(
    reward_state: dict[str, Any],
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reset_mask: torch.Tensor,
) -> None:
    s = step_state["frenet"]["s"].reshape(-1)

    prev_s = reward_state["prev_s"]
    if prev_s is None or prev_s.numel() != s.numel():
        reward_state["prev_s"] = s.detach().clone()
    else:
        prev_s[reset_mask] = s[reset_mask].detach()

    step_now = episode_steps_buf.to(dtype=rt.tc_float)
    prev_step_counter = reward_state["prev_step_counter"]
    if prev_step_counter is None or prev_step_counter.numel() != s.numel():
        reward_state["prev_step_counter"] = step_now.detach().clone()
    else:
        prev_step_counter[reset_mask] = step_now[reset_mask].detach()

    reward_state["last_progress_ds"][reset_mask] = 0.0

    prev_off = reward_state.get("prev_off_track")
    if prev_off is not None and prev_off.numel() == s.numel():
        prev_off[reset_mask] = False


def ensure_opp_progress_delta(
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
) -> dict[str, Any]:
    """Per-step opponent arc-length delta, mirroring ``ensure_progress_delta``.

    Tracks the opponent's previous ``s`` (and step counter) so the delta wraps the
    start/finish line, is clamped, and is zeroed on reset - exactly like the ego
    progress delta. Requires ``step_state['opp_s']`` to be present.
    """
    if "opp_progress_ds" in step_state:
        return step_state

    opp_s = step_state["opp_s"].reshape(-1)
    length = step_state["frenet"]["L"]
    step_now = episode_steps_buf.to(dtype=rt.tc_float)
    batch = opp_s.shape[0]

    prev_step_counter = reward_state.get("prev_opp_step_counter")
    prev_s = reward_state.get("prev_opp_s")
    if prev_step_counter is None or prev_step_counter.numel() != batch:
        prev_step_counter = step_now.detach().clone()
    if prev_s is None or prev_s.numel() != batch:
        prev_s = opp_s.detach().clone()

    reset_mask = step_now < prev_step_counter.reshape(-1)

    ds = opp_s - prev_s.reshape(-1)
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)

    max_step_frac = float(reward_cfg.get("progress_max_step_frac", 0.05))
    max_ds = max_step_frac * length
    ds = ds.clamp(min=-max_ds, max=max_ds)
    ds = torch.where(reset_mask, torch.zeros_like(ds), ds)

    reward_state["prev_opp_s"] = opp_s.detach().clone()
    reward_state["prev_opp_step_counter"] = step_now.detach().clone()

    step_state["opp_progress_ds"] = ds
    return step_state


def reward_passing(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
) -> torch.Tensor:
    """Reward gaining track position on the opponent: ``k * (ego_ds - opp_ds)``.

    Gated to opponents within ``[-behind_m, +ahead_m]`` on the centerline (GT
    Sophy ``1_(b,f)`` locality; current-state indicator only). Built from the
    per-step arc-length deltas of both cars, so it is naturally zeroed on reset
    (both deltas are) and never spikes at the start/finish line. Returns zeros
    when no opponent is present.
    """
    if "opp_s" not in step_state:
        return torch.zeros_like(step_state["progress_ds"])

    ensure_opp_progress_delta(
        step_state=step_state,
        episode_steps_buf=episode_steps_buf,
        reward_cfg=reward_cfg,
        reward_state=reward_state,
    )
    ego_ds = step_state["progress_ds"]
    opp_ds = step_state["opp_progress_ds"]
    k = float(reward_cfg.get("passing_k", 5.0))
    passing = k * (ego_ds - opp_ds)

    ego_s = step_state["frenet"]["s"].reshape(-1)
    opp_s = step_state["opp_s"].reshape(-1)
    length = step_state["frenet"]["L"]
    gap = opp_s - ego_s
    half = 0.5 * length
    gap = torch.where(gap > half, gap - length, gap)
    gap = torch.where(gap < -half, gap + length, gap)

    ahead_m = float(reward_cfg.get("passing_gate_ahead_m", 40.0))
    behind_m = float(reward_cfg.get("passing_gate_behind_m", 20.0))
    in_window = (gap <= ahead_m) & (gap >= -behind_m)
    return passing * in_window.to(passing.dtype)


def reward_collision(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """GT Sophy any-collision penalty ``Rc = -k * c`` for car-to-car contact.

    ``c`` is the binary overlap indicator (the same ego-frame box predicate used
    for collision termination). The penalty fires on every step the cars overlap,
    regardless of fault. Returns zeros when no opponent is present.
    """
    mask = step_state.get("car_collision")
    if mask is None:
        return torch.zeros_like(step_state["progress_ds"])
    k = float(reward_cfg.get("collision_k", 5.0))
    return -k * mask.to(dtype=step_state["progress_ds"].dtype)


def reward_rear_end(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """GT Sophy rear-end penalty ``Rr`` (Wurman et al., Nature 2022).

    ``Rr = -k * c * 1(opp ahead) * ||v_ego - v_opp||^2``: it fires only when the
    agent is in a car-car collision (same overlap predicate as ``Rc``) with an
    opponent that is ahead of it on the centerline, and scales with the squared
    closing speed (relative velocity magnitude) so high-speed rear-ends are
    punished far harder than gentle taps. Returns zeros when no opponent is
    present (1v0) or the velocity/arc-length state is unavailable.
    """
    mask = step_state.get("car_collision")
    opp_s = step_state.get("opp_s")
    opp_vel = step_state.get("opp_vel_world")
    # World-frame ego velocity (NOT body-frame base_lin_vel) so the closing speed
    # is computed in the same frame as opp_vel_world.
    ego_vel = step_state.get("ego_vel_world")
    if mask is None or opp_s is None or opp_vel is None or ego_vel is None:
        return torch.zeros_like(step_state["progress_ds"])

    ego_s = step_state["frenet"]["s"].reshape(-1)
    length = step_state["frenet"]["L"]
    gap = opp_s.reshape(-1) - ego_s
    half = 0.5 * length
    gap = torch.where(gap > half, gap - length, gap)
    gap = torch.where(gap < -half, gap + length, gap)
    opp_ahead = gap > 0.0

    rel_v = ego_vel[:, :2] - opp_vel[:, :2]
    closing_sq = (rel_v * rel_v).sum(dim=-1)

    k = float(reward_cfg.get("rear_end_k", 5.0))
    fire = mask.to(closing_sq.dtype) * opp_ahead.to(closing_sq.dtype)
    return -k * fire * closing_sq


def reward_progress(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    frenet_state = step_state["frenet"]
    ds = step_state["progress_ds"]
    pos = frenet_state["pos"].reshape(-1, 2)
    proj = frenet_state["proj"].reshape(-1, 2)

    k_fwd = float(reward_cfg.get("progress_k_fwd", 5.0))
    k_back = float(reward_cfg.get("progress_k_back", 5.0))
    fwd = torch.clamp(ds, min=0.0)
    back = torch.clamp(ds, max=0.0)
    reward = k_fwd * fwd + k_back * back

    max_lateral_m = reward_cfg.get("progress_max_lateral_m", 1.0)
    if max_lateral_m is not None:
        e_lat = torch.linalg.norm(pos - proj, dim=-1).reshape(-1)
        reward = torch.where(
            e_lat <= float(max_lateral_m), reward, torch.zeros_like(reward)
        )

    return reward


def reward_lateral(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Penalize lateral offset from centerline (track-aligned ey)."""
    ey = step_state["boundary"]["ey"].reshape(-1)
    k = float(reward_cfg.get("lateral_k", 0.5))
    return -k * ey * ey


def reward_oob_penalty(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    # GT Sophy off-course penalty: R_soc = -(time off course) * speed^2. The time
    # off course over a single control step is constant, so it folds into oob_k;
    # what remains is a penalty proportional to squared speed while off course (and
    # exactly zero while on course). This punishes fast excursions far harder than
    # slow ones and lets the boundary bind without an explicit speed cap.
    margin_m = float(reward_cfg.get("oob_margin_m", 0.5))
    k_oob = float(reward_cfg.get("oob_k", 0.15))
    oob_mask, _ = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=margin_m
    )
    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    return -k_oob * oob_mask.to(v.dtype) * v * v


def reward_speed(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """
    Forward (track-aligned) speed reward, capped at a target speed.

    Rewards velocity projected onto the track tangent (so spinning/sliding does
    not pay), normalized to [0, 1] and saturated at speed_target_mps. The cap
    keeps the agent below the unstable high-speed spin-out regime instead of
    letting it farm raw speed.
    """
    v_xy = step_state["base_lin_vel"][:, :2]
    seg_dir = step_state["frenet"]["seg_dir"]
    seg_dir = seg_dir / torch.linalg.norm(seg_dir, dim=-1, keepdim=True).clamp_min(1e-6)
    v_long = (v_xy * seg_dir).sum(dim=-1)

    target = float(reward_cfg.get("speed_target_mps", 3.0))
    return torch.clamp(v_long, min=0.0, max=target) / target


def reward_smoothness_penalty(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Penalize large action changes (jerk) to discourage bang-bang control."""
    actions = step_state.get("actions")
    last_actions = step_state.get("last_actions")
    if actions is None or last_actions is None:
        ref = step_state["boundary"]["ey"].reshape(-1)
        return torch.zeros_like(ref)
    delta = actions - last_actions
    return -torch.sum(delta * delta, dim=-1)


def reward_tyre_slip_penalty(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
) -> torch.Tensor:
    """Additive combined-slip penalty over the four tyres.

    Reads the per-wheel slip already computed by the env (single source of truth)
    and penalizes the longitudinal (slip ratio) and lateral (slip angle) channels
    additively, so pure wheelspin and pure drift are each penalized on their own.
    A per-channel deadzone leaves a controlled grip-limit regime unpenalized:

        R_ts = -( sum_i relu(min(|ratio_i|, 1) - d_ratio)
                  + w_angle * sum_i relu(|angle_i| - d_angle) )

    With both deadzones 0 this is the plain additive all-slip penalty.
    """
    slip = step_state["tyre_slip"]
    slip_ratio_mag = torch.clamp(torch.abs(slip[:, :4]), max=1.0)
    slip_angle_mag = torch.abs(slip[:, 4:])

    d_ratio = float(reward_cfg.get("slip_deadzone_ratio", 0.0))
    d_angle = float(reward_cfg.get("slip_deadzone_angle", 0.0))
    w_angle = float(reward_cfg.get("slip_angle_weight", 1.0))

    ratio_excess = torch.clamp(slip_ratio_mag - d_ratio, min=0.0)
    angle_excess = torch.clamp(slip_angle_mag - d_angle, min=0.0)

    return -(ratio_excess.sum(dim=1) + w_angle * angle_excess.sum(dim=1))


def compute_rewards(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    lap_count_buf: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    step_state = ensure_progress_delta(
        step_state=step_state,
        episode_steps_buf=episode_steps_buf,
        reward_cfg=reward_cfg,
        reward_state=reward_state,
        lap_count_buf=lap_count_buf,
    )

    num_envs = episode_steps_buf.shape[0]
    device = episode_steps_buf.device
    reward_buf = torch.zeros((num_envs,), dtype=rt.tc_float, device=device)

    progress = reward_progress(step_state, reward_cfg)
    lateral = reward_lateral(step_state, reward_cfg)
    oob_penalty = reward_oob_penalty(step_state, reward_cfg)
    tyre_slip_penalty = reward_tyre_slip_penalty(step_state, reward_cfg)
    smoothness_penalty = reward_smoothness_penalty(step_state, reward_cfg)

    # 1v1 passing reward (gated: only when a 'passing' scale is configured). Off
    # for 1v0, keeping the solo reward byte-for-byte unchanged.
    scales = reward_cfg["reward_scales"]
    passing_enabled = "passing" in scales
    if passing_enabled:
        passing = reward_passing(
            step_state, reward_cfg, reward_state, episode_steps_buf
        )

    # 1v1 collision penalty (gated like passing): GT Sophy any-collision term.
    collision_enabled = "collision" in scales
    if collision_enabled:
        collision = reward_collision(step_state, reward_cfg)

    # 1v1 rear-end penalty (gated like collision): GT Sophy Rr term.
    rear_end_enabled = "rear_end" in scales
    if rear_end_enabled:
        rear_end = reward_rear_end(step_state, reward_cfg)

    # GT Sophy masks course progress whenever the agent is off course (anti
    # corner-cutting). Derive the mask directly from the boundary state: the
    # off-course penalty is ~v^2 and goes to zero at low speed, so it can no longer
    # be used as a reliable off-track indicator.
    off_track, _ = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=float(reward_cfg.get("oob_margin_m", 0.5))
    )
    prev_off = reward_state.get("prev_off_track")
    if prev_off is None or prev_off.numel() != off_track.numel():
        prev_off = torch.zeros_like(off_track, dtype=torch.bool)
    on_track = ~off_track
    rejoin = on_track & prev_off
    progress = torch.where(off_track | rejoin, torch.zeros_like(progress), progress)
    reward_state["prev_off_track"] = off_track.detach().clone()

    progress *= scales["progress"]
    lateral *= scales.get("lateral", 0.0)
    oob_penalty *= scales["oob_penalty"]
    tyre_slip_penalty *= scales["tyre_slip_penalty"]
    smoothness_penalty *= scales.get("smoothness", 0.0)
    if passing_enabled:
        passing *= scales["passing"]
    if collision_enabled:
        collision *= scales["collision"]
    if rear_end_enabled:
        rear_end *= scales["rear_end"]

    # Single global knob to shrink overall reward magnitude (keeps the relative
    # balance between terms intact) so returns / critic targets stay O(1).
    global_scale = float(reward_cfg.get("global_reward_scale", 1.0))
    progress *= global_scale
    lateral *= global_scale
    oob_penalty *= global_scale
    tyre_slip_penalty *= global_scale
    smoothness_penalty *= global_scale
    if passing_enabled:
        passing *= global_scale
    if collision_enabled:
        collision *= global_scale
    if rear_end_enabled:
        rear_end *= global_scale

    last_terms: dict[str, torch.Tensor] = {
        "progress": progress.clone(),
        "lateral": lateral.clone(),
        "oob_penalty": oob_penalty.clone(),
        "tyre_slip_penalty": tyre_slip_penalty.clone(),
        "smoothness": smoothness_penalty.clone(),
    }

    reward_buf += progress + lateral + oob_penalty + tyre_slip_penalty + smoothness_penalty
    if passing_enabled:
        reward_buf += passing
        last_terms["passing"] = passing.clone()
    if collision_enabled:
        reward_buf += collision
        last_terms["collision"] = collision.clone()
    if rear_end_enabled:
        reward_buf += rear_end
        last_terms["rear_end"] = rear_end.clone()

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state

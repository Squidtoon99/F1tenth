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
        "prev_wall_contact": torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        ),
        "wall_impact_done": torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        ),
        "oob_term_streak": None,
        "prev_oob_term_done": None,
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

    prev_wall = reward_state.get("prev_wall_contact")
    if prev_wall is not None and prev_wall.numel() == s.numel():
        prev_wall[reset_mask] = False

    oob_streak = reward_state.get("oob_term_streak")
    if oob_streak is not None and oob_streak.numel() == s.numel():
        oob_streak[reset_mask] = 0

    prev_oob_term = reward_state.get("prev_oob_term_done")
    if prev_oob_term is not None and prev_oob_term.numel() == s.numel():
        prev_oob_term[reset_mask] = False


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
    """Raw GT Sophy passing component ``(gap_prev - gap_cur) = (ego_ds - opp_ds)``.

    Gated to opponents within ``[-behind_m, +ahead_m]`` on the centerline with the
    gate active when the opponent was in range in either the previous or current
    state (``max(gate_prev, gate_cur)``). Built from the per-step arc-length deltas
    of both cars, so it is naturally zeroed on reset (both deltas are) and never
    spikes at the start/finish line. Returns zeros when no opponent is present; the
    single coefficient is applied by ``compute_rewards``.
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
    passing = ego_ds - opp_ds

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

    step_now = episode_steps_buf.reshape(-1)
    prev_in = reward_state.get("prev_opp_in_window")
    prev_step = reward_state.get("prev_passing_step")
    if prev_in is None or prev_in.numel() != in_window.numel():
        prev_in = in_window.clone()
    if prev_step is None or prev_step.numel() != step_now.numel():
        prev_step = step_now.clone()
    reset_mask = step_now < prev_step
    gate = in_window | (prev_in & (~reset_mask))

    reward_state["prev_opp_in_window"] = in_window.detach().clone()
    reward_state["prev_passing_step"] = step_now.detach().clone()
    return passing * gate.to(passing.dtype)


def reward_overtake(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
) -> torch.Tensor:
    """One-time bonus for completing a pass: the opponent transitions from ahead
    to behind on the centerline while the two cars are close (``|gap| <
    overtake_gap_m``), so it fires on a genuine overtake, not a lap-count wrap
    (large gap) or a reset. Returns zeros when no opponent is present.
    """
    if "opp_s" not in step_state:
        return torch.zeros_like(step_state["progress_ds"])

    ego_s = step_state["frenet"]["s"].reshape(-1)
    opp_s = step_state["opp_s"].reshape(-1)
    length = step_state["frenet"]["L"]
    gap = opp_s - ego_s
    half = 0.5 * length
    gap = torch.where(gap > half, gap - length, gap)
    gap = torch.where(gap < -half, gap + length, gap)
    opp_ahead = gap > 0.0

    step_now = episode_steps_buf.reshape(-1)
    prev_ahead = reward_state.get("prev_opp_ahead")
    prev_step = reward_state.get("prev_overtake_step")
    if prev_ahead is None or prev_ahead.numel() != opp_ahead.numel():
        prev_ahead = opp_ahead.clone()
    if prev_step is None or prev_step.numel() != step_now.numel():
        prev_step = step_now.clone()
    reset_mask = step_now < prev_step

    gap_gate = float(reward_cfg.get("overtake_gap_m", 5.0))
    completed = prev_ahead & (~opp_ahead) & (gap.abs() < gap_gate) & (~reset_mask)
    k = float(reward_cfg.get("overtake_bonus_k", 1.0))

    reward_state["prev_opp_ahead"] = opp_ahead.detach().clone()
    reward_state["prev_overtake_step"] = step_now.detach().clone()
    return k * completed.to(step_state["progress_ds"].dtype)


def _cadence(reward_cfg: dict[str, Any]) -> float:
    return float(reward_cfg.get("control_dt", 0.1)) / 0.1


def reward_collision(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Raw GT Sophy any-collision component ``Rc = -cadence * c``.

    ``c`` is the binary car-to-car overlap indicator; ``cadence`` normalizes the
    per-step penalty to Sophy's 10 Hz reward rate. Returns zeros when no opponent
    is present; the single coefficient is applied by ``compute_rewards``.
    """
    mask = step_state.get("car_collision")
    if mask is None:
        return torch.zeros_like(step_state["progress_ds"])
    return -_cadence(reward_cfg) * mask.to(dtype=step_state["progress_ds"].dtype)


def reward_rear_end(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Raw GT Sophy rear-end component ``Rr`` (Wurman et al., Nature 2022).

    ``Rr = -cadence * c * 1(opp ahead) * ||v_ego - v_opp||^2``: it fires only when
    the agent is in a car-car collision (same overlap predicate as ``Rc``) with an
    opponent that is ahead of it on the centerline, and scales with the squared
    closing speed (relative velocity magnitude) so high-speed rear-ends are
    punished far harder than gentle taps. ``cadence`` normalizes to Sophy's 10 Hz
    rate. Returns zeros when no opponent is present (1v0) or the velocity/arc-length
    state is unavailable; the single coefficient is applied by ``compute_rewards``.
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

    fire = mask.to(closing_sq.dtype) * opp_ahead.to(closing_sq.dtype)
    return -_cadence(reward_cfg) * fire * closing_sq


def reward_progress(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Raw GT Sophy course-progress component ``Rcp = delta_s`` (off-course masked
    by ``compute_rewards``). The single coefficient is applied by ``compute_rewards``.
    """
    del reward_cfg
    return step_state["progress_ds"]


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
    """Raw GT Sophy off-course component ``Rsoc = -elapsed * (3.6 * speed)^2``.

    ``elapsed`` is the control-step duration (constant per step) and the speed is
    expressed in km/h, so a fast excursion is punished far harder than a slow one
    and the penalty is exactly zero while on course. The single coefficient is
    applied by ``compute_rewards``.
    """
    margin_m = float(reward_cfg.get("oob_margin_m", 0.2))
    control_dt = float(reward_cfg.get("control_dt", 0.1))
    oob_mask, _ = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=margin_m
    )
    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    speed_kmh = 3.6 * v
    return -control_dt * oob_mask.to(v.dtype) * speed_kmh * speed_kmh


def wall_contact_from_boundary(
    boundary_state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Footprint contact with the actual corridor boundary (no reward margin)."""
    ey = boundary_state["ey"].reshape(-1)
    w_l_s = boundary_state["w_l_s"].reshape(-1)
    w_r_s = boundary_state["w_r_s"].reshape(-1)
    half = boundary_state.get("oob_half_extent_m", 0.0)
    if not torch.is_tensor(half):
        half = torch.full_like(ey, float(half))
    else:
        half = half.reshape(-1).to(dtype=ey.dtype, device=ey.device)
    left = (ey + half) >= w_l_s
    right = (ey - half) <= -w_r_s
    return left | right, left, right


def wall_normal_speed(
    step_state: dict[str, Any],
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
) -> torch.Tensor:
    """Inward speed into the contacted boundary normal (0 when separating)."""
    vel = step_state.get("ego_vel_world")
    if vel is None:
        vel = step_state["base_lin_vel"]
    vel_xy = vel[:, :2]
    t_hat = step_state["frenet"]["seg_dir"]
    n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)
    v_lat = (vel_xy * n_hat).sum(dim=-1)
    into_left = torch.clamp(v_lat, min=0.0)
    into_right = torch.clamp(-v_lat, min=0.0)
    v_normal = torch.zeros_like(v_lat)
    v_normal = torch.where(left_contact, into_left, v_normal)
    v_normal = torch.where(right_contact, torch.maximum(v_normal, into_right), v_normal)
    return v_normal


def reward_wall_penalty(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    """Raw continuous wall term ``Rw = -elapsed * (3.6 * speed)^2`` while contacting."""
    control_dt = float(reward_cfg.get("control_dt", 0.1))
    contact, _, _ = wall_contact_from_boundary(step_state["boundary"])
    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    speed_kmh = 3.6 * v
    return -control_dt * contact.to(v.dtype) * speed_kmh * speed_kmh


def reward_oob_impact(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
) -> torch.Tensor:
    """One-shot terminal OOB impact ``-speed`` when OOB termination fires."""
    term_margin = float(reward_cfg.get("term_oob_margin_m", 0.15))
    max_consecutive = int(reward_cfg.get("term_oob_max_consecutive", 3))

    boundary = step_state["boundary"]
    ey = boundary["ey"].reshape(-1)
    w_l = boundary["w_l_s"].reshape(-1)
    w_r = boundary["w_r_s"].reshape(-1)
    batch = ey.shape[0]
    severe_oob = (ey > w_l - term_margin) | (ey < -(w_r - term_margin))

    streak = reward_state.get("oob_term_streak")
    if streak is None or streak.numel() != batch:
        streak = torch.zeros((batch,), dtype=torch.int32, device=ey.device)
    streak = torch.where(
        severe_oob,
        streak + 1,
        torch.zeros_like(streak),
    )
    reward_state["oob_term_streak"] = streak

    center_penetration = (ey >= w_l) | (ey <= -w_r)
    oob_done = (streak >= max_consecutive) | center_penetration

    prev_done = reward_state.get("prev_oob_term_done")
    if prev_done is None or prev_done.numel() != batch:
        prev_done = torch.zeros((batch,), dtype=torch.bool, device=ey.device)
    first_term = oob_done & (~prev_done)
    reward_state["prev_oob_term_done"] = oob_done.detach().clone()

    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    return -v * first_term.to(v.dtype)


def reward_wall_impact(
    step_state: dict[str, Any],
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
) -> torch.Tensor:
    """One-shot first-contact impact ``-v_normal^2``; latches until reset/leave."""
    contact, left, right = wall_contact_from_boundary(step_state["boundary"])
    v_normal = wall_normal_speed(step_state, left, right)
    prev = reward_state.get("prev_wall_contact")
    if prev is None or prev.numel() != contact.numel():
        prev = torch.zeros_like(contact)
    first = contact & (~prev)
    impact = -v_normal * v_normal * first.to(v_normal.dtype)

    term_speed = float(reward_cfg.get("wall_impact_term_speed_mps", 4.0))
    reward_state["wall_impact_done"] = contact & (v_normal >= term_speed)
    reward_state["prev_wall_contact"] = contact.detach().clone()
    reward_state["last_wall_contact"] = contact.detach().clone()
    reward_state["last_v_normal"] = v_normal.detach().clone()
    return impact


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
    """Raw GT Sophy tyre-slip component ``Rts = -cadence * sum_i min(|ratio_i|, 1) * |angle_i|``.

    Reads the per-wheel slip already computed by the env (single source of truth):
    the longitudinal slip ratio (clamped at 1 so a spinning wheel cannot dominate)
    multiplied by the lateral slip angle, summed over the four tyres. ``cadence``
    normalizes to Sophy's 10 Hz rate. The single coefficient is applied by
    ``compute_rewards``.
    """
    slip = step_state["tyre_slip"]
    slip_ratio_mag = torch.clamp(torch.abs(slip[:, :4]), max=1.0)
    slip_angle_mag = torch.abs(slip[:, 4:])
    product = (slip_ratio_mag * slip_angle_mag).sum(dim=1)
    return -_cadence(reward_cfg) * product


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

    # 1v1 overtake-completed bonus (gated): one-time reward for finishing a pass.
    overtake_enabled = "overtake" in scales
    if overtake_enabled:
        overtake = reward_overtake(
            step_state, reward_cfg, reward_state, episode_steps_buf
        )

    oob_impact_enabled = "oob_impact" in scales
    if oob_impact_enabled:
        oob_impact = reward_oob_impact(step_state, reward_cfg, reward_state)

    wall_enabled = "wall_penalty" in scales or "wall_impact" in scales
    if wall_enabled:
        wall_penalty = reward_wall_penalty(step_state, reward_cfg)
        wall_impact = reward_wall_impact(step_state, reward_cfg, reward_state)
    else:
        reward_state["wall_impact_done"] = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
        )

    # GT Sophy masks course progress whenever the agent is off course (anti
    # corner-cutting). Derive the mask directly from the boundary state; solid
    # walls prevent shortcutting so only the off-course mask (no lateral/rejoin
    # shaping) is applied.
    off_track, _ = compute_oob_from_boundary_state(
        step_state["boundary"], margin_m=float(reward_cfg.get("oob_margin_m", 0.2))
    )
    progress = torch.where(off_track, torch.zeros_like(progress), progress)

    progress *= scales["progress"]
    lateral *= scales.get("lateral", 0.0)
    oob_penalty *= scales["oob_penalty"]
    if oob_impact_enabled:
        oob_impact *= scales["oob_impact"]
    tyre_slip_penalty *= scales["tyre_slip_penalty"]
    smoothness_penalty *= scales.get("smoothness", 0.0)
    if passing_enabled:
        passing *= scales["passing"]
    if collision_enabled:
        collision *= scales["collision"]
    if rear_end_enabled:
        rear_end *= scales["rear_end"]
    if overtake_enabled:
        overtake *= scales["overtake"]
    if wall_enabled:
        wall_penalty *= scales.get("wall_penalty", 0.0)
        wall_impact *= scales.get("wall_impact", 0.0)

    # Single global knob to shrink overall reward magnitude (keeps the relative
    # balance between terms intact) so returns / critic targets stay O(1).
    global_scale = float(reward_cfg.get("global_reward_scale", 1.0))
    progress *= global_scale
    lateral *= global_scale
    oob_penalty *= global_scale
    if oob_impact_enabled:
        oob_impact *= global_scale
    tyre_slip_penalty *= global_scale
    smoothness_penalty *= global_scale
    if passing_enabled:
        passing *= global_scale
    if collision_enabled:
        collision *= global_scale
    if rear_end_enabled:
        rear_end *= global_scale
    if overtake_enabled:
        overtake *= global_scale
    if wall_enabled:
        wall_penalty *= global_scale
        wall_impact *= global_scale

    last_terms: dict[str, torch.Tensor] = {
        "progress": progress.clone(),
        "lateral": lateral.clone(),
        "oob_penalty": oob_penalty.clone(),
        "tyre_slip_penalty": tyre_slip_penalty.clone(),
        "smoothness": smoothness_penalty.clone(),
    }

    reward_buf += progress + lateral + oob_penalty + tyre_slip_penalty + smoothness_penalty
    if oob_impact_enabled:
        reward_buf += oob_impact
        last_terms["oob_impact"] = oob_impact.clone()
    if passing_enabled:
        reward_buf += passing
        last_terms["passing"] = passing.clone()
    if collision_enabled:
        reward_buf += collision
        last_terms["collision"] = collision.clone()
    if rear_end_enabled:
        reward_buf += rear_end
        last_terms["rear_end"] = rear_end.clone()
    if overtake_enabled:
        reward_buf += overtake
        last_terms["overtake"] = overtake.clone()
    if wall_enabled:
        reward_buf += wall_penalty + wall_impact
        last_terms["wall_penalty"] = wall_penalty.clone()
        last_terms["wall_impact"] = wall_impact.clone()

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state

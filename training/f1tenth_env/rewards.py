from typing import Any

import torch

from . import runtime as rt


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


def ensure_opp_progress_delta(
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    reward_cfg: dict[str, Any],
    reward_state: dict[str, Any],
) -> dict[str, Any]:
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
    mask = step_state.get("car_collision")
    opp_s = step_state.get("opp_s")
    opp_vel = step_state.get("opp_vel_world")
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


def _shape_forward_progress_ds(
    ds: torch.Tensor, reward_cfg: dict[str, Any]
) -> torch.Tensor:
    control_dt = float(reward_cfg.get("control_dt", 0.1))
    ds_thr = float(reward_cfg.get("progress_speed_threshold_mps", 0.0)) * control_dt
    multiplier = float(reward_cfg.get("progress_high_speed_multiplier", 1.0))
    saturation_mps = float(reward_cfg.get("progress_speed_saturation_mps", 0.0))
    ds_sat = saturation_mps * control_dt
    forward = torch.clamp(ds, min=0.0)
    if saturation_mps > 0.0:
        boosted = ds_thr + multiplier * (forward - ds_thr)
        saturated = ds_thr + multiplier * (ds_sat - ds_thr) + (forward - ds_sat)
        return torch.where(
            forward <= ds_thr,
            forward,
            torch.where(forward > ds_sat, saturated, boosted),
        )
    boosted = ds_thr + multiplier * (forward - ds_thr)
    return torch.where(forward <= ds_thr, forward, boosted)


def reward_progress(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    return _shape_forward_progress_ds(step_state["progress_ds"], reward_cfg)


def reward_wall_contact(
    step_state: dict[str, Any], reward_cfg: dict[str, Any]
) -> torch.Tensor:
    v = torch.linalg.norm(step_state["base_lin_vel"][:, :2], dim=-1)
    contact = step_state["wall_contact"].to(v.dtype)
    coefficient = float(reward_cfg["wall_contact_coefficient"])
    return -coefficient * contact * v


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
    scales = reward_cfg["reward_scales"]
    global_scale = float(reward_cfg.get("global_reward_scale", 1.0))

    progress = reward_progress(step_state, reward_cfg)
    wall_contact = reward_wall_contact(step_state, reward_cfg)
    progress = torch.where(
        step_state["wall_contact"], torch.zeros_like(progress), progress
    )

    progress = progress * float(scales.get("progress", 1.0)) * global_scale
    wall_contact = wall_contact * global_scale
    reward_buf = reward_buf + progress + wall_contact
    last_terms: dict[str, torch.Tensor] = {
        "progress": progress.clone(),
        "wall_contact": wall_contact.clone(),
    }

    if "passing" in scales:
        passing = reward_passing(
            step_state, reward_cfg, reward_state, episode_steps_buf
        )
        passing = passing * float(scales["passing"]) * global_scale
        reward_buf = reward_buf + passing
        last_terms["passing"] = passing.clone()
    if "collision" in scales:
        collision = reward_collision(step_state, reward_cfg)
        collision = collision * float(scales["collision"]) * global_scale
        reward_buf = reward_buf + collision
        last_terms["collision"] = collision.clone()
    if "rear_end" in scales:
        rear_end = reward_rear_end(step_state, reward_cfg)
        rear_end = rear_end * float(scales["rear_end"]) * global_scale
        reward_buf = reward_buf + rear_end
        last_terms["rear_end"] = rear_end.clone()

    reward_state["last_reward_terms"] = last_terms
    return reward_buf, step_state

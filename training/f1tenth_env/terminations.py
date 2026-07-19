import math
from typing import Any

import torch

from . import runtime as rt
from .geom import quat_to_xyz
from .utils import compute_oob_from_boundary_state


def init_termination_params(env_cfg: dict[str, Any], dt: float) -> dict[str, Any]:
    term_not_moving_time_s = float(env_cfg.get("term_not_moving_time_s", 2.0))
    return {
        "term_oob_margin_m": float(env_cfg.get("term_oob_margin_m", 0.15)),
        "term_oob_max_consecutive": int(env_cfg.get("term_oob_max_consecutive", 15)),
        "term_speed_threshold": float(env_cfg.get("term_speed_threshold", 0.2)),
        "term_not_moving_time_s": term_not_moving_time_s,
        "term_not_moving_min_ds": float(env_cfg.get("term_not_moving_min_ds", 1e-3)),
        "term_heading_error_rad": float(env_cfg.get("term_heading_error_rad", math.pi)),
        "not_moving_steps_threshold": max(
            1, int(math.ceil(term_not_moving_time_s / dt))
        ),
    }


def init_termination_state(
    num_envs: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "oob_consecutive_buf": torch.zeros(
            (num_envs,), dtype=torch.int32, device=device
        ),
        "not_moving_steps_buf": torch.zeros(
            (num_envs,), dtype=torch.int32, device=device
        ),
    }


def reset_termination_state(
    term_state: dict[str, torch.Tensor], reset_mask: torch.Tensor
) -> None:
    term_state["oob_consecutive_buf"].masked_fill_(reset_mask, 0)
    term_state["not_moving_steps_buf"].masked_fill_(reset_mask, 0)


def invalid_state_mask(
    step_state: dict[str, Any],
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    term_heading_error_rad: float,
) -> torch.Tensor:
    finite_ok = (
        torch.isfinite(base_pos).all(dim=1)
        & torch.isfinite(base_quat).all(dim=1)
        & torch.isfinite(base_lin_vel).all(dim=1)
        & torch.isfinite(base_ang_vel).all(dim=1)
    )

    cached = step_state.get("centerline_angle")
    if cached is not None:
        heading_err = cached.squeeze(-1)
    else:
        track_angle = torch.atan2(
            step_state["frenet"]["seg_dir"][:, 1], step_state["frenet"]["seg_dir"][:, 0]
        )
        yaw = quat_to_xyz(base_quat, rpy=True, degrees=False)[:, 2]
        heading_err = yaw - track_angle
        heading_err = torch.atan2(torch.sin(heading_err), torch.cos(heading_err))

    heading_bad = torch.abs(heading_err) > term_heading_error_rad
    return (~finite_ok) | heading_bad


def _obb_axes(yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit heading (+x) and left (+y) axes for each yaw, shape (B, 2)."""
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    heading = torch.stack([cos, sin], dim=-1)
    left = torch.stack([-sin, cos], dim=-1)
    return heading, left


def obb_overlap_mtv(
    pa: torch.Tensor,
    ya: torch.Tensor,
    pb: torch.Tensor,
    yb: torch.Tensor,
    car_length: float,
    car_width: float,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oriented-box overlap between two equal cars via the separating-axis test.

    Each car is a ``car_length`` x ``car_width`` rectangle centred on its position
    and rotated by its own yaw. Returns ``(overlap, normal, depth)`` where
    ``normal`` (B, 2) is the minimum-translation direction pointing from b toward a
    and ``depth`` (B,) its positive penetration; both are zero where the boxes are
    apart. ``margin`` inflates the combined extent (additive slack). With equal
    headings this reduces to the ego-frame box test.
    """
    hl = 0.5 * float(car_length)
    hw = 0.5 * float(car_width)
    ax_a, ay_a = _obb_axes(ya)
    ax_b, ay_b = _obb_axes(yb)
    d = pa - pb

    separated = torch.zeros(pa.shape[0], dtype=torch.bool, device=pa.device)
    best_depth = torch.full(
        (pa.shape[0],), float("inf"), device=pa.device, dtype=pa.dtype
    )
    best_normal = torch.zeros_like(pa)
    for axis in (ax_a, ay_a, ax_b, ay_b):
        proj = (d * axis).sum(dim=-1)
        rad_a = hl * (ax_a * axis).sum(-1).abs() + hw * (ay_a * axis).sum(-1).abs()
        rad_b = hl * (ax_b * axis).sum(-1).abs() + hw * (ay_b * axis).sum(-1).abs()
        overlap_amt = (rad_a + rad_b + float(margin)) - proj.abs()
        separated = separated | (overlap_amt <= 0)
        sign = torch.where(proj >= 0, torch.ones_like(proj), -torch.ones_like(proj))
        cand_normal = axis * sign.unsqueeze(-1)
        take = overlap_amt < best_depth
        best_depth = torch.where(take, overlap_amt, best_depth)
        best_normal = torch.where(take.unsqueeze(-1), cand_normal, best_normal)

    overlap = ~separated
    depth = torch.where(
        overlap, best_depth.clamp_min(0.0), torch.zeros_like(best_depth)
    )
    normal = torch.where(
        overlap.unsqueeze(-1), best_normal, torch.zeros_like(best_normal)
    )
    return overlap, normal, depth


def collision_mask(
    ego_pos_xy: torch.Tensor,
    opp_pos_xy: torch.Tensor,
    ego_yaw: torch.Tensor,
    opp_yaw: torch.Tensor,
    car_length: float,
    car_width: float,
    collision_margin_m: float = 0.0,
) -> torch.Tensor:
    """1v1 collision predicate: oriented-box (OBB) overlap for two equal cars.

    Each car is a ``car_length`` x ``car_width`` rectangle rotated by its own yaw;
    ``collision_margin_m`` inflates the combined extent. Reduces to the ego-frame
    box test when both cars share a heading.
    """
    overlap, _, _ = obb_overlap_mtv(
        ego_pos_xy,
        ego_yaw,
        opp_pos_xy,
        opp_yaw,
        car_length,
        car_width,
        collision_margin_m,
    )
    return overlap


def compute_terminations(
    step_state: dict[str, Any],
    episode_steps_buf: torch.Tensor,
    max_episode_steps: int,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    term_state: dict[str, torch.Tensor],
    term_params: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    time_out = episode_steps_buf >= max_episode_steps

    oob, _ = compute_oob_from_boundary_state(
        step_state["boundary"],
        margin_m=term_params["term_oob_margin_m"],
    )

    term_state["oob_consecutive_buf"] = torch.where(
        oob,
        term_state["oob_consecutive_buf"] + 1,
        torch.zeros_like(term_state["oob_consecutive_buf"]),
    )
    out_of_bounds = (
        term_state["oob_consecutive_buf"] >= term_params["term_oob_max_consecutive"]
    )
    boundary = step_state["boundary"]
    center_penetration = (
        boundary["ey"] >= boundary["w_l_s"]
    ) | (
        boundary["ey"] <= -boundary["w_r_s"]
    )
    out_of_bounds = out_of_bounds | center_penetration

    speed_xy = torch.linalg.norm(base_lin_vel[:, :2], dim=-1)
    ds = step_state.get("progress_ds")
    if ds is None:
        ds = torch.zeros_like(speed_xy)

    not_moving_now = (speed_xy < term_params["term_speed_threshold"]) & (
        torch.abs(ds) < term_params["term_not_moving_min_ds"]
    )
    term_state["not_moving_steps_buf"] = torch.where(
        not_moving_now,
        term_state["not_moving_steps_buf"] + 1,
        torch.zeros_like(term_state["not_moving_steps_buf"]),
    )
    not_moving = (
        term_state["not_moving_steps_buf"] >= term_params["not_moving_steps_threshold"]
    )

    invalid_state = invalid_state_mask(
        step_state=step_state,
        base_pos=base_pos,
        base_quat=base_quat,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        term_heading_error_rad=term_params["term_heading_error_rad"],
    )

    reset = time_out | out_of_bounds | not_moving | invalid_state

    termination_extras = {
        "time_out": time_out.to(dtype=rt.tc_float),
        "out_of_bounds": out_of_bounds.to(dtype=rt.tc_float),
        "not_moving": not_moving.to(dtype=rt.tc_float),
        "invalid_state": invalid_state.to(dtype=rt.tc_float),
    }
    return reset, termination_extras, time_out.to(dtype=rt.tc_float)

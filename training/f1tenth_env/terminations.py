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


def collision_mask(
    ego_pos_xy: torch.Tensor,
    opp_pos_xy: torch.Tensor,
    ego_yaw: torch.Tensor,
    car_length: float,
    car_width: float,
    collision_margin_m: float = 0.0,
) -> torch.Tensor:
    """1v1 collision predicate: axis-aligned boxes in the ego heading frame.

    Both cars are treated as rectangles aligned with the ego heading (cheap
    interim, not full OBB-OBB).  Collision when longitudinal and lateral
    separations are both below their respective half-extent sums plus margin.
    """
    dp = opp_pos_xy - ego_pos_xy
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    d_long = dp[:, 0] * cos_yaw + dp[:, 1] * sin_yaw
    d_lat = -dp[:, 0] * sin_yaw + dp[:, 1] * cos_yaw

    long_thresh = float(car_length) + float(collision_margin_m)
    lat_thresh = float(car_width) + float(collision_margin_m)
    return (torch.abs(d_long) < long_thresh) & (torch.abs(d_lat) < lat_thresh)


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

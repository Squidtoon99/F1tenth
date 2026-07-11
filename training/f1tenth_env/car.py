"""Vehicle observation helpers."""

from __future__ import annotations

import torch

from . import geom as gu


def compute_tyre_slip(
    wheel_state: dict[str, torch.Tensor],
    wheel_radius: float,
    active: torch.Tensor | None = None,
    min_lat: float = 0.2,
    min_active_long: float = 0.1,
    min_passive_long: float = 0.4,
) -> torch.Tensor:
    """Compute per-wheel slip ratios and angles in LR, RR, LF, RF order."""
    lin_vel = wheel_state["motion_link_vel"]
    frame_quat = wheel_state.get("frame_quat")
    if frame_quat is not None:
        lin_vel_local = gu.inv_transform_by_quat(lin_vel, frame_quat)
    else:
        lin_vel_local = lin_vel
    spin_rate = wheel_state["dof_vel"]

    v_fwd = lin_vel_local[:, :, 0]
    v_lat = lin_vel_local[:, :, 1]
    v_fwd_abs = torch.abs(v_fwd)
    slip_angle = torch.atan(v_lat / (v_fwd_abs + min_lat))

    wheel_speed = wheel_radius * spin_rate
    if active is None:
        min_long = min_active_long
    else:
        active_batch = active.reshape(active.shape[0], -1).to(
            device=v_fwd.device
        ).bool()
        min_long = torch.where(
            active_batch,
            v_fwd.new_full((), min_active_long),
            v_fwd.new_full((), min_passive_long),
        )
    slip_ratio = (wheel_speed - v_fwd) / (v_fwd_abs + min_long)
    return torch.cat([slip_ratio, slip_angle], dim=-1)

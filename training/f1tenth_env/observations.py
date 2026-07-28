import math
from typing import Any

import torch

from . import runtime as rt
from .geom import quat_to_xyz


def obs_track_progress(
    step_state: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    frenet = step_state["frenet"]
    track_len = frenet["L"].clamp(min=1e-6)
    progress_ratio = frenet["s"] / track_len
    angle = (2.0 * math.pi) * progress_ratio
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def obs_centerline_angle(
    step_state: dict[str, Any],
    base_quat: torch.Tensor,
) -> torch.Tensor:
    cached = step_state.get("centerline_angle")
    if cached is not None:
        return cached

    frenet_state = step_state["frenet"]
    track_angle = torch.atan2(
        frenet_state["seg_dir"][:, 1], frenet_state["seg_dir"][:, 0]
    )

    euler_xyz = quat_to_xyz(base_quat, rpy=True, degrees=False)
    yaw = euler_xyz[:, 2]

    theta_err = yaw - track_angle
    theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))
    out = theta_err.unsqueeze(-1)
    step_state["centerline_angle"] = out
    return out


def obs_centerline_distance(
    step_state: dict[str, Any],
) -> torch.Tensor:
    return step_state["boundary"]["ey"].unsqueeze(-1)


def obs_contact_flag(
    step_state: dict[str, Any], obs_cfg: dict[str, Any]
) -> torch.Tensor:
    boundary_dist = step_state["boundary"]["boundary_dist"]
    contact_margin = float(obs_cfg.get("contact_margin_m", 0.08))
    return (boundary_dist < contact_margin).float().unsqueeze(-1)


def obs_future_track_points(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    obs_cfg: dict[str, Any],
    device: torch.device,
    step_state: dict[str, Any],
) -> torch.Tensor:
    obs_track = step_state["obs_track"]
    centerline_t = obs_track["centerline_t"]
    seg_len = obs_track["seg_len"]
    cumlen = obs_track["cumlen"]
    n = obs_track["n"]

    robot_pos = base_pos[:, :2]
    lin_vel = base_lin_vel[:, :2]
    yaw = quat_to_xyz(base_quat, rpy=True, degrees=False)[:, 2]

    batch = robot_pos.shape[0]
    samples = int(obs_cfg.get("future_track_num_points", 60))
    horizon_s = float(obs_cfg.get("future_track_horizon_s", 6.0))

    frenet = step_state["frenet"]
    s0 = frenet["s"]
    total_len = frenet["L"].clamp(min=1e-6)

    min_lookahead = float(obs_cfg.get("future_track_min_lookahead_m", 5.0))
    speed = torch.linalg.vector_norm(lin_vel, dim=-1)
    lookahead = torch.clamp(speed * horizon_s, min=min_lookahead)

    steps = torch.arange(1, samples + 1, device=device, dtype=rt.tc_float) / samples
    s_targets = s0.unsqueeze(1) + lookahead.unsqueeze(1) * steps.unsqueeze(0)
    s_targets = torch.remainder(s_targets, total_len)

    seg_idx = torch.searchsorted(cumlen, s_targets, right=True) - 1
    seg_idx = seg_idx.clamp(min=0, max=n - 2)

    seg_idx_flat = seg_idx.reshape(-1)
    p0 = centerline_t[seg_idx_flat]
    p1 = centerline_t[seg_idx_flat + 1]

    seg_len_sel = seg_len[seg_idx_flat].clamp(min=1e-8)
    s_base = cumlen[seg_idx_flat]
    alpha = ((s_targets.reshape(-1) - s_base) / seg_len_sel).unsqueeze(-1)

    center_pts = p0 + alpha * (p1 - p0)
    tangents = (p1 - p0) / seg_len_sel.unsqueeze(-1)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)

    w_tr_left = obs_track["w_tr_left"]
    w_tr_right = obs_track["w_tr_right"]
    alpha_t = alpha.squeeze(-1)
    w_l = w_tr_left[seg_idx_flat] + alpha_t * (
        w_tr_left[seg_idx_flat + 1] - w_tr_left[seg_idx_flat]
    )
    w_r = w_tr_right[seg_idx_flat] + alpha_t * (
        w_tr_right[seg_idx_flat + 1] - w_tr_right[seg_idx_flat]
    )
    left_pts = center_pts + w_l.unsqueeze(-1) * normals
    right_pts = center_pts - w_r.unsqueeze(-1) * normals

    center_pts = center_pts.view(batch, samples, 2)
    left_pts = left_pts.view(batch, samples, 2)
    right_pts = right_pts.view(batch, samples, 2)

    cos_y = torch.cos(yaw).view(batch, 1, 1)
    sin_y = torch.sin(yaw).view(batch, 1, 1)

    def world_to_ego(points: torch.Tensor) -> torch.Tensor:
        d = points - robot_pos.unsqueeze(1)
        x = d[..., 0:1]
        y = d[..., 1:2]
        x_p = cos_y * x + sin_y * y
        y_p = -sin_y * x + cos_y * y
        return torch.cat([x_p, y_p], dim=-1)

    center_ego = world_to_ego(center_pts)
    left_ego = world_to_ego(left_pts)
    right_ego = world_to_ego(right_pts)

    all_ego = torch.stack([center_ego, left_ego, right_ego], dim=1)
    return all_ego.reshape(batch, -1)


def obs_opponent(
    self_agent: dict[str, torch.Tensor],
    other_agent: dict[str, torch.Tensor],
    obs_cfg: dict[str, Any],
) -> torch.Tensor:
    """Symmetric opponent-relative observation block (8 dims).

    Built from "self"'s ego frame with "other" as the opponent. Components:

    0,1  other position relative to self, rotated into self's body frame (m)
    2,3  other velocity relative to self, rotated into self's body frame (m/s)
    4,5  other acceleration relative to self, rotated into self's body frame
         (m/s^2). Body-frame ``ax,ay`` of each vehicle are rotated to world,
         subtracted, then rotated into self's frame.
    6    signed along-track gap ``s_other - s_self`` wrapped to ``[-L/2, L/2]`` and
         normalized by ``L/2`` (positive => other ahead)
    7    other's signed lateral offset from the centerline ``ey_other`` (m)

    Masking (range gate in training, detection certainty on deploy) is applied by
    the caller; an all-zero block is the sole "no relevant opponent" sentinel.
    """
    pos_s = self_agent["pos_xy"]
    yaw_s = self_agent["yaw"].reshape(-1)
    vel_s = self_agent["vel_xy"]
    acc_s = self_agent["acc_xy"]
    s_s = self_agent["s"].reshape(-1)

    pos_o = other_agent["pos_xy"]
    yaw_o = other_agent["yaw"].reshape(-1)
    vel_o = other_agent["vel_xy"]
    acc_o = other_agent["acc_xy"]
    s_o = other_agent["s"].reshape(-1)
    ey_o = other_agent["ey"].reshape(-1)

    track_len = self_agent["L"]
    if not torch.is_tensor(track_len):
        track_len = torch.as_tensor(track_len, dtype=s_s.dtype, device=s_s.device)
    track_len = track_len.reshape(-1).to(s_s.dtype)

    cos_y = torch.cos(yaw_s)
    sin_y = torch.sin(yaw_s)
    cos_o = torch.cos(yaw_o)
    sin_o = torch.sin(yaw_o)

    d = pos_o - pos_s
    rel_x = cos_y * d[:, 0] + sin_y * d[:, 1]
    rel_y = -sin_y * d[:, 0] + cos_y * d[:, 1]

    dv = vel_o - vel_s
    rel_vx = cos_y * dv[:, 0] + sin_y * dv[:, 1]
    rel_vy = -sin_y * dv[:, 0] + cos_y * dv[:, 1]

    ax_w_s = cos_y * acc_s[:, 0] - sin_y * acc_s[:, 1]
    ay_w_s = sin_y * acc_s[:, 0] + cos_y * acc_s[:, 1]
    ax_w_o = cos_o * acc_o[:, 0] - sin_o * acc_o[:, 1]
    ay_w_o = sin_o * acc_o[:, 0] + cos_o * acc_o[:, 1]
    dax = ax_w_o - ax_w_s
    day = ay_w_o - ay_w_s
    rel_ax = cos_y * dax + sin_y * day
    rel_ay = -sin_y * dax + cos_y * day

    gap = s_o - s_s
    half = 0.5 * track_len
    gap = torch.where(gap > half, gap - track_len, gap)
    gap = torch.where(gap < -half, gap + track_len, gap)
    gap_norm = gap / half.clamp_min(1e-6)

    return torch.stack(
        [rel_x, rel_y, rel_vx, rel_vy, rel_ax, rel_ay, gap_norm, ey_o], dim=-1
    )


def build_observation(
    num_obs: int,
    num_envs: int,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_lin_acc: torch.Tensor,
    last_actions: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    obs_cfg: dict[str, Any],
    step_state: dict[str, Any],
    device: torch.device,
    opponent_block: torch.Tensor | None = None,
) -> torch.Tensor:
    obs_scales = obs_cfg.get("obs_scales", {})
    lin_vel_scale = float(obs_scales.get("lin_vel", 1.0))
    ang_vel_scale = float(obs_scales.get("ang_vel", 1.0))
    lin_acc_scale = float(obs_scales.get("lin_acc", 1.0))

    components = (
        base_lin_vel[:, :2] * lin_vel_scale,
        base_ang_vel[:, 2:3] * ang_vel_scale,
        base_lin_acc[:, :2] * lin_acc_scale,
        last_actions,
        obs_track_progress(step_state, device),
        obs_centerline_angle(step_state, base_quat),
        obs_centerline_distance(step_state),
        obs_contact_flag(step_state, obs_cfg),
        obs_future_track_points(
            base_pos,
            base_quat,
            base_lin_vel,
            obs_cfg,
            device,
            step_state,
        ),
        step_state["tyre_slip"],
        step_state["tyre_load"],
    )

    opp_dim = int(obs_cfg.get("opponent_obs_dim", 8))
    if opponent_block is None or not bool(obs_cfg.get("enable_opponent_obs", True)):
        opponent_block = base_lin_vel.new_zeros((num_envs, opp_dim))
    components = components + (opponent_block,)

    # Write components into a single freshly-allocated buffer via slice copies
    # instead of torch.concatenate. This drops the concatenate output allocation
    # and its copy kernel while remaining numerically identical. The buffer is
    # freshly allocated each step (never reused in place), so returned/stored
    # observations stay independent of subsequent steps.
    obs = base_lin_vel.new_empty((num_envs, num_obs))
    offset = 0
    for comp in components:
        width = comp.shape[1]
        next_offset = offset + width
        if next_offset > num_obs:
            raise ValueError(
                "Observation shape mismatch: components exceed "
                f"num_obs={num_obs} (overflow at width {width}). "
                "Check obs_cfg['num_obs'] and observation component sizes."
            )
        obs[:, offset:next_offset] = comp
        offset = next_offset

    if offset != num_obs:
        raise ValueError(
            "Observation shape mismatch: "
            f"expected num_obs={num_obs}, got {offset}. "
            "Check obs_cfg['num_obs'] and observation component sizes."
        )

    # Final safety net: bound the whole observation so a transient physics
    # blow-up (e.g. a high-speed spin sending ang_vel/lin_acc or speed-scaled
    # future-track points to huge magnitudes) can never feed extreme values into
    # the networks. Disabled when clip_obs <= 0.
    clip_obs = float(obs_cfg.get("clip_obs", 0.0))
    if clip_obs > 0.0:
        obs = torch.clamp(obs, min=-clip_obs, max=clip_obs)

    return obs

"""Genesis-free port of the F1TENTH observation pipeline.

This module reproduces ``f1tenth_env/observations.py`` and the Frenet / boundary /
future-point math from ``f1tenth_env/utils.py`` using only ``torch`` and ``numpy``
(no ``genesis`` dependency), so it can run inside a ROS 2 Humble container for
inference.

The output of :func:`build_observation` is numerically identical (within float
tolerance) to ``f1tenth_env.observations.build_observation`` for the same inputs.
A parity test in ``test/test_obs_parity.py`` enforces this.

All tensors are CPU float32. Functions accept batched inputs (shape ``[B, ...]``)
but the ROS deployment uses ``B = 1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

TC_FLOAT = torch.float32


# --- quaternion helpers -------------------------------------------------------
def quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about z) from a (w, x, y, z) quaternion. Shape [B, 4] -> [B]."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


# --- track cache + Frenet projection (port of utils.py) -----------------------
def build_track_cache(
    centerline: np.ndarray,
    device: torch.device,
    coarse_stride: int = 10,
) -> dict[str, Any]:
    cl = torch.as_tensor(centerline, device=device, dtype=TC_FLOAT)
    if torch.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        cl = torch.cat([cl, cl[0:1]], dim=0)

    c = cl[:-1]
    d = cl[1:]
    seg = d - c
    seg_len = torch.linalg.norm(seg, dim=-1).clamp_min(1e-8)
    cumlen = torch.zeros_like(seg_len)
    cumlen[1:] = torch.cumsum(seg_len[:-1], dim=0)
    length = seg_len.sum()
    m = int(c.shape[0])

    coarse_idx = torch.arange(0, m, coarse_stride, device=device)
    coarse_pts = c[coarse_idx]

    return {
        "C": c,
        "seg": seg,
        "seg_len": seg_len,
        "cumlen": cumlen,
        "L": length,
        "M": m,
        "coarse_stride": coarse_stride,
        "coarse_idx": coarse_idx,
        "coarse_pts": coarse_pts,
    }


def frenet_projection(
    base_pos: torch.Tensor,
    geom: dict[str, Any],
    device: torch.device,
    window: int = 40,
) -> dict[str, Any]:
    pos = base_pos[:, :2].to(device=device, dtype=TC_FLOAT)
    batch = pos.shape[0]

    c_all = geom["C"]
    seg_all = geom["seg"]
    seg_len_all = geom["seg_len"]
    cumlen_all = geom["cumlen"]
    length = geom["L"]
    m = geom["M"]
    coarse_pts = geom["coarse_pts"]
    coarse_idx = geom["coarse_idx"]

    diffc = coarse_pts.unsqueeze(0) - pos.unsqueeze(1)
    dist2c = (diffc * diffc).sum(dim=-1)
    j = dist2c.argmin(dim=-1)
    i0 = coarse_idx[j]

    offsets = torch.arange(-window, window + 1, device=device)
    cand = (i0.unsqueeze(1) + offsets.unsqueeze(0)) % m

    c = c_all[cand]
    seg = seg_all[cand]
    seg_len2 = (seg * seg).sum(dim=-1).clamp_min(1e-10)

    p = pos.unsqueeze(1)
    t = ((p - c) * seg).sum(dim=-1) / seg_len2
    t = t.clamp(0.0, 1.0)

    proj = c + t.unsqueeze(-1) * seg
    dist2 = ((proj - p) ** 2).sum(dim=-1)

    k = dist2.argmin(dim=-1)
    ar = torch.arange(batch, device=device)
    best_idx = cand[ar, k]
    best_t = t[ar, k]
    best_proj = proj[ar, k]

    best_seg = seg_all[best_idx]
    seg_dir = best_seg / torch.linalg.norm(best_seg, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    s = cumlen_all[best_idx] + best_t * seg_len_all[best_idx]

    return {
        "pos": pos,
        "best_idx": best_idx,
        "best_t": best_t,
        "proj": best_proj,
        "seg_dir": seg_dir,
        "s": s,
        "L": length,
    }


def interp_width_at_s(
    best_idx: torch.Tensor,
    best_t: torch.Tensor,
    widths: torch.Tensor,
) -> torch.Tensor:
    w0 = widths[best_idx]
    w1 = widths[(best_idx + 1) % widths.shape[0]]
    return w0 + best_t * (w1 - w0)


def build_boundary_state(
    frenet_state: dict[str, Any],
    w_tr_left: torch.Tensor,
    w_tr_right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t_hat = frenet_state["seg_dir"]
    n_hat = torch.stack([-t_hat[:, 1], t_hat[:, 0]], dim=-1)
    ey = ((frenet_state["pos"] - frenet_state["proj"]) * n_hat).sum(-1)

    w_l_s = interp_width_at_s(frenet_state["best_idx"], frenet_state["best_t"], w_tr_left)
    w_r_s = interp_width_at_s(frenet_state["best_idx"], frenet_state["best_t"], w_tr_right)

    d_left = w_l_s - ey
    d_right = w_r_s + ey
    boundary_dist = torch.minimum(d_left, d_right)

    return {
        "ey": ey,
        "w_l_s": w_l_s,
        "w_r_s": w_r_s,
        "boundary_dist": boundary_dist,
    }


# --- observation components (port of observations.py) -------------------------
def obs_track_progress(
    frenet_state: dict[str, Any],
) -> torch.Tensor:
    track_len = frenet_state["L"].clamp(min=1e-6)
    progress_ratio = frenet_state["s"] / track_len
    angle = (2.0 * np.pi) * progress_ratio
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def obs_centerline_angle(
    frenet_state: dict[str, Any],
    base_quat_wxyz: torch.Tensor,
) -> torch.Tensor:
    track_angle = torch.atan2(
        frenet_state["seg_dir"][:, 1], frenet_state["seg_dir"][:, 0]
    )
    yaw = quat_wxyz_to_yaw(base_quat_wxyz)
    theta_err = yaw - track_angle
    theta_err = torch.atan2(torch.sin(theta_err), torch.cos(theta_err))
    return theta_err.unsqueeze(-1)


def obs_centerline_distance(boundary_state: dict[str, Any]) -> torch.Tensor:
    return boundary_state["ey"].unsqueeze(-1)


def obs_contact_flag(
    boundary_state: dict[str, Any], obs_cfg: dict[str, Any]
) -> torch.Tensor:
    boundary_dist = boundary_state["boundary_dist"]
    contact_margin = float(obs_cfg.get("contact_margin_m", 0.08))
    return (boundary_dist < contact_margin).float().unsqueeze(-1)


def obs_future_track_points(
    centerline: np.ndarray,
    w_tr_left: torch.Tensor,
    w_tr_right: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat_wxyz: torch.Tensor,
    base_lin_vel: torch.Tensor,
    obs_cfg: dict[str, Any],
    device: torch.device,
    frenet_state: dict[str, Any],
) -> torch.Tensor:
    centerline_t = torch.as_tensor(centerline, device=device, dtype=TC_FLOAT)
    robot_pos = base_pos[:, :2]
    lin_vel = base_lin_vel[:, :2]
    yaw = quat_wxyz_to_yaw(base_quat_wxyz)

    n = centerline_t.shape[0]
    seg = centerline_t[1:] - centerline_t[:-1]
    seg_len = torch.linalg.vector_norm(seg, dim=-1)
    cumlen = torch.cat(
        [
            torch.zeros(1, device=device, dtype=TC_FLOAT),
            torch.cumsum(seg_len, dim=0),
        ],
        dim=0,
    )

    batch = robot_pos.shape[0]
    samples = int(obs_cfg.get("future_track_num_points", 60))
    horizon_s = float(obs_cfg.get("future_track_horizon_s", 6.0))
    # future_track_width is deprecated: corridor edges use per-vertex CSV widths.

    # Use the Frenet arc-length and closed-loop track length (matches training
    # observations.py), not the nearest open-polyline vertex / open total length.
    s0 = frenet_state["s"]
    total_len = frenet_state["L"].clamp(min=1e-6)

    min_lookahead = float(obs_cfg.get("future_track_min_lookahead_m", 5.0))
    speed = torch.linalg.vector_norm(lin_vel, dim=-1)
    lookahead = torch.clamp(speed * horizon_s, min=min_lookahead)

    steps = torch.arange(1, samples + 1, device=device, dtype=TC_FLOAT) / samples
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


def quasi_static_load_ratio(
    ax: float,
    ay: float,
    *,
    h_cg: float = 0.05,
    lf: float = 0.1584,
    lr: float = 0.1666,
    track_width: float = 0.253,
    roll_stiffness_front: float = 0.47,
    gravity: float = 9.81,
) -> np.ndarray:
    """Per-wheel normal-load ratio Fz / Fz_static from body specific forces.

    Closed-form of ``f1tenth_sim/suspension.py::quasi_static_loads`` divided by the
    static per-wheel load; vehicle mass cancels, so only geometry is needed. ``ax``
    (forward, m/s^2) shifts load rearward; ``ay`` (leftward, m/s^2) shifts load onto
    the right (outer) wheels. Wheel order [LR, RR, LF, RF]. This is the on-car /
    gym estimate for the tyre-load block when no wheel-load sensing is available.
    """
    g = max(gravity, 1e-6)
    lf = max(lf, 1e-6)
    lr = max(lr, 1e-6)
    tw = max(track_width, 1e-6)
    wheelbase = lf + lr

    rear_base = 1.0 + ax * h_cg / (g * lf)
    front_base = 1.0 - ax * h_cg / (g * lr)
    lat_rear = (1.0 - roll_stiffness_front) * ay * h_cg * 2.0 * wheelbase / (g * tw * lf)
    lat_front = roll_stiffness_front * ay * h_cg * 2.0 * wheelbase / (g * tw * lr)

    ratio = np.array(
        [
            rear_base - lat_rear,
            rear_base + lat_rear,
            front_base - lat_front,
            front_base + lat_front,
        ],
        dtype=np.float32,
    )
    return np.clip(ratio, 0.0, None)


def obs_opponent(
    self_agent: dict[str, torch.Tensor],
    other_agent: dict[str, torch.Tensor],
    obs_cfg: dict[str, Any],
) -> torch.Tensor:
    """Port of ``f1tenth_env.observations.obs_opponent`` (6-dim opponent block)."""
    pos_s = self_agent["pos_xy"]
    yaw_s = self_agent["yaw"].reshape(-1)
    vel_s = self_agent["vel_xy"]

    pos_o = other_agent["pos_xy"]
    vel_o = other_agent["vel_xy"]
    s_o = other_agent["s"].reshape(-1)
    ey_o = other_agent["ey"].reshape(-1)

    s_s = self_agent["s"].reshape(-1)
    track_len = self_agent["L"]
    if not torch.is_tensor(track_len):
        track_len = torch.as_tensor(track_len, dtype=s_s.dtype, device=s_s.device)
    track_len = track_len.reshape(-1).to(s_s.dtype)

    cos_y = torch.cos(yaw_s)
    sin_y = torch.sin(yaw_s)

    d = pos_o - pos_s
    rel_x = cos_y * d[:, 0] + sin_y * d[:, 1]
    rel_y = -sin_y * d[:, 0] + cos_y * d[:, 1]

    dv = vel_o - vel_s
    rel_vx = cos_y * dv[:, 0] + sin_y * dv[:, 1]
    rel_vy = -sin_y * dv[:, 0] + cos_y * dv[:, 1]

    gap = s_o - s_s
    half = 0.5 * track_len
    gap = torch.where(gap > half, gap - track_len, gap)
    gap = torch.where(gap < -half, gap + track_len, gap)
    gap_norm = gap / half.clamp_min(1e-6)

    return torch.stack(
        [rel_x, rel_y, rel_vx, rel_vy, gap_norm, ey_o], dim=-1
    )


class ObservationBuilder:
    """Stateful helper that owns the track geometry and builds observations.

    Construct once with the track centerline + widths, then call
    :meth:`build` each control step with the current pose, velocities and last action.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        w_tr_left: np.ndarray,
        w_tr_right: np.ndarray,
        obs_cfg: dict[str, Any] | None = None,
        device: torch.device | None = None,
        coarse_stride: int = 10,
    ):
        self.device = device or torch.device("cpu")
        self.centerline = np.asarray(centerline, dtype=np.float32)
        self.obs_cfg = obs_cfg or {}
        self.num_obs = int(self.obs_cfg.get("num_obs", 384))
        self.base_num_obs = int(self.obs_cfg.get("base_num_obs", self.num_obs))
        self.w_tr_left = torch.as_tensor(w_tr_left, device=self.device, dtype=TC_FLOAT)
        self.w_tr_right = torch.as_tensor(w_tr_right, device=self.device, dtype=TC_FLOAT)
        self.geom = build_track_cache(
            self.centerline, device=self.device, coarse_stride=coarse_stride
        )

    def build(
        self,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
        base_lin_acc: torch.Tensor,
        last_actions: torch.Tensor,
        base_pos: torch.Tensor,
        base_quat_wxyz: torch.Tensor,
        tyre_slip: torch.Tensor | None = None,
        tyre_load: torch.Tensor | None = None,
        opponent_block: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frenet_state = frenet_projection(base_pos, self.geom, self.device)
        boundary_state = build_boundary_state(
            frenet_state, self.w_tr_left, self.w_tr_right
        )

        obs_scales = self.obs_cfg.get("obs_scales", {})
        lin_vel_scale = float(obs_scales.get("lin_vel", 1.0))
        ang_vel_scale = float(obs_scales.get("ang_vel", 1.0))
        lin_acc_scale = float(obs_scales.get("lin_acc", 1.0))

        batch = base_lin_vel.shape[0]
        if tyre_slip is None:
            tyre_slip = base_lin_vel.new_zeros((batch, 8))
        if tyre_load is None:
            # At-rest normal-load ratio Fz / Fz_static is 1.0 on each wheel.
            tyre_load = base_lin_vel.new_ones((batch, 4))

        obs = torch.cat(
            (
                base_lin_vel[:, :2] * lin_vel_scale,
                base_ang_vel[:, 2:3] * ang_vel_scale,
                base_lin_acc[:, :2] * lin_acc_scale,
                last_actions,
                obs_track_progress(frenet_state),
                obs_centerline_angle(frenet_state, base_quat_wxyz),
                obs_centerline_distance(boundary_state),
                obs_contact_flag(boundary_state, self.obs_cfg),
                obs_future_track_points(
                    self.centerline,
                    self.w_tr_left,
                    self.w_tr_right,
                    base_pos,
                    base_quat_wxyz,
                    base_lin_vel,
                    self.obs_cfg,
                    self.device,
                    frenet_state,
                ),
                tyre_slip,
                tyre_load,
            ),
            dim=-1,
        )

        if bool(self.obs_cfg.get("enable_opponent_obs", False)):
            opp_dim = int(self.obs_cfg.get("opponent_obs_dim", 6))
            if opponent_block is None:
                opponent_block = base_lin_vel.new_zeros((batch, opp_dim))
            obs = torch.cat((obs, opponent_block), dim=-1)

        clip_obs = float(self.obs_cfg.get("clip_obs", 0.0))
        if clip_obs > 0.0:
            obs = torch.clamp(obs, min=-clip_obs, max=clip_obs)

        actual_obs_dim = int(obs.shape[1])
        if actual_obs_dim != self.num_obs:
            raise ValueError(
                f"Observation shape mismatch: expected num_obs={self.num_obs}, "
                f"got {actual_obs_dim}."
            )
        return obs

    def build_opponent_block(
        self,
        ego_pos: torch.Tensor,
        ego_yaw: torch.Tensor,
        ego_vel_world: torch.Tensor,
        opp_pos: torch.Tensor,
        opp_vel_world: torch.Tensor,
    ) -> torch.Tensor:
        ego_frenet = frenet_projection(ego_pos, self.geom, self.device)
        ego_boundary = build_boundary_state(ego_frenet, self.w_tr_left, self.w_tr_right)
        opp_frenet = frenet_projection(opp_pos, self.geom, self.device)
        opp_boundary = build_boundary_state(opp_frenet, self.w_tr_left, self.w_tr_right)

        track_len = ego_frenet["L"]
        self_agent = {
            "pos_xy": ego_pos[:, :2],
            "yaw": ego_yaw.reshape(-1),
            "vel_xy": ego_vel_world[:, :2],
            "s": ego_frenet["s"],
            "ey": ego_boundary["ey"],
            "L": track_len,
        }
        other_agent = {
            "pos_xy": opp_pos[:, :2],
            "vel_xy": opp_vel_world[:, :2],
            "s": opp_frenet["s"],
            "ey": opp_boundary["ey"],
            "L": track_len,
        }
        return obs_opponent(self_agent, other_agent, self.obs_cfg)

    def frenet(self, base_pos: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        """Expose Frenet + boundary state (used by evaluation_node)."""
        frenet_state = frenet_projection(base_pos, self.geom, self.device)
        boundary_state = build_boundary_state(
            frenet_state, self.w_tr_left, self.w_tr_right
        )
        return frenet_state, boundary_state

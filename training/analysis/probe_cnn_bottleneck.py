#!/usr/bin/env python3
"""Zero-training linear probe for the 1D LiDAR CNN spatial bottleneck.

Rolls out a trained policy checkpoint (plus a random orthogonal-init CNN
structural baseline) in the multi-car simulator on Austin with active
opponents, collects 16,384 multi-car frames where at least one opponent is
within [1.0 m, 12.0 m] and inside the 270-degree LiDAR FOV, and fits closed-form
Ridge regression from each candidate representation to ground-truth opponent
and passable lateral gap geometry across Near (1-4 m), Mid (4-8 m), and Far
(8-12 m) distance buckets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TRAINING_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TRAINING_DIR.parent
_POLICY_LIB = _REPO_ROOT / "libs" / "f1tenth_policy"
if str(_POLICY_LIB) not in sys.path:
    sys.path.insert(0, str(_POLICY_LIB))
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from f1tenth_policy.actor import CNN_CONV_CHANNELS, LidarCNNEncoder  # noqa: E402
from f1tenth_policy.layout import (  # noqa: E402
    LIDAR_DIM,
    LIDAR_FOV_DEG,
    executed_plant_action,
)

DEFAULT_TARGET_FRAMES = 16_384
DEFAULT_DIST_MIN_M = 1.0
DEFAULT_DIST_MAX_M = 12.0
DEFAULT_FOV_DEG = LIDAR_FOV_DEG
DEFAULT_ALPHAS = (1e-2, 1.0, 10.0)
DEFAULT_TRAIN_FRAC = 0.80
DEFAULT_CAR_WIDTH_M = 0.296
DEFAULT_CAR_LENGTH_M = 0.568

DISTANCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Near (1-4 m)", 1.0, 4.0),
    ("Mid (4-8 m)", 4.0, 8.0),
    ("Far (8-12 m)", 8.0, 12.0),
    ("Mid+Far (4-12 m)", 4.0, 12.0),
    ("All (1-12 m)", 1.0, 12.0),
)

TARGET_SPECS: tuple[tuple[str, str, float], ...] = (
    ("rel_x", "m", 1.0),
    ("rel_y", "m", 1.0),
    ("dist_m", "m", 1.0),
    ("theta_opp_deg", "deg", 1.0),
    ("delta_psi_deg", "deg", 1.0),
    ("gap_left_m", "m", 1.0),
    ("gap_right_m", "m", 1.0),
)

REPRESENTATION_ORDER: tuple[str, ...] = (
    "Raw LiDAR (1081-D)",
    "z_256 (AvgPool32->256)",
    "AvgPool1d(32) (2048-D)",
    "MaxPool1d(32) (2048-D)",
    "MaxPool1d(64)->512 (512-D)",
    "MaxPool1d(64) (4096-D)",
    "Unpooled conv (64x91=5824-D)",
)


def wrap_angle_rad(angle: np.ndarray | torch.Tensor):
    """Wrap angle in radians to (-pi, pi]."""
    if torch.is_tensor(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return np.arctan2(np.sin(angle), np.cos(angle))


def build_orthogonal_cnn_encoder(
    *,
    lidar_dim: int = LIDAR_DIM,
    pool_bins: int = 32,
    projection_dim: int = 256,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> LidarCNNEncoder:
    """Construct a random orthogonal-init LidarCNNEncoder structural baseline."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    encoder = LidarCNNEncoder(
        lidar_dim=lidar_dim,
        pool_bins=pool_bins,
        projection_dim=projection_dim,
    )
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.orthogonal_(
                    module.weight, gain=math.sqrt(2.0), generator=gen
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return encoder.to(device=device, dtype=torch.float32).eval()


def build_orthogonal_projection(
    in_features: int,
    out_features: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> nn.Linear:
    """Deterministic orthogonal random projection (e.g. 4096 -> 512)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    proj = nn.Linear(int(in_features), int(out_features), bias=False)
    with torch.no_grad():
        nn.init.orthogonal_(proj.weight, gain=1.0, generator=gen)
    return proj.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def extract_candidate_representations(
    encoder: LidarCNNEncoder,
    lidar: torch.Tensor,
    *,
    proj_4096_to_512: nn.Linear | None = None,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Compute all candidate pooled/projected representations from normalized LiDAR."""
    if lidar.dim() != 2 or lidar.shape[1] != encoder.lidar_dim:
        raise ValueError(
            f"Expected lidar shape (N, {encoder.lidar_dim}), got {tuple(lidar.shape)}"
        )
    device = next(encoder.parameters()).device
    if proj_4096_to_512 is None:
        proj_4096_to_512 = build_orthogonal_projection(
            CNN_CONV_CHANNELS[-1] * 64,
            512,
            seed=101,
            device=device,
        )
    else:
        proj_4096_to_512 = proj_4096_to_512.to(device=device)

    avg_pool_32 = nn.AdaptiveAvgPool1d(32)
    max_pool_32 = nn.AdaptiveMaxPool1d(32)
    max_pool_64 = nn.AdaptiveMaxPool1d(64)

    n = lidar.shape[0]
    out_lists: dict[str, list[np.ndarray]] = {
        key: [] for key in REPRESENTATION_ORDER
    }

    for start in range(0, n, int(batch_size)):
        chunk = lidar[start : start + int(batch_size)].to(
            device=device, dtype=torch.float32
        )
        conv_feat = encoder.conv(chunk.unsqueeze(1))  # [B, 64, 91]
        avg32 = avg_pool_32(conv_feat).flatten(1)  # [B, 2048]
        z_256 = F.relu(encoder.projection(avg32))  # [B, 256]
        max32 = max_pool_32(conv_feat).flatten(1)  # [B, 2048]
        max64 = max_pool_64(conv_feat).flatten(1)  # [B, 4096]
        max64_512 = F.relu(proj_4096_to_512(max64))  # [B, 512]
        unpooled = conv_feat.flatten(1)  # [B, 5824]

        out_lists["Raw LiDAR (1081-D)"].append(chunk.cpu().numpy())
        out_lists["z_256 (AvgPool32->256)"].append(z_256.cpu().numpy())
        out_lists["AvgPool1d(32) (2048-D)"].append(avg32.cpu().numpy())
        out_lists["MaxPool1d(32) (2048-D)"].append(max32.cpu().numpy())
        out_lists["MaxPool1d(64)->512 (512-D)"].append(max64_512.cpu().numpy())
        out_lists["MaxPool1d(64) (4096-D)"].append(max64.cpu().numpy())
        out_lists["Unpooled conv (64x91=5824-D)"].append(unpooled.cpu().numpy())

    return {
        key: np.concatenate(chunks, axis=0) for key, chunks in out_lists.items()
    }


def project_positions_to_track(
    track_host,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world (pos_x, pos_y) onto track_host.

    Returns (s, ey, w_left, w_right) as float32 arrays.
    """
    pts = np.asarray(track_host.point, dtype=np.float64)  # [M, 2]
    normals = np.asarray(track_host.normal, dtype=np.float64)  # [M, 2]
    seg_lens = np.asarray(track_host.segment_length, dtype=np.float64)  # [M]
    cum_lens = np.asarray(track_host.cumulative_length, dtype=np.float64)  # [M]
    w_left_arr = np.asarray(track_host.width_left, dtype=np.float64)  # [M]
    w_right_arr = np.asarray(track_host.width_right, dtype=np.float64)  # [M]
    m = pts.shape[0]
    nxt = (np.arange(m) + 1) % m
    edges = pts[nxt] - pts  # [M, 2]
    edge_len2 = np.maximum(np.sum(edges * edges, axis=1), 1e-10)  # [M]

    px = np.asarray(pos_x, dtype=np.float64).reshape(-1)
    py = np.asarray(pos_y, dtype=np.float64).reshape(-1)
    pos = np.stack([px, py], axis=-1)  # [B, 2]
    b = pos.shape[0]

    diff = pos[:, None, :] - pts[None, :, :]  # [B, M, 2]
    alpha = np.sum(diff * edges[None, :, :], axis=-1) / edge_len2[None, :]
    alpha = np.clip(alpha, 0.0, 1.0)  # [B, M]
    proj = pts[None, :, :] + alpha[:, :, None] * edges[None, :, :]  # [B, M, 2]
    delta = pos[:, None, :] - proj  # [B, M, 2]
    dist2 = np.sum(delta * delta, axis=-1)  # [B, M]
    best_seg = np.argmin(dist2, axis=1)  # [B]

    rows = np.arange(b)
    best_alpha = alpha[rows, best_seg]
    best_delta = delta[rows, best_seg]
    best_normal = normals[best_seg]
    best_nxt = (best_seg + 1) % m

    s = cum_lens[best_seg] + best_alpha * seg_lens[best_seg]
    ey = np.sum(best_delta * best_normal, axis=-1)
    w_left = w_left_arr[best_seg] + best_alpha * (
        w_left_arr[best_nxt] - w_left_arr[best_seg]
    )
    w_right = w_right_arr[best_seg] + best_alpha * (
        w_right_arr[best_nxt] - w_right_arr[best_seg]
    )
    return (
        s.astype(np.float32),
        ey.astype(np.float32),
        w_left.astype(np.float32),
        w_right.astype(np.float32),
    )


def check_lidar_opponent_obb_hit(
    *,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    opp_x: np.ndarray,
    opp_y: np.ndarray,
    opp_yaw: np.ndarray,
    raw_lidar_ranges: np.ndarray,
    fov_deg: float = DEFAULT_FOV_DEG,
    car_length_m: float = DEFAULT_CAR_LENGTH_M,
    car_width_m: float = DEFAULT_CAR_WIDTH_M,
    lidar_offset_x: float = 0.0,
    lidar_offset_y: float = 0.0,
    lidar_offset_yaw: float = 0.0,
    hit_tol_m: float = 0.15,
) -> np.ndarray:
    """Return boolean mask [B] where at least one LiDAR beam hits the opponent OBB."""
    ex = np.asarray(ego_x, dtype=np.float64)
    ey = np.asarray(ego_y, dtype=np.float64)
    eyaw = np.asarray(ego_yaw, dtype=np.float64)
    ox = np.asarray(opp_x, dtype=np.float64)
    oy = np.asarray(opp_y, dtype=np.float64)
    oyaw = np.asarray(opp_yaw, dtype=np.float64)
    ranges = np.asarray(raw_lidar_ranges, dtype=np.float64)
    b, num_beams = ranges.shape

    cos_e = np.cos(eyaw)
    sin_e = np.sin(eyaw)
    orig_x = ex + cos_e * float(lidar_offset_x) - sin_e * float(lidar_offset_y)
    orig_y = ey + sin_e * float(lidar_offset_x) + cos_e * float(lidar_offset_y)

    fov_rad = math.radians(float(fov_deg))
    beam_offsets = np.linspace(-0.5 * fov_rad, 0.5 * fov_rad, num_beams)
    beam_angles = (
        eyaw[:, None] + float(lidar_offset_yaw) + beam_offsets[None, :]
    )  # [B, M]
    dir_x = np.cos(beam_angles)
    dir_y = np.sin(beam_angles)

    cos_o = np.cos(oyaw)[:, None]
    sin_o = np.sin(oyaw)[:, None]
    dx = (orig_x - ox)[:, None]
    dy = (orig_y - oy)[:, None]

    local_ox = cos_o * dx + sin_o * dy  # [B, 1]
    local_oy = -sin_o * dx + cos_o * dy  # [B, 1]
    local_dx = cos_o * dir_x + sin_o * dir_y  # [B, M]
    local_dy = -sin_o * dir_x + cos_o * dir_y  # [B, M]

    half_l = 0.5 * float(car_length_m)
    half_w = 0.5 * float(car_width_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        tx0 = (-half_l - local_ox) / local_dx
        tx1 = (half_l - local_ox) / local_dx
        ty0 = (-half_w - local_oy) / local_dy
        ty1 = (half_w - local_oy) / local_dy

    tmin_x = np.minimum(tx0, tx1)
    tmax_x = np.maximum(tx0, tx1)
    tmin_y = np.minimum(ty0, ty1)
    tmax_y = np.maximum(ty0, ty1)

    t_enter = np.maximum(tmin_x, tmin_y)
    t_exit = np.minimum(tmax_x, tmax_y)
    valid_obb = (t_exit >= np.maximum(t_enter, 0.0)) & np.isfinite(t_enter)
    t_hit = np.where(valid_obb, np.maximum(t_enter, 0.0), np.inf)

    # Beam actually reached the opponent OBB (was not blocked earlier by a wall)
    unoccluded_hit = valid_obb & (ranges >= (t_hit - float(hit_tol_m)))
    return np.any(unoccluded_hit, axis=1)


def interpolate_track_widths_at_s(
    track_host,
    s_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w_left, w_right, track_yaw) at along-track positions s_values."""
    length = float(track_host.length)
    s_wrapped = np.mod(np.asarray(s_values, dtype=np.float64), length)
    cum = np.asarray(track_host.cumulative_length, dtype=np.float64)
    idx = np.searchsorted(cum, s_wrapped, side="right") - 1
    n_pts = int(cum.shape[0])
    idx = np.clip(idx, 0, n_pts - 1)
    seg_len = np.asarray(track_host.segment_length, dtype=np.float64)[idx]
    alpha = np.where(seg_len > 1e-9, (s_wrapped - cum[idx]) / seg_len, 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    nxt = (idx + 1) % n_pts

    w_left_arr = np.asarray(track_host.width_left, dtype=np.float64)
    w_right_arr = np.asarray(track_host.width_right, dtype=np.float64)
    w_left = w_left_arr[idx] + alpha * (w_left_arr[nxt] - w_left_arr[idx])
    w_right = w_right_arr[idx] + alpha * (w_right_arr[nxt] - w_right_arr[idx])

    tangents = np.asarray(track_host.tangent, dtype=np.float64)[idx]
    track_yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    return (
        w_left.astype(np.float32),
        w_right.astype(np.float32),
        track_yaw.astype(np.float32),
    )


def compute_privileged_targets(
    *,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    opp_x: np.ndarray,
    opp_y: np.ndarray,
    opp_yaw: np.ndarray,
    opp_s: np.ndarray,
    opp_ey: np.ndarray,
    w_left_at_opp: np.ndarray,
    w_right_at_opp: np.ndarray,
    car_width_m: float = DEFAULT_CAR_WIDTH_M,
) -> dict[str, np.ndarray]:
    """Compute ground-truth opponent pose and passable left/right gap targets."""
    dx = np.asarray(opp_x, dtype=np.float64) - np.asarray(ego_x, dtype=np.float64)
    dy = np.asarray(opp_y, dtype=np.float64) - np.asarray(ego_y, dtype=np.float64)
    yaw_e = np.asarray(ego_yaw, dtype=np.float64)
    yaw_o = np.asarray(opp_yaw, dtype=np.float64)

    cos_y = np.cos(yaw_e)
    sin_y = np.sin(yaw_e)
    rel_x = cos_y * dx + sin_y * dy
    rel_y = -sin_y * dx + cos_y * dy
    dist_m = np.hypot(rel_x, rel_y)
    theta_opp_rad = np.arctan2(rel_y, rel_x)
    delta_psi_rad = wrap_angle_rad(yaw_o - yaw_e)

    half_car_w = 0.5 * float(car_width_m)
    ey_o = np.asarray(opp_ey, dtype=np.float64)
    w_l = np.asarray(w_left_at_opp, dtype=np.float64)
    w_r = np.asarray(w_right_at_opp, dtype=np.float64)

    gap_left_m = np.maximum(0.0, w_l - (ey_o + half_car_w))
    gap_right_m = np.maximum(0.0, w_r + (ey_o - half_car_w))

    return {
        "rel_x": rel_x.astype(np.float32),
        "rel_y": rel_y.astype(np.float32),
        "dist_m": dist_m.astype(np.float32),
        "theta_opp_rad": theta_opp_rad.astype(np.float32),
        "theta_opp_deg": np.degrees(theta_opp_rad).astype(np.float32),
        "delta_psi_rad": delta_psi_rad.astype(np.float32),
        "delta_psi_deg": np.degrees(delta_psi_rad).astype(np.float32),
        "gap_left_m": gap_left_m.astype(np.float32),
        "gap_right_m": gap_right_m.astype(np.float32),
        "opp_s": np.asarray(opp_s, dtype=np.float32),
        "opp_ey": ey_o.astype(np.float32),
    }


def filter_fov_and_distance_mask(
    dist_m: np.ndarray,
    theta_opp_rad: np.ndarray,
    *,
    dist_min_m: float = DEFAULT_DIST_MIN_M,
    dist_max_m: float = DEFAULT_DIST_MAX_M,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> np.ndarray:
    """Boolean mask for frames with opponent in [dist_min_m, dist_max_m] and inside FOV."""
    half_fov_rad = math.radians(0.5 * float(fov_deg))
    return (
        np.isfinite(dist_m)
        & np.isfinite(theta_opp_rad)
        & (dist_m >= float(dist_min_m))
        & (dist_m <= float(dist_max_m))
        & (np.abs(theta_opp_rad) <= half_fov_rad)
    )


def split_by_episode(
    episode_keys: np.ndarray,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint episode split returning boolean (train_mask, test_mask)."""
    unique_eps = np.unique(episode_keys)
    if unique_eps.size < 2:
        raise ValueError(
            f"Need at least 2 distinct episodes for train/test split, got {unique_eps.size}"
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(unique_eps)
    n_train = int(round(float(train_frac) * unique_eps.size))
    n_train = max(1, min(unique_eps.size - 1, n_train))
    train_eps = set(perm[:n_train].tolist())
    train_mask = np.isin(episode_keys, list(train_eps))
    test_mask = ~train_mask
    return train_mask, test_mask


@dataclass
class RidgeFitResult:
    weights: np.ndarray  # [D, K]
    bias: np.ndarray  # [K]
    x_mean: np.ndarray  # [1, D]
    x_std: np.ndarray  # [1, D]
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        return (x_norm @ self.weights + self.bias).astype(np.float64)


def fit_ridge_closed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> RidgeFitResult:
    """Fit closed-form Ridge regression with feature standardization via dual/primal Cholesky."""
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n, d = x.shape
    if n == 0:
        raise ValueError("fit_ridge_closed_form requires at least 1 training sample")

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_c = (x - x_mean) / x_std

    y_mean = y.mean(axis=0, keepdims=True)
    y_c = y - y_mean

    reg = float(alpha)
    if reg <= 0.0:
        raise ValueError(f"Ridge alpha must be positive, got {alpha}")

    if n >= d:
        gram = x_c.T @ x_c
        gram.flat[:: d + 1] += reg
        rhs = x_c.T @ y_c
        weights = np.linalg.solve(gram, rhs)
    else:
        gram_dual = x_c @ x_c.T
        gram_dual.flat[:: n + 1] += reg
        dual_sol = np.linalg.solve(gram_dual, y_c)
        weights = x_c.T @ dual_sol

    bias = y_mean.reshape(-1)
    return RidgeFitResult(
        weights=weights,
        bias=bias,
        x_mean=x_mean,
        x_std=x_std,
        alpha=reg,
    )


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute held-out R^2 and MAE."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return {"r2": float("nan"), "mae": float("nan"), "n": 0}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err * err))
    yt_centered = yt - float(np.mean(yt))
    ss_tot = float(np.sum(yt_centered * yt_centered))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12)) if ss_tot > 1e-12 else 0.0
    return {"r2": float(r2), "mae": mae, "n": int(yt.size)}


def evaluate_probe_on_features(
    features_by_name: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    episode_keys: np.ndarray | None = None,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Fit Ridge probes per representation and evaluate across distance buckets."""
    target_names = [spec[0] for spec in TARGET_SPECS]
    y_all = np.column_stack([targets[name] for name in target_names]).astype(
        np.float64
    )

    # Split train into 85% inner-train / 15% val by episode (when episode_keys is
    # available and has >= 2 train episodes) to pick the best alpha in {1e-2, 1.0, 10.0}
    # without touching the held-out test set.
    train_indices = np.flatnonzero(train_mask)
    if (
        episode_keys is not None
        and np.unique(episode_keys[train_mask]).size >= 2
    ):
        inner_sub, val_sub = split_by_episode(
            episode_keys[train_mask], train_frac=0.85, seed=0
        )
        inner_idx = train_indices[inner_sub]
        val_idx = (
            train_indices[val_sub] if np.any(val_sub) else inner_idx
        )
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(train_indices.size)
        n_inner = max(1, int(0.85 * train_indices.size))
        inner_idx = train_indices[perm[:n_inner]]
        val_idx = (
            train_indices[perm[n_inner:]]
            if n_inner < train_indices.size
            else inner_idx
        )

    dist_m = targets["dist_m"]
    results: dict[str, Any] = {}

    for rep_name, x_all in features_by_name.items():
        best_alpha_per_target: dict[str, float] = {}
        preds_test = np.zeros(
            (int(test_mask.sum()), len(target_names)), dtype=np.float64
        )
        dim = int(x_all.shape[1])
        n_inner_samples = max(1, int(inner_idx.size))
        n_train_samples = max(1, int(train_mask.sum()))
        # Since fit_ridge_closed_form standardizes each of the D feature columns to
        # unit variance, trace(X_c^T X_c) = N * D (average eigenvalue = N * D / min(N, D)).
        # Scaling alpha by (N * D / 256.0) makes alpha in {1e-2, 1.0, 10.0} invariant
        # to feature dimension D relative to the 256-D baseline.
        reg_scale_inner = (n_inner_samples * dim) / 256.0
        reg_scale_full = (n_train_samples * dim) / 256.0

        fits_by_alpha: dict[float, RidgeFitResult] = {}
        val_preds_by_alpha: dict[float, np.ndarray] = {}
        for alpha in alphas:
            fit_inner = fit_ridge_closed_form(
                x_all[inner_idx],
                y_all[inner_idx],
                alpha=float(alpha) * reg_scale_inner,
            )
            val_preds_by_alpha[float(alpha)] = fit_inner.predict(x_all[val_idx])
            fits_by_alpha[float(alpha)] = fit_ridge_closed_form(
                x_all[train_mask],
                y_all[train_mask],
                alpha=float(alpha) * reg_scale_full,
            )

        for col_idx, t_name in enumerate(target_names):
            best_a = float(alphas[0])
            best_val_mae = float("inf")
            for alpha in alphas:
                val_mae = float(
                    np.mean(
                        np.abs(
                            val_preds_by_alpha[float(alpha)][:, col_idx]
                            - y_all[val_idx, col_idx]
                        )
                    )
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_a = float(alpha)
            best_alpha_per_target[t_name] = best_a
            full_pred = fits_by_alpha[best_a].predict(x_all[test_mask])
            preds_test[:, col_idx] = full_pred[:, col_idx]

        y_test = y_all[test_mask]
        dist_test = dist_m[test_mask]

        bucket_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for bucket_label, d_lo, d_hi in DISTANCE_BUCKETS:
            b_mask = (dist_test >= d_lo) & (dist_test <= d_hi)
            bucket_metrics[bucket_label] = {}
            for col_idx, t_name in enumerate(target_names):
                bucket_metrics[bucket_label][t_name] = score_predictions(
                    y_test[b_mask, col_idx],
                    preds_test[b_mask, col_idx],
                )

        results[rep_name] = {
            "dim": int(x_all.shape[1]),
            "best_alpha": best_alpha_per_target,
            "buckets": bucket_metrics,
        }

    return results


def evaluate_go_no_go_gate(probe_results: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the Phase 1 Go/No-Go criterion in the 4-12 m bucket.

    Go criterion: `conv_feat (64x91)` or `MaxPool1d(64)` (or `MaxPool1d(64)->512`)
    achieves >= 15% lower MAE or >= +0.08 higher R^2 than `AvgPool1d(32) -> 256-D`
    in the `4-12 m` bucket on opponent lateral position `rel_y`, bearing
    `theta_opp_deg`, or gap width (`gap_left_m` / `gap_right_m`).
    """
    baseline_key = "z_256 (AvgPool32->256)"
    candidate_keys = (
        "MaxPool1d(64)->512 (512-D)",
        "MaxPool1d(64) (4096-D)",
        "Unpooled conv (64x91=5824-D)",
    )
    gate_targets = ("rel_y", "theta_opp_deg", "gap_left_m", "gap_right_m")
    bucket_name = "Mid+Far (4-12 m)"

    base_bucket = probe_results[baseline_key]["buckets"][bucket_name]
    comparisons: dict[str, dict[str, dict[str, float | bool]]] = {}
    any_go = False

    for cand_key in candidate_keys:
        if cand_key not in probe_results:
            continue
        cand_bucket = probe_results[cand_key]["buckets"][bucket_name]
        comparisons[cand_key] = {}
        for t_name in gate_targets:
            b_mae = float(base_bucket[t_name]["mae"])
            b_r2 = float(base_bucket[t_name]["r2"])
            c_mae = float(cand_bucket[t_name]["mae"])
            c_r2 = float(cand_bucket[t_name]["r2"])
            mae_rel_reduction = (
                (b_mae - c_mae) / max(b_mae, 1e-12) if b_mae > 0 else 0.0
            )
            r2_delta = c_r2 - b_r2
            passes = (mae_rel_reduction >= 0.15) or (r2_delta >= 0.08)
            if passes:
                any_go = True
            comparisons[cand_key][t_name] = {
                "baseline_mae": b_mae,
                "candidate_mae": c_mae,
                "mae_rel_reduction": mae_rel_reduction,
                "baseline_r2": b_r2,
                "candidate_r2": c_r2,
                "r2_delta": r2_delta,
                "passes_gate": passes,
            }

    return {
        "decision": "GO" if any_go else "STOP",
        "bucket": bucket_name,
        "Baseline": baseline_key,
        "comparisons": comparisons,
    }


def collect_multicar_probe_dataset(
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    num_envs: int = 128,
    dist_min_m: float = DEFAULT_DIST_MIN_M,
    dist_max_m: float = DEFAULT_DIST_MAX_M,
    fov_deg: float = DEFAULT_FOV_DEG,
    device_name: str = "cpu",
    seed: int = 42,
    max_rollout_steps: int = 8_000,
) -> dict[str, Any]:
    """Roll out a checkpoint on Austin with active opponents and collect filtered frames."""
    from eval_full_lap import load_actor_bundle, resolve_eval_config
    from evaluation import actor_is_recurrent
    from f1tenth_env import F1tenthEnv, runtime as rt
    from standalone_trainer import build_env_cfg

    device = torch.device(device_name)
    rt.configure(dev=device)

    cfg, resolved_cfg_path = resolve_eval_config(checkpoint_path, config_path)
    cfg = json.loads(json.dumps(cfg))
    cfg["env"]["track"] = "Austin"
    cfg["env"]["episode_length"] = 25.0
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_spawn_gap_min_m"] = 2.0
    cfg["env"]["opponent_spawn_gap_max_m"] = 12.0
    cfg["env"]["opponent_spawn_behind_prob"] = 0.20
    cfg["env"]["opponent_target_speed_range"] = [4.0, 7.5]
    cfg["env"]["opponent_lateral_offset_m"] = 0.45
    cfg["env"]["seed"] = int(seed)
    dr = dict(cfg["env"].get("domain_randomization") or {})
    dr["enabled"] = False
    cfg["env"]["domain_randomization"] = dr

    env_cfg = build_env_cfg(
        cfg,
        launch_strategy="uniform_jittered",
        launch_strategy_data={"num_cars": int(num_envs)},
    )
    env = F1tenthEnv(
        num_envs=int(num_envs),
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )

    actor, normalizer, _arch, payload = load_actor_bundle(
        checkpoint_path, device, cfg
    )
    normalize = normalizer.normalize
    actor.eval()
    recurrent = actor_is_recurrent(actor)
    clip_actions = float(cfg["env"].get("clip_actions", 1.0))
    car_width_m = float(cfg["env"].get("car_width", DEFAULT_CAR_WIDTH_M))
    car_length_m = float(cfg["env"].get("car_length", DEFAULT_CAR_LENGTH_M))
    sensor_cfg = env.sensor_cfg
    lidar_offset_x = float(sensor_cfg.get("lidar_offset_x", 0.0))
    lidar_offset_y = float(sensor_cfg.get("lidar_offset_y", 0.0))
    lidar_offset_yaw = float(sensor_cfg.get("lidar_offset_yaw", 0.0))

    raw_obs, _extras = env.reset(seed=int(seed), with_sensors=True)
    obs = raw_obs["actor"].to(device=device, dtype=torch.float32)
    hidden = (
        actor.initial_hidden(int(num_envs), device=device, dtype=torch.float32)
        if recurrent
        else None
    )

    lidar_chunks: list[np.ndarray] = []
    raw_actor_obs_chunks: list[np.ndarray] = []
    ego_x_chunks: list[np.ndarray] = []
    ego_y_chunks: list[np.ndarray] = []
    ego_yaw_chunks: list[np.ndarray] = []
    opp_x_chunks: list[np.ndarray] = []
    opp_y_chunks: list[np.ndarray] = []
    opp_yaw_chunks: list[np.ndarray] = []
    ep_key_chunks: list[np.ndarray] = []

    collected = 0
    total_env_steps = 0

    with torch.no_grad():
        for _step in range(int(max_rollout_steps)):
            if collected >= int(target_frames):
                break
            model_obs = normalize(obs)
            ego_t = env._ego.tensor
            opp_t = env._opponent.tensor
            env_t = env._env.tensor

            ego_x = ego_t["x"].detach().cpu().numpy()
            ego_y = ego_t["y"].detach().cpu().numpy()
            ego_yaw = ego_t["yaw"].detach().cpu().numpy()
            opp_x = opp_t["x"].detach().cpu().numpy()
            opp_y = opp_t["y"].detach().cpu().numpy()
            opp_yaw = opp_t["yaw"].detach().cpu().numpy()
            ep_id = env_t["episode_id"].detach().cpu().numpy().astype(np.int64)
            env_idx = np.arange(int(num_envs), dtype=np.int64)
            ep_key = env_idx * 1_000_000 + ep_id

            dx = opp_x - ego_x
            dy = opp_y - ego_y
            cos_y = np.cos(ego_yaw)
            sin_y = np.sin(ego_yaw)
            rel_x = cos_y * dx + sin_y * dy
            rel_y = -sin_y * dx + cos_y * dy
            dist_m = np.hypot(rel_x, rel_y)
            theta_opp_rad = np.arctan2(rel_y, rel_x)

            fov_mask = filter_fov_and_distance_mask(
                dist_m,
                theta_opp_rad,
                dist_min_m=dist_min_m,
                dist_max_m=dist_max_m,
                fov_deg=fov_deg,
            )
            raw_actor_np = obs.detach().cpu().numpy()
            los_mask = check_lidar_opponent_obb_hit(
                ego_x=ego_x,
                ego_y=ego_y,
                ego_yaw=ego_yaw,
                opp_x=opp_x,
                opp_y=opp_y,
                opp_yaw=opp_yaw,
                raw_lidar_ranges=raw_actor_np[:, :LIDAR_DIM],
                fov_deg=fov_deg,
                car_length_m=car_length_m,
                car_width_m=car_width_m,
                lidar_offset_x=lidar_offset_x,
                lidar_offset_y=lidar_offset_y,
                lidar_offset_yaw=lidar_offset_yaw,
            )
            mask = fov_mask & los_mask
            if np.any(mask):
                lidar_norm = (
                    model_obs[:, :LIDAR_DIM].detach().cpu().numpy()[mask]
                )
                lidar_chunks.append(lidar_norm)
                raw_actor_obs_chunks.append(raw_actor_np[mask])
                ego_x_chunks.append(ego_x[mask])
                ego_y_chunks.append(ego_y[mask])
                ego_yaw_chunks.append(ego_yaw[mask])
                opp_x_chunks.append(opp_x[mask])
                opp_y_chunks.append(opp_y[mask])
                opp_yaw_chunks.append(opp_yaw[mask])
                ep_key_chunks.append(ep_key[mask])
                collected += int(mask.sum())

            if recurrent:
                actions, _, hidden = actor.step(
                    model_obs,
                    hidden,
                    reset_mask=None,
                    deterministic=True,
                    with_logprob=False,
                )
            else:
                actions, _ = actor(
                    model_obs, deterministic=True, with_logprob=False
                )
            actions = actions.clamp(-clip_actions, clip_actions)
            plant_actions = executed_plant_action(actions)
            raw_obs, _rew, done, _extras = env.step(
                plant_actions.to(rt.tc_float),
                n_steps=env.control_interval,
                with_sensors=True,
            )
            obs = raw_obs["actor"].to(device=device, dtype=torch.float32)
            if recurrent and hidden is not None:
                done_b = done.to(device=device, dtype=torch.bool)
                hidden[done_b] = 0.0
            total_env_steps += int(num_envs)

    if collected < int(target_frames):
        raise RuntimeError(
            f"Only collected {collected}/{target_frames} filtered multi-car frames "
            f"after {max_rollout_steps} steps"
        )

    n_keep = int(target_frames)
    lidar_arr = np.concatenate(lidar_chunks, axis=0)[:n_keep]
    raw_actor_obs_arr = np.concatenate(raw_actor_obs_chunks, axis=0)[:n_keep]
    ego_x_arr = np.concatenate(ego_x_chunks, axis=0)[:n_keep]
    ego_y_arr = np.concatenate(ego_y_chunks, axis=0)[:n_keep]
    ego_yaw_arr = np.concatenate(ego_yaw_chunks, axis=0)[:n_keep]
    opp_x_arr = np.concatenate(opp_x_chunks, axis=0)[:n_keep]
    opp_y_arr = np.concatenate(opp_y_chunks, axis=0)[:n_keep]
    opp_yaw_arr = np.concatenate(opp_yaw_chunks, axis=0)[:n_keep]
    ep_key_arr = np.concatenate(ep_key_chunks, axis=0)[:n_keep]

    opp_s_arr, opp_ey_arr, w_left_opp, w_right_opp = (
        project_positions_to_track(env._track_host, opp_x_arr, opp_y_arr)
    )
    targets = compute_privileged_targets(
        ego_x=ego_x_arr,
        ego_y=ego_y_arr,
        ego_yaw=ego_yaw_arr,
        opp_x=opp_x_arr,
        opp_y=opp_y_arr,
        opp_yaw=opp_yaw_arr,
        opp_s=opp_s_arr,
        opp_ey=opp_ey_arr,
        w_left_at_opp=w_left_opp,
        w_right_at_opp=w_right_opp,
        car_width_m=car_width_m,
    )

    return {
        "lidar": lidar_arr,
        "raw_actor_obs": raw_actor_obs_arr,
        "targets": targets,
        "episode_keys": ep_key_arr,
        "actor": actor,
        "checkpoint": str(checkpoint_path),
        "resolved_config": resolved_cfg_path,
        "env_transitions": int(payload.get("env_transitions", -1)),
        "total_env_steps_rolled": total_env_steps,
        "n_frames": n_keep,
        "n_episodes": int(np.unique(ep_key_arr).size),
    }


def format_markdown_report(
    *,
    checkpoint_label: str,
    checkpoint_path: str,
    n_frames: int,
    n_episodes: int,
    n_train: int,
    n_test: int,
    trained_probe: dict[str, Any],
    ortho_probe: dict[str, Any],
    trained_gate: dict[str, Any],
    ortho_gate: dict[str, Any],
    extra_checkpoints: list[dict[str, Any]] | None = None,
) -> str:
    """Format the full Phase 1 diagnostic report in Markdown."""
    lines: list[str] = []
    lines.append("# Phase 1: CNN Spatial Bottleneck Zero-Training Linear Probe Results")
    lines.append("")
    lines.append("## Executive Summary & Go/No-Go Gate Decision")
    lines.append("")
    overall_go = (
        trained_gate["decision"] == "GO" or ortho_gate["decision"] == "GO"
    )
    decision_str = (
        "GO TO PHASE 2/3 GPU A/B"
        if overall_go
        else "STOP (NO GPU RENTAL NEEDED)"
    )
    lines.append(f"- **Phase 1 Gate Decision:** **{decision_str}**")
    lines.append(
        f"- **Primary Evaluated Checkpoint:** `{checkpoint_label}` "
        f"(`{checkpoint_path}`)"
    )
    lines.append(
        f"- **Dataset:** `{n_frames:,}` multi-car frames on `Austin` with "
        "active opponents inside `[1.0 m, 12.0 m]` and within the "
        f"$270^\\circ$ LiDAR FOV across `{n_episodes:,}` episodes "
        f"(`{n_train:,}` train / `{n_test:,}` held-out test frames, "
        "split disjointly by episode)."
    )
    lines.append(
        "- **Go/No-Go Criterion:** `>= 15%` lower MAE or `>= +0.08` higher "
        "$R^2$ in the `4-12 m` opponent distance bucket for `MaxPool1d(64)` / "
        "`MaxPool1d(64)->512` / unpooled `conv_feat (64x91)` vs current "
        "`z_256 = AvgPool1d(32)->256` on opponent lateral position `rel_y`, "
        "bearing $\\theta_{\\text{opp}}$, or passable gap width "
        "(`gap_left_m`, `gap_right_m`)."
    )
    lines.append("")

    def _append_gate_summary(title: str, gate: dict[str, Any]) -> None:
        lines.append(f"### {title} (`4-12 m` Mid+Far Gate Summary)")
        lines.append("")
        lines.append(
            "| Candidate Representation | Target | Baseline `z_256` MAE | "
            "Candidate MAE | MAE Reduction | Baseline $R^2$ | "
            "Candidate $R^2$ | $\\Delta R^2$ | "
            "Gate (`>=15%` MAE or `>=+0.08` $R^2$) |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
        )
        for cand_name, t_map in gate["comparisons"].items():
            for t_name, row in t_map.items():
                unit = "deg" if t_name.endswith("_deg") else "m"
                b_mae = float(row["baseline_mae"])
                c_mae = float(row["candidate_mae"])
                red_pct = 100.0 * float(row["mae_rel_reduction"])
                b_r2 = float(row["baseline_r2"])
                c_r2 = float(row["candidate_r2"])
                d_r2 = float(row["r2_delta"])
                status = "**PASS (GO)**" if row["passes_gate"] else "fail"
                lines.append(
                    f"| `{cand_name}` | `{t_name}` ({unit}) | "
                    f"{b_mae:.4f} | {c_mae:.4f} | {red_pct:+.1f}% | "
                    f"{b_r2:.4f} | {c_r2:.4f} | {d_r2:+.4f} | {status} |"
                )
        lines.append("")

    _append_gate_summary(
        f"1. Frozen Trained Checkpoint (`{checkpoint_label}`)", trained_gate
    )
    _append_gate_summary(
        "2. Random Orthogonal-Init CNN Structural Baseline", ortho_gate
    )

    def _append_full_tables(title: str, probe: dict[str, Any]) -> None:
        lines.append(
            f"## {title} — Held-Out Test $R^2$ and MAE by Distance Bucket"
        )
        lines.append("")
        for bucket_label, _, _ in DISTANCE_BUCKETS:
            first_rep = next(iter(probe.values()))
            n_bucket = first_rep["buckets"][bucket_label]["rel_x"]["n"]
            lines.append(
                f"### Bucket: `{bucket_label}` (`n = {n_bucket:,}` held-out frames)"
            )
            lines.append("")
            lines.append(
                "| Representation | Dim | `rel_x` MAE (m) / $R^2$ | "
                "`rel_y` MAE (m) / $R^2$ | `dist` MAE (m) / $R^2$ | "
                "`$\\theta_{\\text{opp}}$` MAE (deg) / $R^2$ | "
                "`$\\Delta\\psi$` MAE (deg) / $R^2$ | "
                "`gap_left` MAE (m) / $R^2$ | `gap_right` MAE (m) / $R^2$ |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for rep_name in REPRESENTATION_ORDER:
                if rep_name not in probe:
                    continue
                rep_data = probe[rep_name]
                b = rep_data["buckets"][bucket_label]
                cells = [f"`{rep_name}`", f"{rep_data['dim']}"]
                for t_name, _unit, _scale in TARGET_SPECS:
                    m = b[t_name]
                    cells.append(f"{m['mae']:.4f} / {m['r2']:.4f}")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    _append_full_tables(
        f"Frozen Trained Checkpoint (`{checkpoint_label}`)", trained_probe
    )
    _append_full_tables(
        "Random Orthogonal-Init CNN Structural Baseline", ortho_probe
    )

    if extra_checkpoints:
        lines.append("## Additional Trained Checkpoint Verification")
        lines.append("")
        for extra in extra_checkpoints:
            _append_gate_summary(
                f"Trained Checkpoint `{extra['label']}` (`{extra['checkpoint']}`)",
                extra["gate"],
            )
            _append_full_tables(
                f"Trained Checkpoint (`{extra['label']}`)",
                extra["probe"],
            )

    return "\n".join(lines) + "\n"


def run_probe_pipeline(
    checkpoint_path: Path,
    *,
    extra_checkpoint_paths: list[Path] | None = None,
    config_path: Path | None = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    num_envs: int = 64,
    device_name: str = "cpu",
    seed: int = 42,
    out_markdown: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Run the full Phase 1 linear probe on primary + optional extra checkpoints."""
    dataset = collect_multicar_probe_dataset(
        checkpoint_path,
        config_path=config_path,
        target_frames=target_frames,
        num_envs=num_envs,
        device_name=device_name,
        seed=seed,
    )
    lidar_t = torch.from_numpy(dataset["lidar"])
    train_mask, test_mask = split_by_episode(
        dataset["episode_keys"], train_frac=DEFAULT_TRAIN_FRAC, seed=seed
    )

    trained_encoder: LidarCNNEncoder = dataset["actor"].encoder
    trained_feats = extract_candidate_representations(trained_encoder, lidar_t)
    trained_probe = evaluate_probe_on_features(
        trained_feats,
        dataset["targets"],
        train_mask,
        test_mask,
        episode_keys=dataset["episode_keys"],
    )
    trained_gate = evaluate_go_no_go_gate(trained_probe)

    ortho_encoder = build_orthogonal_cnn_encoder(
        lidar_dim=LIDAR_DIM,
        pool_bins=32,
        projection_dim=256,
        seed=seed,
        device=device_name,
    )
    ortho_feats = extract_candidate_representations(ortho_encoder, lidar_t)
    ortho_probe = evaluate_probe_on_features(
        ortho_feats,
        dataset["targets"],
        train_mask,
        test_mask,
        episode_keys=dataset["episode_keys"],
    )
    ortho_gate = evaluate_go_no_go_gate(ortho_probe)

    extra_results: list[dict[str, Any]] = []
    for extra_ckpt in extra_checkpoint_paths or []:
        if not extra_ckpt.is_file():
            continue
        from eval_full_lap import load_actor_bundle, resolve_eval_config

        extra_cfg, _ = resolve_eval_config(extra_ckpt, config_path)
        extra_actor, extra_norm, _arch, _payload = load_actor_bundle(
            extra_ckpt, torch.device(device_name), extra_cfg
        )
        extra_actor.eval()
        raw_obs_t = torch.from_numpy(dataset["raw_actor_obs"]).to(
            device=torch.device(device_name), dtype=torch.float32
        )
        extra_lidar_t = extra_norm.normalize(raw_obs_t)[:, :LIDAR_DIM]
        extra_feats = extract_candidate_representations(
            extra_actor.encoder, extra_lidar_t
        )
        extra_probe = evaluate_probe_on_features(
            extra_feats,
            dataset["targets"],
            train_mask,
            test_mask,
            episode_keys=dataset["episode_keys"],
        )
        extra_gate = evaluate_go_no_go_gate(extra_probe)
        extra_results.append(
            {
                "label": f"{extra_ckpt.parent.parent.name}/{extra_ckpt.name}",
                "checkpoint": str(extra_ckpt),
                "probe": extra_probe,
                "gate": extra_gate,
            }
        )

    ckpt_label = f"{checkpoint_path.parent.parent.name}/{checkpoint_path.name}"
    report_md = format_markdown_report(
        checkpoint_label=ckpt_label,
        checkpoint_path=str(checkpoint_path),
        n_frames=int(dataset["n_frames"]),
        n_episodes=int(dataset["n_episodes"]),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        trained_probe=trained_probe,
        ortho_probe=ortho_probe,
        trained_gate=trained_gate,
        ortho_gate=ortho_gate,
        extra_checkpoints=extra_results,
    )

    if out_markdown is not None:
        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(report_md)

    summary_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_label": ckpt_label,
        "n_frames": int(dataset["n_frames"]),
        "n_episodes": int(dataset["n_episodes"]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "trained_gate": trained_gate,
        "ortho_gate": ortho_gate,
        "trained_probe": trained_probe,
        "ortho_probe": ortho_probe,
        "extra_checkpoints": extra_results,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary_payload, indent=2))

    return summary_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-training linear probe for the 1D LiDAR CNN bottleneck."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to primary trained sensor-policy checkpoint (.pt).",
    )
    parser.add_argument(
        "--extra-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Optional additional checkpoint(s) to evaluate on the collected frames.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config JSON override.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help="Number of filtered multi-car frames to collect (default: 16384).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel environments for rollout (default: 64).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for rollout and feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=None,
        help="Output path for the Markdown probe report.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output path for the JSON probe results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_probe_pipeline(
        args.checkpoint.resolve(),
        extra_checkpoint_paths=[p.resolve() for p in args.extra_checkpoint],
        config_path=args.config.resolve() if args.config else None,
        target_frames=args.target_frames,
        num_envs=args.num_envs,
        device_name=args.device,
        seed=args.seed,
        out_markdown=args.out_markdown.resolve() if args.out_markdown else None,
        out_json=args.out_json.resolve() if args.out_json else None,
    )
    print(
        f"Probe finished: n_frames={summary['n_frames']} "
        f"trained_gate={summary['trained_gate']['decision']} "
        f"ortho_gate={summary['ortho_gate']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

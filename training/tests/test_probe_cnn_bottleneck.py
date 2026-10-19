from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from analysis.probe_cnn_bottleneck import (
    REPRESENTATION_ORDER,
    build_orthogonal_cnn_encoder,
    compute_privileged_targets,
    evaluate_go_no_go_gate,
    evaluate_probe_on_features,
    extract_candidate_representations,
    filter_fov_and_distance_mask,
    fit_ridge_closed_form,
    split_by_episode,
)
from f1tenth_policy.layout import LIDAR_DIM


def test_extract_candidate_representations_shapes():
    encoder = build_orthogonal_cnn_encoder(
        lidar_dim=LIDAR_DIM,
        pool_bins=32,
        projection_dim=256,
        seed=7,
    )
    lidar = torch.randn(12, LIDAR_DIM, dtype=torch.float32)
    reps = extract_candidate_representations(encoder, lidar, batch_size=5)
    assert set(reps.keys()) == set(REPRESENTATION_ORDER)
    assert reps["Raw LiDAR (1081-D)"].shape == (12, LIDAR_DIM)
    assert reps["z_256 (AvgPool32->256)"].shape == (12, 256)
    assert reps["AvgPool1d(32) (2048-D)"].shape == (12, 2048)
    assert reps["MaxPool1d(32) (2048-D)"].shape == (12, 2048)
    assert reps["MaxPool1d(64)->512 (512-D)"].shape == (12, 512)
    assert reps["MaxPool1d(64) (4096-D)"].shape == (12, 4096)
    assert reps["Unpooled conv (64x91=5824-D)"].shape == (12, 5824)


def test_compute_privileged_targets_and_fov_filter():
    ego_x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ego_yaw = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    opp_x = np.array([5.0, 0.5, -6.0], dtype=np.float32)
    opp_y = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    opp_yaw = np.array([0.0, 0.25, math.pi], dtype=np.float32)
    opp_s = np.array([10.0, 12.0, 15.0], dtype=np.float32)
    opp_ey = np.array([0.2, -0.1, 0.0], dtype=np.float32)
    w_left = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    w_right = np.array([1.5, 1.5, 1.5], dtype=np.float32)

    targets = compute_privileged_targets(
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        opp_x=opp_x,
        opp_y=opp_y,
        opp_yaw=opp_yaw,
        opp_s=opp_s,
        opp_ey=opp_ey,
        w_left_at_opp=w_left,
        w_right_at_opp=w_right,
        car_width_m=0.30,
    )
    assert np.allclose(targets["rel_x"], [5.0, 0.5, -6.0], atol=1e-5)
    assert np.allclose(targets["rel_y"], [0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(targets["dist_m"], [5.0, 0.5, 6.0], atol=1e-5)
    assert targets["gap_left_m"][0] == pytest.approx(1.5 - (0.2 + 0.15), abs=1e-5)
    assert targets["gap_right_m"][0] == pytest.approx(1.5 + (0.2 - 0.15), abs=1e-5)

    mask = filter_fov_and_distance_mask(
        targets["dist_m"],
        targets["theta_opp_rad"],
        dist_min_m=1.0,
        dist_max_m=12.0,
        fov_deg=270.0,
    )
    # Frame 0: d=5.0m, theta=0 -> True
    # Frame 1: d=0.5m < 1.0m -> False
    # Frame 2: d=6.0m, theta=180 deg > 135 deg -> False
    assert mask.tolist() == [True, False, False]


def test_ridge_closed_form_recovers_linear_function():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(200, 16))
    true_w = rng.normal(size=(16, 2))
    true_b = np.array([1.5, -0.7])
    y = x @ true_w + true_b
    fit = fit_ridge_closed_form(x[:150], y[:150], alpha=1e-2)
    pred = fit.predict(x[150:])
    assert np.allclose(pred, y[150:], atol=1e-2)


def test_split_by_episode_and_go_no_go_gate():
    ep_keys = np.repeat(np.arange(20), 10)
    train_mask, test_mask = split_by_episode(ep_keys, train_frac=0.8, seed=0)
    assert not np.any(train_mask & test_mask)
    assert np.all(train_mask | test_mask)
    train_eps = set(ep_keys[train_mask].tolist())
    test_eps = set(ep_keys[test_mask].tolist())
    assert train_eps.isdisjoint(test_eps)

    rng = np.random.default_rng(99)
    n = ep_keys.size
    dist_m = rng.uniform(1.0, 12.0, size=n).astype(np.float32)
    rel_y = rng.normal(0.0, 0.8, size=n).astype(np.float32)
    targets = {
        "rel_x": dist_m,
        "rel_y": rel_y,
        "dist_m": dist_m,
        "theta_opp_deg": np.degrees(np.arctan2(rel_y, dist_m)).astype(np.float32),
        "delta_psi_deg": np.zeros(n, dtype=np.float32),
        "gap_left_m": (1.2 - rel_y).astype(np.float32),
        "gap_right_m": (1.2 + rel_y).astype(np.float32),
    }

    # Construct synthetic features where unpooled / MaxPool64 carries exact rel_y
    # while AvgPool32->256 is heavily corrupted by noise.
    clean_signal = np.column_stack(
        [rel_y, dist_m, targets["theta_opp_deg"], rng.normal(size=(n, 13))]
    )
    noisy_signal = rng.normal(size=(n, 16))
    features = {
        "z_256 (AvgPool32->256)": noisy_signal,
        "AvgPool1d(32) (2048-D)": noisy_signal,
        "MaxPool1d(32) (2048-D)": clean_signal,
        "MaxPool1d(64)->512 (512-D)": clean_signal,
        "MaxPool1d(64) (4096-D)": clean_signal,
        "Unpooled conv (64x91=5824-D)": clean_signal,
    }
    probe = evaluate_probe_on_features(features, targets, train_mask, test_mask)
    gate = evaluate_go_no_go_gate(probe)
    assert gate["decision"] == "GO"

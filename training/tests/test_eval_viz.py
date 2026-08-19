"""Focused tests for eval rollout color utilities (no rendering backends)."""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from config import DEFAULT_CONFIG
from eval_visualize import build_eval_config
from f1tenth_env.eval_viz import _C_OPP, _OPP_EGO_BLEND, _car_color, _opp_color
from f1tenth_env.eval_viz import RolloutVisualizer


def test_opp_color_matches_ego_index():
    for i in range(6):
        ego = _car_color(i)
        opp = _opp_color(i)
        for c in range(3):
            expected = int(ego[c] * _OPP_EGO_BLEND + _C_OPP[c] * (1.0 - _OPP_EGO_BLEND))
            assert opp[c] == expected


def test_opp_colors_differ_across_envs():
    colors = {_opp_color(i) for i in range(4)}
    assert len(colors) == 4


def test_opp_color_is_muted_vs_ego():
    for i in range(4):
        ego = _car_color(i)
        opp = _opp_color(i)
        assert all(abs(opp[c] - _C_OPP[c]) < abs(ego[c] - _C_OPP[c]) for c in range(3))


def test_visualizer_writes_finite_rollout_frames(tmp_path):
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    centerline = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)
    width = np.full(64, 0.3, dtype=np.float32)
    path = tmp_path / "rollout.mp4"
    visualizer = RolloutVisualizer(
        centerline,
        width,
        width,
        num_show=2,
        mp4_path=str(path),
        img_size=128,
        has_opponent=True,
    )
    for step in range(3):
        visualizer.render(
            ego_xy=np.array([[1.0, 0.01 * step], [0.9, 0.1]], dtype=np.float32),
            ego_yaw=np.array([0.0, 0.1], dtype=np.float32),
            speed=np.array([2.0, 1.5], dtype=np.float32),
            opp_xy=np.array([[0.8, 0.0], [0.7, 0.1]], dtype=np.float32),
            opp_yaw=np.array([0.0, 0.1], dtype=np.float32),
            lidar_points=np.array(
                [
                    [[1.2, 0.1], [1.1, -0.1]],
                    [[0.8, 0.2], [0.8, -0.1]],
                ],
                dtype=np.float32,
            ),
        )
    assert visualizer.close() == str(path)
    assert path.stat().st_size > 0


def test_eval_config_can_disable_training_speed_cap(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "policy.pt"
    checkpoint.touch()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["env"]["ego_speed_cap_mps"] = 4.0
    (run_dir / "config.json").write_text(json.dumps({"config": config}))

    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        track="Austin",
        opponent_ckpt=None,
        ego_speed_cap_mps=0.0,
    )
    eval_config = build_eval_config(args)

    assert eval_config["env"]["ego_speed_cap_mps"] == 0.0

"""Focused tests for eval rollout color utilities (no rendering backends)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
        )
    assert visualizer.close() == str(path)
    assert path.stat().st_size > 0


def _follow_targets():
    """Eight cars share a start. Seven keep moving; one crashes early."""
    import numpy as np

    frames = 24
    step_m = 0.15
    targets = []
    crash_at = 4
    for t in range(frames):
        alive_x = step_m * t
        targets.append(np.array([alive_x, 0.0]))
    crash_xy = np.array([step_m * (crash_at - 1), 0.0])
    return frames, crash_at, crash_xy, targets


def _write_follow_npz(path):
    import numpy as np

    frames, crash_at, crash_xy, targets = _follow_targets()
    square = np.array(
        [[-8.0, -8.0], [8.0, -8.0], [8.0, 8.0], [-8.0, 8.0]],
        dtype=np.float64,
    )
    payload = {
        "lengths": np.array([frames] * 7 + [crash_at], dtype=np.int32),
        "crashed": np.array([False] * 7 + [True]),
        "car_length": np.float64(0.58),
        "car_width": np.float64(0.30),
        "left": square,
        "right": square * 0.7,
        "centerline": square * 0.85,
    }
    opp = np.array([0.0, 2.0])
    for i in range(8):
        n = int(payload["lengths"][i])
        xy = np.array([targets[t] for t in range(n)], dtype=np.float64)
        if i == 7:
            xy[-1] = crash_xy
        payload[f"ego_xy_{i}"] = xy
        payload[f"ego_yaw_{i}"] = np.zeros(n, dtype=np.float64)
        payload[f"ego_speed_{i}"] = np.full(n, 3.0, dtype=np.float64)
        payload[f"opp_xy_{i}"] = np.repeat(opp[None, :], n, axis=0)
        payload[f"opp_yaw_{i}"] = np.zeros(n, dtype=np.float64)
    np.savez(path, **payload)


def _eased_cameras(targets):
    import numpy as np

    cam = None
    cams = []
    for target in targets:
        target = np.asarray(target, dtype=np.float64)
        cam = target.copy() if cam is None else 0.82 * cam + 0.18 * target
        cams.append(cam.copy())
    return cams


def test_follow_renderer_eases_keeps_trail_and_marks_crash(tmp_path):
    """Drive the recovered renderer. Do not reimplement its camera."""
    import subprocess
    import sys

    import imageio.v2 as imageio
    import numpy as np

    npz = tmp_path / "follow.npz"
    mp4 = tmp_path / "overlay_follow.mp4"
    _write_follow_npz(npz)
    script = Path(__file__).resolve().parents[1] / "render_follow.py"
    subprocess.run(
        [sys.executable, str(script), "--npz", str(npz), "--mp4", str(mp4),
         "--fps", "10", "--width", "1120", "--height", "700"],
        check=True,
    )
    reader = imageio.get_reader(mp4)
    frame = reader.get_data(reader.count_frames() - 1)
    reader.close()
    assert frame.shape == (700, 1120, 3)

    _frames, _crash_at, crash_xy, targets = _follow_targets()
    cam = _eased_cameras(targets)[-1]
    scale = 1120 / 26.0
    live_px = (targets[-1][0] - cam[0]) * scale + 1120 / 2
    crash_px = (crash_xy[0] - cam[0]) * scale + 1120 / 2
    crash_py = (cam[1] - crash_xy[1]) * scale + 700 / 2
    static_px = (targets[-1][0] - targets[0][0]) * scale + 1120 / 2

    mid = 350
    band = frame[mid - 50:mid + 50]
    live = np.argwhere(
        (band[:, :, 1] > 200) & (band[:, :, 2] > 180) & (band[:, :, 0] < 180)
    )
    assert len(live) > 20
    centroid = live.mean(axis=0)
    assert abs(centroid[1] - live_px) < 35
    assert abs(centroid[1] - static_px) > 60

    from render_follow import CAR_COLORS, OPP_COLOR

    opp_xy = np.array([0.0, 2.0])
    opp_px = (opp_xy[0] - cam[0]) * scale + 1120 / 2
    opp_py = (cam[1] - opp_xy[1]) * scale + 700 / 2
    ox = int(np.clip(opp_px, 0, 1119))
    oy = int(np.clip(opp_py, 0, 699))
    patch = frame[max(0, oy - 22):oy + 22, max(0, ox - 22):ox + 22]
    ego0 = np.array(CAR_COLORS[0])
    border = np.abs(patch.astype(int) - ego0).sum(axis=2) <= 24
    assert border.any()
    fill = np.abs(patch.astype(int) - np.array(OPP_COLOR)).sum(axis=2) <= 24
    assert fill.any()

    legend = frame[20, 1120 - 202]
    assert legend.min() > 180

    x0 = int(np.clip(crash_px, 0, 1119))
    y0 = int(np.clip(crash_py, 0, 699))
    window = frame[max(0, y0 - 18):y0 + 18, max(0, x0 - 18):x0 + 18]
    background = np.array([16, 17, 20])
    marked = np.abs(window.astype(int) - background).max(axis=2) > 25
    assert marked.any()
    filled = (
        (window[:, :, 0] > 220)
        & (window[:, :, 1] > 140)
        & (window[:, :, 2] > 100)
    )
    assert int(filled.sum()) < 20

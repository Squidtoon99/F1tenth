"""Warp plant + sensor_racer packing should clear Austin's first corner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for path in (
    _ROOT,
    _ROOT / "training",
    _ROOT / "src/racing_rl/f1tenth_rl_agent",
    _ROOT / "libs/f1tenth_policy",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.warp_sensor_lockstep import run_lockstep  # noqa: E402

_CKPT = Path(
    "/home/ubuntu/projects/F1tenth-baseline-v5/outputs/runs/"
    "baseline-v5-s42/checkpoints/policy_43520000.pt"
)


@pytest.mark.skipif(not _CKPT.is_file(), reason="baseline-v5 s42 checkpoint missing")
def test_warp_ros_packing_clears_first_austin_corner(warp_runtime):
    del warp_runtime
    result = run_lockstep(_CKPT, steps=160)
    first = result["first_crash"] or result["first_offtrack"]
    if first is not None:
        assert first["s_m"] > 8.0, first
        assert first["t_s"] > 4.0, first
    assert result["spawn_speed_mps"] == pytest.approx(2.0)
    assert result["progress_m"] > 12.0
    assert result["crashes"] == 0 or result["progress_m"] > 12.0
    assert result["laps"] == 0
    assert result["lap_times_s"] == []
    assert result["steps"] == 160

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from conftest import (
    brute_force_frenet,
    build_track_state,
    make_circle_track,
)
from f1tenth_env.kernel import TrackData, project_track_kernel
from f1tenth_env.utils import build_warp_track_data


def _upload(host, device):
    track = TrackData()
    track.point = wp.array(host.point, dtype=wp.vec2f, device=device)
    track.tangent = wp.array(host.tangent, dtype=wp.vec2f, device=device)
    track.normal = wp.array(host.normal, dtype=wp.vec2f, device=device)
    track.segment_length = wp.array(
        host.segment_length, dtype=wp.float32, device=device
    )
    track.cumulative_length = wp.array(
        host.cumulative_length, dtype=wp.float32, device=device
    )
    track.width_left = wp.array(host.width_left, dtype=wp.float32, device=device)
    track.width_right = wp.array(host.width_right, dtype=wp.float32, device=device)
    track.nearest_segment_lut = wp.array(
        host.nearest_segment_lut, dtype=wp.int32, device=device
    )
    track.count = host.point.shape[0]
    track.length = host.length
    track.lut_width = host.lut_width
    track.lut_height = host.lut_height
    track.lut_origin = wp.vec2f(*host.lut_origin)
    track.lut_resolution = host.lut_resolution
    return track


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_warp_frenet_matches_brute_force(real_modules, device):
    centerline, width_left, width_right = make_circle_track(radius=20.0, n=360)
    state = build_track_state(
        real_modules.utils, centerline, width_left, width_right
    )
    host = build_warp_track_data(state, lut_resolution=0.25)
    track = _upload(host, device)
    angle = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False, dtype=np.float32)
    radius = 20.0 + 0.8 * np.sin(7.0 * angle)
    positions = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    oracle = brute_force_frenet(positions, centerline)

    torch_device = torch.device(device)
    position_tensor = torch.as_tensor(
        positions, device=torch_device, dtype=torch.float32
    )
    seeds = torch.zeros(positions.shape[0], device=torch_device, dtype=torch.int32)
    segments = torch.empty_like(seeds)
    s = torch.empty(positions.shape[0], device=torch_device)
    ey = torch.empty_like(s)
    boundary = torch.empty_like(s)
    wp.launch(
        project_track_kernel,
        dim=positions.shape[0],
        inputs=[
            wp.from_torch(position_tensor, dtype=wp.vec2f),
            wp.from_torch(seeds),
            track,
            wp.from_torch(segments),
            wp.from_torch(s),
            wp.from_torch(ey),
            wp.from_torch(boundary),
        ],
        device=device,
    )
    wp.synchronize_device(device)

    assert np.max(np.abs(s.cpu().numpy() - oracle["s"])) < 5.0e-4
    assert np.max(np.abs(ey.cpu().numpy() - oracle["ey"])) < 2.0e-4
    assert torch.isfinite(boundary).all()

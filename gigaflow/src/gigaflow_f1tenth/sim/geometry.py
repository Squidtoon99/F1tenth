"""Track geometry helpers over PackedTrackAtlasView (+ derived normals/segments)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from gigaflow_f1tenth.tracks import (
    PackedTrackAtlasView,
    build_track_arrays,
    make_synthetic_oval_table,
    pack_built_tracks,
)

FRENET_WINDOW = 24


@wp.struct
class FrenetState:
    segment: wp.int32
    t: wp.float32
    s: wp.float32
    ey: wp.float32
    width_left: wp.float32
    width_right: wp.float32
    boundary_distance: wp.float32
    distance_sq: wp.float32


@dataclass
class SimTrackGeometry:
    """CPU/GPU geometry pack: atlas fields plus derived normals/segment lengths."""

    num_tracks: int
    offsets: np.ndarray
    centerline_xy: np.ndarray
    tangents_xy: np.ndarray
    normals_xy: np.ndarray
    widths_rl: np.ndarray
    cum_length: np.ndarray
    segment_length: np.ndarray
    track_length: np.ndarray
    capacity: np.ndarray
    edt_distance: np.ndarray
    edt_offsets: np.ndarray
    edt_width: np.ndarray
    edt_height: np.ndarray
    edt_origin: np.ndarray
    edt_resolution: np.ndarray
    track_ids: tuple[str, ...]
    lut_offsets: np.ndarray
    nearest_segment_lut: np.ndarray
    lut_width: np.ndarray
    lut_height: np.ndarray
    lut_origin_xy: np.ndarray
    lut_resolution: np.ndarray


def _as_numpy(array, dtype) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return np.asarray(array, dtype=dtype)
    if hasattr(array, "detach"):
        return np.asarray(array.detach().cpu().numpy(), dtype=dtype)
    if hasattr(array, "numpy"):
        return np.asarray(array.numpy(), dtype=dtype)
    return np.asarray(array, dtype=dtype)


def derive_normals(tangents: np.ndarray) -> np.ndarray:
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0e-8)
    return (normals / norms).astype(np.float32)


def derive_segment_lengths(cum_length: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    seg = np.zeros_like(cum_length, dtype=np.float32)
    for t in range(len(offsets) - 1):
        a = int(offsets[t])
        b = int(offsets[t + 1])
        if b - a < 2:
            continue
        seg[a : b - 1] = cum_length[a + 1 : b] - cum_length[a : b - 1]
        last = max(float(cum_length[b - 1] - cum_length[b - 2]), 1.0e-3)
        seg[b - 1] = last
    return seg


def build_sim_geometry_from_atlas(atlas: PackedTrackAtlasView) -> SimTrackGeometry:
    offsets = _as_numpy(atlas.offsets, np.int32)
    centerline = _as_numpy(atlas.centerline_xy, np.float32).reshape(-1, 2)
    tangents = _as_numpy(atlas.tangents_xy, np.float32).reshape(-1, 2)
    widths = _as_numpy(atlas.widths_rl, np.float32).reshape(-1, 2)
    cum = _as_numpy(atlas.cum_length, np.float32).reshape(-1)
    normals = derive_normals(tangents)
    seg_len = derive_segment_lengths(cum, offsets)
    # Fix closing segment lengths from actual centerline chords.
    for t in range(int(atlas.num_tracks)):
        a = int(offsets[t])
        b = int(offsets[t + 1])
        if b - a < 2:
            continue
        seg_len[b - 1] = float(
            np.linalg.norm(centerline[a] - centerline[b - 1])
        )
    return SimTrackGeometry(
        num_tracks=int(atlas.num_tracks),
        offsets=offsets,
        centerline_xy=centerline,
        tangents_xy=tangents,
        normals_xy=normals,
        widths_rl=widths,
        cum_length=cum,
        segment_length=seg_len,
        track_length=_as_numpy(atlas.lengths, np.float32).reshape(-1),
        capacity=_as_numpy(atlas.capacity, np.int32).reshape(-1),
        edt_distance=_as_numpy(atlas.edt_distance, np.float32).reshape(-1),
        edt_offsets=_as_numpy(atlas.edt_offsets, np.int32).reshape(-1),
        edt_width=_as_numpy(atlas.edt_width, np.int32).reshape(-1),
        edt_height=_as_numpy(atlas.edt_height, np.int32).reshape(-1),
        edt_origin=_as_numpy(atlas.edt_origin_xy, np.float32).reshape(-1, 2),
        edt_resolution=_as_numpy(atlas.edt_resolution, np.float32).reshape(-1),
        track_ids=tuple(atlas.track_ids),
        lut_offsets=_as_numpy(atlas.lut_offsets, np.int32).reshape(-1),
        nearest_segment_lut=_as_numpy(atlas.nearest_segment_lut, np.int32).reshape(-1),
        lut_width=_as_numpy(atlas.lut_width, np.int32).reshape(-1),
        lut_height=_as_numpy(atlas.lut_height, np.int32).reshape(-1),
        lut_origin_xy=_as_numpy(atlas.lut_origin_xy, np.float32).reshape(-1, 2),
        lut_resolution=_as_numpy(atlas.lut_resolution, np.float32).reshape(-1),
    )


def make_synthetic_oval_atlas(
    *,
    radius: float = 8.0,
    half_width: float = 1.1,
    num_points: int = 64,
    name: str = "synthetic_oval",
    max_agents: int = 8,
) -> PackedTrackAtlasView:
    table = make_synthetic_oval_table(radius=radius, n=num_points, width=half_width)
    built = build_track_arrays(
        table,
        name=name,
        source_sha="synthetic",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    return pack_built_tracks([built]).view()


def make_two_track_atlas(*, max_agents: int = 8) -> PackedTrackAtlasView:
    a = build_track_arrays(
        make_synthetic_oval_table(radius=8.0, n=64, width=1.1),
        name="oval_a",
        source_sha="synthetic-a",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    b = build_track_arrays(
        make_synthetic_oval_table(radius=6.0, n=48, width=1.0),
        name="oval_b",
        source_sha="synthetic-b",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    return pack_built_tracks([a, b]).view()


def make_offset_mismatch_atlas(*, max_agents: int = 8) -> PackedTrackAtlasView:
    """Two-track atlas that exposes global-vs-local ``project_window`` seeds.

    Track-1 must be longer than ``2 * FRENET_WINDOW + 1`` segments (else a
    wrong seed still covers the whole loop), and
    ``min(offset % count, count - offset % count)`` must exceed
    ``FRENET_WINDOW`` so the true local segment falls outside the search.
    """
    a = build_track_arrays(
        make_synthetic_oval_table(radius=12.0, n=120, width=1.2),
        name="oval_long",
        source_sha="synthetic-long",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    b = build_track_arrays(
        make_synthetic_oval_table(radius=7.0, n=80, width=1.0),
        name="oval_short",
        source_sha="synthetic-short",
        max_agents=max_agents,
        license_id="local-fixture",
        optional_local=True,
    )
    return pack_built_tracks([a, b]).view()


@wp.func
def wrap_segment(index: wp.int32, count: wp.int32) -> wp.int32:
    wrapped = index % count
    if wrapped < 0:
        wrapped = wrapped + count
    return wrapped


@wp.func
def wrapped_delta(delta: wp.float32, length: wp.float32) -> wp.float32:
    half = 0.5 * length
    out = delta
    if out > half:
        out = out - length
    if out < -half:
        out = out + length
    return out


@wp.func
def project_window(
    position: wp.vec2f,
    seed_segment: wp.int32,
    point: wp.array(dtype=wp.vec2f),
    tangent: wp.array(dtype=wp.vec2f),
    normal: wp.array(dtype=wp.vec2f),
    segment_length: wp.array(dtype=wp.float32),
    cumulative_length: wp.array(dtype=wp.float32),
    width_left: wp.array(dtype=wp.float32),
    width_right: wp.array(dtype=wp.float32),
    count: wp.int32,
    offset: wp.int32,
) -> FrenetState:
    best = FrenetState()
    best.distance_sq = wp.float32(1.0e30)
    best.segment = 0
    for relative in range(-FRENET_WINDOW, FRENET_WINDOW + 1):
        segment = wrap_segment(seed_segment + relative, count)
        following = wrap_segment(segment + 1, count)
        i0 = offset + segment
        i1 = offset + following
        start = point[i0]
        edge = point[i1] - start
        alpha = wp.clamp(
            wp.dot(position - start, edge) / wp.max(wp.dot(edge, edge), 1.0e-10),
            0.0,
            1.0,
        )
        projection = start + alpha * edge
        delta = position - projection
        distance_sq = wp.dot(delta, delta)
        if distance_sq < best.distance_sq - 1.0e-10 or (
            wp.abs(distance_sq - best.distance_sq) <= 1.0e-10 and segment < best.segment
        ):
            ey = wp.dot(delta, normal[i0])
            wl = width_left[i0] + alpha * (width_left[i1] - width_left[i0])
            wr = width_right[i0] + alpha * (width_right[i1] - width_right[i0])
            best.segment = segment
            best.t = alpha
            best.s = cumulative_length[i0] + alpha * segment_length[i0]
            best.ey = ey
            best.width_left = wl
            best.width_right = wr
            best.boundary_distance = wp.min(wl - ey, wr + ey)
            best.distance_sq = distance_sq
    return best


def project_frenet_numpy(
    position: np.ndarray,
    seed_segment: int,
    centerline: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    widths_rl: np.ndarray,
    cum_length: np.ndarray,
    segment_length: np.ndarray,
) -> dict[str, float | int]:
    count = len(centerline)
    best_dsq = 1.0e30
    best = {
        "segment": 0,
        "t": 0.0,
        "s": 0.0,
        "ey": 0.0,
        "width_left": 0.0,
        "width_right": 0.0,
        "boundary_distance": 0.0,
    }
    for relative in range(-FRENET_WINDOW, FRENET_WINDOW + 1):
        segment = (seed_segment + relative) % count
        following = (segment + 1) % count
        start = centerline[segment]
        edge = centerline[following] - start
        denom = float(np.dot(edge, edge))
        alpha = 0.0
        if denom > 1.0e-10:
            alpha = float(np.clip(np.dot(position - start, edge) / denom, 0.0, 1.0))
        projection = start + alpha * edge
        delta = position - projection
        dsq = float(np.dot(delta, delta))
        if dsq < best_dsq - 1.0e-10 or (
            abs(dsq - best_dsq) <= 1.0e-10 and segment < int(best["segment"])
        ):
            ey = float(np.dot(delta, normals[segment]))
            wl = float(
                widths_rl[segment, 1]
                + alpha * (widths_rl[following, 1] - widths_rl[segment, 1])
            )
            wr = float(
                widths_rl[segment, 0]
                + alpha * (widths_rl[following, 0] - widths_rl[segment, 0])
            )
            best_dsq = dsq
            best = {
                "segment": segment,
                "t": alpha,
                "s": float(cum_length[segment] + alpha * segment_length[segment]),
                "ey": ey,
                "width_left": wl,
                "width_right": wr,
                "boundary_distance": float(min(wl - ey, wr + ey)),
            }
    return best

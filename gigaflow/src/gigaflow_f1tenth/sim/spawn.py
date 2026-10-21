"""Capacity-aware deterministic spawn / async reset hooks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from gigaflow_f1tenth.config import ExperimentConfig, episode_steps
from gigaflow_f1tenth.sim.vehicle import VehicleParams
from gigaflow_f1tenth.sim.geometry import SimTrackGeometry, project_frenet_numpy
from gigaflow_f1tenth.tracks import sample_active_counts as atlas_sample_active_counts
from gigaflow_f1tenth.tracks import sample_track_ids as atlas_sample_track_ids

# Deterministic placement constraints for immobile "static opponent" slots,
# shared by the viewer (interactive Off/Light/Heavy presets) and training
# (fixed light-preset density). See ``place_static_opponents``.
STATIC_BOUNDARY_MARGIN_M = 0.15
STATIC_LONGITUDINAL_GAP_CAR_LENGTHS = 4.0
STATIC_PASSING_MARGIN_M = 0.15
HEAD_TO_HEAD_BEHIND_GAP_M = 8.0
HEAD_TO_HEAD_LATERAL_M = 0.14
HEAD_TO_HEAD_SPEED_DELTA_MPS = 2.5
HEAD_TO_HEAD_DRIVE_SCALE = 1.25


def mix_seed(global_seed: int, world_id: int, slot: int, episode_id: int) -> int:
    mixed = (
        int(global_seed)
        ^ (int(world_id) * 73244475)
        ^ (int(slot) * 19349663)
        ^ (int(episode_id) * 295075153)
    )
    return mixed & 0x7FFFFFFF


def assign_world_tracks(
    cfg: ExperimentConfig,
    geom: SimTrackGeometry,
    seed: int,
) -> np.ndarray:
    return atlas_sample_track_ids(
        geom.num_tracks,
        cfg.worlds.num_worlds,
        seed,
        sampling=cfg.tracks.sampling,
        lengths=geom.track_length.astype(np.float64),
    )


def sample_active_counts(
    cfg: ExperimentConfig,
    geom: SimTrackGeometry,
    track_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Learner-agent counts per world, reserving room for static opponents.

    Static opponents count toward ``max_agents_per_world`` rather than
    growing it, so the learner-count ceiling shrinks by the reserved amount
    instead of raising the per-world slot budget the VRAM estimate is built
    on.
    """
    reserved = int(cfg.worlds.static_opponents_per_world)
    effective_max = max(int(cfg.worlds.max_agents_per_world) - reserved, 1)
    return atlas_sample_active_counts(
        track_ids,
        geom.capacity,
        density_bins=cfg.worlds.density_bins,
        solo_world_fraction=cfg.worlds.solo_world_fraction,
        max_agents=effective_max,
        seed=seed,
    )


def _pose_clear_of_others(
    x: float,
    y: float,
    yaw: float,
    others: list[tuple[float, float, float]],
    car_length: float,
    car_width: float,
    clearance: float,
) -> bool:
    del yaw
    r = 0.5 * math.hypot(car_length, car_width) + clearance
    for ox, oy, _oyaw in others:
        if (x - ox) ** 2 + (y - oy) ** 2 < (2.0 * r) ** 2:
            return False
    return True


def spawn_agent_pose(
    rng: np.random.Generator,
    geom: SimTrackGeometry,
    track_id: int,
    existing: list[tuple[float, float, float]],
    *,
    car_length: float,
    car_width: float,
    spawn_margin: float = 0.2,
    speed_min: float = 0.5,
    speed_max: float = 3.0,
    yaw_jitter: float = 0.15,
    max_rejects: int = 64,
    width_scale: float = 1.0,
) -> dict[str, float | int]:
    a = int(geom.offsets[track_id])
    b = int(geom.offsets[track_id + 1])
    centerline = geom.centerline_xy[a:b]
    tangents = geom.tangents_xy[a:b]
    normals = geom.normals_xy[a:b]
    widths = geom.widths_rl[a:b]
    cum = geom.cum_length[a:b]
    seg = geom.segment_length[a:b]
    count = b - a
    scale = float(width_scale)
    for _ in range(max_rejects):
        segment = int(rng.integers(0, count))
        alpha = float(rng.random())
        following = (segment + 1) % count
        wr = max(float(widths[segment, 0]) * scale - spawn_margin, 0.0)
        wl = max(float(widths[segment, 1]) * scale - spawn_margin, 0.0)
        lateral = float(rng.uniform(-wr, wl))
        center = centerline[segment] + alpha * (
            centerline[following] - centerline[segment]
        )
        x = float(center[0] + lateral * normals[segment, 0])
        y = float(center[1] + lateral * normals[segment, 1])
        yaw = float(math.atan2(tangents[segment, 1], tangents[segment, 0]))
        yaw += float(rng.uniform(-yaw_jitter, yaw_jitter))
        if not _pose_clear_of_others(
            x, y, yaw, existing, car_length, car_width, clearance=0.25
        ):
            continue
        speed = float(rng.uniform(speed_min, speed_max))
        fr = project_frenet_numpy(
            np.array([x, y], dtype=np.float32),
            segment,
            centerline,
            tangents,
            normals,
            widths,
            cum,
            seg,
        )
        return {
            "x": x,
            "y": y,
            "yaw": yaw,
            "vx": speed,
            "segment": int(fr["segment"]),
            "s": float(fr["s"]),
            "ey": float(fr["ey"]),
            "boundary_distance": float(
                min(
                    float(fr["width_left"]) * scale - float(fr["ey"]),
                    float(fr["width_right"]) * scale + float(fr["ey"]),
                )
            ),
            "rejected": 0,
        }
    segment = int(rng.integers(0, count))
    x = float(centerline[segment, 0])
    y = float(centerline[segment, 1])
    yaw = float(math.atan2(tangents[segment, 1], tangents[segment, 0]))
    fr = project_frenet_numpy(
        np.array([x, y], dtype=np.float32),
        segment,
        centerline,
        tangents,
        normals,
        widths,
        cum,
        seg,
    )
    return {
        "x": x,
        "y": y,
        "yaw": yaw,
        "vx": float(speed_min),
        "segment": int(fr["segment"]),
        "s": float(fr["s"]),
        "ey": float(fr["ey"]),
        "boundary_distance": float(
            min(
                float(fr["width_left"]) * scale - float(fr["ey"]),
                float(fr["width_right"]) * scale + float(fr["ey"]),
            )
        ),
        "rejected": 1,
    }


def pose_on_track(
    geom: SimTrackGeometry,
    track_id: int,
    s: float,
    ey: float,
) -> dict[str, float | int]:
    a = int(geom.offsets[track_id])
    b = int(geom.offsets[track_id + 1])
    length = float(geom.track_length[track_id])
    s = float(s) % length
    if s < 0.0:
        s += length
    center = geom.centerline_xy[a:b]
    tangents = geom.tangents_xy[a:b]
    normals = geom.normals_xy[a:b]
    widths = geom.widths_rl[a:b]
    cum = geom.cum_length[a:b]
    seglen = geom.segment_length[a:b]
    segment = int(np.searchsorted(cum, s, side="right") - 1)
    segment = min(max(segment, 0), b - a - 1)
    sl = max(float(seglen[segment]), 1e-6)
    alpha = min(max((s - float(cum[segment])) / sl, 0.0), 1.0)
    following = (segment + 1) % (b - a)
    mid = center[segment] + alpha * (center[following] - center[segment])
    xy = mid + float(ey) * normals[segment]
    yaw = float(math.atan2(tangents[segment, 1], tangents[segment, 0]))
    fr = project_frenet_numpy(
        np.asarray(xy, dtype=np.float32),
        segment,
        center,
        tangents,
        normals,
        widths,
        cum,
        seglen,
    )
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "yaw": yaw,
        "tx": float(tangents[segment, 0]),
        "ty": float(tangents[segment, 1]),
        "segment": int(fr["segment"]),
        "s": float(fr["s"]),
        "ey": float(fr["ey"]),
        "boundary_distance": float(fr["boundary_distance"]),
    }


def head_to_head_chaser_pose(
    geom: SimTrackGeometry,
    track_id: int,
    *,
    leader_s: float,
    leader_speed: float,
    gap_m: float = HEAD_TO_HEAD_BEHIND_GAP_M,
    lateral_m: float = HEAD_TO_HEAD_LATERAL_M,
    speed_delta_mps: float = HEAD_TO_HEAD_SPEED_DELTA_MPS,
) -> dict[str, float | int]:
    length = float(geom.track_length[track_id])
    s = (float(leader_s) - float(gap_m)) % length
    if s < 0.0:
        s += length
    pose = pose_on_track(geom, track_id, s, float(lateral_m))
    pose["speed"] = float(leader_speed) + float(speed_delta_mps)
    return pose


def apply_racer_pose(arrays: Any, slot: int, pose: dict[str, float | int]) -> None:
    slot = int(slot)
    speed = float(pose.get("speed", pose.get("vx", 0.0)))
    arrays.x[slot] = float(pose["x"])
    arrays.y[slot] = float(pose["y"])
    arrays.yaw[slot] = float(pose["yaw"])
    arrays.prev_x[slot] = float(pose["x"])
    arrays.prev_y[slot] = float(pose["y"])
    arrays.prev_yaw[slot] = float(pose["yaw"])
    arrays.vx[slot] = speed
    arrays.vy[slot] = 0.0
    arrays.yaw_rate[slot] = 0.0
    arrays.steer[slot] = 0.0
    arrays.frenet_segment[slot] = int(pose["segment"])
    arrays.frenet_s[slot] = float(pose["s"])
    arrays.prev_s[slot] = float(pose["s"])
    arrays.frenet_ey[slot] = float(pose["ey"])
    arrays.boundary_distance[slot] = float(pose["boundary_distance"])
    arrays.done[slot] = 0
    arrays.active[slot] = 1
    arrays.trainable[slot] = 1
    arrays.contact[slot] = 0
    arrays.wall_contact[slot] = 0


def sample_lidar_corruption(rng: np.random.Generator) -> dict[str, float | int]:
    sector_width = 0
    sector_start = 0
    if rng.random() < 0.25:
        sector_width = int(rng.integers(8, 64))
        sector_start = int(rng.integers(0, 1081))
    return {
        "range_noise_std": float(rng.uniform(0.0, 0.05)),
        "dropout_prob": float(rng.uniform(0.0, 0.02)),
        "far_dropout_prob": float(rng.uniform(0.0, 0.08)),
        "angle_bias": float(rng.uniform(-0.01, 0.01)),
        "extrinsic_x": float(rng.uniform(-0.02, 0.02)),
        "extrinsic_y": float(rng.uniform(-0.02, 0.02)),
        "extrinsic_yaw": float(rng.uniform(-0.02, 0.02)),
        "sector_start": sector_start,
        "sector_width": sector_width,
    }


def horizon_for_track(cfg: ExperimentConfig, geom: SimTrackGeometry, track_id: int) -> int:
    return episode_steps(cfg, float(geom.track_length[track_id]))


def place_static_opponents(
    geom: SimTrackGeometry,
    track_id: int,
    count: int,
    *,
    seed: int,
    racer_poses: list[tuple[float, float, float]],
    car_length: float,
    car_width: float,
) -> list[dict[str, float | int]]:
    """Deterministically place ``count`` immobile opponents on one track.

    Shared by the viewer (interactive Off/Light/Heavy presets) and training
    (fixed light-preset density): both need the same boundary-margin,
    passing-corridor, and longitudinal/lateral spacing constraints so a
    static opponent never blocks the track or overlaps a racer.
    """
    if count <= 0:
        return []
    a = int(geom.offsets[track_id])
    b = int(geom.offsets[track_id + 1])
    length = float(geom.track_length[track_id])
    centerline = geom.centerline_xy[a:b]
    tangents = geom.tangents_xy[a:b]
    normals = geom.normals_xy[a:b]
    widths = geom.widths_rl[a:b]
    cumulative = geom.cum_length[a:b]
    segments = geom.segment_length[a:b]
    rng = np.random.default_rng(int(seed))
    placed: list[dict[str, float | int]] = []
    min_arc = max(
        STATIC_LONGITUDINAL_GAP_CAR_LENGTHS * car_length,
        length / max(2.5 * count, 1.0),
    )
    min_xy = 3.0 * math.hypot(car_length, car_width)
    racer_clearance = 4.0 * math.hypot(car_length, car_width)
    passing_width = car_width + 2.0 * STATIC_PASSING_MARGIN_M
    station_count = max(64, 24 * count)
    stations = np.linspace(0.15 * length, 0.95 * length, station_count)
    stations += rng.uniform(
        -0.25 * length / station_count,
        0.25 * length / station_count,
        size=station_count,
    )
    rng.shuffle(stations)
    lateral_fractions = [0.12, -0.12, 0.38, -0.38]
    rng.shuffle(lateral_fractions)
    lateral_fractions.insert(0, 0.0)

    for obstacle_index in range(count):
        target_fraction = lateral_fractions[obstacle_index % len(lateral_fractions)]
        found = False
        for s_raw in stations:
            s = float(s_raw)
            if any(
                min(abs(s - float(item["s"])), length - abs(s - float(item["s"])))
                < min_arc
                for item in placed
            ):
                continue
            segment = int(np.searchsorted(cumulative, s, side="right") - 1)
            segment = min(max(segment, 0), b - a - 1)
            seg_len = max(float(segments[segment]), 1e-6)
            alpha = min(
                max((s - float(cumulative[segment])) / seg_len, 0.0), 1.0
            )
            following = (segment + 1) % (b - a)
            center = centerline[segment] + alpha * (
                centerline[following] - centerline[segment]
            )
            right = float(widths[segment, 0])
            left = float(widths[segment, 1])
            lateral_min = (
                -right + 0.5 * car_width + STATIC_BOUNDARY_MARGIN_M
            )
            lateral_max = (
                left - 0.5 * car_width - STATIC_BOUNDARY_MARGIN_M
            )
            if lateral_min > lateral_max:
                continue
            side_width = right if target_fraction < 0.0 else left
            lateral = target_fraction * side_width
            lateral = min(max(lateral, lateral_min), lateral_max)
            left_corridor = left - (lateral + 0.5 * car_width)
            right_corridor = right + (lateral - 0.5 * car_width)
            if max(left_corridor, right_corridor) < passing_width:
                continue
            xy = center + lateral * normals[segment]
            if any(
                math.hypot(float(xy[0]) - x, float(xy[1]) - y)
                < racer_clearance
                for x, y, _ in racer_poses
            ):
                continue
            if any(
                math.hypot(
                    float(xy[0]) - float(item["x"]),
                    float(xy[1]) - float(item["y"]),
                )
                < min_xy
                for item in placed
            ):
                continue
            yaw = float(
                math.atan2(tangents[segment, 1], tangents[segment, 0])
            )
            fr = project_frenet_numpy(
                np.asarray(xy, dtype=np.float32),
                segment,
                centerline,
                tangents,
                normals,
                widths,
                cumulative,
                segments,
            )
            placed.append(
                {
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "yaw": yaw,
                    "segment": int(fr["segment"]),
                    "s": float(fr["s"]),
                    "ey": float(fr["ey"]),
                    "boundary_distance": float(fr["boundary_distance"]),
                }
            )
            found = True
            break
        if not found:
            break
    return placed


def apply_static_pins(
    arrays: Any,
    mask: torch.Tensor,
    pin_x: torch.Tensor,
    pin_y: torch.Tensor,
    pin_yaw: torch.Tensor,
    pin_segment: torch.Tensor,
    pin_s: torch.Tensor,
    pin_ey: torch.Tensor,
    pin_boundary: torch.Tensor,
    *,
    reset_contact: bool,
) -> None:
    """Vectorized full-brake pin/restore for static-opponent slots.

    Writes the same fields the viewer and the training runtime both need to
    hold a slot at zero motion after physics/contact: pose, zeroed velocity
    and command state, frenet projection, and a cleared terminal/reset
    lifecycle so the slot is never mistaken for an episode boundary.
    ``reset_contact`` is True only at placement time; the per-step restore
    leaves ``contact``/``wall_contact`` alone so a colliding learner still
    observes this step's collision.
    """
    if not bool(mask.any()):
        return
    arrays.x[mask] = pin_x[mask]
    arrays.y[mask] = pin_y[mask]
    arrays.yaw[mask] = pin_yaw[mask]
    arrays.prev_x[mask] = pin_x[mask]
    arrays.prev_y[mask] = pin_y[mask]
    arrays.prev_yaw[mask] = pin_yaw[mask]
    arrays.vx[mask] = 0.0
    arrays.vy[mask] = 0.0
    arrays.yaw_rate[mask] = 0.0
    arrays.steer[mask] = 0.0
    arrays.steer_logical[mask] = 0.0
    arrays.i_drive[mask] = 0.0
    arrays.i_brake[mask] = float(VehicleParams.i_brake_max_a)
    arrays.i_drive_cmd[mask] = 0.0
    arrays.i_brake_cmd[mask] = float(VehicleParams.i_brake_max_a)
    arrays.applied_effort[mask] = -1.0
    arrays.ax[mask] = 0.0
    arrays.ay[mask] = 0.0
    arrays.frenet_segment[mask] = pin_segment[mask].to(dtype=arrays.frenet_segment.dtype)
    arrays.frenet_s[mask] = pin_s[mask]
    arrays.prev_s[mask] = pin_s[mask]
    arrays.frenet_ey[mask] = pin_ey[mask]
    arrays.boundary_distance[mask] = pin_boundary[mask]
    arrays.progress_s[mask] = 0.0
    arrays.rewards[mask] = 0.0
    arrays.episode_step[mask] = 0
    arrays.stalled_steps[mask] = 0
    arrays.done[mask] = 0
    arrays.timeout[mask] = 0
    arrays.reset_mask[mask] = 0
    arrays.active[mask] = 1
    arrays.trainable[mask] = 0
    if reset_contact:
        arrays.contact[mask] = 0
        arrays.wall_contact[mask] = 0

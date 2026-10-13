"""Gym certify helpers: Warp longitudinal plant and eval_full_lap net progress."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROGRESS_EPS_M = 1e-4
DEFAULT_MAX_STEP_FRAC = 0.1
WARP_LIDAR_TF_XYYAW = (0.27, 0.0, 0.0)
GYM_LIDAR_TF_XYYAW = (0.27, 0.0, 0.0)
GYM_OCCUPIED_PIXEL = 0.0
GYM_UNKNOWN_PIXEL = 205.0
GYM_FREE_PIXEL = 255.0


def lidar_frame_pose(
    x: float,
    y: float,
    yaw: float,
    tf_xyyaw: tuple[float, float, float] = WARP_LIDAR_TF_XYYAW,
) -> tuple[float, float, float]:
    """World pose of the LiDAR mount from a base_link / CoG pose."""
    dx, dy, dtheta = (float(v) for v in tf_xyyaw)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return (
        float(x) + dx * cosine - dy * sine,
        float(y) + dx * sine + dy * cosine,
        float(yaw) + dtheta,
    )


def spawn_pose_from_centerline(path: str | Path) -> tuple[float, float, float]:
    """First centerline vertex and heading of the following segment."""
    xy, _cum, _length = load_centerline(path)
    heading = math.atan2(float(xy[1, 1] - xy[0, 1]), float(xy[1, 0] - xy[0, 0]))
    return float(xy[0, 0]), float(xy[0, 1]), heading


def pose_along_heading(
    pose_xyyaw: tuple[float, float, float], distance_m: float
) -> tuple[float, float, float]:
    x, y, yaw = (float(v) for v in pose_xyyaw)
    return (
        x + float(distance_m) * math.cos(yaw),
        y + float(distance_m) * math.sin(yaw),
        yaw,
    )


def ros_trinary_occupancy(
    image: np.ndarray,
    occupied_thresh: float,
    free_thresh: float,
    negate: int = 0,
) -> np.ndarray:
    """0 occupied, 205 unknown, 255 free — same ROS rules Warp uses."""
    pixels = np.asarray(image, dtype=np.float32)
    probability = pixels / 255.0
    if int(negate) == 0:
        probability = 1.0 - probability
    occupied = probability >= float(occupied_thresh)
    free = probability <= float(free_thresh)
    unknown = ~occupied & ~free
    out = np.full(pixels.shape, GYM_FREE_PIXEL, dtype=np.float32)
    out[occupied] = GYM_OCCUPIED_PIXEL
    out[unknown] = GYM_UNKNOWN_PIXEL
    return out


def gym_midgrey_occupancy(image: np.ndarray) -> np.ndarray:
    """f1tenth_gym Track default: split at 128, wiping ROS unknown cells to free."""
    pixels = np.asarray(image, dtype=np.float32)
    out = np.zeros_like(pixels)
    out[pixels > 128.0] = GYM_FREE_PIXEL
    return out


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ang_diff(a: float, b: float) -> float:
    return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi


def pose_error_xy_yaw(
    x: float,
    y: float,
    yaw: float,
    spawn_x: float,
    spawn_y: float,
    spawn_yaw: float,
) -> tuple[float, float]:
    return (
        math.hypot(float(x) - float(spawn_x), float(y) - float(spawn_y)),
        abs(ang_diff(yaw, spawn_yaw)),
    )


REPEAT_SCORE_KEYS = (
    "progress_m",
    "laps",
    "crashes",
    "deadline_misses",
    "max_speed_mps",
    "offtrack_samples",
    "desired_hz",
)


def summarize_repeat_scores(scores: list[dict]) -> dict:
    if not scores:
        raise ValueError("need at least one score")
    out: dict = {
        "repeats": len(scores),
        "pass": all(bool(item.get("pass")) for item in scores),
        "runs": scores,
    }
    for key in REPEAT_SCORE_KEYS:
        vals = np.asarray([float(item[key]) for item in scores], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_var"] = float(vals.var(ddof=1)) if vals.size > 1 else 0.0
    return out


def wrapped_delta_s(current, previous, length) -> float:
    """Shortest-arc along-track delta in (-L/2, L/2]."""
    length = float(length)
    delta = float(current) - float(previous)
    half = 0.5 * length
    if delta > half:
        delta -= length
    elif delta < -half:
        delta += length
    return delta


def load_centerline(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    pts = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("x_m"):
                continue
            parts = line.split(",")
            pts.append((float(parts[0]), float(parts[1])))
    xy = np.asarray(pts, dtype=np.float64)
    if xy.shape[0] < 2:
        raise ValueError(f"{path}: need at least two centerline rows")
    if float(np.hypot(*(xy[0] - xy[-1]))) > 1e-6:
        xy = np.vstack([xy, xy[0]])
    deltas = np.diff(xy, axis=0)
    seglen = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    return xy, cum, float(cum[-1])


def project_s(x: float, y: float, xy: np.ndarray, cum: np.ndarray) -> tuple[float, float]:
    point = np.array([x, y], dtype=np.float64)
    dxy = xy - point
    dist = np.hypot(dxy[:, 0], dxy[:, 1])
    i = int(np.argmin(dist))
    n = xy.shape[0]
    best_s = float(cum[i])
    best_d = float(dist[i])
    for a, b in ((max(i - 1, 0), i), (i, min(i + 1, n - 1))):
        if a == b:
            continue
        p0 = xy[a]
        p1 = xy[b]
        span = p1 - p0
        denom = float(span @ span)
        if denom < 1e-12:
            continue
        t = float(np.clip(((point - p0) @ span) / denom, 0.0, 1.0))
        proj = p0 + t * span
        d = float(np.hypot(proj[0] - x, proj[1] - y))
        if d < best_d:
            best_d = d
            best_s = float(cum[a] + t * (cum[b] - cum[a]))
    return best_s, best_d


@dataclass
class NetLapCounter:
    length: float
    max_step_frac: float = DEFAULT_MAX_STEP_FRAC

    def __post_init__(self) -> None:
        self.length = float(self.length)
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError(f"track length must be positive, got {self.length}")
        self.max_ds = float(self.max_step_frac) * self.length
        self.prev_s: float | None = None
        self.progress = 0.0
        self.laps = 0
        self.lap_times: list[float] = []
        self.crashes = 0
        self.offtrack = False
        self.lap_t0 = 0.0

    def reset_progress(self, s: float, t: float) -> None:
        self.prev_s = float(s)
        self.progress = 0.0
        self.lap_t0 = float(t)

    def clear(self) -> None:
        self.prev_s = None
        self.progress = 0.0
        self.laps = 0
        self.lap_times.clear()
        self.crashes = 0
        self.offtrack = False
        self.lap_t0 = 0.0

    def _on_crash(self, s: float, t: float) -> None:
        self.crashes += 1
        self.reset_progress(s, t)

    def update(self, s: float, offtrack: bool, t: float) -> None:
        s = float(s)
        t = float(t)
        if self.prev_s is None:
            self.prev_s = s
            self.lap_t0 = t
            self.offtrack = bool(offtrack)
            return
        if offtrack and not self.offtrack:
            self._on_crash(s, t)
            self.offtrack = True
            return
        self.offtrack = bool(offtrack)
        if self.offtrack:
            self.prev_s = s
            return
        raw = wrapped_delta_s(s, self.prev_s, self.length)
        if abs(raw) > self.max_ds:
            self._on_crash(s, t)
            return
        self.progress += raw
        self.prev_s = s
        while self.progress >= self.length - PROGRESS_EPS_M:
            self.laps += 1
            self.lap_times.append(t - self.lap_t0)
            self.lap_t0 = t
            self.progress -= self.length


@dataclass
class WarpLongitudinalParams:
    mass: float = 3.444
    f_drive_max: float = 26.5
    f_brake_max: float = 23.1
    i_drive_max: float = 80.0
    i_brake_max: float = 40.0
    i_slew: float = 200.0
    c_coast: float = 0.0
    v_eps: float = 0.1
    sim_dt: float = 0.005
    control_dt: float = 0.05


def _slew_channel(prev: float, tgt: float, max_step: float) -> float:
    delta = tgt - prev
    if abs(delta) <= max_step:
        return tgt
    return prev + max(-max_step, min(max_step, delta))


def apply_current_slew(
    i_drive: float,
    i_brake: float,
    i_drive_cmd: float,
    i_brake_cmd: float,
    dt: float,
    params: WarpLongitudinalParams,
) -> tuple[float, float]:
    max_step = abs(params.i_slew) * dt
    i_drive = _slew_channel(i_drive, i_drive_cmd, max_step)
    i_brake = _slew_channel(i_brake, i_brake_cmd, max_step)
    if i_drive > 0.0 and i_brake > 0.0:
        if i_drive_cmd >= i_brake_cmd:
            i_brake = 0.0
        else:
            i_drive = 0.0
    return i_drive, i_brake


def longitudinal_force(
    vx: float,
    i_drive: float,
    i_brake: float,
    params: WarpLongitudinalParams,
) -> float:
    drive_force = (
        i_drive * params.f_drive_max / max(params.i_drive_max, 1.0e-6)
    )
    brake_force = (
        i_brake * params.f_brake_max / max(params.i_brake_max, 1.0e-6)
    )
    if vx > params.v_eps:
        return drive_force - brake_force - params.c_coast
    if vx < -params.v_eps:
        return drive_force + brake_force + params.c_coast
    return drive_force


def integrate_vx(
    vx: float,
    i_drive: float,
    i_brake: float,
    dt: float,
    params: WarpLongitudinalParams,
) -> float:
    drive_force = (
        i_drive * params.f_drive_max / max(params.i_drive_max, 1.0e-6)
    )
    ax = longitudinal_force(vx, i_drive, i_brake, params) / params.mass
    vx_dyn = vx + dt * ax
    if vx * vx_dyn < 0.0 and vx_dyn * drive_force <= 0.0:
        vx_dyn = 0.0
    return vx_dyn


class WarpLongitudinalPlant:
    def __init__(self, params: WarpLongitudinalParams | None = None) -> None:
        self.params = params or WarpLongitudinalParams()
        self.i_drive = 0.0
        self.i_brake = 0.0

    def reset(self) -> None:
        self.i_drive = 0.0
        self.i_brake = 0.0

    def step(
        self,
        vx: float,
        i_drive_cmd: float,
        i_brake_cmd: float,
        control_dt: float | None = None,
    ) -> float:
        p = self.params
        dt = float(p.sim_dt)
        horizon = float(p.control_dt if control_dt is None else control_dt)
        steps = max(1, int(round(horizon / dt)))
        vx_n = float(vx)
        cmd_d = float(i_drive_cmd)
        cmd_b = float(i_brake_cmd)
        for _ in range(steps):
            self.i_drive, self.i_brake = apply_current_slew(
                self.i_drive, self.i_brake, cmd_d, cmd_b, dt, p
            )
            vx_n = integrate_vx(vx_n, self.i_drive, self.i_brake, dt, p)
        return vx_n

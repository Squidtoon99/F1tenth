"""Pure episode-monitoring logic for the evaluation node (no ROS imports).

Tracks lap completion, out-of-bounds, and stuck detection from a stream of Frenet
progress / lateral-error / speed samples so it can be unit-tested deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def sample_centerline_pose(
    centerline: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Pick a random on-track pose aligned with the local centerline tangent."""
    if len(centerline) < 2:
        raise ValueError("centerline must have at least two points")
    idx = int(rng.integers(0, len(centerline)))
    x, y = float(centerline[idx, 0]), float(centerline[idx, 1])
    next_idx = (idx + 1) % len(centerline)
    dx = float(centerline[next_idx, 0] - centerline[idx, 0])
    dy = float(centerline[next_idx, 1] - centerline[idx, 1])
    if math.hypot(dx, dy) < 1e-6:
        prev_idx = (idx - 1) % len(centerline)
        dx = float(centerline[idx, 0] - centerline[prev_idx, 0])
        dy = float(centerline[idx, 1] - centerline[prev_idx, 1])
    yaw = math.atan2(dy, dx)
    return x, y, yaw




def opponent_pose_ahead(
    centerline: np.ndarray,
    ego_x: float,
    ego_y: float,
    gap_m: float = 7.0,
) -> tuple[float, float, float]:
    """Place the opponent ``gap_m`` arc-length ahead on the closed centerline (training parity)."""
    if len(centerline) < 2:
        raise ValueError("centerline must have at least two points")
    cl = np.asarray(centerline, dtype=np.float64)
    seg = cl[1:] - cl[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    mean_seg = float(np.mean(seg_len)) if seg_len.size else 1.0
    gap_pts = max(1, int(round(float(gap_m) / max(mean_seg, 1e-6))))
    d2 = (cl[:, 0] - ego_x) ** 2 + (cl[:, 1] - ego_y) ** 2
    ego_idx = int(np.argmin(d2))
    opp_idx = (ego_idx + gap_pts) % len(cl)
    ox, oy = float(cl[opp_idx, 0]), float(cl[opp_idx, 1])
    next_idx = (opp_idx + 1) % len(cl)
    dx = float(cl[next_idx, 0] - cl[opp_idx, 0])
    dy = float(cl[next_idx, 1] - cl[opp_idx, 1])
    if math.hypot(dx, dy) < 1e-6:
        prev_idx = (opp_idx - 1) % len(cl)
        dx = float(cl[opp_idx, 0] - cl[prev_idx, 0])
        dy = float(cl[opp_idx, 1] - cl[prev_idx, 1])
    yaw = math.atan2(dy, dx)
    return ox, oy, yaw
@dataclass
class EpisodeEvent:
    progress_ratio: float
    lap_completed: bool
    oob: bool
    stuck: bool
    lap_time: float | None
    lap_count: int
    max_progress: float


@dataclass
class EpisodeMonitor:
    oob_margin_m: float = 0.0
    stuck_speed_mps: float = 0.2
    stuck_timeout_s: float = 3.0
    lap_hi: float = 0.75
    lap_lo: float = 0.25

    lap_count: int = 0
    max_progress: float = 0.0
    last_lap_time: float | None = None
    _prev_progress: float | None = field(default=None, repr=False)
    _lap_start_t: float | None = field(default=None, repr=False)
    _last_move_t: float | None = field(default=None, repr=False)

    def reset(self, t: float):
        self.max_progress = 0.0
        self._prev_progress = None
        self._lap_start_t = t
        self._last_move_t = t

    def update(
        self,
        s: float,
        track_len: float,
        ey: float,
        w_left: float,
        w_right: float,
        speed: float,
        t: float,
    ) -> EpisodeEvent:
        if self._lap_start_t is None:
            self._lap_start_t = t
        if self._last_move_t is None:
            self._last_move_t = t

        progress = (s / track_len) % 1.0 if track_len > 1e-6 else 0.0
        self.max_progress = max(self.max_progress, progress)

        lap_completed = False
        lap_time = None
        if (
            self._prev_progress is not None
            and self._prev_progress > self.lap_hi
            and progress < self.lap_lo
        ):
            lap_completed = True
            lap_time = t - self._lap_start_t
            self.last_lap_time = lap_time
            self.lap_count += 1
            self._lap_start_t = t
            self.max_progress = progress
        self._prev_progress = progress

        oob = (ey > (w_left - self.oob_margin_m)) or (
            ey < -(w_right - self.oob_margin_m)
        )

        if speed >= self.stuck_speed_mps:
            self._last_move_t = t
        stuck = (t - self._last_move_t) > self.stuck_timeout_s

        return EpisodeEvent(
            progress_ratio=progress,
            lap_completed=lap_completed,
            oob=oob,
            stuck=stuck,
            lap_time=lap_time,
            lap_count=self.lap_count,
            max_progress=self.max_progress,
        )

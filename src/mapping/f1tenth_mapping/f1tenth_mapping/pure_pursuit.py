"""Pure pursuit path tracking (no ROS imports)."""

from __future__ import annotations

import math

import numpy as np


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def nearest_path_index(path: np.ndarray, pose_xy: np.ndarray) -> int:
    """Return index of the closest point on ``path`` (N, 2) to ``pose_xy`` (2,)."""
    if path.shape[0] == 0:
        return 0
    diffs = path - pose_xy[None, :]
    return int(np.argmin(np.sum(diffs * diffs, axis=1)))


def lookahead_point(
    path: np.ndarray,
    pose_xy: np.ndarray,
    lookahead_m: float,
) -> np.ndarray | None:
    """Return the first point on ``path`` at least ``lookahead_m`` from ``pose_xy``."""
    if path.shape[0] == 0:
        return None
    start = nearest_path_index(path, pose_xy)
    for idx in range(start, path.shape[0]):
        delta = path[idx] - pose_xy
        dist = float(np.linalg.norm(delta))
        if dist >= lookahead_m:
            return path[idx]
    return path[-1]


def compute_steering(
    pose_xy: np.ndarray,
    yaw_rad: float,
    path: np.ndarray,
    wheelbase_m: float,
    lookahead_m: float,
    max_steer_rad: float,
) -> tuple[float, float]:
    """Return ``(steering_angle_rad, cross_track_error_m)`` via pure pursuit."""
    target = lookahead_point(path, pose_xy, lookahead_m)
    if target is None:
        return 0.0, 0.0

    dx = float(target[0] - pose_xy[0])
    dy = float(target[1] - pose_xy[1])
    alpha = _wrap_angle(math.atan2(dy, dx) - yaw_rad)
    cross_track = math.sin(alpha) * math.hypot(dx, dy)

    lookahead = max(lookahead_m, 1e-3)
    curvature = 2.0 * math.sin(alpha) / lookahead
    steering = math.atan(wheelbase_m * curvature)
    steering = max(-max_steer_rad, min(max_steer_rad, steering))
    return steering, cross_track


class SpeedPID:
    """Simple PID controller for longitudinal speed tracking."""

    def __init__(self, kp: float, ki: float, kd: float, limit: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def compute(self, target_speed: float, current_speed: float, dt: float) -> float:
        error = target_speed - current_speed
        self._integral = max(-self.limit, min(self.limit, self._integral + error * dt))
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.limit, min(self.limit, output))

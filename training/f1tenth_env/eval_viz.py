"""Backend-agnostic top-down rollout visualizer for *evaluation only*.

Renders one or more environments' rollouts as a top-down 2D scene (track corridor,
car poses + heading, breadcrumb trails, live speed) from generic quantities only:
the track boundaries and the per-step car pose(s). It never touches the training
loop or replay buffer.

Because every parallel env shares the same track, any number of envs can be drawn
overlaid on a single view (a "swarm" plot showing the spread of behaviours). Pass
scalars/(x, y) to draw one car, or arrays of shape ``(K, 2)`` / ``(K,)`` to draw K.

Two independent sinks (either, both, or neither):

- ``live=True`` -> streams native 2D primitives to **Rerun** so you can watch the
  rollout as it happens. Nothing is buffered to a file first; frames appear in the
  Rerun viewer in real time (spawn a local viewer, or connect one remotely).
- ``mp4_path=...`` -> draws frames with OpenCV and encodes an mp4 via imageio,
  suitable for ``wandb.Video`` or offline inspection.

Coordinates: world is x-right / y-up (north-up). The mp4 raster flips y so north
is up; the Rerun sink logs ``(x, -y)`` so its y-down 2D view also renders north-up
with correct turn chirality (a left turn looks like a left turn in both sinks).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from f1tenth_env.utils import compute_track_boundaries

# Distinct RGB colors cycled per visualized env (frames are RGB for imageio/Rerun).
_PALETTE = [
    (240, 60, 60), (60, 200, 90), (90, 140, 255), (240, 180, 40),
    (200, 90, 220), (60, 210, 210), (250, 120, 40), (160, 160, 60),
    (230, 90, 150), (120, 200, 120),
]
_C_LEFT = (60, 170, 255)
_C_RIGHT = (255, 140, 60)
_C_CENTER = (110, 110, 120)
_C_OPP = (150, 150, 160)
_C_BG = (18, 18, 22)
_C_TEXT = (235, 235, 235)


def yaw_from_quat_wxyz(quat) -> float:
    """Planar yaw (rad) from a (w, x, y, z) quaternion; valid for both backends."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _car_color(i: int) -> tuple:
    return _PALETTE[i % len(_PALETTE)]


_OPP_EGO_BLEND = 0.42


def _opp_color(i: int) -> tuple:
    """Muted gray tint of the ego color for env *i* (eval viz only)."""
    ego = _car_color(i)
    return tuple(
        int(ego[c] * _OPP_EGO_BLEND + _C_OPP[c] * (1.0 - _OPP_EGO_BLEND))
        for c in range(3)
    )


def _dim(color: tuple, f: float = 0.55) -> tuple:
    return tuple(int(c * f) for c in color)


def _to_cars(xy, yaw, speed) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize (x, y)/yaw/speed inputs to arrays of shape (K, 2)/(K,)/(K,)."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy[None, :]
    k = xy.shape[0]
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    if yaw.size == 1:
        yaw = np.full(k, float(yaw[0]))
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    if speed.size == 1:
        speed = np.full(k, float(speed[0]))
    return xy, yaw, speed, k


def _rect_corners(cx: float, cy: float, yaw: float, length: float,
                  width: float) -> np.ndarray:
    """Four world-frame corners of the car rectangle."""
    hl, hw = 0.5 * length, 0.5 * width
    base = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return base @ rot.T + np.array([cx, cy], dtype=np.float64)


class RolloutVisualizer:
    """Render an eval rollout of one or more envs to Rerun (live) and/or an mp4."""

    def __init__(
        self,
        centerline: np.ndarray,
        w_tr_left: np.ndarray,
        w_tr_right: np.ndarray,
        car_length: float = 0.568,
        car_width: float = 0.296,
        *,
        num_show: int = 1,
        live: bool = False,
        mp4_path: Optional[str] = None,
        fps: int = 10,
        img_size: int = 900,
        pad_px: int = 24,
        trail_len: int = 400,
        has_opponent: bool = False,
        rr_app_id: str = "f1tenth_eval",
        rr_spawn: bool = True,
        rr_save_path: Optional[str] = None,
    ) -> None:
        self.car_length = float(car_length)
        self.car_width = float(car_width)
        self.num_show = max(1, int(num_show))
        self.live = bool(live)
        self.mp4_path = mp4_path
        self.fps = int(fps)
        self.has_opponent = bool(has_opponent)
        self._trail_len = int(trail_len)
        self._trails: list[list[tuple[float, float]]] = [
            [] for _ in range(self.num_show)
        ]
        self._step = 0

        cl = np.asarray(centerline, dtype=np.float64)[:, :2]
        left, right = compute_track_boundaries(
            centerline.astype(np.float32),
            np.asarray(w_tr_left, dtype=np.float32),
            np.asarray(w_tr_right, dtype=np.float32),
        )
        self._center = cl
        self._left = np.asarray(left, dtype=np.float64)[:, :2]
        self._right = np.asarray(right, dtype=np.float64)[:, :2]

        allpts = np.concatenate([self._left, self._right, cl], axis=0)
        self._xmin, self._ymin = allpts.min(axis=0)
        self._xmax, self._ymax = allpts.max(axis=0)
        span_x = max(self._xmax - self._xmin, 1e-3)
        span_y = max(self._ymax - self._ymin, 1e-3)
        self._pad = int(pad_px)
        self._scale = (int(img_size) - 2 * self._pad) / max(span_x, span_y)
        self._W = int(span_x * self._scale) + 2 * self._pad
        self._H = int(span_y * self._scale) + 2 * self._pad

        self._writer = None
        self._bg = None
        if self.mp4_path is not None:
            self._init_mp4()
        if self.live:
            self._init_rerun(rr_app_id, rr_spawn, rr_save_path)

    def _poly_px(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty((pts.shape[0], 2), dtype=np.int32)
        out[:, 0] = ((pts[:, 0] - self._xmin) * self._scale).astype(np.int32) + self._pad
        out[:, 1] = ((self._ymax - pts[:, 1]) * self._scale).astype(np.int32) + self._pad
        return out

    def _init_mp4(self) -> None:
        import os

        import cv2  # noqa: F401  (imported to fail fast if unavailable)
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(self.mp4_path)) or ".",
                    exist_ok=True)
        self._writer = imageio.get_writer(
            self.mp4_path, fps=self.fps, macro_block_size=2
        )
        bg = np.zeros((self._H, self._W, 3), dtype=np.uint8)
        bg[:] = _C_BG
        self._draw_polyline(bg, self._center, _C_CENTER, 1, closed=True)
        self._draw_polyline(bg, self._left, _C_LEFT, 2, closed=True)
        self._draw_polyline(bg, self._right, _C_RIGHT, 2, closed=True)
        self._bg = bg

    def _draw_polyline(self, img, pts, color, thickness, closed=False) -> None:
        import cv2

        cv2.polylines(img, [self._poly_px(pts)], closed, color, thickness,
                      lineType=cv2.LINE_AA)

    def _init_rerun(self, app_id: str, spawn: bool, save_path: Optional[str]) -> None:
        import rerun as rr

        self._rr = rr
        rr.init(app_id, spawn=spawn and save_path is None)
        if save_path is not None:
            rr.save(save_path)
        rr.log("track/left", rr.LineStrips2D([self._rr_pts(self._left)],
               colors=[_C_LEFT]), static=True)
        rr.log("track/right", rr.LineStrips2D([self._rr_pts(self._right)],
               colors=[_C_RIGHT]), static=True)
        rr.log("track/center", rr.LineStrips2D([self._rr_pts(self._center)],
               colors=[_C_CENTER]), static=True)

    @staticmethod
    def _rr_pts(pts: np.ndarray) -> np.ndarray:
        out = np.asarray(pts, dtype=np.float32).copy()
        out[:, 1] = -out[:, 1]  # y-up world -> y-down 2D view, north-up render
        return out

    def render(
        self,
        ego_xy,
        ego_yaw,
        speed=0.0,
        opp_xy=None,
        opp_yaw=0.0,
        done=False,
        extra_text: str = "",
    ) -> None:
        """Draw one frame.

        ``ego_xy`` is either a single ``(x, y)`` or an ``(K, 2)`` array to draw K
        cars overlaid; ``ego_yaw``/``speed``/``done`` may be scalars (broadcast) or
        length-K arrays. Only the first ``num_show`` cars keep breadcrumb trails.
        """
        xy, yaw, spd, k = _to_cars(ego_xy, ego_yaw, speed)
        done_arr = np.asarray(done).reshape(-1)
        if done_arr.size == 1:
            done_arr = np.full(k, bool(done_arr.item()))

        for i in range(min(k, self.num_show)):
            if bool(done_arr[i]):
                self._trails[i].clear()
            self._trails[i].append((float(xy[i, 0]), float(xy[i, 1])))
            if len(self._trails[i]) > self._trail_len:
                del self._trails[i][:-self._trail_len]

        rects = [
            _rect_corners(xy[i, 0], xy[i, 1], yaw[i], self.car_length, self.car_width)
            for i in range(k)
        ]
        opp_rects = None
        if opp_xy is not None:
            oxy, oyaw, _, ok = _to_cars(opp_xy, opp_yaw, 0.0)
            opp_rects = [
                _rect_corners(oxy[i, 0], oxy[i, 1], oyaw[i],
                              self.car_length, self.car_width)
                for i in range(ok)
            ]

        if self._writer is not None:
            self._render_mp4(rects, opp_rects, spd, extra_text)
        if self.live:
            self._render_rerun(rects, opp_rects, spd)
        self._step += 1

    def _render_mp4(self, rects, opp_rects, spd, extra_text) -> None:
        import cv2

        frame = self._bg.copy()
        for i, trail in enumerate(self._trails):
            if len(trail) > 1:
                cv2.polylines(frame, [self._poly_px(np.array(trail))], False,
                              _dim(_car_color(i)), 2, lineType=cv2.LINE_AA)
        if opp_rects is not None:
            for i, r in enumerate(opp_rects):
                cv2.fillPoly(frame, [self._poly_px(r)], _opp_color(i))
        for i, r in enumerate(rects):
            cv2.fillPoly(frame, [self._poly_px(r)], _car_color(i))
        label = (f"step {self._step}  cars={len(rects)}  "
                 f"v[mean={spd.mean():4.1f} max={spd.max():4.1f}] m/s")
        if extra_text:
            label += f"  {extra_text}"
        cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    _C_TEXT, 1, cv2.LINE_AA)
        self._writer.append_data(frame)

    def _render_rerun(self, rects, opp_rects, spd) -> None:
        rr = self._rr
        rr.set_time("step", sequence=self._step)
        strips, colors = [], []
        for i, r in enumerate(rects):
            strips.append(self._rr_pts(np.vstack([r, r[:1]])))
            colors.append(_car_color(i))
        rr.log("cars/ego", rr.LineStrips2D(strips, colors=colors))
        trail_strips, trail_colors = [], []
        for i, trail in enumerate(self._trails):
            if len(trail) > 1:
                trail_strips.append(self._rr_pts(np.array(trail)))
                trail_colors.append(_dim(_car_color(i)))
        if trail_strips:
            rr.log("cars/trails", rr.LineStrips2D(trail_strips, colors=trail_colors))
        if opp_rects is not None:
            opp_strips = [self._rr_pts(np.vstack([r, r[:1]])) for r in opp_rects]
            opp_colors = [_opp_color(i) for i in range(len(opp_rects))]
            rr.log("cars/opp", rr.LineStrips2D(opp_strips, colors=opp_colors))
        rr.log("metrics/speed_mean_mps", rr.Scalars(float(spd.mean())))
        rr.log("metrics/speed_max_mps", rr.Scalars(float(spd.max())))

    def close(self) -> Optional[str]:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.mp4_path

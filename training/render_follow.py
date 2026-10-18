"""Render a zoomed follow-camera overlay video from saved rollout trajectories."""

import argparse
import math

import cv2
import imageio.v2 as imageio
import numpy as np

CAR_COLORS = [
    (255, 128, 60), (80, 220, 255), (120, 255, 140), (255, 110, 220),
    (255, 235, 90), (170, 150, 255), (110, 255, 235), (255, 170, 130),
]
OPP_COLOR = (235, 235, 235)
BG = (16, 17, 20)
C_EDGE = (90, 110, 200)
C_CENTER = (70, 72, 82)
C_TEXT = (235, 235, 235)
C_THROTTLE = (70, 210, 90)
C_BRAKE = (230, 55, 55)
C_BAR_BG = (38, 40, 48)
C_PANEL = (22, 24, 30)
C_TICK = (140, 142, 150)
WHEEL_LOCK_DEG = 90.0


def rect(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _bar(frame, x, y, w, h, fill, color):
    fill = float(np.clip(fill, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_BAR_BG, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 92, 102), 1, cv2.LINE_AA)
    fh = int(round((h - 2) * fill))
    if fh > 0:
        cv2.rectangle(
            frame,
            (x + 1, y + h - 1 - fh),
            (x + w - 1, y + h - 1),
            color,
            -1,
        )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ty = y + h - 1 - int(round((h - 2) * frac))
        cv2.line(frame, (x - 4, ty), (x, ty), C_TICK, 1, cv2.LINE_AA)
        cv2.line(frame, (x + w, ty), (x + w + 4, ty), C_TICK, 1, cv2.LINE_AA)


def _wheel_rgba(radius, lock_frac):
    pad = 8
    size = 2 * radius + 2 * pad
    img = np.zeros((size, size, 4), dtype=np.uint8)
    c = size // 2
    rim = (245, 245, 248, 255)
    spoke = (220, 222, 230, 255)
    hub = (200, 202, 210, 255)
    cv2.circle(img, (c, c), radius, rim, 6, cv2.LINE_AA)
    cv2.circle(img, (c, c), radius - 9, (70, 72, 80, 255), 2, cv2.LINE_AA)
    inner, outer = 11, radius - 7
    for deg in (0.0, 180.0, 90.0):
        rad = math.radians(deg)
        x0 = int(round(c + inner * math.cos(rad)))
        y0 = int(round(c + inner * math.sin(rad)))
        x1 = int(round(c + outer * math.cos(rad)))
        y1 = int(round(c + outer * math.sin(rad)))
        cv2.line(img, (x0, y0), (x1, y1), spoke, 5, cv2.LINE_AA)
    cv2.circle(img, (c, c), 10, hub, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), 10, rim, 2, cv2.LINE_AA)
    marker = np.array(
        [
            [c, c - radius + 1],
            [c - 8, c - radius + 18],
            [c + 8, c - radius + 18],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, marker, (90, 255, 130, 255), cv2.LINE_AA)
    angle = float(np.clip(lock_frac, -1.0, 1.0)) * WHEEL_LOCK_DEG
    rot = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    return cv2.warpAffine(
        img,
        rot,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _blit_rgba(dst, overlay, x, y):
    h, w = overlay.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    patch = overlay[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(
        alpha * patch[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi, 0, 255
    ).astype(np.uint8)


def draw_controls_hud(frame, throttle_frac, brake_frac, steer_rad, max_steer_rad):
    bar_w, bar_h = 20, 112
    gap = 18
    wheel_r = 48
    margin = 14
    panel_w = 250
    panel_h = 168
    h, w = frame.shape[:2]
    x1 = w - margin
    y0 = margin
    x0 = x1 - panel_w
    cv2.rectangle(frame, (x0, y0), (x1, y0 + panel_h), C_PANEL, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + panel_h), (70, 72, 82), 1, cv2.LINE_AA)
    bx = x0 + 28
    by = y0 + 18
    _bar(frame, bx, by, bar_w, bar_h, throttle_frac, C_THROTTLE)
    _bar(frame, bx + bar_w + gap, by, bar_w, bar_h, brake_frac, C_BRAKE)
    for frac, text in ((1.0, "100"), (0.5, "50"), (0.0, "0")):
        ty = by + bar_h - 1 - int(round((bar_h - 2) * frac))
        cv2.putText(
            frame, text, (x0 + 4, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TICK, 1, cv2.LINE_AA,
        )
    t_pct = int(round(100.0 * float(np.clip(throttle_frac, 0.0, 1.0))))
    b_pct = int(round(100.0 * float(np.clip(brake_frac, 0.0, 1.0))))
    cv2.putText(
        frame, f"T {t_pct}%", (bx - 8, by + bar_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_THROTTLE, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"B {b_pct}%", (bx + bar_w + gap - 10, by + bar_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_BRAKE, 1, cv2.LINE_AA,
    )
    lock_frac = float(steer_rad) / max(float(max_steer_rad), 1e-6)
    wheel = _wheel_rgba(wheel_r, lock_frac)
    wx = x0 + 118
    wy = y0 + 8
    _blit_rgba(frame, wheel, wx, wy)
    steer_deg = float(steer_rad) * 180.0 / math.pi
    cv2.putText(
        frame, f"{steer_deg:+5.1f} deg", (wx + 6, y0 + panel_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT, 1, cv2.LINE_AA,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--mp4", required=True)
    p.add_argument("--view-m", type=float, default=26.0)
    p.add_argument("--width", type=int, default=1120)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--tail-s", type=float, default=3.0)
    args = p.parse_args()

    d = np.load(args.npz)
    n = int(len(d["lengths"]))
    lengths = d["lengths"]
    crashed = d["crashed"]
    car_l, car_w = float(d["car_length"]), float(d["car_width"])
    left, right, center = d["left"], d["right"], d["centerline"]
    ego_xy = [d[f"ego_xy_{i}"] for i in range(n)]
    ego_yaw = [d[f"ego_yaw_{i}"] for i in range(n)]
    ego_sp = [d[f"ego_speed_{i}"] for i in range(n)]
    opp_xy = [d[f"opp_xy_{i}"] for i in range(n)]
    opp_yaw = [d[f"opp_yaw_{i}"] for i in range(n)]
    has_hud = all(
        key in d for key in (
            "ego_i_drive_cmd_0", "ego_i_brake_cmd_0", "ego_steer_logical_0",
            "i_drive_max_a", "i_brake_max_a", "max_steer_rad",
        )
    )
    if has_hud:
        i_drive_max = float(d["i_drive_max_a"])
        i_brake_max = float(d["i_brake_max_a"])
        max_steer = float(d["max_steer_rad"])
        throttle = np.clip(
            d["ego_i_drive_cmd_0"] / max(i_drive_max, 1e-6), 0.0, 1.0
        )
        brake = np.clip(
            d["ego_i_brake_cmd_0"] / max(i_brake_max, 1e-6), 0.0, 1.0
        )
        steer_rad = np.asarray(d["ego_steer_logical_0"], dtype=np.float64)

    total = int(lengths.max())
    stop = min(total, int(lengths.max()) if crashed.all() else total)
    tail = int(args.tail_s * args.fps)
    if not crashed.all():
        stop = total
    W, H = args.width, args.height
    scale = W / args.view_m
    writer = imageio.get_writer(args.mp4, fps=args.fps, macro_block_size=2)

    cam = None
    for t in range(stop):
        idx = [min(t, lengths[i] - 1) for i in range(n)]
        alive = [i for i in range(n) if t < lengths[i]]
        focus = alive if alive else list(range(n))
        target = np.mean([ego_xy[i][idx[i]] for i in focus], axis=0)
        cam = target if cam is None else 0.82 * cam + 0.18 * target

        def to_px(pts):
            pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
            out = np.empty((pts.shape[0], 2), dtype=np.int32)
            out[:, 0] = ((pts[:, 0] - cam[0]) * scale + W / 2).astype(np.int32)
            out[:, 1] = ((cam[1] - pts[:, 1]) * scale + H / 2).astype(np.int32)
            return out

        frame = np.empty((H, W, 3), dtype=np.uint8)
        frame[:] = BG
        cv2.polylines(frame, [to_px(center)], True, C_CENTER, 1, cv2.LINE_AA)
        for edge in (left, right):
            cv2.polylines(frame, [to_px(edge)], True, C_EDGE, 3, cv2.LINE_AA)

        for i in range(n):
            lo = max(0, idx[i] - tail)
            trail = ego_xy[i][lo:idx[i] + 1]
            if len(trail) > 1:
                col = tuple(int(0.55 * c) for c in CAR_COLORS[i % len(CAR_COLORS)])
                cv2.polylines(frame, [to_px(trail)], False, col, 2, cv2.LINE_AA)

        for i in range(n):
            col = CAR_COLORS[i % len(CAR_COLORS)]
            x, y = ego_xy[i][idx[i]]
            r = to_px(rect(x, y, ego_yaw[i][idx[i]], car_l, car_w))
            if t < lengths[i]:
                cv2.fillPoly(frame, [r], col)
            else:
                cv2.polylines(frame, [r], True, tuple(int(0.5 * c) for c in col),
                              1, cv2.LINE_AA)
                cx, cy = to_px([[x, y]])[0]
                for dx, dy in ((-7, -7, ), (-7, 7)):
                    cv2.line(frame, (cx + dx, cy + dy), (cx - dx, cy - dy),
                             (90, 90, 240), 2, cv2.LINE_AA)

        for i in range(n):
            o = to_px(rect(opp_xy[i][idx[i]][0], opp_xy[i][idx[i]][1],
                           opp_yaw[i][idx[i]], car_l, car_w))
            cv2.fillPoly(frame, [o], OPP_COLOR)
            cv2.polylines(
                frame, [o], True, CAR_COLORS[i % len(CAR_COLORS)], 2, cv2.LINE_AA
            )

        n_alive = len(alive)
        cv2.putText(frame, f"t = {t / args.fps:5.1f} s", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"running {n_alive}/{n}", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 255, 140) if n_alive == n else (255, 140, 140),
                    1, cv2.LINE_AA)
        v = [ego_sp[i][idx[i]] for i in alive]
        if v:
            cv2.putText(frame, f"v {np.mean(v):4.1f} m/s", (14, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 1, cv2.LINE_AA)
        if has_hud:
            fi = min(idx[0], len(throttle) - 1)
            draw_controls_hud(
                frame, float(throttle[fi]), float(brake[fi]),
                float(steer_rad[fi]), max_steer,
            )
            legend_y = 196
        else:
            legend_y = 14
        cv2.rectangle(frame, (W - 210, legend_y), (W - 194, legend_y + 12),
                      OPP_COLOR, -1)
        cv2.putText(frame, "centerline opponent", (W - 186, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
        writer.append_data(frame)

    writer.close()
    print(f"wrote {args.mp4} ({stop} frames)")


if __name__ == "__main__":
    main()

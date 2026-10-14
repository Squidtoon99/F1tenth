"""Render a zoomed follow-camera overlay video from saved rollout trajectories."""

import argparse

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


def rect(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    pts = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


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
            cv2.polylines(frame, [o], True, (40, 40, 40), 1, cv2.LINE_AA)

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
        cv2.rectangle(frame, (W - 210, 14), (W - 194, 26), OPP_COLOR, -1)
        cv2.putText(frame, "centerline opponent", (W - 186, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
        writer.append_data(frame)

    writer.close()
    print(f"wrote {args.mp4} ({stop} frames)")


if __name__ == "__main__":
    main()

"""PNG snapshots for a run page. The pixels are not stored as history."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from seed_data import EPISODE_MARKS

BG = (22, 22, 24)
ORANGE = (58, 160, 200)
ORANGE_2 = (110, 231, 240)
MUTED = (154, 154, 158)
TEXT = (244, 244, 245)
W, H = 960, 420


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _blend(top, bottom_rgb, alpha: float):
    return tuple(int(top[i] * alpha + bottom_rgb[i] * (1 - alpha)) for i in range(3))


def series_for(wandb_run_id: str, evals: list[dict]) -> tuple[str, list[tuple[float, float]]]:
    marks = EPISODE_MARKS.get(wandb_run_id) or []
    if marks:
        return "Episode length", [(m["step"] / 1_000_000, m["seconds"]) for m in marks]
    points = []
    for ev in evals:
        if ev["flying_lap_s"] is not None and ev["step"] is not None:
            points.append((ev["step"] / 1_000_000, ev["flying_lap_s"]))
    points.sort()
    if points:
        return "Flying lap", points
    return "No score yet", []


def render_chart(title: str, points: list[tuple[float, float]], unit: str) -> bytes:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((32, 24), title, fill=TEXT, font=_font(22))
    if not points:
        draw.text((32, 180), "No score yet", fill=MUTED, font=_font(28))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    left, right, top, bottom = 72, W - 32, 80, H - 56
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
        ymin = max(0, ymin - 1)
    pad_y = (ymax - ymin) * 0.12
    ymin -= pad_y
    ymax += pad_y

    def xy(x, y):
        px = left + (x - xmin) / (xmax - xmin) * (right - left)
        py = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
        return px, py

    coords = [xy(x, y) for x, y in points]
    floor = [(right, bottom), (left, bottom)]
    polygon = coords + floor
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        steps = max(int(abs(x1 - x0)), 1)
        for s in range(steps):
            t = s / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            fade = (y - top) / max(bottom - top, 1)
            color = _blend(ORANGE_2 if fade < 0.5 else ORANGE, BG, 0.85 - 0.45 * fade)
            draw.line([(x, y), (x, bottom)], fill=color)
    if len(coords) == 1:
        x, y = coords[0]
        draw.line([(x, y), (x, bottom)], fill=ORANGE, width=8)
    draw.line(coords, fill=ORANGE, width=3)
    for x, y in coords:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ORANGE_2)
    draw.text((left, H - 36), f"{xmin:.0f}M", fill=MUTED, font=_font(16))
    draw.text((right - 40, H - 36), f"{xmax:.0f}M", fill=MUTED, font=_font(16))
    draw.text((16, top), f"{ymax:.0f}{unit}", fill=MUTED, font=_font(16))
    del polygon
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

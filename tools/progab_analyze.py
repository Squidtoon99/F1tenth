#!/usr/bin/env python3
"""Extract progab run metrics for validity checks and A/B verdict."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PCPLUS600_600M_REF = 5.38
PCPLUS600_600M_MIN = 5.05
CONTROL_PROGRESS_BASELINE = 0.510


def parse_log(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    rows: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        tm = re.search(
            r"transitions=(\d+).*episode_lifespan=([\d.]+)s",
            line,
        )
        if tm and "ticks=" in line:
            current = {
                "transitions": int(tm.group(1)),
                "lifespan": float(tm.group(2)),
            }
            rows.append(current)
            continue
        sm = re.search(r"env: speed=([\d.]+)", line)
        if sm and current:
            current["speed"] = float(sm.group(1))
        pm = re.search(r"progress=([\d.]+)", line)
        if pm and "rewards:" in line and current:
            current["progress"] = float(pm.group(1))
    return rows


def nearest(rows: list[dict], target: int) -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["transitions"] - target))


def bucket_mean(rows: list[dict], lo: int, hi: int, key: str) -> float | None:
    vals = [r[key] for r in rows if lo <= r["transitions"] <= hi and key in r]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pct_above_speed(rows: list[dict], threshold: float, lo: int, hi: int) -> float | None:
    speeds = [r["speed"] for r in rows if lo <= r["transitions"] <= hi and "speed" in r]
    if not speeds:
        return None
    return 100.0 * sum(s > threshold for s in speeds) / len(speeds)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--milestone", type=int, default=600_000_000)
    p.add_argument("--window", type=int, default=5_000_000)
    args = p.parse_args()
    rows = parse_log(args.log)
    if not rows:
        print(f"NO_DATA {args.log}")
        return 2
    m = args.milestone
    w = args.window
    lo, hi = m - w, m
    speed = bucket_mean(rows, lo, hi, "speed")
    life = bucket_mean(rows, lo, hi, "lifespan")
    progress = bucket_mean(rows, lo, hi, "progress")
    near = nearest(rows, m)
    above5 = pct_above_speed(rows, 5.0, lo, hi)
    print(f"log={args.log}")
    print(f"milestone={m} window=[{lo},{hi}]")
    print(f"speed_mean={speed}")
    print(f"lifespan_mean={life}")
    print(f"progress_mean={progress}")
    print(f"pct_ticks_speed_gt_5={above5}")
    if near:
        print(
            f"nearest@{near['transitions']}: speed={near.get('speed')} "
            f"life={near.get('lifespan')} progress={near.get('progress')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

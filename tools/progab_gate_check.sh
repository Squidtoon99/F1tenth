#!/usr/bin/env bash
# Evaluate warm-start unfreeze+10M gate from a run log. Exit 0 pass, 1 fail.
set -euo pipefail

LOG="${1:-/home/ubuntu/projects/F1tenth/training/outputs/runs/progab-control-a001/run.log}"
MIN_TRANS="${2:-29000000}"
MAX_TRANS="${3:-31000000}"

python3 - "$LOG" "$MIN_TRANS" "$MAX_TRANS" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
lo, hi = int(sys.argv[2]), int(sys.argv[3])
text = path.read_text(errors="replace")
speed = None
life = None
progress = None
for line in text.splitlines():
    tm = re.search(
        r"transitions=(\d+).*episode_lifespan=([\d.]+)s",
        line,
    )
    if tm:
        t, l = int(tm.group(1)), float(tm.group(2))
        if lo <= t <= hi:
            life = l
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and "transitions=" in line:
        tm2 = re.search(r"transitions=(\d+)", line)
        if tm2 and lo <= int(tm2.group(1)) <= hi:
            speed = float(sm.group(1))
    pm = re.search(r"progress=([\d.]+)", line)
    if pm and "rewards:" in line:
        tm2 = re.search(r"transitions=(\d+)", line)
        if tm2 and lo <= int(tm2.group(1)) <= hi:
            progress = float(pm.group(1))

if life is None or speed is None:
    print(f"NO_DATA life={life} speed={speed} window={lo}-{hi}")
    raise SystemExit(2)

ok = life >= 100.0 and speed >= 5.2
print(f"gate life={life:.1f}s speed={speed:.2f} progress={progress} pass={ok}")
raise SystemExit(0 if ok else 1)
PY

#!/usr/bin/env bash
# Sourceable progab milestone gates (400M interim, 1B validity). No side effects.

# pcplus2b-a001 anchors (from-noise self-play reproduction run)
PCPLUS2B_400M_SPEED=4.91
PCPLUS2B_400M_LIFE=54.6
PCPLUS2B_400M_SPEED_MIN=4.76
PCPLUS2B_400M_LIFE_MIN=45.0
PCPLUS2B_1B_SPEED=5.158
PCPLUS2B_1B_LIFE=227.2
PCPLUS2B_1B_SPEED_MIN=5.05
PCPLUS2B_1B_LIFE_MIN=200.0

latest_transitions() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
    raise SystemExit(0)
vals = re.findall(
    r"standalone_trainer INFO: ticks=\d+ transitions=(\d+)",
    path.read_text(errors="replace"),
)
print(int(vals[-1]) if vals else 0)
PY
}

check_interim_400m_gate() {
  local log_file="$1"
  python3 - "$log_file" \
    "$PCPLUS2B_400M_SPEED" "$PCPLUS2B_400M_LIFE" \
    "$PCPLUS2B_400M_SPEED_MIN" "$PCPLUS2B_400M_LIFE_MIN" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
ref_speed, ref_life = float(sys.argv[2]), float(sys.argv[3])
min_speed, min_life = float(sys.argv[4]), float(sys.argv[5])
rows = []
cur = {}
for line in path.read_text(errors="replace").splitlines():
    m = re.search(r"transitions=(\d+).*episode_lifespan=([\d.]+)s", line)
    if m and "ticks=" in line:
        cur = {"t": int(m.group(1)), "life": float(m.group(2))}
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and cur:
        cur["s"] = float(sm.group(1))
        rows.append(dict(cur))
life_band = (ref_life - 10.0, ref_life + 10.0)
w = [
    r for r in rows
    if 375_000_000 <= r["t"] <= 425_000_000
    and life_band[0] <= r["life"] <= life_band[1]
    and "s" in r
]
if not w:
    print(f"INTERIM_400M NO_DATA life_band={life_band}")
    raise SystemExit(1)
speed = sum(r["s"] for r in w) / len(w)
life = sum(r["life"] for r in w) / len(w)
ok = speed >= min_speed and life >= min_life
print(
    f"INTERIM_400M speed={speed:.3f} life={life:.1f} "
    f"ref={ref_speed}@{ref_life} min_speed={min_speed} pass={ok}"
)
raise SystemExit(0 if ok else 1)
PY
}

check_validity_1b_gate() {
  local log_file="$1"
  python3 - "$log_file" \
    "$PCPLUS2B_1B_SPEED" "$PCPLUS2B_1B_LIFE" \
    "$PCPLUS2B_1B_SPEED_MIN" "$PCPLUS2B_1B_LIFE_MIN" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
ref_speed, ref_life = float(sys.argv[2]), float(sys.argv[3])
min_speed, min_life = float(sys.argv[4]), float(sys.argv[5])
rows = []
cur = {}
for line in path.read_text(errors="replace").splitlines():
    m = re.search(r"transitions=(\d+).*episode_lifespan=([\d.]+)s", line)
    if m and "ticks=" in line:
        cur = {"t": int(m.group(1)), "life": float(m.group(2))}
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and cur:
        cur["s"] = float(sm.group(1))
        rows.append(dict(cur))
speed_w = [r for r in rows if 950_000_000 <= r["t"] <= 1_000_000_000 and "s" in r]
life_w = [r for r in rows if 950_000_000 <= r["t"] <= 1_000_000_000]
if not speed_w or not life_w:
    print("VALIDITY_1B NO_DATA")
    raise SystemExit(1)
speed = sum(r["s"] for r in speed_w) / len(speed_w)
life = sum(r["life"] for r in life_w) / len(life_w)
ok = speed >= min_speed and life >= min_life
print(
    f"VALIDITY_1B speed={speed:.3f} life={life:.1f} "
    f"ref={ref_speed}@{ref_life} min_speed={min_speed} min_life={min_life} pass={ok}"
)
raise SystemExit(0 if ok else 1)
PY
}

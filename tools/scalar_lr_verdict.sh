#!/usr/bin/env bash
# Read scalar-LR validation run and print PASS|FAIL|GRAY with metrics.
set -euo pipefail

LOG="${1:?usage: scalar_lr_verdict.sh run.log}"
python3 - "$LOG" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
rows, cur = [], {}
for line in log_path.read_text(errors="replace").splitlines():
    m = re.search(r"transitions=(\d+).*episode_lifespan=([\d.]+)s", line)
    if m and "ticks=" in line:
        cur = {"t": int(m.group(1)), "life": float(m.group(2))}
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and cur:
        rows.append({"t": cur["t"], "life": cur["life"], "speed": float(sm.group(1))})

if not rows:
    print("VERDICT=NO_DATA")
    raise SystemExit(2)

max_t = max(r["t"] for r in rows)
print(f"max_transitions={max_t}")

# For a 100M validation run, anchor on 95–100M only (not the whole tail of longer logs).
if max_t <= 101_000_000:
    eval_lo, eval_hi = 95_000_000, 100_000_000
else:
    eval_lo, eval_hi = max_t - 5_000_000, max_t
    print(f"note: run exceeded 100M — using last 5M window [{eval_lo},{eval_hi}]")

print("=== trajectory (10M buckets, speed / lifespan) ===")
for m in range(10_000_000, min(max_t, 100_000_000) + 1, 10_000_000):
    w = [r for r in rows if m - 5_000_000 <= r["t"] <= m + 5_000_000]
    if w:
        spd = sum(r["speed"] for r in w) / len(w)
        life = sum(r["life"] for r in w) / len(w)
        print(f"  @{m/1e6:.0f}M: speed={spd:.2f} m/s  life={life:.1f}s  n={len(w)}")

tail = [r for r in rows if eval_lo <= r["t"] <= eval_hi]
tail_life22 = [r for r in tail if r["life"] >= 22.0]
tail_short = [r for r in tail if r["life"] <= 20.0]

tail_spd = sum(r["speed"] for r in tail) / len(tail) if tail else 0.0
tail_life = sum(r["life"] for r in tail) / len(tail) if tail else 0.0
spd_at_life22 = (
    sum(r["speed"] for r in tail_life22) / len(tail_life22) if tail_life22 else None
)
life_at_life22 = (
    sum(r["life"] for r in tail_life22) / len(tail_life22) if tail_life22 else None
)
short_spd = sum(r["speed"] for r in tail_short) / len(tail_short) if tail_short else None
short_life = sum(r["life"] for r in tail_short) / len(tail_short) if tail_short else None

print(f"=== @{eval_lo/1e6:.0f}–{eval_hi/1e6:.0f}M ===")
print(f"  speed_mean={tail_spd:.3f}  lifespan_mean={tail_life:.1f}s  n={len(tail)}")
if spd_at_life22 is not None:
    print(
        f"  @comparable lifespan (life>=22s): speed={spd_at_life22:.3f}  "
        f"life={life_at_life22:.1f}s  n={len(tail_life22)}"
    )
else:
    print("  @comparable lifespan (life>=22s): NO_TICKS")
if short_spd is not None:
    print(
        f"  @short-lived (life<=20s): speed={short_spd:.3f}  "
        f"life={short_life:.1f}s  n={len(tail_short)}"
    )

# Reference anchors
print("=== reference ===")
print("  pcplus2b-a001 @100M: 4.26 m/s @ 26.7 s (good)")
print("  progab-control @100M: 2.72 m/s @ 17.0 s (regressed)")

if spd_at_life22 is not None and spd_at_life22 >= 3.8 and life_at_life22 >= 22.0:
    verdict = "PASS"
elif short_spd is not None and short_spd <= 3.0 and short_life <= 20.0:
    verdict = "FAIL"
elif tail_spd <= 3.0 and tail_life <= 20.0:
    verdict = "FAIL"
else:
    verdict = "GRAY"

print(f"=== VERDICT={verdict} ===")
PY

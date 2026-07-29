#!/usr/bin/env bash
# Tight poll near 600M: stop chain immediately if control validity fails (attempt 7).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_LOG="$REPO/training/outputs/runs/progab-control-a001/run.log"
WLOG="$REPO/training/outputs/runs/progab-validity-watch.log"
MIN=5.05

log() { echo "[$(date -Is)] validity-watch: $*" | tee -a "$WLOG"; }

latest_trans() {
  python3 - "$CONTROL_LOG" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0); raise SystemExit
vals = re.findall(r"transitions=(\d+)", p.read_text(errors="replace"))
print(int(vals[-1]) if vals else 0)
PY
}

speed_at_600m() {
  python3 - "$CONTROL_LOG" <<'PY'
import re, sys
from pathlib import Path
rows, cur = [], {}
for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    m = re.search(r"transitions=(\d+).*episode_lifespan=", line)
    if m and "ticks=" in line:
        cur = {"t": int(m.group(1))}
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and cur:
        cur["s"] = float(sm.group(1))
        rows.append(cur)
w = [r for r in rows if 595_000_000 <= r["t"] <= 600_000_000 and "s" in r]
print(sum(r["s"] for r in w) / len(w) if w else 0)
PY
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
log "started (poll 5s when trans>=590M; stop chain on fail at Training finished)"

while true; do
  trans=$(latest_trans)
  if [ "$trans" -ge 590000000 ]; then
    sleep 5
    if grep -q "Training finished" "$CONTROL_LOG" 2>/dev/null; then
      spd=$(speed_at_600m)
      log "control finished trans=$trans speed_mean=$spd min=$MIN"
      if python3 - "$spd" "$MIN" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
      then
        log "VALIDITY PASS — chain may start Arm B"
      else
        log "VALIDITY FAIL — stopping f1tenth-progab-a001.service"
        systemctl --user stop f1tenth-progab-a001.service 2>/dev/null || true
        pkill -f 'standalone_trainer.*progab-super' 2>/dev/null || true
        echo "CONTROL_VALIDITY_FAIL speed=$spd" >> "$WLOG"
      fi
      exit 0
    fi
  else
    sleep 60
  fi
done
fi

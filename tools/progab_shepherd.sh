#!/usr/bin/env bash
# Autonomous progab attempt-7 shepherd: validity @600M control, verdict @600M B, follow-up.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$REPO/training"
CONTROL_LOG="$TRAINING/outputs/runs/progab-control-a001/run.log"
SUPER_LOG="$TRAINING/outputs/runs/progab-super-a001/run.log"
SHEPHERD_LOG="$TRAINING/outputs/runs/progab-shepherd.log"
PCPLUS600_600M=5.38
PCPLUS600_MIN=5.05
PROGRESS_BASE=0.510

log() { echo "[$(date -Is)] shepherd: $*" | tee -a "$SHEPHERD_LOG"; }

append_experiment_log() {
  local block="$1"
  {
    echo ""
    echo "$block"
  } >> "$REPO/training/outputs/experiments/long-horizon-2b-sensors/EXPERIMENT_LOG.md"
}

wait_training_finished() {
  local log_file="$1" target="$2"
  while true; do
    if grep -q "Training finished" "$log_file" 2>/dev/null; then
      local trans
      trans=$(python3 - "$log_file" <<'PY'
import re, sys
from pathlib import Path
vals = re.findall(r"transitions=(\d+)", Path(sys.argv[1]).read_text(errors="replace"))
print(int(vals[-1]) if vals else 0)
PY
)
      if [ "$trans" -ge "$target" ]; then
        return 0
      fi
    fi
    sleep 120
  done
}

analyze_arm() {
  local log_file="$1"
  "$REPO/tools/progab_analyze.py" "$log_file" --milestone 600000000 --window 5000000
}

log "shepherd started (attempt 7 from-noise)"

# --- Phase 1: wait for control 600M ---
log "waiting for control Arm A to finish 600M..."
wait_training_finished "$CONTROL_LOG" 600000000

log "control finished — validity check"
analyze_arm "$CONTROL_LOG" | tee -a "$SHEPHERD_LOG"

CTRL_SPEED=$(python3 - "$CONTROL_LOG" <<'PY'
import re, sys
from pathlib import Path
rows = []
cur = {}
for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    m = re.search(r"transitions=(\d+).*episode_lifespan=", line)
    if m and "ticks=" in line:
        cur = {"t": int(m.group(1))}
    sm = re.search(r"env: speed=([\d.]+)", line)
    if sm and cur:
        cur["s"] = float(sm.group(1))
        rows.append(cur)
window = [r for r in rows if 595_000_000 <= r["t"] <= 600_000_000 and "s" in r]
print(sum(r["s"] for r in window)/len(window) if window else 0)
PY
)

VALID=1
if ! python3 - "$CTRL_SPEED" <<'PY'
import sys
speed = float(sys.argv[1])
ref, mn = 5.38, 5.05
ok = speed >= mn
print(f"validity speed={speed:.3f} ref={ref} min={mn} pass={ok}")
sys.exit(0 if ok else 1)
PY
then
  VALID=0
fi

if [ "$VALID" -eq 0 ]; then
  log "VALIDITY FAIL — stopping chain before/during Arm B; GPU better spent diagnosing control"
  systemctl --user stop f1tenth-progab-a001.service 2>/dev/null || true
  pkill -f 'standalone_trainer.*progab-super' 2>/dev/null || true
  echo "CONTROL_VALIDITY_FAIL speed=$CTRL_SPEED" >> "$SHEPHERD_LOG"
  append_experiment_log "### Attempt 7 control validity @600M ($(date '+%Y-%m-%d %H:%M %Z'))

**Result: FAIL** — from-noise control did not reproduce \`pcplus600\` baseline.

| Metric | Value | Gate |
| --- | --- | --- |
| speed_mean @595–600M | **${CTRL_SPEED}** m/s | ≥ **5.05** (pcplus600−0.15; ref **5.38**) |

**Decision:** stopped chain; Arm B not started (A/B contrast would be on sand).

**Follow-up:** GPU available for control divergence diagnosis (not auto-launched — validity failure)."
  exit 1
fi

log "VALIDITY PASS speed=$CTRL_SPEED — allowing Arm B to complete"
append_experiment_log "### Attempt 7 control validity @600M ($(date '+%Y-%m-%d %H:%M %Z'))

**Result: PASS** — control reproduced baseline within tolerance.

| Metric | Value | Gate |
| --- | --- | --- |
| speed_mean @595–600M | **${CTRL_SPEED}** m/s | ≥ **5.05** (ref pcplus600 **5.38**) |

Arm B (treatment) proceeding."

# --- Phase 2: wait for treatment 600M (chain should run B) ---
if [ ! -f "$SUPER_LOG" ]; then
  log "waiting for super log to appear..."
  while [ ! -f "$SUPER_LOG" ]; do sleep 60; done
fi
wait_training_finished "$SUPER_LOG" 600000000

log "treatment finished — verdict"
analyze_arm "$CONTROL_LOG" | tee -a "$SHEPHERD_LOG"
analyze_arm "$SUPER_LOG" | tee -a "$SHEPHERD_LOG"

VERDICT=$(python3 - "$CONTROL_LOG" "$SUPER_LOG" <<'PY'
import re, sys
from pathlib import Path

def metrics(path):
    rows = []
    cur = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.search(r"transitions=(\d+).*episode_lifespan=([\d.]+)s", line)
        if m and "ticks=" in line:
            cur = {"t": int(m.group(1)), "life": float(m.group(2))}
            rows.append(cur)
        sm = re.search(r"env: speed=([\d.]+)", line)
        if sm and rows:
            rows[-1]["speed"] = float(sm.group(1))
        pm = re.search(r"progress=([\d.]+)", line)
        if pm and "rewards:" in line and rows:
            rows[-1]["progress"] = float(pm.group(1))
    w = [r for r in rows if 595_000_000 <= r["t"] <= 600_000_000]
    if not w:
        return None
    def mean(k):
        v = [r[k] for r in w if k in r]
        return sum(v)/len(v) if v else None
    above5 = sum(1 for r in w if r.get("speed", 0) > 5.0) / len(w) * 100
    return {"speed": mean("speed"), "life": mean("life"), "progress": mean("progress"), "above5": above5}

c, t = metrics(sys.argv[1]), metrics(sys.argv[2])
if not c or not t:
    print("void"); raise SystemExit(0)
prog_delta = (t["progress"] or 0) - (c["progress"] or 0)
if abs(prog_delta) < 0.02 and t["above5"] < 10:
    print("void"); raise SystemExit(0)
if t["speed"] < 5.20 or t["life"] < 120:
    print("harm"); raise SystemExit(0)
if t["speed"] >= c["speed"] + 0.10 and t["life"] >= 200:
    print("win"); raise SystemExit(0)
if abs(t["speed"] - c["speed"]) < 0.10 and abs(t["life"] - c["life"]) < 30:
    print("null"); raise SystemExit(0)
if t["speed"] > c["speed"] and t["life"] < 150:
    print("harm"); raise SystemExit(0)
print("null")
PY
)

log "VERDICT=$VERDICT"
systemctl --user stop f1tenth-progab-a001.service 2>/dev/null || true
sleep 5

METRICS=$(python3 - "$CONTROL_LOG" "$SUPER_LOG" <<'PY'
import re, sys
from pathlib import Path

def metrics(path):
    rows = []
    cur = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.search(r"transitions=(\d+).*episode_lifespan=([\d.]+)s", line)
        if m and "ticks=" in line:
            cur = {"t": int(m.group(1)), "life": float(m.group(2))}
            rows.append(cur)
        sm = re.search(r"env: speed=([\d.]+)", line)
        if sm and rows:
            rows[-1]["speed"] = float(sm.group(1))
        pm = re.search(r"progress=([\d.]+)", line)
        if pm and "rewards:" in line and rows:
            rows[-1]["progress"] = float(pm.group(1))
    w = [r for r in rows if 595_000_000 <= r["t"] <= 600_000_000]
    if not w:
        return None
    def mean(k):
        v = [r[k] for r in w if k in r]
        return sum(v)/len(v) if v else None
    above5 = sum(1 for r in w if r.get("speed", 0) > 5.0) / len(w) * 100
    return {"speed": mean("speed"), "life": mean("life"), "progress": mean("progress"), "above5": above5}

c, t = metrics(sys.argv[1]), metrics(sys.argv[2])
if not c or not t:
    print("MISSING"); raise SystemExit(0)
print(f"control speed={c['speed']:.3f} life={c['life']:.1f} progress={c['progress']:.3f} above5={c['above5']:.1f}%")
print(f"treatment speed={t['speed']:.3f} life={t['life']:.1f} progress={t['progress']:.3f} above5={t['above5']:.1f}%")
PY
)

case "$VERDICT" in
  win)
    FOLLOWUP="progab-super-extend-a001 (2B, init from treatment 600M ckpt)"
    "$REPO/tools/progab_followup.sh" win
    ;;
  null|harm)
    FOLLOWUP="pcplus2b-a002 (2B from noise, pcplus2b-a001 recipe)"
    "$REPO/tools/progab_followup.sh" "$VERDICT"
    ;;
  void)
    FOLLOWUP="progab-super-lowthr-a001 (2B, threshold 4.5 m/s — bonus active in policy regime)"
    "$REPO/tools/progab_followup.sh" void 4.5
    ;;
  *) log "unknown verdict"; exit 1;;
esac

append_experiment_log "### Attempt 7 A/B verdict @600M ($(date '+%Y-%m-%d %H:%M %Z'))

**Verdict: \`${VERDICT^^}\`**

\`\`\`
$METRICS
\`\`\`

**Decision:** launched **$FOLLOWUP** immediately (no idle GPU).

**Reasoning:**
$(case "$VERDICT" in
  win) echo "- Treatment beat control on speed (+≥0.10 m/s) with lifespan ≥200 s — extend winning config to 2B.";;
  null) echo "- Speed/lifespan within null band vs control — default to strongest known-good \`pcplus2b-a001\` recipe at 2B.";;
  harm) echo "- Treatment harmed speed (<5.20 m/s) or lifespan (<120 s) — revert to \`pcplus2b-a001\` at 2B.";;
  void) echo "- \`reward/progress\` did not diverge from control ~0.510 and policy rarely >5 m/s — threshold inert; relaunch treatment at 4.5 m/s.";;
esac)"

log "follow-up launched for verdict=$VERDICT ($FOLLOWUP)"

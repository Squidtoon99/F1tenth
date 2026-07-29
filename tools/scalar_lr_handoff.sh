#!/usr/bin/env bash
# Poll validation to completion, verdict, launch next stage with no idle GPU.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/training/outputs/runs/scalar-lr-fix-validate-a001/run.log"
HANDOFF_LOG="$REPO/training/outputs/runs/scalar-lr-handoff.log"

log() { echo "[$(date -Is)] handoff: $*" | tee -a "$HANDOFF_LOG"; }

latest_transitions() {
  python3 - "$LOG" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0); raise SystemExit(0)
vals = re.findall(r"standalone_trainer INFO: ticks=\d+ transitions=(\d+)", p.read_text(errors="replace"))
print(int(vals[-1]) if vals else 0)
PY
}

launch_seed43() {
  log "GRAY -> launching seed 43 validation immediately"
  systemctl --user stop f1tenth-scalar-lr-fix-validate.service 2>/dev/null || true
  sleep 3
  cat >"$HOME/.config/systemd/user/f1tenth-scalar-lr-fix-validate.service" <<EOF
[Unit]
Description=F1TENTH scalar-LR fix validation seed 43
After=default.target

[Service]
Type=simple
WorkingDirectory=$REPO/training
Environment=WANDB_MODE=online
Environment=WANDB_GROUP=long-horizon-2b-sensors
Environment=PYTHONUNBUFFERED=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=RUN_ID=scalar-lr-fix-validate-a001-seed43
Environment=SEED=43
Environment=PATH=/home/ubuntu/miniconda3/bin:$REPO/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$REPO/tools/scalar_lr_fix_validate.sh
Restart=no
TimeoutStopSec=120
StandardOutput=append:$REPO/training/outputs/runs/scalar-lr-fix-validate-seed43-chain.log
StandardError=append:$REPO/training/outputs/runs/scalar-lr-fix-validate-seed43-chain.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user start f1tenth-scalar-lr-fix-validate.service
}

launch_ab() {
  log "PASS -> launching 1B progab A/B"
  systemctl --user stop f1tenth-scalar-lr-fix-validate.service 2>/dev/null || true
  sleep 3
  # Archive warm-start contamination from earlier aborted chain attempt.
  for rid in progab-control-a001 progab-super-a001; do
    if [ -d "$REPO/training/outputs/runs/$rid" ]; then
      dest="$REPO/training/outputs/runs/${rid}.archived-$(date +%Y%m%d%H%M%S)"
      log "archiving $rid -> $(basename "$dest")"
      mv "$REPO/training/outputs/runs/$rid" "$dest"
    fi
  done
  cp "$REPO/tools/systemd/f1tenth-progab-a001.service" "$HOME/.config/systemd/user/"
  systemctl --user daemon-reload
  systemctl --user start f1tenth-progab-a001.service
}

launch_known_good() {
  log "FAIL -> launching pcplus2b-a002 (known-good recipe) while investigating"
  systemctl --user stop f1tenth-scalar-lr-fix-validate.service 2>/dev/null || true
  sleep 3
  cat >"$HOME/.config/systemd/user/f1tenth-pcplus2b-a002.service" <<EOF
[Unit]
Description=F1TENTH pcplus2b-a002 known-good 2B from-noise
After=default.target

[Service]
Type=simple
WorkingDirectory=$REPO/training
Environment=WANDB_MODE=online
Environment=WANDB_GROUP=long-horizon-2b-sensors
Environment=PYTHONUNBUFFERED=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=PATH=/home/ubuntu/miniconda3/bin:$REPO/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$REPO/tools/progab_followup.sh harm
Restart=no
TimeoutStopSec=120
StandardOutput=append:$REPO/training/outputs/runs/pcplus2b-a002-chain.log
StandardError=append:$REPO/training/outputs/runs/pcplus2b-a002-chain.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user start f1tenth-pcplus2b-a002.service
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
log "waiting for validation to reach 100M..."
while true; do
  if grep -q "Training finished" "$LOG" 2>/dev/null; then
    break
  fi
  trans=$(latest_transitions)
  if [ "$trans" -ge 100000000 ]; then
    sleep 30
    if grep -q "Training finished" "$LOG" 2>/dev/null; then
      break
    fi
  fi
  sleep 60
done

log "validation complete — running verdict"
VERDICT_OUT=$("$REPO/tools/scalar_lr_verdict.sh" "$LOG" 2>&1 | tee -a "$HANDOFF_LOG")
VERDICT=$(echo "$VERDICT_OUT" | grep -oP 'VERDICT=\K\w+' | tail -1)
log "verdict=$VERDICT"

case "$VERDICT" in
  PASS) launch_ab ;;
  GRAY) launch_seed43 ;;
  FAIL) launch_known_good ;;
  *) log "unknown verdict; launching known-good as fallback"; launch_known_good ;;
esac

log "next stage launched"
fi

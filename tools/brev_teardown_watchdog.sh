#!/usr/bin/env bash
# Safety net that tears down rented Brev GPU instances without depending on any
# agent staying alive. Two independent triggers:
#   1. hard deadline  - tear down at HARD_DEADLINE (see below), when the
#                       provisioned budget expires and the instance is forfeit
#   2. idle watchdog   - tear down after IDLE_CHECKS_TO_STOP consecutive checks
#                       with no trainer process (generous, so it never races a
#                       normal run handoff or an artifact harvest)
#
# This provider (massedcompute_L40S) rejects `brev stop` with "does not support
# stop", so delete is the only way to end billing, and teardown is irreversible.
# An idle teardown therefore requires a per-instance marker written by whoever
# harvested the artifacts; without it the watchdog keeps paying rather than
# destroy unharvested runs. The hard deadline ignores the marker because the
# budget is exhausted at that point either way.
set -uo pipefail

BREV=/home/ubuntu/.local/bin/brev
INSTANCES=(educational-harlequin-lark-1 educational-harlequin-lark-2 educational-harlequin-lark-3)
STATE_DIR=/home/ubuntu/.local/state/brev-teardown
LOG="$STATE_DIR/watchdog.log"
IDLE_CHECKS_TO_STOP=6

# Set HARD_DEADLINE before enabling the cron/systemd timer (e.g.
# HARD_DEADLINE='2026-07-25 09:30:00 PDT'). When unset, only the idle
# watchdog runs; the hard deadline is disabled.
HARD_DEADLINE_EPOCH=9223372036854775807
if [[ -n "${HARD_DEADLINE:-}" ]]; then
  HARD_DEADLINE_EPOCH=$(date -d "$HARD_DEADLINE" +%s)
fi

mkdir -p "$STATE_DIR"
now=$(date +%s)
log() { echo "[$(date -Is)] $*" >>"$LOG"; }

teardown() {
  local inst=$1
  if "$BREV" delete "$inst" >>"$LOG" 2>&1; then
    log "$inst: delete issued"
  else
    log "$inst: DELETE FAILED"
  fi
}

live_names=$("$BREV" ls 2>/dev/null | awk '$2 ~ /RUNNING|STARTING|STOPPED/ {print $1}')

for inst in "${INSTANCES[@]}"; do
  if ! grep -qx "$inst" <<<"$live_names"; then
    log "$inst: gone, nothing to do"
    rm -f "$STATE_DIR/$inst.idle"
    continue
  fi

  if ((now >= HARD_DEADLINE_EPOCH)); then
    log "$inst: HARD DEADLINE reached, tearing down"
    teardown "$inst"
    continue
  fi

  # The bracket keeps the pattern from matching the wrapper shell that carries
  # it, which would otherwise report every instance as permanently busy.
  procs=$("$BREV" exec "$inst" "pgrep -c -f 'python.*[s]tandalone_trainer' || true" 2>/dev/null \
    | tr -dc '0-9\n' | grep -E '^[0-9]+$' | tail -1)

  if [[ -z "$procs" ]]; then
    log "$inst: unreachable, not counting toward idle"
    continue
  fi

  if ((procs > 0)); then
    log "$inst: busy ($procs trainer proc(s))"
    rm -f "$STATE_DIR/$inst.idle"
    continue
  fi

  idle=$(cat "$STATE_DIR/$inst.idle" 2>/dev/null || echo 0)
  idle=$((idle + 1))
  echo "$idle" >"$STATE_DIR/$inst.idle"
  log "$inst: idle ($idle/$IDLE_CHECKS_TO_STOP)"

  if ((idle >= IDLE_CHECKS_TO_STOP)); then
    if [[ -f "$STATE_DIR/$inst.harvested" ]]; then
      log "$inst: idle threshold reached and harvest confirmed, tearing down"
      teardown "$inst"
      rm -f "$STATE_DIR/$inst.idle"
    else
      log "$inst: idle threshold reached but NO HARVEST MARKER, still billing - leaving it up"
    fi
  fi
done

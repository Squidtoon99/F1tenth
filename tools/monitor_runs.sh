#!/usr/bin/env bash
# Poll Brev runs until all complete or timeout; print transition counts.
set -euo pipefail
HOSTS=(educational-harlequin-lark-1:oobhalf301:300000000
       educational-harlequin-lark-2:oobhalf-d2-300:300000000
       educational-harlequin-lark-3:pcplus600:600000000)
REPO="/home/shadeform/F1tenth/training/outputs/runs"
INTERVAL="${1:-900}"
DEADLINE="${2:-18000}"

start=$(date +%s)
while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  all_done=1
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed=${elapsed}s ==="
  for entry in "${HOSTS[@]}"; do
    IFS=: read -r host run target <<< "$entry"
    out=$(brev exec "$host" "grep 'transitions=' $REPO/$run/run.log 2>/dev/null | tail -1; ps aux | grep -c '[s]tandalone_trainer.*$run'" 2>&1 | tail -3)
    trans=$(echo "$out" | grep -oP 'transitions=\K[0-9]+' | tail -1 || true)
    running=$(echo "$out" | tail -1)
    if [ -n "$trans" ] && [ "$trans" -ge "$target" ]; then
      status="DONE"
    elif [ "$running" = "0" ] && [ -n "$trans" ]; then
      status="STOPPED@${trans}"
      all_done=0
    elif [ -n "$trans" ]; then
      pct=$(awk "BEGIN {printf \"%.1f\", 100*$trans/$target}")
      status="${trans}/${target} (${pct}%)"
      all_done=0
    else
      status="unknown"
      all_done=0
    fi
    echo "  $host $run: $status"
  done
  if [ "$all_done" = "1" ]; then
    echo "All runs complete."
    exit 0
  fi
  if [ "$elapsed" -ge "$DEADLINE" ]; then
    echo "Deadline reached."
    exit 1
  fi
  sleep "$INTERVAL"
done

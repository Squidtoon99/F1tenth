#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
while IFS= read -r -d '' script; do
  if ! bash -n "$script"; then
    echo "syntax error: $script" >&2
    fail=1
  fi
done < <(
  find "$REPO/tools" "$REPO/training/outputs/experiments" -name '*.sh' -print0 2>/dev/null \
    | sort -z
)
exit "$fail"

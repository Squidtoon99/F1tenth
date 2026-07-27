#!/usr/bin/env bash
# Apply a temporary training source overlay; restore tracked files on exit.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
TRAIN="$REPO/training"
MARKER="$TRAIN/.overlay-active"

OVERLAY_FILES=(
  f1tenth_env/kernel.py
  f1tenth_env/warp_env.py
  f1tenth_env/rewards.py
  standalone_trainer.py
  config.py
)

restore_overlay() {
  if [[ ! -f "$MARKER" ]]; then
    return 0
  fi
  mapfile -t files < <(
    python3 -c "import json; print('\n'.join(json.load(open('$MARKER'))['files']))"
  )
  backup="$(
    python3 -c "import json; print(json.load(open('$MARKER'))['backup'])"
  )"
  if [[ -d "$backup" ]]; then
    for rel in "${files[@]}"; do
      cp "$backup/$(basename "$rel")" "$TRAIN/$rel"
    done
    rm -rf "$backup"
  fi
  rm -f "$MARKER"
}

if [[ -f "$MARKER" ]]; then
  echo "overlay_launch: stale marker at $MARKER; restoring" >&2
  restore_overlay
fi

OVERLAY_DIR="${1:?usage: overlay_launch.sh <overlay_dir> -- trainer args...}"
shift
if [[ "${1:-}" != "--" ]]; then
  echo "usage: overlay_launch.sh <overlay_dir> -- trainer args..." >&2
  exit 2
fi
shift

BACKUP="$TRAIN/.overlay-backup-$$"
mkdir -p "$BACKUP"
applied=()
for rel in "${OVERLAY_FILES[@]}"; do
  src="$OVERLAY_DIR/$rel"
  if [[ ! -f "$src" ]]; then
    continue
  fi
  cp "$TRAIN/$rel" "$BACKUP/$(basename "$rel")"
  cp "$src" "$TRAIN/$rel"
  applied+=("$rel")
done

if [[ "${#applied[@]}" -eq 0 ]]; then
  rmdir "$BACKUP"
  echo "overlay_launch: no overlay files found under $OVERLAY_DIR" >&2
  exit 1
fi

python3 - "$MARKER" "$BACKUP" "${applied[@]}" <<'PY'
import json
import sys

marker, backup, *files = sys.argv[1:]
with open(marker, "w", encoding="utf-8") as fh:
    json.dump({"backup": backup, "files": files}, fh)
print(f"overlay_launch: active marker {marker} ({len(files)} files)", flush=True)
PY

trap restore_overlay EXIT INT TERM

cd "$TRAIN"
exec "$REPO/.venv/bin/python" standalone_trainer.py "$@"

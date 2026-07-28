#!/usr/bin/env bash
# Harvest a complete run directory from Brev as a single tar archive with checksum verification.
set -euo pipefail
HOST="${1:?usage: harvest-run.sh <brev-host> <run_id>}"
RUN_ID="${2:?usage: harvest-run.sh <brev-host> <run_id>}"
REMOTE_REPO="/home/shadeform/F1tenth"
REMOTE_RUN="$REMOTE_REPO/training/outputs/runs/$RUN_ID"
LOCAL_RUN="/home/ubuntu/projects/F1tenth/training/outputs/runs/$RUN_ID"
ARCHIVE="/tmp/${RUN_ID}.tar.gz"

echo "=== Harvesting $RUN_ID from $HOST ==="

REMOTE_SHA=$(brev exec "$HOST" "cd $REMOTE_REPO/training/outputs/runs && tar czf $ARCHIVE $RUN_ID && sha256sum $ARCHIVE" 2>&1 | grep -oE '^[a-f0-9]{64}' | head -1)
echo "Remote archive sha256: $REMOTE_SHA"

mkdir -p "$LOCAL_RUN"
brev copy "$HOST:$ARCHIVE" "$ARCHIVE"
LOCAL_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
echo "Local archive sha256:  $LOCAL_SHA"

if [ "$REMOTE_SHA" != "$LOCAL_SHA" ]; then
  echo "ERROR: archive checksum mismatch" >&2
  exit 1
fi

REMOTE_FILES=$(brev exec "$HOST" "find $REMOTE_RUN -type f | wc -l" 2>&1 | grep -oE '^[0-9]+' | head -1)
TMP=$(mktemp -d)
tar xzf "$ARCHIVE" -C "$TMP"
EXTRACTED=$(find "$TMP/$RUN_ID" -type f | wc -l)
echo "Remote files: $REMOTE_FILES, extracted: $EXTRACTED"

if [ "$REMOTE_FILES" != "$EXTRACTED" ]; then
  echo "ERROR: file count mismatch" >&2
  exit 1
fi

rm -rf "$LOCAL_RUN"
mv "$TMP/$RUN_ID" "$LOCAL_RUN"
rm -rf "$TMP" "$ARCHIVE"

REMOTE_CKPTS=$(brev exec "$HOST" "ls $REMOTE_RUN/checkpoints 2>/dev/null | wc -l" 2>&1 | grep -oE '^[0-9]+' | head -1)
LOCAL_CKPTS=$(ls "$LOCAL_RUN/checkpoints" 2>/dev/null | wc -l)
echo "Checkpoints: remote=$REMOTE_CKPTS local=$LOCAL_CKPTS"
echo "=== Harvest verified: $RUN_ID ==="

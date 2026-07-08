#!/usr/bin/env bash
# Load base + dev images from an archive produced by save_dev_images.sh.
#
# Usage: deploy/scripts/load_dev_images.sh [path/to/dev-images.tar.gz]
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ARCHIVE="${1:-deploy/snapshots/dev-images.tar.gz}"
if [ ! -f "${ARCHIVE}" ]; then
  echo "Archive not found: ${ARCHIVE}" >&2
  exit 1
fi

echo "==> Loading images from ${ARCHIVE}"
gunzip -c "${ARCHIVE}" | docker load
echo "==> Done. You can now run ./tools/dev.sh shell"

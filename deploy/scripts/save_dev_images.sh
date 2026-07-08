#!/usr/bin/env bash
# Save the prebuilt base + dev images to a portable archive so teammates can load
# them without rebuilding (no registry yet).
#
# Usage: deploy/scripts/save_dev_images.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT_DIR="${OUT_DIR:-deploy/snapshots}"   # gitignored
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/dev-images.tar.gz"

echo "==> Saving f1tenth-base:latest + f1tenth-dev:latest -> ${OUT}"
docker save f1tenth-base:latest f1tenth-dev:latest | gzip > "${OUT}"
echo "==> Done. Share ${OUT}; load it with deploy/scripts/load_dev_images.sh"

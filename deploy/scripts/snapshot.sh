#!/usr/bin/env bash
# Freeze a built image into a portable, SHA-named artifact for manual sharing/deploy
# (no registry yet). Optionally also produce an Apptainer .sif.
#
# Usage:   deploy/scripts/snapshot.sh
# Env:     IMAGE (default f1tenth-racing), GITSHA (default: current), OUT_DIR
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

IMAGE="${IMAGE:-f1tenth-racing}"
GITSHA="${GITSHA:-$(git rev-parse --short HEAD)}"
OUT_DIR="${OUT_DIR:-deploy/snapshots}"   # gitignored
mkdir -p "${OUT_DIR}"

TAR="${OUT_DIR}/${IMAGE}-${GITSHA}.tar.gz"
echo "==> Saving ${IMAGE}:${GITSHA} -> ${TAR}"
docker save "${IMAGE}:${GITSHA}" | gzip > "${TAR}"

if command -v shasum >/dev/null 2>&1; then
  SHA256="$(shasum -a 256 "${TAR}" | awk '{print $1}')"
else
  SHA256="$(sha256sum "${TAR}" | awk '{print $1}')"
fi

echo "==> Snapshot: ${TAR}"
echo "    sha256:  ${SHA256}"
echo "    Record this in deploy/releases/manifest.csv (git_sha,image_sha256,car_id,...)."

# Optional Apptainer image (uncomment if apptainer is installed):
# apptainer build "${OUT_DIR}/${IMAGE}-${GITSHA}.sif" "docker-daemon://${IMAGE}:${GITSHA}"

#!/usr/bin/env bash
# Build the GENERIC car image (no per-car config, no weights) for the Jetson arch.
# Tags it by git SHA and with a moving :develop tag.
#
# Usage:   deploy/scripts/build_image.sh
# Env:     ARCH (default linux/arm64), ROS_DISTRO (default humble), IMAGE (default f1tenth-racing)
#
# Cross-arch builds use buildx + QEMU emulation on non-arm64 hosts.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ARCH="${ARCH:-linux/arm64}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
IMAGE="${IMAGE:-f1tenth-racing}"
GITSHA="$(git rev-parse --short HEAD)"

echo "==> Building base image (deps + tooling) for ${ARCH}"
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.base \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  -t f1tenth-base:latest .

echo "==> Building generic runtime (car) image ${IMAGE}:${GITSHA}"
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.runtime \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  -t "${IMAGE}:${GITSHA}" \
  -t "${IMAGE}:develop" .

echo "==> Built ${IMAGE}:${GITSHA} (and :develop) for ${ARCH}"
echo "    Next: deploy/scripts/snapshot.sh to freeze it into a portable artifact."

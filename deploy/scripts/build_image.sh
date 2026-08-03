#!/usr/bin/env bash
# Build the GENERIC car image (no per-car config, no weights) for the Jetson arch.
# Tags it by git SHA and with a moving :develop tag.
#
# Usage:   deploy/scripts/build_image.sh
# Env:     ARCH (default linux/arm64), ROS_DISTRO (default humble), IMAGE (default f1tenth-racing)
#          RANGE_LIBC_WITH_CUDA  OFF (default) | ON — GPU raycast for particle_filter
#          CUDA_ARCH             default sm_87 (Jetson Orin)
#
# Cross-arch builds use buildx + QEMU emulation on non-arm64 hosts. RANGE_LIBC_WITH_CUDA=ON
# requires nvcc and must be built natively on the Jetson (JetPack /usr/local/cuda).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ARCH="${ARCH:-linux/arm64}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
IMAGE="${IMAGE:-f1tenth-racing}"
GITSHA="$(git rev-parse --short HEAD)"
RANGE_LIBC_WITH_CUDA="${RANGE_LIBC_WITH_CUDA:-}"
CUDA_ARCH="${CUDA_ARCH:-sm_87}"

if [ -z "${RANGE_LIBC_WITH_CUDA}" ]; then
  if [ "$(uname -m)" = "aarch64" ] && [ -d /usr/local/cuda ]; then
    RANGE_LIBC_WITH_CUDA="ON"
  else
    RANGE_LIBC_WITH_CUDA="OFF"
  fi
fi

if [ "${RANGE_LIBC_WITH_CUDA}" = "ON" ] && [ ! -d /usr/local/cuda ] && ! command -v nvcc >/dev/null 2>&1; then
  echo "RANGE_LIBC_WITH_CUDA=ON but /usr/local/cuda and nvcc are missing; build on Jetson with JetPack" >&2
  exit 1
fi

echo "==> Building base image (deps + tooling) for ${ARCH}"
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.base \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  -t f1tenth-base:latest .

echo "==> Building generic runtime (car) image ${IMAGE}:${GITSHA}"
echo "    range_libc: WITH_CUDA=${RANGE_LIBC_WITH_CUDA} CUDA_ARCH=${CUDA_ARCH}"
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.runtime \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  --build-arg RANGE_LIBC_WITH_CUDA="${RANGE_LIBC_WITH_CUDA}" \
  --build-arg CUDA_ARCH="${CUDA_ARCH}" \
  -t "${IMAGE}:${GITSHA}" \
  -t "${IMAGE}:develop" .

echo "==> Built ${IMAGE}:${GITSHA} (and :develop) for ${ARCH}"
echo "    Next: deploy/scripts/snapshot.sh to freeze it into a portable artifact."

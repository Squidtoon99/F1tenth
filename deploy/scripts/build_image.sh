#!/usr/bin/env bash
# Build the GENERIC car image (no per-car config, no weights) for the Jetson arch.
# Tags it by git SHA and with a moving :develop tag.
#
# Usage:   deploy/scripts/build_image.sh
# Env:     ARCH (default linux/arm64), ROS_DISTRO (default humble), IMAGE (default f1tenth-racing)
#          RANGE_LIBC_WITH_CUDA  OFF | ON — GPU raycast for particle_filter (auto ON on Jetson)
#          CUDA_ARCH             default sm_87 (Jetson Orin)
#          CUDA_BASE_IMAGE       NGC compile stage for CUDA path (arm64, cached on Jetson)
#
# Cross-arch builds use buildx + QEMU emulation on non-arm64 hosts. RANGE_LIBC_WITH_CUDA=ON
# compiles range_libc in an NGC stage (nvcc inside the container) and must run natively on
# linux/arm64; it is not certified under QEMU. Apple Silicon reports uname -m=arm64 (not
# aarch64) and lacks /usr/local/cuda, so CUDA stays OFF there by default.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

ARCH="${ARCH:-linux/arm64}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
IMAGE="${IMAGE:-f1tenth-racing}"
GITSHA="$(git rev-parse --short HEAD)"
CUDA_ARCH="${CUDA_ARCH:-sm_87}"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvcr.io/nvidia/pytorch@sha256:c652f021080c2d327fe7c14ae898fef1ba7dd3b756f15bff1455612f32b7cc0c}"
RANGE_LIBC_WITH_CUDA="${RANGE_LIBC_WITH_CUDA:-}"

if [ -z "${RANGE_LIBC_WITH_CUDA}" ]; then
  if [ "$(uname -m)" = "aarch64" ] && [ -d /usr/local/cuda ]; then
    RANGE_LIBC_WITH_CUDA="ON"
  else
    RANGE_LIBC_WITH_CUDA="OFF"
  fi
fi

if [ "${RANGE_LIBC_WITH_CUDA}" = "ON" ] && [ "${ARCH}" != "linux/arm64" ]; then
  echo "RANGE_LIBC_WITH_CUDA=ON requires ARCH=linux/arm64 (got ${ARCH})" >&2
  exit 1
fi

echo "==> Building base image (deps + tooling) for ${ARCH}"
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.base \
  --build-arg ROS_DISTRO="${ROS_DISTRO}" \
  -t f1tenth-base:latest .

RUNTIME_TARGET="runtime"
RUNTIME_ARGS=(
  --build-arg "ROS_DISTRO=${ROS_DISTRO}"
)
if [ "${RANGE_LIBC_WITH_CUDA}" = "ON" ]; then
  RUNTIME_TARGET="runtime-cuda"
  RUNTIME_ARGS+=(
    --build-arg "CUDA_ARCH=${CUDA_ARCH}"
    --build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}"
  )
fi

echo "==> Building generic runtime (car) image ${IMAGE}:${GITSHA}"
echo "    target:     ${RUNTIME_TARGET}"
echo "    range_libc: WITH_CUDA=${RANGE_LIBC_WITH_CUDA} CUDA_ARCH=${CUDA_ARCH}"
if [ "${RANGE_LIBC_WITH_CUDA}" = "ON" ]; then
  echo "    cuda base:  ${CUDA_BASE_IMAGE}"
fi
docker buildx build --platform "${ARCH}" --load \
  -f deploy/docker/Dockerfile.runtime \
  --target "${RUNTIME_TARGET}" \
  "${RUNTIME_ARGS[@]}" \
  -t "${IMAGE}:${GITSHA}" \
  -t "${IMAGE}:develop" .

echo "==> Built ${IMAGE}:${GITSHA} (and :develop) for ${ARCH}"
if [ "${RANGE_LIBC_WITH_CUDA}" = "ON" ]; then
  echo "    Run on Jetson with: docker run ... --runtime=nvidia ... ${IMAGE}:develop"
  echo "    (see docs/deployment.md — CPU-only images omit --runtime=nvidia)"
fi
echo "    Next: deploy/scripts/snapshot.sh to freeze it into a portable artifact."

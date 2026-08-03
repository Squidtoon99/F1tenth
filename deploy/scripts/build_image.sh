#!/usr/bin/env bash
# Build a GENERIC car image (no per-car config, no weights) for the Jetson arch.
# Tags it by git SHA. Legacy racing keeps :develop; sensor-policy uses a separate
# image name and :sensor-policy moving tag so known-good racing tags are never
# overwritten.
#
# Usage:
#   deploy/scripts/build_image.sh
#   TARGET=sensor_policy deploy/scripts/build_image.sh
#
# Env:
#   TARGET       racing (default) | sensor_policy
#   ARCH         default linux/arm64
#   ROS_DISTRO   default humble
#   IMAGE        override image name
#   BASE_IMAGE   sensor_policy only — pin NGC iGPU base (tag or @sha256:)
#   SMOKE        sensor_policy only — if 1, run smoke_sensor_policy.sh after build
#   REQUIRE_CUDA smoke only — pass through (default 1)
#   GITSHA       default: current short HEAD
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

TARGET="${TARGET:-racing}"
ARCH="${ARCH:-linux/arm64}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
GITSHA="${GITSHA:-$(git rev-parse --short HEAD)}"

case "${TARGET}" in
  racing)
    IMAGE="${IMAGE:-f1tenth-racing}"
    DOCKERFILE="deploy/docker/Dockerfile.runtime"
    MOVING_TAG="develop"
    ;;
  sensor_policy)
    IMAGE="${IMAGE:-f1tenth-sensor-policy}"
    DOCKERFILE="deploy/docker/Dockerfile.sensor_policy"
    MOVING_TAG="sensor-policy"
    if [ "${ROS_DISTRO}" != "humble" ]; then
      echo "sensor_policy is locked to ROS_DISTRO=humble" >&2
      exit 1
    fi
    # Ubuntu 22.04 + CUDA 12.6 iGPU base for ROS 2 Humble on JetPack 6.
    BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch@sha256:c652f021080c2d327fe7c14ae898fef1ba7dd3b756f15bff1455612f32b7cc0c}"
    ;;
  *)
    echo "TARGET must be racing or sensor_policy (got: ${TARGET})" >&2
    exit 1
    ;;
esac

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "docker not available in this environment (need a working daemon)." >&2
  echo "Build on a Docker-capable arm64 host (or the Jetson) with:" >&2
  echo "  TARGET=${TARGET} ARCH=${ARCH} $0" >&2
  exit 1
fi

BUILD_ARGS=()
if [ "${TARGET}" = "sensor_policy" ]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
  echo "==> Building sensor-policy runtime ${IMAGE}:${GITSHA}"
  echo "    Dockerfile: ${DOCKERFILE}"
  echo "    Base:       ${BASE_IMAGE}"
  echo "    Platform:   ${ARCH} (igpu base is arm64-only; prefer native arm64)"
  if [ "${ARCH}" != "linux/arm64" ]; then
    echo "warning: ARCH=${ARCH} — NGC -igpu base is arm64-only; build will not be certified" >&2
  fi
  docker buildx build --platform "${ARCH}" --load \
    -f "${DOCKERFILE}" \
    "${BUILD_ARGS[@]}" \
    --label "f1tenth.git_sha=${GITSHA}" \
    --label "f1tenth.target=sensor_policy" \
    --label "f1tenth.base_image=${BASE_IMAGE}" \
    -t "${IMAGE}:${GITSHA}" \
    -t "${IMAGE}:${MOVING_TAG}" .
else
  BUILD_ARGS+=(--build-arg "ROS_DISTRO=${ROS_DISTRO}")
  echo "==> Building base image (deps + tooling) for ${ARCH}"
  docker buildx build --platform "${ARCH}" --load \
    -f deploy/docker/Dockerfile.base \
    --build-arg ROS_DISTRO="${ROS_DISTRO}" \
    -t f1tenth-base:latest .

  echo "==> Building generic runtime (car) image ${IMAGE}:${GITSHA}"
  docker buildx build --platform "${ARCH}" --load \
    -f "${DOCKERFILE}" \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE}:${GITSHA}" \
    -t "${IMAGE}:${MOVING_TAG}" .
fi

if command -v sha256sum >/dev/null 2>&1; then
  IMAGE_SHA="$(docker image inspect --format '{{index .Id}}' "${IMAGE}:${GITSHA}")"
else
  IMAGE_SHA="$(docker image inspect --format '{{index .Id}}' "${IMAGE}:${GITSHA}")"
fi

echo "==> Built ${IMAGE}:${GITSHA} (and :${MOVING_TAG}) for ${ARCH}"
echo "    image id: ${IMAGE_SHA}"

if [ "${TARGET}" = "sensor_policy" ]; then
  echo "==> Version pins (from image)"
  docker run --rm --entrypoint cat "${IMAGE}:${GITSHA}" \
    /etc/f1tenth/sensor_policy_pins.json 2>/dev/null || \
    echo "    (pins file unavailable — image may not have finished runtime stage)"
  IMAGE_SIZE="$(docker image inspect --format '{{.Size}}' "${IMAGE}:${GITSHA}")"
  echo "    size bytes: ${IMAGE_SIZE}"
  if [ "${SMOKE:-0}" = "1" ]; then
    echo "==> Running non-powered smoke (SMOKE=1)"
    SMOKE_ENV=()
    if [ -n "${REQUIRE_CUDA:-}" ]; then
      SMOKE_ENV+=(-e "REQUIRE_CUDA=${REQUIRE_CUDA}")
    fi
    if [ -n "${CHECKPOINT_PATH:-}" ]; then
      SMOKE_ENV+=(-e "CHECKPOINT_PATH=${CHECKPOINT_PATH}")
      SMOKE_MOUNT=(-v "${CHECKPOINT_PATH}:${CHECKPOINT_PATH}:ro")
    else
      SMOKE_MOUNT=()
    fi
    docker run --rm --runtime nvidia "${SMOKE_ENV[@]}" "${SMOKE_MOUNT[@]}" \
      --entrypoint smoke_sensor_policy.sh "${IMAGE}:${GITSHA}"
  else
    echo "    Smoke: SMOKE=1 TARGET=sensor_policy $0 (after native arm64 build on Jetson)"
  fi
fi

echo "    Next: IMAGE=${IMAGE} GITSHA=${GITSHA} deploy/scripts/snapshot.sh"
echo "    Then: deploy/scripts/load_to_jetson.sh ... (TARGET=${TARGET})"
